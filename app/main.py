import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import secrets
from time import perf_counter
import uuid

import httpx
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .inference import InvalidTritonOutput, TritonClient, predictions
from .logging_config import configure_logging, log_event
from .metrics import ApiMetrics
from .preprocessing import ImageValidationError, UnsupportedImageFormat, preprocess_image
from .settings import Settings


class RequestBodyTooLarge(Exception):
    pass


def create_app(settings: Settings | None = None, triton_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    config = settings or Settings()
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        labels = json.loads(Path(config.labels_path).read_text())
        if not isinstance(labels, list) or len(labels) != 1000 or not all(isinstance(x, str) for x in labels):
            raise RuntimeError("labels file must contain exactly 1000 strings")
        app.state.labels = labels
        app.state.logger = configure_logging(config.log_path)
        app.state.semaphore = asyncio.Semaphore(config.max_concurrency)
        async with httpx.AsyncClient(base_url=config.triton_url, timeout=httpx.Timeout(config.request_timeout_seconds), transport=triton_transport) as client:
            app.state.triton = TritonClient(client, config.model_name, config.model_version)
            yield

    app = FastAPI(lifespan=lifespan)
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = perf_counter()
        request.state.request_id = str(uuid.uuid4())
        metrics.inflight.inc()
        is_classify = request.method == "POST" and request.url.path == "/v1/classify"
        admitted = False
        try:
            if is_classify:
                body_limit = config.max_upload_bytes + 64 * 1024
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        if int(content_length) > body_limit:
                            raise RequestBodyTooLarge
                    except ValueError:
                        response = JSONResponse(status_code=400, content={"request_id": request.state.request_id, "detail": "invalid Content-Length"})
                    else:
                        response = None
                else:
                    response = None
                if response is None:
                    semaphore = request.app.state.semaphore
                    if semaphore.locked():
                        response = JSONResponse(status_code=429, content={"request_id": request.state.request_id, "detail": "inference capacity is full"})
                        log_event(request.app.state.logger, logging.ERROR, failure_event(
                            request.state.request_id, "admission_rejected", "inference capacity is full", 429, started
                        ))
                        request.state.inference_logged = True
                    else:
                        await semaphore.acquire()
                        admitted = True
                        received = 0
                        chunks: list[bytes] = []
                        while True:
                            message = await request._receive()
                            if message["type"] == "http.request":
                                chunk = message.get("body", b"")
                                received += len(chunk)
                                if received > body_limit:
                                    raise RequestBodyTooLarge
                                chunks.append(chunk)
                                if not message.get("more_body", False):
                                    break
                            elif message["type"] == "http.disconnect":
                                break

                        replayed = False

                        async def limited_receive():
                            nonlocal replayed
                            if replayed:
                                return {"type": "http.request", "body": b"", "more_body": False}
                            replayed = True
                            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

                        request._receive = limited_receive
                        response = await call_next(request)
            else:
                response = await call_next(request)
        except RequestBodyTooLarge:
            response = JSONResponse(status_code=413, content={"request_id": request.state.request_id, "detail": "request body is too large"})
        except Exception as exc:
            event = failure_event(request.state.request_id, type(exc).__name__, "internal server error", 500, started)
            request.app.state.logger.exception("", extra={"event": event})
            request.state.inference_logged = True
            response = JSONResponse(status_code=500, content={"request_id": request.state.request_id, "detail": "internal server error"})
        finally:
            if admitted:
                request.app.state.semaphore.release()
            metrics.inflight.dec()
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        metrics.requests.labels(request.method, route_name, str(response.status_code)).inc()
        metrics.request_duration.labels(request.method, route_name).observe(perf_counter() - started)
        if request.url.path == "/v1/classify" and response.status_code >= 400 and not getattr(request.state, "inference_logged", False):
            log_event(request.app.state.logger, logging.ERROR, failure_event(
                request.state.request_id, "request_validation", "classification request rejected", response.status_code, started
            ))
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"request_id": request.state.request_id, "detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"request_id": request.state.request_id, "detail": exc.errors()})

    def failure_event(request_id: str, error_type: str, message: str, status: int, started: float) -> dict:
        total_ms = (perf_counter() - started) * 1000
        return {"event.action": "inference.failed", "event.outcome": "failure", "request.id": request_id,
                "model.name": config.model_name, "model.version": config.model_version, "latency.total_ms": total_ms,
                "event.duration": int(total_ms * 1_000_000), "http.response.status_code": status,
                "error.type": error_type, "error.message": message}

    @app.post("/v1/classify")
    async def classify(request: Request, file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=10)):
        started = perf_counter()
        request_id = request.state.request_id
        try:
            data = await file.read(config.max_upload_bytes + 1)
            if len(data) > config.max_upload_bytes:
                raise HTTPException(413, "upload exceeds 10 MiB")
            if not data:
                raise HTTPException(400, "empty image")
            preprocess_started = perf_counter()
            try:
                tensor = await run_in_threadpool(preprocess_image, data, config.max_image_pixels)
            except UnsupportedImageFormat as exc:
                raise HTTPException(415, str(exc)) from exc
            except ImageValidationError as exc:
                raise HTTPException(400, str(exc)) from exc
            preprocess_ms = (perf_counter() - preprocess_started) * 1000
            triton_started = perf_counter()
            try:
                logits = await request.app.state.triton.infer(tensor, request_id)
            except httpx.TimeoutException as exc:
                metrics.upstream_errors.labels("timeout").inc()
                raise HTTPException(504, "Triton inference timed out") from exc
            except httpx.HTTPError as exc:
                metrics.upstream_errors.labels("unavailable").inc()
                raise HTTPException(503, "Triton is unavailable") from exc
            except InvalidTritonOutput as exc:
                metrics.upstream_errors.labels("invalid_response").inc()
                raise HTTPException(502, str(exc)) from exc
            triton_seconds = perf_counter() - triton_started
            metrics.inference_duration.labels(config.model_name).observe(triton_seconds)
            post_started = perf_counter()
            try:
                result = predictions(logits, request.app.state.labels, top_k)
            except InvalidTritonOutput as exc:
                metrics.upstream_errors.labels("invalid_response").inc()
                raise HTTPException(502, str(exc)) from exc
            postprocess_ms = (perf_counter() - post_started) * 1000
            total_ms = (perf_counter() - started) * 1000
            timing = {"preprocess_ms": preprocess_ms, "triton_roundtrip_ms": triton_seconds * 1000, "postprocess_ms": postprocess_ms, "total_ms": total_ms}
            event = {"event.action": "inference.completed", "event.outcome": "success", "request.id": request_id,
                     "model.name": config.model_name, "model.version": config.model_version,
                     "classification.label": result[0]["label"], "classification.class_id": result[0]["class_id"],
                     "classification.probability": result[0]["probability"], "classification.top_k": result,
                     **{f"latency.{key}": value for key, value in timing.items()}, "event.duration": int(total_ms * 1_000_000),
                     "http.response.status_code": 200}
            log_event(request.app.state.logger, logging.INFO, event)
            return {"request_id": request_id, "model": config.model_name, "version": config.model_version, "predictions": result, "timing": timing}
        except HTTPException as exc:
            log_event(request.app.state.logger, logging.ERROR, failure_event(request_id, type(exc.__cause__).__name__ if exc.__cause__ else "request_error", str(exc.detail), exc.status_code, started))
            request.state.inference_logged = True
            raise

    @app.get("/health/live")
    async def live(request: Request):
        return {"request_id": request.state.request_id, "status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request):
        try:
            is_ready = await request.app.state.triton.ready()
        except httpx.HTTPError:
            is_ready = False
        if not is_ready:
            raise HTTPException(503, "model is not ready")
        return {"request_id": request.state.request_id, "status": "ready", "model": config.model_name, "version": config.model_version}

    @app.get("/v1/models/mobilenet_v2")
    async def model(request: Request):
        try:
            metadata = await request.app.state.triton.metadata()
            is_ready = await request.app.state.triton.ready()
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Triton is unavailable") from exc
        return {"request_id": request.state.request_id, "model": config.model_name, "version": config.model_version, "ready": is_ready, "metadata": metadata}

    def authorize(api_key: str | None) -> None:
        if not config.admin_api_key:
            raise HTTPException(404, "model controls are disabled")
        if api_key is None or not secrets.compare_digest(api_key, config.admin_api_key):
            raise HTTPException(401, "invalid API key")

    @app.post("/v1/models/mobilenet_v2/load")
    async def load_model(request: Request, x_api_key: str | None = Header(None)):
        authorize(x_api_key)
        try:
            await request.app.state.triton.load()
            if not await request.app.state.triton.ready():
                raise HTTPException(503, "model did not become ready")
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Triton is unavailable") from exc
        return {"request_id": request.state.request_id, "model": config.model_name, "version": config.model_version, "ready": True}

    @app.post("/v1/models/mobilenet_v2/unload")
    async def unload_model(request: Request, x_api_key: str | None = Header(None)):
        authorize(x_api_key)
        try:
            await request.app.state.triton.unload()
        except httpx.HTTPError as exc:
            raise HTTPException(503, "Triton is unavailable") from exc
        return {"request_id": request.state.request_id, "model": config.model_name, "version": config.model_version, "ready": False}

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
