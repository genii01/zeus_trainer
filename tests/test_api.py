from io import BytesIO
import asyncio
import importlib
import json
import threading

import httpx
import pytest
from PIL import Image

from app.main import create_app
from app.settings import Settings


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(output, "PNG")
    return output.getvalue()


def settings(tmp_path, **overrides) -> Settings:
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps([f"class-{i}" for i in range(1000)]))
    values = {"labels_path": labels, "log_path": tmp_path / "events.jsonl", "triton_url": "http://triton", "max_concurrency": 1}
    values.update(overrides)
    return Settings(**values)


async def request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_classify_returns_correlated_predictions(tmp_path) -> None:
    async def triton(request_: httpx.Request) -> httpx.Response:
        header_length = int(request_.headers["Inference-Header-Content-Length"])
        body = json.loads(request_.content[:header_length])
        logits = [0.0] * 1000
        logits[7] = 4.0
        return httpx.Response(200, json={"id": body["id"], "outputs": [{"name": "logits", "datatype": "FP32", "shape": [1, 1000], "data": logits}]})

    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(triton))
    response = await request(app, "POST", "/v1/classify?top_k=2", files={"file": ("x.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert response.headers["x-request-id"] == body["request_id"]
    assert body["model"] == "mobilenet_v2" and body["version"] == "1"
    assert body["predictions"][0]["class_id"] == 7
    assert list(body["timing"]) == ["preprocess_ms", "triton_roundtrip_ms", "postprocess_ms", "total_ms"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "file", "status"),
    [("?top_k=0", png_bytes(), 422), ("", b"broken", 400), ("", b"x" * (10 * 1024 * 1024 + 1), 413)],
)
async def test_invalid_classification_requests_include_request_id(tmp_path, query, file, status) -> None:
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(app, "POST", "/v1/classify" + query, files={"file": ("x.png", file, "image/png")})
    assert response.status_code == status
    assert response.headers["x-request-id"] == response.json()["request_id"]


@pytest.mark.asyncio
async def test_unsupported_decoded_format_is_415(tmp_path) -> None:
    output = BytesIO()
    Image.new("RGB", (4, 4)).save(output, "GIF")
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(app, "POST", "/v1/classify", files={"file": ("x.gif", output.getvalue(), "image/gif")})
    assert response.status_code == 415


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,status", [(httpx.ReadTimeout("slow"), 504), (httpx.ConnectError("down"), 503)])
async def test_upstream_failures_are_mapped(tmp_path, exc, status) -> None:
    async def triton(_: httpx.Request) -> httpx.Response:
        raise exc
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(triton))
    response = await request(app, "POST", "/v1/classify", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == status
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_invalid_triton_tensor_is_502(tmp_path) -> None:
    async def triton(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "wrong-id", "outputs": []})
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(triton))
    response = await request(app, "POST", "/v1/classify", files={"file": ("x.png", png_bytes(), "image/png")})
    assert response.status_code == 502
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_full_inference_capacity_is_rejected_without_waiting(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def triton(request_: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        header_length = int(request_.headers["Inference-Header-Content-Length"])
        request_body = json.loads(request_.content[:header_length])
        return httpx.Response(200, json={"id": request_body["id"], "outputs": [{"name": "logits", "datatype": "FP32", "shape": [1, 1000], "data": [0.0] * 1000}]})

    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(triton))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            first = asyncio.create_task(client.post("/v1/classify", files={"file": ("a.png", png_bytes(), "image/png")}))
            await entered.wait()
            second = await asyncio.wait_for(client.post("/v1/classify", files={"file": ("b.png", png_bytes(), "image/png")}), timeout=0.2)
            release.set()
            assert (await first).status_code == 200
    assert second.status_code == 429


@pytest.mark.asyncio
async def test_model_controls_require_configured_key(tmp_path) -> None:
    app = create_app(settings(tmp_path, admin_api_key=None), triton_transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    response = await request(app, "POST", "/v1/models/mobilenet_v2/load")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_blank_admin_key_disables_model_controls(tmp_path) -> None:
    app = create_app(settings(tmp_path, admin_api_key=""), triton_transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    response = await request(app, "POST", "/v1/models/mobilenet_v2/load", headers={"X-API-Key": ""})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_model_controls_reject_wrong_key_and_accept_right_key(tmp_path) -> None:
    async def triton(request_: httpx.Request) -> httpx.Response:
        if request_.url.path.endswith("/ready"):
            return httpx.Response(200)
        return httpx.Response(200)
    app = create_app(settings(tmp_path, admin_api_key="secret"), triton_transport=httpx.MockTransport(triton))
    denied = await request(app, "POST", "/v1/models/mobilenet_v2/load", headers={"X-API-Key": "wrong"})
    allowed = await request(app, "POST", "/v1/models/mobilenet_v2/load", headers={"X-API-Key": "secret"})
    assert denied.status_code == 401
    assert allowed.status_code == 200


@pytest.mark.asyncio
async def test_readiness_reflects_triton_model_readiness(tmp_path) -> None:
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    response = await request(app, "GET", "/health/ready")
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_metrics_expose_exact_api_metric_names(tmp_path) -> None:
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    response = await request(app, "GET", "/metrics")
    assert response.status_code == 200
    for name in ["api_requests_total", "api_request_duration_seconds", "api_inference_duration_seconds", "api_inflight_requests", "api_upstream_errors_total"]:
        assert name in response.text


@pytest.mark.asyncio
async def test_validation_failure_emits_one_correlated_structured_event(tmp_path) -> None:
    configured = settings(tmp_path)
    app = create_app(configured, triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(app, "POST", "/v1/classify?top_k=99", files={"file": ("x.png", png_bytes(), "image/png")})

    records = [json.loads(line) for line in configured.log_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["event.action"] == "inference.failed"
    assert records[0]["event.outcome"] == "failure"
    assert records[0]["request.id"] == response.json()["request_id"]
    assert records[0]["http.response.status_code"] == 422


@pytest.mark.asyncio
async def test_declared_request_body_over_cap_is_rejected_before_parsing(tmp_path) -> None:
    app = create_app(settings(tmp_path, max_upload_bytes=16), triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(
        app, "POST", "/v1/classify", content=b"small",
        headers={"Content-Type": "multipart/form-data; boundary=x", "Content-Length": "70000"},
    )
    assert response.status_code == 413
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_chunked_request_body_over_cap_is_rejected_while_streaming(tmp_path) -> None:
    boundary = b"boundary"
    body = (
        b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n"
        b"Content-Type: image/png\r\n\r\n" + b"x" * 70_000 + b"\r\n--" + boundary + b"--\r\n"
    )

    async def chunks():
        for offset in range(0, len(body), 1024):
            yield body[offset:offset + 1024]

    app = create_app(settings(tmp_path, max_upload_bytes=16), triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(
        app, "POST", "/v1/classify", content=chunks(),
        headers={"Content-Type": "multipart/form-data; boundary=boundary"},
    )
    assert response.status_code == 413
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_malformed_multipart_error_has_request_id(tmp_path) -> None:
    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    response = await request(
        app, "POST", "/v1/classify", content=b"malformed",
        headers={"Content-Type": "multipart/form-data"},
    )
    assert response.status_code == 400
    assert response.json()["request_id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_health_remains_responsive_while_preprocessing_runs(tmp_path, monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    real_preprocess = importlib.import_module("app.main").preprocess_image

    def held_preprocess(data, max_pixels):
        entered.set()
        release.wait(timeout=1)
        return real_preprocess(data, max_pixels)

    monkeypatch.setattr("app.main.preprocess_image", held_preprocess)

    async def triton(request_: httpx.Request) -> httpx.Response:
        header_length = int(request_.headers["Inference-Header-Content-Length"])
        body = json.loads(request_.content[:header_length])
        return httpx.Response(200, json={"id": body["id"], "outputs": [{"name": "logits", "datatype": "FP32", "shape": [1, 1000], "data": [0.0] * 1000}]})

    app = create_app(settings(tmp_path), triton_transport=httpx.MockTransport(triton))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            started = asyncio.get_running_loop().time()
            classify = asyncio.create_task(client.post("/v1/classify", files={"file": ("a.png", png_bytes(), "image/png")}))
            while not entered.is_set():
                await asyncio.sleep(0.005)
            try:
                health = await asyncio.wait_for(client.get("/health/live"), timeout=0.2)
                assert health.status_code == 200
                assert asyncio.get_running_loop().time() - started < 0.2
            finally:
                release.set()
            assert (await classify).status_code == 200
