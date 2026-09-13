# MobileNet Serving Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Track steps with checkboxes.

**Goal:** 실제 MobileNet Triton 추론을 FastAPI로 호출하고 Kibana/Grafana에서 관측한다.
**Architecture:** PRD의 CPU ONNX Triton + FastAPI 및 공유 JSONL 로그 파이프라인.
**Tech Stack:** Python 3.11, FastAPI, Triton 25.08, torch/torchvision, Elastic 8.17.3, Prometheus 3.2.1, Grafana 11.6.0.
**Spec:** docs/superpowers/specs/2026-09-13-mobilenet-serving-design.md

## Global Constraints

- Model mobilenet_v2/version 1; IMAGENET1K_V2; input `input` FP32 [1,3,224,224], output `logits` FP32 [1,1000].
- GPU 없이 ARM64 CPU. FastAPI 8000, Triton 8000 internally, metrics 8002.
- API label path /models/labels.json; TRITON_URL=http://triton:8000; LOG_PATH=/var/log/inference/events.jsonl.
- JSON fields exactly follow PRD. Local loopback ports; no raw image or secret logging.
- Tests precede production logic. Unit versus live integration evidence kept separately.
- Keep implementation in current empty checkout on a feature branch; no existing implementation requires worktree isolation.

## Task 1: Model export and runtime integration (controller)

Files: scripts/export_model.py, scripts/validate_model.py, model_repository/mobilenet_v2/config.pbtxt, Dockerfile.export, docker-compose.yml, Dockerfile, requirements.txt.
Produces: artifacts/labels.json, artifacts/manifest.json, artifacts/reference.json, artifacts/sample.jpg, model_repository/mobilenet_v2/1/model.onnx.
Consumes: Task 2 app.main:app, Task 3 observability configs.

- [x] Write ONNX runtime equivalence check asserting logits allclose rtol=1e-3 atol=1e-4 and top-1 agreement. Run before artifacts exist to verify failure.
- [x] Export eval MobileNetV2 with fixed batch, opset 17 and named input/output; generate labels and checksums.
- [x] Start ARM64 Triton image 25.08-py3 with explicit model control, CPU instance and only required backend load; record any platform issues.
- [x] Run model check and a real Triton HTTP inference; compare full outputs with reference.

## Task 2: API and unit tests (API implementer)

Files: app/__init__.py, main.py, settings.py, inference.py, preprocessing.py, logging_config.py, metrics.py; tests/test_api.py, test_inference.py, test_preprocessing.py, test_logging.py.
Produces: app.main:app, ASGI endpoint contract from PRD. `preprocess_image(data: bytes) -> np.ndarray`, `predictions(logits: np.ndarray, labels: list[str], top_k: int) -> list[dict]`.
Consumes: /models/labels.json, HTTP V2 Triton protocol, settings from Global Constraints.

- [x] Write failing behavior tests: red PNG normalization, non-image rejection, stable softmax from large equal logits yielding tied indices 0 then 1, NaN rejection, log JSON numeric roundtrip.
- [x] Run pytest and preserve RED evidence; implement preprocessing, output validation and JSON logging.
- [x] Write API tests with external HTTP transport double: validate response IDs/results, auth, timeout/unavailable, invalid request outcomes.
- [x] Implement lifespan AsyncClient, endpoint validation, bounded concurrency and metrics. Use semaphore with immediate admission rejection; no unbounded waiting.
- [x] Run all owned tests. Document exact metric names for Task 3.

## Task 3: Observability (observability implementer)

Files: observability/{filebeat.yml,prometheus.yml,elasticsearch-template.json,kibana.ndjson,grafana/**}, scripts/bootstrap_observability.py.
Produces: repeatable Elastic template/ILM/Kibana saved objects; provisioned Grafana dashboard.
Consumes: JSON schema in PRD; api metrics `api_requests_total{method,route,status}`, `api_request_duration_seconds_bucket{method,route}`, `api_inference_duration_seconds_bucket{model}`, `api_inflight_requests`, `api_upstream_errors_total{reason}`; Triton standard metrics.

- [x] Create Filebeat filestream ndjson file input reading /var/log/inference/events.jsonl*, persistent registry, event.id document ID.
- [x] Implement tested mapping builder/bootstrap using standard-library HTTP; fail clearly on unsuccessful provisioning; rerun idempotently.
- [x] Provision Kibana data view and saved search/dashboard using top-1 fields; retain nested top-k mapping.
- [x] Provision Prometheus 5s scrape api:8000 and triton:8002; Grafana dashboard queries use histogram for API p95 and rates for Triton means.
- [x] Validate actual configs with service CLI and live API when Compose is available.

## Task 4: End-to-end evidence and documentation (controller)

Files: scripts/smoke_test.py, scripts/load_test.py, README.md, docs/verification.md.

- [x] Before live start, run smoke script and observe connection failure (no false-positive result).
- [x] Start Compose and bootstrap; POST sample; assert exact label/probability equality in Elastic within 60s.
- [x] Verify all Prometheus targets and actual metric counter increase; query Grafana and inspect Kibana/Grafana UI.
- [x] Test unload/load recovery, Filebeat restart catch-up and repeatability.
- [x] Warm up 5 requests then 20 requests each at concurrency 1 and 4; record success/p50/p95.
- [x] Independent review, address findings, rerun impacted tests; publish README and actual evidence including limitations.
