# MobileNet + Triton + FastAPI 관측 실습

이미지 한 장을 FastAPI Docker 컨테이너에 보내면 **실제 Triton ONNX Runtime CPU 추론**을 거쳐 ImageNet top-k 레이블과 softmax 확률을 반환합니다. 요청별 결과는 Filebeat → Elasticsearch → Kibana, 운영 메트릭은 Prometheus → Grafana로 조회합니다.

[시각화 HTML 가이드](docs/index.html) · [PRD](docs/superpowers/specs/2026-09-13-mobilenet-serving-design.md) · [구현 계획](docs/superpowers/plans/2026-09-13-mobilenet-serving.md) · [검증 보고서](docs/verification.md)

## 1. 실행 환경

검증 장비: Apple M1 Max MacBook Pro, arm64, 64GB RAM. Docker Desktop에는 약 15.6 GiB가 할당되어 있습니다. Triton을 포함한 이미지는 ARM64로 실행했고 GPU나 x86 에뮬레이션을 사용하지 않았습니다. NVIDIA GPU가 없어도 CPU instance group으로 동작합니다. 최초 이미지 다운로드에는 수 GB 이상의 디스크와 인터넷 연결이 필요합니다.

버전: Triton 25.08 / server 2.60.0, PyTorch 2.6.0, torchvision 0.21.0, ONNX Runtime export 검사 1.21.0, FastAPI 0.115.11, Elastic Stack 8.17.3, Prometheus 3.2.1, Grafana 11.6.0. API는 Python 3.11 Docker 이미지입니다. Python 의존성은 lock 파일로 고정했습니다.

## 2. 처음부터 실행

프로젝트 루트에서 실행합니다.

```bash
python3 scripts/init_env.py
docker compose build model-export api
docker compose run --rm model-export
docker compose up -d
docker compose ps
```

`model-export`는 MobileNet IMAGENET1K_V2 가중치를 다운로드하고 모델·레이블·샘플·PyTorch 기준 출력·SHA256 manifest를 생성합니다. 샘플은 PyTorch Hub의 dog.jpg이며 체크섬이 달라지면 중단합니다. ONNX checker와 PyTorch/ONNX 수치 비교가 성공해야 종료 코드 0이 됩니다.

`observability-init`는 Elasticsearch template/ILM 및 Kibana 객체를 생성한 뒤 종료합니다. **Exited (0)은 정상**입니다. Filebeat는 이 작업이 성공해야 시작합니다. API 준비 상태는 다음으로 확인합니다.

```bash
curl --fail http://localhost:8000/health/ready
```

모델 export와 부트스트랩은 다시 실행할 수 있습니다. 모델을 다시 export할 때는 추론 요청을 중지하고 이후 Triton을 재시작해 디스크 모델과 로드된 모델을 일치시킵니다.

## 3. 로컬 이미지 분류

```bash
curl --fail-with-body \
  -F 'file=@artifacts/sample.jpg' \
  'http://localhost:8000/v1/classify?top_k=5'

# 자신의 JPEG 또는 PNG 사용
curl --fail-with-body \
  -F 'file=@/absolute/path/to/image.jpg' \
  'http://localhost:8000/v1/classify?top_k=3'
```

샘플의 실제 top-1은 `Samoyed`, 확률은 약 `0.3091233`입니다. 아래는 응답 형태를 줄인 예시이며 시간은 요청마다 달라집니다.

```json
{
  "request_id": "서버가-생성한-UUID",
  "model": "mobilenet_v2",
  "version": "1",
  "predictions": [
    {"rank": 1, "class_id": 258, "label": "Samoyed", "probability": 0.3091233}
  ],
  "timing": {
    "preprocess_ms": 20.0,
    "triton_roundtrip_ms": 15.0,
    "postprocess_ms": 0.2,
    "total_ms": 35.4
  }
}
```

지원: JPEG/PNG, 이미지 파일 10 MiB 이하, 20 megapixels 이하(리사이즈 후에도 같은 픽셀 예산 적용). top_k는 1~10, 기본 5입니다. multipart 전체는 파일 한도 + 64 KiB로 제한하며 실제 스트림 바이트를 파싱 전에 제한합니다. 동시에 4개 요청을 허용하며 초과 요청은 429입니다.

400: 빈/손상 이미지 또는 잘못된 multipart, 413: 크기 초과, 415: 이미지 형식 미지원, 422: top_k 등 입력 오류, 502: 비정상 Triton 출력, 503: 모델/서버 미준비, 504: Triton 타임아웃. 실패 응답도 `request_id`와 `X-Request-ID`를 반환합니다.

## 4. 서버 및 모델 제어

| 목적 | 주소 |
|---|---|
| API 문서와 Try it out | http://localhost:8000/docs |
| API 생존 / 모델 준비 | /health/live, /health/ready |
| 모델 메타데이터 | /v1/models/mobilenet_v2 |
| 모델 로드 / 언로드 | POST /v1/models/mobilenet_v2/load, /unload |

`scripts/init_env.py`가 생성한 `.env`의 `ADMIN_API_KEY`를 `X-API-Key` 헤더로 보내야 모델을 제어할 수 있습니다. 키가 없거나 빈 값이면 제어 기능은 비활성화됩니다. 다음은 키를 출력하지 않는 예시입니다.

```bash
set -a
source .env
set +a
curl --fail-with-body -X POST \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  http://localhost:8000/v1/models/mobilenet_v2/unload
curl --fail-with-body -X POST \
  -H "X-API-Key: ${ADMIN_API_KEY}" \
  http://localhost:8000/v1/models/mobilenet_v2/load
```

언로드 상태에서도 `/health/live`는 200이며 `/health/ready`와 추론은 503입니다. 로드하면 다시 준비 상태로 바뀝니다. API에 Docker socket을 마운트하지 않으며 컨테이너 자체의 시작/종료는 Compose로 수행합니다.

## 5. Kibana: label, probability, latency 조회

[Kibana 대시보드](http://localhost:5601/app/dashboards#/view/inference-overview) · [Discover 저장 검색](http://localhost:5601/app/discover#/view/inference-events-discover)

초기 접속이 Elasticsearch 시작 안내 화면으로 이동하면 대시보드 링크를 다시 여세요. Data view는 `inference-events-*`, 시간 필드는 `@timestamp`, 기본 범위는 최근 24시간입니다.

KQL 검색 예시:

```text
request.id : "응답에서-복사한-UUID"
classification.label : "Samoyed"
classification.probability < 0.5
event.outcome : "failure"
```

대시보드는 레이블 분포, 확률 분포, p50/p95 전체 처리 지연, 성공/실패, 요청별 표를 제공합니다. 표에서 문서를 펼치면 nested `classification.top_k`의 순위·클래스·레이블·확률 쌍을 볼 수 있습니다.

### Logger를 어디에 어떻게 설정했는가

핵심은 **Triton 응답을 후처리하는 FastAPI에서 추론 완료 이벤트를 기록하는 것**입니다. Triton 운영 로그 옵션만으로 레이블·확률 업무 로그가 생성되지는 않습니다.

`app/logging_config.py`의 `JsonFormatter`는 한 이벤트를 한 줄 JSON으로 출력합니다. `app/main.py`의 성공 경로가 아래와 같은 필드를 보냅니다.

```python
event = {
    "event.action": "inference.completed",
    "event.outcome": "success",
    "request.id": request_id,
    "model.name": "mobilenet_v2",
    "model.version": "1",
    "classification.label": result[0]["label"],
    "classification.probability": result[0]["probability"],
    "classification.top_k": result,
    "latency.triton_roundtrip_ms": triton_seconds * 1000,
    "latency.total_ms": total_ms,
    "event.duration": int(total_ms * 1_000_000),
}
log_event(logger, logging.INFO, event)
```

Formatter가 UTC `@timestamp`, `log.level`, `service.name`, 이벤트 UUID를 추가합니다. 예외 stack trace도 줄바꿈이 이스케이프된 JSON 문자열입니다. NaN은 허용하지 않으며 이미지 바이트·파일명·API 키는 기록하지 않습니다.

동일 이벤트를 stdout과 `/var/log/inference/events.jsonl`에 기록하지만 **Filebeat는 파일만 수집**합니다. `RotatingFileHandler`는 10 MiB마다 회전하고 백업 5개를 보존합니다. 단일 worker를 전제로 하며 worker 수를 늘리기 전에 로깅 방식을 바꿔야 합니다.

`observability/filebeat.yml`의 중요 설정:

```yaml
parsers:
  - ndjson:
      target: ""
      expand_keys: true
      add_error_key: true
      overwrite_keys: true
prospector.scanner.fingerprint:
  enabled: true
  offset: 0
  length: 256
file_identity.fingerprint: ~
```

`expand_keys`는 dotted 키를 Elasticsearch `_source`의 중첩 객체로 만듭니다. fingerprint에는 요청 UUID가 포함되어 로그 회전 시 같은 성공 이벤트 접두사끼리 충돌하지 않습니다. `event.id`를 문서 ID로 복사해 재전송 중복을 억제합니다. registry는 볼륨에 유지합니다. 수집은 exactly-once 보장이 아니며 Filebeat 중단이 회전 로그 보관 용량을 초과하면 손실될 수 있습니다.

`observability/elasticsearch-template.json`이 probability와 latency를 `double`, label과 request.id를 `keyword`, top_k를 `nested`로 고정합니다. ILM은 일자별 색인을 생성한 뒤 7일이 지난 색인을 삭제합니다.

### 지연시간을 해석하는 법

| 필드/메트릭 | 측정 구간 |
|---|---|
| latency.preprocess_ms | 이미지 디코딩·변환·정규화 및 threadpool 대기 |
| latency.triton_roundtrip_ms | FastAPI의 Triton 요청 직전부터 응답 수신·검증까지: 네트워크·큐·모델 실행 포함 |
| latency.postprocess_ms | softmax·top-k·레이블 매핑 |
| latency.total_ms | classify endpoint 시작부터 결과 생성까지; multipart 수신/파싱 및 최종 응답 전송 제외 |
| api_request_duration_seconds | HTTP middleware 처리 시간: 업로드 수신/파싱 포함, 최종 응답 전송 제외 |
| nv_inference_compute_infer_duration_us | Triton 내부 모델 계산 누적시간, microseconds |

API의 roundtrip을 순수 모델 계산시간으로 부르지 않습니다. probability는 1,000개 logits의 softmax 점수이며 실제 정확도나 보정된 신뢰도를 보장하지 않습니다.

## 6. Grafana: FastAPI와 Triton 모니터링

[Grafana 대시보드](http://localhost:3000/d/inference-overview) · [Prometheus](http://localhost:9090)

Grafana 사용자명은 `admin`, 비밀번호는 `.env`의 `GRAFANA_ADMIN_PASSWORD`입니다. 로그인 후 대시보드 링크를 다시 열면 됩니다. 모든 호스트 공개 포트는 `127.0.0.1`로 제한했습니다. Elasticsearch/Kibana는 이 로컬 실습에서 인증을 비활성화했으므로 공개 네트워크 배포용 설정이 아닙니다.

Prometheus는 5초마다 `api:8000/metrics`, `triton:8002/metrics`를 수집합니다. Grafana datasource와 대시보드는 자동 provisioning됩니다. 요청량, 오류율, API p50/p95, Triton 왕복 p95, 진행 중 요청, Triton 성공/실패·큐·계산 평균지연과 scrape 상태를 봅니다.

Triton 계산 평균 ms의 예시:

```promql
rate(nv_inference_compute_infer_duration_us[1m])
/
rate(nv_inference_request_success[1m])
/ 1000
```

API p95는 histogram bucket으로 계산합니다. Triton 누적시간 카운터로 개별 요청 p95를 만들지 않습니다. 메트릭 label에는 request_id나 이미지 분류 label을 넣지 않아 시계열 수가 요청마다 증가하지 않습니다. 요청이 없는 구간의 지연값은 No data/NaN일 수 있습니다. 실제 요청을 연속으로 보내고 최근 15분 범위에서 확인하세요.

## 7. 자동 검증

API와 같은 Python 3.11/Linux ARM64 환경에서 단위 테스트만 실행하려면 모델 export 후 `docker compose run --build --rm test`를 사용합니다.

호스트 Python 3.10 이상에서:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt -c requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/acceptance_test.py
.venv/bin/python scripts/load_test.py
```

`smoke_test.py`: 실제 API 결과와 PyTorch top-1/확률 비교, 동일 request.id 로그를 60초 내 확인, 숫자 매핑, Prometheus target 및 Triton 성공 카운터 검사.

`acceptance_test.py`: 모델 unload/load, Filebeat 중단 중 요청과 재시작 수집, API writer를 잠시 중지한 안전한 로그 파일 rename 회전, Grafana 패널 쿼리 및 Kibana 객체/필드 확인. 이 스크립트는 API와 Filebeat를 잠시 재시작하며 데이터 볼륨을 삭제하지 않습니다.

`load_test.py`: 워밍업 5회 후 동시성 1/4에서 각각 20회, 클라이언트 관측 p50/p95 기록. 처리 용량의 운영 보장이 아닌 작은 로컬 비교 실험입니다.

PyTorch와 실제 Triton의 전체 logits 비교:

```bash
docker compose run --rm \
  -v "$PWD/scripts:/work/scripts:ro" \
  model-export python scripts/validate_triton.py
```

검증 결과는 `artifacts/*-result.json`, 모델 출처는 `artifacts/manifest.json`에 저장됩니다. 공식 전처리 테스트는 export된 fixture를 필요로 하므로, 전체 검증은 반드시 export 후 실행합니다. 모델을 준비하지 않았다면 테스트가 실패합니다.

## 8. Docker 종료·재실행·변경 반영

아래 명령은 **이 저장소의 루트 디렉터리**에서 실행합니다. Docker Desktop이 먼저 실행되어 있어야 합니다. 일반적으로는 `down` → `up -d`를 사용하면 됩니다.

### 8.1 일상적인 종료와 다시 실행 (데이터 보존)

```bash
# 컨테이너와 Compose 네트워크 제거. 이름 있는 볼륨은 보존
docker compose down

# 다음 작업 때 동일한 프로젝트 디렉터리에서 다시 생성·시작
docker compose up -d

# 시작 상태 확인: 초기화 작업까지 보려면 -a 사용
docker compose ps -a

# Triton에 모델이 실제 준비되었는지 확인
curl --fail http://localhost:8000/health/ready

# 실제 추론 확인
curl --fail-with-body -F 'file=@artifacts/sample.jpg' \
  'http://localhost:8000/v1/classify?top_k=5'
```

시작 직후에는 준비 중이라 curl이 실패할 수 있습니다. `docker compose ps -a`에서 API/Triton/Elasticsearch/Kibana의 health를 확인한 후 다시 호출하세요. `observability-init`의 **Exited (0)은 정상 완료**이고, 0이 아닌 종료 코드는 초기화 실패입니다. 모델이 언로드되었으면 health/live가 정상이더라도 health/ready가 503이므로 4절의 load를 호출합니다.

보존되는 데이터:

| 저장 위치 | 내용 | down 이후 |
|---|---|---|
| 호스트 `model_repository/`, `artifacts/` | ONNX 모델·레이블·샘플·검증 출력 | 유지 |
| 호스트 `.env` | 관리자 키·Grafana 초기 비밀번호 | 유지 |
| `inference-logs` 볼륨 | FastAPI JSONL 및 회전 로그 | 유지 |
| `es-data` 볼륨 | Elasticsearch 색인·Kibana 저장 객체 | 유지 |
| `filebeat-data` 볼륨 | 파일 읽기 위치 registry | 유지 |
| `prometheus-data` 볼륨 | 수집된 메트릭 | 유지 |
| `grafana-data` 볼륨 | Grafana DB·사용자 설정 | 유지 |
| `torch-cache` 볼륨 | export 가중치 다운로드 캐시 | 유지 |

모델과 artifacts가 남아 있다면 재export할 필요가 없습니다. `.env`도 매번 새로 생성하지 않습니다. 현재 Compose에는 `restart` 정책이 없으므로 Docker Desktop/호스트 재시작 뒤에는 `docker compose up -d`를 실행하고 준비 상태를 확인합니다.

### 8.2 잠시 중지하거나 특정 모듈만 재시작

```bash
# 컨테이너를 지우지 않고 잠시 중지/재개
docker compose stop
docker compose start

# Filebeat만 중지/재개: 중단 중 로그는 회전 보관량 안에서 누적
docker compose stop filebeat
docker compose start filebeat

# API 프로세스 재시작 (새 코드 이미지/.env 반영 용도가 아님)
docker compose restart api
```

`start`는 기존 컨테이너만 시작합니다. 이미 `down`을 했다면 컨테이너가 없으므로 `up -d`를 사용해야 합니다. Filebeat가 재개되어도 보관 용량을 넘어 삭제된 로그는 복구할 수 없습니다. Prometheus 중지 중의 관측치는 소급 수집되지 않습니다.

### 8.3 수정한 코드·설정을 적용하는 명령

| 변경 대상 | 적용 명령 | 주의 사항 |
|---|---|---|
| API Python 코드, requirements, Dockerfile | `docker compose up -d --build api` | API 코드는 이미지에 포함되므로 rebuild 필요 |
| `.env`의 컨테이너 환경변수 | `docker compose up -d --force-recreate api grafana` | restart만으로 새 환경변수 적용 안 됨 |
| Filebeat 설정 | `docker compose restart filebeat` | 로그·registry 볼륨은 유지 |
| Prometheus 설정 | `docker compose restart prometheus` | 수집 설정 재로딩 |
| Grafana provisioning/dashboard 파일 | `docker compose restart grafana` | 저장소 정의를 다시 읽음 |
| ES template, ILM, Kibana saved objects | `docker compose run --rm observability-init` | 초기화 재실행. 기존 ES 필드 타입을 변경하는 명령은 아님 |
| Compose의 이미지/포트/메모리 등 | `docker compose up -d` | 변경된 서비스 컨테이너 재생성 |

Grafana의 관리자 비밀번호는 최초 실행 시 DB에 생성됩니다. 기존 `grafana-data`가 있으면 `.env`의 초기 비밀번호를 바꿔도 기존 계정 비밀번호는 바뀌지 않습니다. Grafana의 사용자 설정에서 별도로 변경하세요. Elasticsearch template의 타입 변경은 기존 색인에 소급 적용되지 않으므로 새 색인/재색인이 필요할 수 있습니다.

모델을 다시 export할 때는 요청을 멈추고 다음 순서를 따릅니다. export가 성공했을 때만 서버를 다시 올립니다.

```bash
docker compose stop api triton
docker compose run --rm model-export
# 위 명령이 종료 코드 0일 때만 실행
docker compose up -d triton api
curl --fail http://localhost:8000/health/ready
```

### 8.4 모든 저장 데이터를 삭제하고 초기화

**다음 명령은 삭제 작업입니다.** Elasticsearch 요청 이력, JSON 로그, Filebeat registry, Prometheus 메트릭, Grafana 사용자 설정과 torch 캐시가 삭제됩니다. 필요한 데이터는 먼저 백업하세요. 일반 종료에는 사용하지 않습니다.

```bash
docker compose down -v

# 호스트의 모델·artifacts·.env가 남아 있다는 전제
docker compose up -d
docker compose ps -a
curl --fail http://localhost:8000/health/ready
```

`-v`도 호스트의 `model_repository/`, `artifacts/`, `.env`와 Docker 이미지는 지우지 않습니다. 호스트 모델 파일까지 없는 새 checkout이라면 2절의 초기 export부터 실행하세요. Kibana 초기 객체와 Grafana provisioning은 다시 생성되지만 삭제된 과거 데이터는 되돌아오지 않습니다.

### 8.5 로그와 장애 진단

```bash
# 전체 상태와 최근 로그
docker compose ps -a
docker compose logs --tail=100 api triton filebeat observability-init

# 실시간 로그 (Ctrl+C는 로그 보기만 종료하며 서버는 유지)
docker compose logs -f --tail=50 api triton

# 초기화 실패 원인을 수정한 후 재실행
docker compose run --rm observability-init
docker compose up -d filebeat
```

- Triton 미준비: export 성공 여부, `model_repository/mobilenet_v2/1/model.onnx`, Triton 로그, 모델 load 상태를 확인합니다.
- Kibana 로그 없음: 실제 분류 요청을 보내고 최대 60초 기다립니다. Filebeat/init 상태, data view `inference-events-*`, 시간 범위, request.id 필터를 확인합니다.
- Grafana No data: [Prometheus Targets](http://localhost:9090/targets)에서 api/triton UP을 확인한 뒤 연속 추론을 보내세요. 요청이 없는 구간의 지연 통계는 비어 있을 수 있습니다.
- API 로그 권한 오류: `inference-logs`는 API 사용자 UID10001이 기록합니다. 다른 사용자로 만든 기존 볼륨은 소유권을 확인하세요.
- Kibana의 “Your data is not secure”: 로컬 인증 비활성화 구성에 대한 안내입니다. 외부 운영 배포에는 인증/TLS/네트워크 접근 제어가 필요합니다.

## 9. 시각화 문서와 GitHub Pages

[시스템 구조 HTML](docs/index.html)은 전체 구조도, 구성 요소별 설명, 요청 순서, JSON logger, KQL·PromQL, 운영·복구 절차를 포함합니다. 경로 강조 버튼과 구성 요소 상세 보기를 제공합니다. 외부 CDN이나 빌드 도구 없이 작동합니다.

로컬 미리보기:

```bash
python3 -m http.server 8088 --bind 127.0.0.1 --directory docs
# 브라우저에서 http://localhost:8088 접속
# Ctrl+C로 문서 서버만 종료
```

GitHub Pages 주소는 배포 성공 후 **https://genii01.github.io/zeus_trainer/** 입니다. 아직 배포 전에는 404일 수 있습니다.

1. 저장소 Settings → Pages → Build and deployment → Source를 **GitHub Actions**로 선택합니다.
2. 문서 PR을 `master`에 병합합니다. `.github/workflows/pages.yml`이 정적 파일을 검증하고 `docs/`를 배포합니다.
3. Actions의 **Documentation Pages** 실행 결과와 `github-pages` environment URL을 확인합니다.
4. 이후 `master`의 `docs/` 변경 때 자동 배포됩니다. 필요하면 master에서 Run workflow로 재배포할 수 있습니다. PR에서는 검증만 하고 배포하지 않습니다.

Pages는 설명 HTML을 호스팅합니다. Docker 컨테이너나 모델을 실행하지 않습니다. 문서의 localhost 링크는 페이지를 보는 컴퓨터에 연결되며 실시간 상태 표시는 하지 않습니다.

배포 구성은 [GitHub 공식 custom workflow 안내](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)를 따릅니다. Pages의 공개 범위는 저장소/계정 설정에 따라 확인하세요.
