# 실행 검증 보고서

검증일: 2026-09-13 (Asia/Seoul). 상태: 실제 CPU 모델 추론 및 로그/메트릭 파이프라인 검증 완료. 이 문서는 코드 테스트와 실제 서비스 검증을 구분합니다.

## 1. 환경과 산출물

- MacBook Pro / Apple M1 Max / ARM64 / 물리 RAM 64GB.
- Docker Desktop Linux aarch64, CPU 10개, Docker 메모리 약 15.6 GiB.
- Triton, Elasticsearch, Kibana, Filebeat, Prometheus, Grafana 모두 **arm64 이미지**. x86 에뮬레이션/GPU 미사용.
- Triton 25.08-py3, 서버 2.60.0, ONNX Runtime backend, CPU instance 1, intra-op thread 2.
- 컨테이너 digest 및 아키텍처: [images.txt](evidence/images.txt).
- 가중치·ONNX·레이블·샘플 SHA256: [manifest.json](evidence/manifest.json).

## 2. 모델 정확성 및 실제 API 결과

`docker compose run --rm model-export`가 두 번 모두 성공했습니다. ONNX checker가 통과했고 동일 입력 PyTorch와 ONNX Runtime logits의 최대 절대오차는 **6.6757e-6**입니다. `rtol=1e-3, atol=1e-4` 비교를 통과했습니다.

별도의 실제 Triton 호출에서도 1,000개 logits를 비교했습니다. 최대 절대오차 **7.1526e-6**, top-1 일치. [Triton 비교 결과](evidence/triton-equivalence.json).

공식 torchvision 전처리 fixture와 API 전처리 결과는 배열 전체가 정확히 같았습니다. 초기 구현의 resize/crop 반올림 차이는 이 비교에서 실패한 뒤 수정됐습니다.

샘플 이미지 실제 top-5:

| 순위 | 클래스 ID | 레이블 | softmax 확률 |
|---|---|---|---|
| 1 | 258 | Samoyed | 0.3091233 |
| 2 | 259 | Pomeranian | 0.0509531 |
| 3 | 261 | keeshond | 0.0159393 |
| 4 | 270 | white wolf | 0.0143294 |
| 5 | 279 | Arctic fox | 0.0108491 |

확률은 1,000개 클래스 전체에 softmax를 적용한 값입니다. top-5 확률의 합이 1이 될 필요는 없습니다.

## 3. 자동 테스트

최종 호스트 검증: `.venv/bin/python -m pytest -q` → **37 passed in 0.74s**. 극단적인 이미지 종횡비 및 Pillow 크기 오류의 회귀 테스트를 포함한다.

실제 API와 같은 Python3.11/Linux ARM64 컨테이너에서도 `docker compose run --build --rm test` → **37 passed in 0.76s**.

확인 범위: 이미지 포맷/크기/전처리, softmax 안정성/순위, NaN/비정상 tensor 거절, HTTP V2 binary 계약, timeout/불가용 처리, 인증, 요청 ID, multipart 파싱 이전의 실제 수신량 제한, threadpool 전처리 중 health 응답, JSONL 필드 및 부트스트랩 계약.

단위 테스트의 Triton HTTP transport double은 외부 통신 경계만 대체합니다. 실제 모델 검증은 위 모델 비교와 아래 smoke 테스트가 별도로 수행했습니다.

## 4. 종단 간 로그 및 메트릭

`.venv/bin/python scripts/smoke_test.py` → 성공. [응답과 실제 색인 이벤트](evidence/smoke-result.json).

- localhost → FastAPI 컨테이너 → Triton 실제 모델 → JSON 응답.
- 동일 request.id의 inference.completed 이벤트가 Filebeat를 통해 60초 이내 색인됨.
- 응답과 색인 문서의 label/probability/top-k/Triton roundtrip 값 일치.
- probability 및 latency의 Elasticsearch 타입 double 확인.
- API/Triton Prometheus scrape target UP. 단일 요청 전후 Triton 성공 카운터52→53, API 성공 카운터0→1 증가 확인.

`.venv/bin/python scripts/acceptance_test.py` → 성공. [복구 및 대시보드 검증](evidence/acceptance-result.json).

- 인증 없는 모델 제어 401.
- unload 후 readiness 503 및 inference 503, load 후 inference 200 복구.
- Filebeat 중단 중 3개 실제 요청 발생 후 재시작하여 모두 수집.
- API writer를 중지한 상태에서 로그 파일을 rename하고 API 재시작. 회전 이전과 이후의 실제 요청이 모두 수집됨. 각 검증 요청의 완료 이벤트는 1개.
- Grafana datasource 경유14개 패널 쿼리 모두 성공하고 실제 finite 숫자 확인. 예: Triton 계산 평균12.863ms, 큐 평균0.0765ms, roundtrip p95 24.25ms, 두 target 값1.
- Kibana saved dashboard 5개 패널 및 numeric 필드 자동 발견 확인.

## 5. 실제 UI 확인

Codex 브라우저로 로컬 Kibana와 Grafana를 열어 확인했습니다.

- Kibana `MobileNet inference overview`: Samoyed 레이블 분포, 확률0.3~0.4 구간의 실제 막대, 성공/실패 및 지연 차트, 요청별 표. 표에서 request.id, probability 0.309, Triton roundtrip 15.719ms 등 실제 숫자 확인.
- Grafana `MobileNet Inference Overview`: 실제 요청 발생 시 처리량 상승, API p50/p95와 Triton roundtrip 그래프 표시. datasource 연결 및 패널 렌더링 확인.

브라우저 뷰포트에 따라 Kibana 패널은 세로로 배치되며 표 오른쪽 열은 가로 스크롤로 확인합니다. 외부 서비스나 별도 시각화로 화면을 대체하지 않았습니다.

## 6. CPU 부하 측정

워밍업 5회 이후 동일 이미지 사용, 각 동시성에서 20개 요청. 클라이언트의 HTTP 왕복시간이며 업로드와 응답 전송을 포함합니다. nearest-rank percentile을 사용합니다.

| 동시성 | 성공 | p50 | p95 |
|---|---|---|---|
| 1 | 20/20 | 39.81ms | 42.51ms |
| 4 | 20/20 | 50.20ms | 74.31ms |

[원본 측정 요약](evidence/load-result.json). 샘플 수가 작고 Docker 자원/동시 작업에 영향을 받으므로 운영 SLA나 최대 처리량으로 해석하지 않습니다.

## 7. 수용 기준 대응

| PRD 기준 | 근거 |
|---|---|
| AC-01 모델 동등성 | ONNX checker + PyTorch/ONNX/Triton 1,000 logits 비교 |
| AC-02 전처리 | 공식 torchvision fixture 배열 전체 equality |
| AC-03 후처리 | softmax·finite·deterministic tie·shape 단위 테스트 |
| AC-04 실제 분류 | sample SHA256 + API top-1/확률 기준값 비교 |
| AC-05 오류 응답 | 크기/형식/top-k/파서/timeout 등 단위 테스트 및 실제 미준비503 |
| AC-06 모델 제어 | 실제401 → unload503 → load200 |
| AC-07 로그 수집 | request.id·결과 일치 및 double field caps |
| AC-08 Kibana | 객체/필드 API 확인 + 실제 UI 표/차트 |
| AC-09 Grafana | scrape/패널 쿼리 확인 + 실제 처리량·지연 UI |
| AC-10 장애 복구 | Filebeat 중단 및 회전 후 누락 없는 검증 요청 수집 |
| AC-11 반복 실행 | export/부트스트랩 재실행 성공, 데이터 볼륨 보존 |
| AC-12 부하 관측 | 동시성1/4 각각20회 성공률·p50/p95 |

## 8. 발견하고 수정한 문제

1. Kibana saved object 최초 생성에 PUT을 사용해404: POST create + overwrite로 변경, 두 번 연속 생성 성공.
2. Filebeat dotted JSON `_source` 불일치: ndjson expand_keys 활성화.
3. Filebeat fingerprint 기본값 및 동일 접두사 충돌 위험: fingerprint identity를 명시하고 UUID가 포함되는256 bytes 사용.
4. 이미지 resize/crop 정수 변환 차이: 공식 torchvision fixture에 맞게 수정.
5. multipart가 크기 검사 전에 무제한 spool되는 경로: 파싱 전 admission 및 실제 수신량 제한 추가.
6. multipart parser400의 request_id 누락: Starlette HTTPException handler 적용.
7. CPU 전처리가 이벤트 루프를 차단: 제한된 admission 안에서 threadpool 실행.
8. 확률 차트의 불완전한 축 설정으로 scale.mode 렌더링 오류: 정상 동작 차트와 같은 최소 series 설정 및 기본 축을 사용해 실제 막대 표시 확인.
9. 매우 가늘고 긴 이미지의 resize 메모리 팽창: resize 이전에도20MP 픽셀 예산 확인, Pillow decompression bomb도400으로 매핑.

API와 관측 구성의 개별 리뷰 후 수정 사항을 재검토했고, 전체 통합 리뷰에서도 중요한 구현 결함은 발견되지 않았습니다.

## 9. 적용 범위와 남은 한계

- 로컬 CPU PoC이며 공개 운영 서비스 설정이 아닙니다. Elasticsearch/Kibana 인증은 비활성화되어 있고 호스트 포트는 loopback으로 제한했습니다.
- 실제 timeout/연결 단절의 API 매핑은 단위 테스트로 검증했으며 네트워크 장애 주입을 통한30초 timeout 테스트는 수행하지 않았습니다.
- ILM 정책 등록은 확인했지만7일 경과 후 실제 삭제를 기다리는 장기 테스트는 하지 않았습니다.
- Filebeat 회전 복구는 writer 중지 후 rename 방식으로 확인했습니다. 10 MiB 자동회전을 유발하는 장시간 부하 테스트는 별도입니다.
- 단일 Uvicorn worker·모델1개·고정batch1입니다. GPU, x86 및 다중 worker 환경은 검증하지 않았습니다.
- 검증 중 의도적으로 만든 오류 로그와 초기 수집 설정 검증 데이터가 로컬 색인에 남아 있습니다. 삭제하지 않았습니다.
