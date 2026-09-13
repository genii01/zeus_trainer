'use strict';
const components = {
  client: ['로컬 클라이언트', 'curl이나 API 문서의 Try it out에서 JPEG/PNG를 업로드합니다. 응답의 request_id로 Kibana에서 같은 요청을 찾습니다.', 'http://localhost:8000/v1/classify?top_k=5', 'README.md · scripts/smoke_test.py'],
  api: ['FastAPI', '이미지를 검증하고 텐서로 변환해 Triton에 전달합니다. logits를 확률·레이블로 후처리한 뒤 응답과 구조화 로그를 생성합니다. 최대 동시 추론 4, Triton timeout 30초입니다.', 'localhost:8000 → api:8000', 'app/main.py · app/settings.py · Dockerfile'],
  triton: ['Triton Inference Server', 'mobilenet_v2 버전 1을 ONNX Runtime CPU backend로 실행합니다. explicit model control로 시작 시 모델을 로드하며 관리자 API로 load/unload할 수 있습니다. 모델 저장소는 읽기 전용입니다.', 'Docker 내부 triton:8000 (HTTP) / triton:8002 (metrics)', 'model_repository/mobilenet_v2/config.pbtxt · docker-compose.yml'],
  logger: ['구조화 JSON 로그', 'FastAPI가 INFO 성공/실패 이벤트를 stdout과 events.jsonl에 씁니다. 공유 inference-logs 볼륨을 Filebeat가 읽기 전용으로 마운트합니다. 파일 로깅은 단일 worker 기준입니다.', '/var/log/inference/events.jsonl · 10 MiB × 현재 파일 + 백업 5개', 'app/logging_config.py · app/main.py'],
  filebeat: ['Filebeat', 'filestream이 JSONL 파일을 읽고 dotted 키를 펼쳐 Elasticsearch에 보냅니다. 256-byte fingerprint로 파일을 식별하고 filebeat-data 볼륨의 registry로 읽은 위치를 유지합니다. init 성공 후 시작합니다.', 'Filebeat → http://elasticsearch:9200', 'observability/filebeat.yml'],
  elasticsearch: ['Elasticsearch', '추론 이벤트를 일자별 inference-events-* 색인에 저장합니다. probability·latency는 double, label·request.id는 keyword, top_k는 nested로 매핑합니다. es-data 볼륨에 영속 저장합니다.', 'Docker 내부 elasticsearch:9200 / 호스트 localhost:9200', 'observability/elasticsearch-template.json · scripts/bootstrap_observability.py'],
  kibana: ['Kibana', 'Elasticsearch를 조회해 요청별 표, 레이블·확률 분포, 지연시간 p50/p95, 성공/실패를 표시합니다. data view의 시간 필드는 @timestamp입니다. 초기 객체는 observability-init가 등록합니다.', 'localhost:5601 · inference-events-*', 'observability/kibana.ndjson'],
  targets: ['메트릭 제공자: FastAPI + Triton', 'FastAPI는 요청 counter·HTTP/왕복 histogram·inflight·upstream 오류를 노출합니다. Triton은 모델 성공/실패·큐·계산 누적시간을 노출합니다. 이 경로에는 개별 이미지 레이블이나 request_id가 없습니다.', 'http://api:8000/metrics · http://triton:8002/metrics', 'app/metrics.py · model_repository/mobilenet_v2/config.pbtxt'],
  prometheus: ['Prometheus', '5초마다 두 대상의 /metrics를 가져옵니다. prometheus-data 볼륨에 시계열을 7일 보관하고 Grafana의 PromQL에 응답합니다. 중단 구간 데이터는 소급 수집되지 않습니다.', 'Docker 내부 prometheus:9090 / 호스트 localhost:9090', 'observability/prometheus.yml'],
  grafana: ['Grafana', 'provisioning된 Prometheus datasource로 RPS, 오류율, p50/p95, inflight, Triton 평균 큐·계산시간을 조회합니다. grafana-data에 사용자와 설정을 보존합니다. 현재 알림 채널은 설정하지 않았습니다.', 'localhost:3000 · admin / .env의 GRAFANA_ADMIN_PASSWORD', 'observability/grafana/provisioning/ · observability/grafana/dashboards/']
};
document.querySelectorAll('[data-flow]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-flow]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelectorAll('[data-lane]').forEach(lane => lane.classList.toggle('muted', button.dataset.flow !== 'all' && lane.dataset.lane !== button.dataset.flow));
  });
});
document.querySelectorAll('[data-component]').forEach(button => {
  button.setAttribute('aria-controls', 'component-detail');
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', () => {
    const [title, body, address, files] = components[button.dataset.component];
    document.getElementById('detail-title').textContent = title;
    document.getElementById('detail-body').textContent = body;
    const fields = document.querySelectorAll('#detail-meta dd');
    fields[0].textContent = address;
    fields[1].textContent = files;
    document.querySelectorAll('[data-component]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.getElementById('component-detail').scrollIntoView({block: 'nearest', behavior: 'auto'});
  });
});
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        document.querySelectorAll('nav a').forEach(link => {
          const selected = link.getAttribute('href') === '#' + entry.target.id;
          link.classList.toggle('active', selected);
          if (selected) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    });
  }, {rootMargin: '-15% 0px -60% 0px'});
  document.querySelectorAll('section[id]').forEach(section => observer.observe(section));
}
