from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class ApiMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("api_requests_total", "HTTP requests", ["method", "route", "status"], registry=self.registry)
        self.request_duration = Histogram("api_request_duration_seconds", "HTTP request duration", ["method", "route"], registry=self.registry)
        self.inference_duration = Histogram("api_inference_duration_seconds", "Triton inference roundtrip", ["model"], registry=self.registry)
        self.inflight = Gauge("api_inflight_requests", "Currently executing HTTP requests", registry=self.registry)
        self.upstream_errors = Counter("api_upstream_errors_total", "Triton upstream errors", ["reason"], registry=self.registry)
