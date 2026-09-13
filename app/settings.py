from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    triton_url: str = field(default_factory=lambda: os.getenv("TRITON_URL", "http://triton:8000"))
    labels_path: Path = field(default_factory=lambda: Path(os.getenv("LABELS_PATH", "/models/labels.json")))
    log_path: Path = field(default_factory=lambda: Path(os.getenv("LOG_PATH", "/var/log/inference/events.jsonl")))
    admin_api_key: str | None = field(default_factory=lambda: os.getenv("ADMIN_API_KEY"))
    request_timeout_seconds: float = 30.0
    max_concurrency: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENCY", "4")))
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    model_name: str = "mobilenet_v2"
    model_version: str = "1"
