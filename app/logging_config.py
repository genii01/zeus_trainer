from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import uuid


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = dict(getattr(record, "event", {}))
        event.setdefault("@timestamp", datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"))
        event.setdefault("log.level", record.levelname)
        event.setdefault("service.name", "fastapi")
        event.setdefault("event.id", str(uuid.uuid4()))
        if record.exc_info:
            event["error.stack_trace"] = self.formatException(record.exc_info)
        return json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def configure_logging(path: Path | str, stream: bool = True) -> logging.Logger:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mobilenet.inference")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = JsonFormatter()
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def log_event(logger: logging.Logger, level: int, event: dict) -> None:
    logger.log(level, "", extra={"event": event})
