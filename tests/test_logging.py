import json
import logging
from pathlib import Path

from app.logging_config import configure_logging, log_event


def test_json_log_preserves_numeric_types_and_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = configure_logging(path, stream=False)
    log_event(logger, logging.INFO, {
        "event.action": "inference.completed",
        "event.outcome": "success",
        "request.id": "req-1",
        "classification.class_id": 4,
        "classification.probability": 0.75,
        "latency.total_ms": 12.5,
    })

    record = json.loads(path.read_text().strip())
    assert record["service.name"] == "fastapi"
    assert record["log.level"] == "INFO"
    assert record["event.action"] == "inference.completed"
    assert isinstance(record["classification.class_id"], int)
    assert isinstance(record["classification.probability"], float)
    assert record["event.id"]
    assert record["@timestamp"].endswith("Z")


def test_reconfiguring_logger_does_not_duplicate_handlers(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = configure_logging(path, stream=False)
    second = configure_logging(path, stream=False)
    assert first is second
    assert len(second.handlers) == 1
