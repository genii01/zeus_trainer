#!/usr/bin/env python3
"""Provision Elasticsearch and Kibana observability objects idempotently."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "observability" / "elasticsearch-template.json"
KIBANA_OBJECTS_PATH = ROOT / "observability" / "kibana.ndjson"


def build_ilm_policy() -> dict:
    return {
        "policy": {
            "phases": {
                "hot": {"actions": {}},
                "delete": {"min_age": "7d", "actions": {"delete": {}}},
            }
        }
    }


def build_index_template() -> dict:
    with TEMPLATE_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def load_saved_objects(path: Path = KIBANA_OBJECTS_PATH) -> list[dict]:
    objects = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid NDJSON on line {line_number}: {error}") from error
    return objects


class HttpClient:
    def __init__(self, timeout: float = 30):
        self.timeout = timeout

    def request(self, method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request_headers = {"Accept": "application/json", **(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace") if error.fp else str(error.reason)
            raise RuntimeError(f"{method} {url} failed with HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"{method} {url} failed: {error.reason}") from error
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{method} {url} returned invalid JSON") from error


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def provision(client: HttpClient, es_url: str, kibana_url: str, saved_objects: list[dict]) -> None:
    client.request("PUT", _join(es_url, "/_ilm/policy/inference-events-7d"), build_ilm_policy())
    client.request("PUT", _join(es_url, "/_index_template/inference-events"), build_index_template())

    for item in saved_objects:
        object_type = urllib.parse.quote(item["type"], safe="")
        object_id = urllib.parse.quote(item["id"], safe="")
        payload = {"attributes": item["attributes"]}
        if "references" in item:
            payload["references"] = item["references"]
        client.request(
            "POST",
            _join(kibana_url, f"/api/saved_objects/{object_type}/{object_id}?overwrite=true"),
            payload,
            {"kbn-xsrf": "true"},
        )


def main() -> int:
    es_url = os.environ.get("ES_URL", "http://elasticsearch:9200")
    kibana_url = os.environ.get("KIBANA_URL", "http://kibana:5601")
    try:
        objects = load_saved_objects()
        provision(HttpClient(), es_url, kibana_url, objects)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"observability bootstrap failed: {error}", file=sys.stderr)
        return 1
    print(f"provisioned Elasticsearch template/ILM and {len(objects)} Kibana saved objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
