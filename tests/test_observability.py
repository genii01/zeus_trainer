import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "bootstrap_observability.py"


def load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("bootstrap_observability", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_index_template_maps_top_k_as_nested_and_applies_retention_policy():
    bootstrap = load_bootstrap_module()

    template = bootstrap.build_index_template()
    mappings = template["template"]["mappings"]["properties"]

    assert template["index_patterns"] == ["inference-events-*"]
    assert template["template"]["settings"]["index.lifecycle.name"] == "inference-events-7d"
    assert mappings["classification"]["properties"]["top_k"] == {
        "type": "nested",
        "properties": {
            "rank": {"type": "integer"},
            "class_id": {"type": "integer"},
            "label": {"type": "keyword"},
            "probability": {"type": "double"},
        },
    }
    assert mappings["event"]["properties"]["duration"] == {"type": "long"}
    assert mappings["error"]["properties"]["message"] == {"type": "text"}


def test_load_saved_objects_includes_data_view_search_visualizations_and_dashboard():
    bootstrap = load_bootstrap_module()

    objects = bootstrap.load_saved_objects(ROOT / "observability" / "kibana.ndjson")
    types = [item["type"] for item in objects]

    assert types.count("index-pattern") == 1
    assert types.count("search") == 1
    assert types.count("visualization") >= 4
    assert types.count("dashboard") == 1
    data_view = next(item for item in objects if item["type"] == "index-pattern")
    assert data_view["attributes"]["title"] == "inference-events-*"
    assert data_view["attributes"]["timeFieldName"] == "@timestamp"


class RecordingClient:
    def __init__(self):
        self.calls = []

    def request(self, method, url, payload=None, headers=None):
        self.calls.append((method, url, payload, headers or {}))
        return {"acknowledged": True}


def test_bootstrap_creates_or_overwrites_saved_objects_idempotently():
    bootstrap = load_bootstrap_module()
    client = RecordingClient()
    saved_objects = [
        {"type": "index-pattern", "id": "inference-events", "attributes": {"title": "inference-events-*"}},
        {"type": "dashboard", "id": "inference-overview", "attributes": {"title": "Inference Overview"}},
    ]

    bootstrap.provision(client, "http://es:9200/", "http://kibana:5601/", saved_objects)

    assert [call[:2] for call in client.calls] == [
        ("PUT", "http://es:9200/_ilm/policy/inference-events-7d"),
        ("PUT", "http://es:9200/_index_template/inference-events"),
        ("POST", "http://kibana:5601/api/saved_objects/index-pattern/inference-events?overwrite=true"),
        ("POST", "http://kibana:5601/api/saved_objects/dashboard/inference-overview?overwrite=true"),
    ]
    assert client.calls[2][3]["kbn-xsrf"] == "true"


def test_http_client_reports_unsuccessful_response(monkeypatch):
    bootstrap = load_bootstrap_module()

    def fail(_request, timeout):
        raise urllib.error.HTTPError(
            "http://es:9200/test", 503, "unavailable", {}, None
        )

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", fail)

    with pytest.raises(RuntimeError, match=r"PUT .* failed with HTTP 503"):
        bootstrap.HttpClient(timeout=2).request("PUT", "http://es:9200/test", {})
