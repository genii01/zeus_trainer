"""Live acceptance checks. Requires the real Compose stack; no service doubles."""
import argparse
import json
import math
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait_document(client, es_url, request_id, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.post(f'{es_url}/inference-events-*/_search', json={
            'query': {'term': {'request.id': request_id}}, 'size': 10,
        })
        response.raise_for_status()
        hits = response.json()['hits']['hits']
        completed = [hit for hit in hits if hit['_source']['event']['action'] == 'inference.completed']
        if completed:
            assert len(completed) == 1, 'Duplicate completion events for request'
            return completed[0]['_source']
        time.sleep(1)
    raise AssertionError(f'No indexed inference event within {timeout}s: {request_id}')


def query(client, prometheus, expression):
    response = client.get(f'{prometheus}/api/v1/query', params={'query': expression})
    response.raise_for_status()
    payload = response.json()
    assert payload['status'] == 'success', payload
    return payload['data']['result']


def metric_total(series):
    """Sum finite values from a Prometheus instant-vector result."""
    values = [float(item['value'][1]) for item in series]
    return sum(value for value in values if math.isfinite(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--es', default='http://127.0.0.1:9200')
    parser.add_argument('--prometheus', default='http://127.0.0.1:9090')
    args = parser.parse_args()
    with httpx.Client(timeout=35) as client:
        client.get(f'{args.api}/health/ready').raise_for_status()
        triton_expression = 'nv_inference_request_success{model="mobilenet_v2"}'
        api_expression = 'api_requests_total{method="POST",route="/v1/classify",status="200"}'
        triton_before = metric_total(query(client, args.prometheus, triton_expression))
        api_before = metric_total(query(client, args.prometheus, api_expression))
        reference = json.loads((ROOT / 'artifacts/reference.json').read_text())
        response = client.post(f'{args.api}/v1/classify', files={
            'file': ('sample.jpg', (ROOT / 'artifacts/sample.jpg').read_bytes(), 'image/jpeg'),
        })
        response.raise_for_status()
        result = response.json()
        assert response.headers['x-request-id'] == result['request_id']
        assert result['model'] == 'mobilenet_v2' and result['version'] == '1'
        top = result['predictions'][0]
        assert top['class_id'] == reference['class_id'] and top['label'] == reference['label']
        assert math.isclose(top['probability'], reference['probability'], abs_tol=1e-4)
        assert len(result['predictions']) == 5
        assert all(0 <= item['probability'] <= 1 for item in result['predictions'])
        assert all(math.isfinite(value) and value >= 0 for value in result['timing'].values())
        document = wait_document(client, args.es, result['request_id'])
        assert document['classification']['label'] == top['label']
        assert document['classification']['probability'] == top['probability']
        assert document['classification']['top_k'] == result['predictions']
        assert document['latency']['triton_roundtrip_ms'] == result['timing']['triton_roundtrip_ms']
        mapping = client.get(f'{args.es}/inference-events-*/_field_caps', params={
            'fields': 'classification.probability,latency.triton_roundtrip_ms',
        }).json()['fields']
        assert 'double' in mapping['classification.probability']
        assert 'double' in mapping['latency.triton_roundtrip_ms']
        deadline = time.monotonic() + 30
        while True:
            up = query(client, args.prometheus, 'up{job=~"fastapi|api|triton"}')
            triton_after = metric_total(query(client, args.prometheus, triton_expression))
            api_after = metric_total(query(client, args.prometheus, api_expression))
            if (len(up) >= 2 and all(float(x['value'][1]) == 1 for x in up)
                    and triton_after > triton_before and api_after > api_before):
                break
            assert time.monotonic() < deadline, (
                'Prometheus targets or request counters did not increase: '
                f'Triton {triton_before}->{triton_after}, API {api_before}->{api_after}'
            )
            time.sleep(1)
        metric_deltas = {
            'triton_success': {'before': triton_before, 'after': triton_after,
                               'delta': triton_after - triton_before},
            'api_classify_200': {'before': api_before, 'after': api_after,
                                 'delta': api_after - api_before},
        }
        report = {'status': 'passed', 'result': result, 'indexed_event': document,
                  'prometheus_up': up, 'prometheus_counter_deltas': metric_deltas}
        (ROOT / 'artifacts/smoke-result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': 'passed', 'request_id': result['request_id'],
                          'label': top['label'], 'probability': top['probability'],
                          'timing': result['timing'], 'log_indexed': True, 'metrics_up': True}, indent=2))


if __name__ == '__main__':
    main()
