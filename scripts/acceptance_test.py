"""Exercise local model controls and recovery; preserves all Docker volumes."""
import json
import math
import subprocess
import time
from pathlib import Path

import httpx

from smoke_test import wait_document

ROOT = Path(__file__).resolve().parents[1]


def finite_values(result):
    values = []
    for series in result:
        value = float(series['value'][1])
        if math.isfinite(value):
            values.append(value)
    return values


def requires_finite_series(panel_title, expression):
    if panel_title in {'FastAPI error rate', 'Upstream errors / second'}:
        return False
    if panel_title == 'Triton successes and failures / second' and 'failure' in expression:
        return False
    return True


def compose(*args):
    subprocess.run(['docker', 'compose', *args], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    secrets = dict(line.split('=', 1) for line in (ROOT / '.env').read_text().splitlines()
                   if line and not line.startswith('#'))
    api = 'http://127.0.0.1:8000'
    es = 'http://127.0.0.1:9200'
    image = (ROOT / 'artifacts/sample.jpg').read_bytes()
    report = {}
    with httpx.Client(timeout=35) as client:
        def classify():
            response = client.post(f'{api}/v1/classify', files={'file': ('sample.jpg', image, 'image/jpeg')})
            response.raise_for_status()
            return response.json()

        control = {'X-API-Key': secrets['ADMIN_API_KEY']}
        assert client.post(f'{api}/v1/models/mobilenet_v2/unload').status_code == 401
        try:
            client.post(f'{api}/v1/models/mobilenet_v2/unload', headers=control).raise_for_status()
            assert client.get(f'{api}/health/ready').status_code == 503
            unavailable = client.post(f'{api}/v1/classify', files={'file': ('sample.jpg', image, 'image/jpeg')})
            assert unavailable.status_code == 503
        finally:
            client.post(f'{api}/v1/models/mobilenet_v2/load', headers=control).raise_for_status()
        restored = classify()
        report['model_control'] = {'unauthorized': 401, 'unloaded_readiness': 503,
                                   'unloaded_inference': 503, 'restored_request_id': restored['request_id']}
        compose('stop', 'filebeat')
        try:
            buffered = [classify() for _ in range(3)]
        finally:
            compose('start', 'filebeat')
        for result in buffered:
            wait_document(client, es, result['request_id'])
        report['filebeat_catchup'] = {'passed': True, 'request_ids': [r['request_id'] for r in buffered]}
        # Stop writers before renaming to avoid dangling writer descriptors.
        compose('stop', 'filebeat')
        try:
            before_rotation = classify()
            compose('stop', 'api')
            try:
                rotation = "from pathlib import Path; import uuid; p=Path('/var/log/inference/events.jsonl'); p.rename(p.with_name(p.name+'.acceptance-'+uuid.uuid4().hex))"
                compose('run', '--rm', '--no-deps', 'api', 'python', '-c', rotation)
            finally:
                compose('start', 'api')
            deadline = time.monotonic() + 30
            while True:
                try:
                    if client.get(f'{api}/health/ready').status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() < deadline, 'API restart failed'
                time.sleep(.5)
            after_rotation = classify()
        finally:
            compose('start', 'filebeat')
        for result in (before_rotation, after_rotation):
            wait_document(client, es, result['request_id'])
        report['filebeat_rotation'] = {'passed': True, 'request_ids': [before_rotation['request_id'], after_rotation['request_id']]}

        grafana_auth = ('admin', secrets['GRAFANA_ADMIN_PASSWORD'])
        dashboard = client.get('http://127.0.0.1:3000/api/dashboards/uid/inference-overview', auth=grafana_auth)
        dashboard.raise_for_status()
        recent = [classify() for _ in range(3)]
        report['grafana_probe_requests'] = [item['request_id'] for item in recent]
        panels = dashboard.json()['dashboard']['panels']
        deadline = time.monotonic() + 45
        while True:
            checks = []
            missing_required = []
            for panel in panels:
                for target in panel.get('targets', []):
                    expression = target.get('expr')
                    if not expression:
                        continue
                    response = client.get('http://127.0.0.1:3000/api/datasources/proxy/uid/prometheus/api/v1/query',
                                          auth=grafana_auth, params={'query': expression})
                    response.raise_for_status()
                    payload = response.json()
                    assert payload['status'] == 'success'
                    values = finite_values(payload['data']['result'])
                    required = requires_finite_series(panel['title'], expression)
                    checks.append({'panel': panel['title'], 'ref_id': target.get('refId'),
                                   'query_valid': True, 'finite_required': required,
                                   'finite_series': len(values), 'sample_values': values[:3]})
                    if required and not values:
                        missing_required.append(f"{panel['title']} ({target.get('refId')})")
            if not missing_required:
                break
            assert time.monotonic() < deadline, f'Grafana queries lack finite series: {missing_required}'
            time.sleep(2)
        assert checks, 'No Grafana panel queries were checked'
        report['grafana_panels'] = checks
        kibana = client.get('http://127.0.0.1:5601/api/saved_objects/dashboard/inference-overview')
        kibana.raise_for_status()
        assert len(kibana.json()['references']) == 5
        fields = client.get('http://127.0.0.1:5601/api/data_views/data_view/inference-events')
        fields.raise_for_status()
        assert 'classification.probability' in fields.json()['data_view']['fields']
        report['kibana'] = {'dashboard_panels': 5, 'numeric_fields_discovered': True}
    (ROOT / 'artifacts/acceptance-result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
