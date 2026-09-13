"""Small reproducible CPU benchmark, not a production capacity claim."""
import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image = (root / 'artifacts/sample.jpg').read_bytes()
    with httpx.Client(timeout=35) as client:
        def request(_):
            start = time.perf_counter()
            response = client.post(f'{args.api}/v1/classify', files={'file': ('sample.jpg', image, 'image/jpeg')})
            duration = (time.perf_counter() - start) * 1000
            return {'status': response.status_code, 'client_ms': duration,
                    'server': response.json().get('timing')}
        for i in range(5):
            assert request(i)['status'] == 200
        report = []
        for concurrency in (1, 4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                rows = list(executor.map(request, range(20)))
            times = sorted(row['client_ms'] for row in rows)
            passed = sum(row['status'] == 200 for row in rows)
            report.append({'concurrency': concurrency, 'requests': 20, 'successes': passed,
                           'p50_client_ms': times[math.ceil(.5 * len(times))-1],
                           'p95_client_ms': times[math.ceil(.95 * len(times))-1], 'rows': rows})
        (root / 'artifacts/load-result.json').write_text(json.dumps(report, indent=2))
        print(json.dumps([{k: v for k, v in row.items() if k != 'rows'} for row in report], indent=2))
        assert all(row['successes'] == 20 for row in report), 'Some benchmark requests failed'


if __name__ == '__main__':
    main()
