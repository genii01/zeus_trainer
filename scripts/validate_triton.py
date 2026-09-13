"""Compare actual Triton CPU inference with the exported PyTorch fixture."""
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

root = Path('/work')
tensor = np.load(root / 'artifacts/sample_input.npy').astype('<f4')
header = json.dumps({'id': 'onnx-equivalence-check', 'inputs': [
    {'name': 'input', 'shape': list(tensor.shape), 'datatype': 'FP32',
     'parameters': {'binary_data_size': tensor.nbytes}},
], 'outputs': [{'name': 'logits', 'parameters': {'binary_data': False}}]}).encode()
request = Request(os.getenv('TRITON_URL', 'http://triton:8000') + '/v2/models/mobilenet_v2/versions/1/infer',
                  data=header + tensor.tobytes(), headers={
                      'Content-Type': 'application/octet-stream',
                      'Inference-Header-Content-Length': str(len(header)),
                  })
with urlopen(request, timeout=30) as response:
    result = json.load(response)
output = result['outputs'][0]
assert output['name'] == 'logits' and output['shape'] == [1, 1000]
logits = np.array(output['data'], dtype=np.float32).reshape(1, 1000)
reference = json.loads((root / 'artifacts/reference.json').read_text())
expected = np.array(reference['logits'], dtype=np.float32)
np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-4)
assert int(logits.argmax()) == reference['class_id']
report = {'triton_equivalence': 'passed', 'top1': reference['label'],
          'max_abs_error': float(abs(logits - expected).max())}
(root / 'artifacts/triton-equivalence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
