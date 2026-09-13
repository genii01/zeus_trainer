"""Fail unless exported ONNX matches independently captured PyTorch outputs."""
from pathlib import Path


def validate(root=Path('.')):
    model = root / 'model_repository/mobilenet_v2/1/model.onnx'
    assert model.is_file(), 'Model export has not produced model.onnx'
    import json
    import numpy as np
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(str(model))
    reference = json.loads((root / 'artifacts/reference.json').read_text())
    tensor = np.load(root / 'artifacts/sample_input.npy')
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    logits = session.run(['logits'], {'input': tensor})[0]
    expected = np.asarray(reference['logits'], dtype=np.float32)
    np.testing.assert_allclose(logits, expected, rtol=1e-3, atol=1e-4)
    assert int(logits.argmax()) == reference['class_id']
    print(json.dumps({'onnx_check': 'passed', 'equivalence': 'passed',
                      'top1': reference['label'], 'max_abs_error': float(abs(logits-expected).max())}))


if __name__ == '__main__':
    validate()
