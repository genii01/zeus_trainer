"""Create a versioned MobileNet ONNX model and independently verifiable fixture."""
import hashlib
import json
import platform
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnxruntime
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from validate_model import validate

ROOT = Path('/work')
ARTIFACTS = ROOT / 'artifacts'
MODEL = ROOT / 'model_repository/mobilenet_v2/1/model.onnx'
SAMPLE_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
SAMPLE_SHA256 = 'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    weights = MobileNet_V2_Weights.IMAGENET1K_V2
    model = mobilenet_v2(weights=weights).eval()
    sample = ARTIFACTS / 'sample.jpg'
    if not sample.exists():
        urlretrieve(SAMPLE_URL, sample)
    if sha256(sample) != SAMPLE_SHA256:
        raise ValueError('Reference image checksum changed; refusing to silently change test input')
    with Image.open(sample) as image:
        tensor = weights.transforms()(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = logits.softmax(dim=1)
    labels = weights.meta['categories']
    class_id = int(logits.argmax())
    (ARTIFACTS / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False))
    np.save(ARTIFACTS / 'sample_input.npy', tensor.numpy())
    (ARTIFACTS / 'reference.json').write_text(json.dumps({
        'class_id': class_id, 'label': labels[class_id],
        'probability': float(probabilities[0, class_id]), 'logits': logits.tolist(),
    }))
    torch.onnx.export(model, tensor, str(MODEL), input_names=['input'],
                      output_names=['logits'], opset_version=17, dynamo=False)
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
    manifest = {
        'model': 'mobilenet_v2', 'version': '1', 'weights': str(weights),
        'weights_url': weights.url, 'weights_sha256': sha256(checkpoint),
        'onnx_sha256': sha256(MODEL), 'labels_sha256': sha256(ARTIFACTS / 'labels.json'),
        'sample_url': SAMPLE_URL, 'sample_sha256': sha256(sample),
        'platform': platform.machine(),
        'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                     'onnx': onnx.__version__, 'onnxruntime': onnxruntime.__version__},
    }
    (ARTIFACTS / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    validate(ROOT)


if __name__ == '__main__':
    main()
