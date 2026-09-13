import json
from typing import Any

import httpx
import numpy as np


class InvalidTritonOutput(ValueError):
    pass


def predictions(logits: np.ndarray, labels: list[str], top_k: int) -> list[dict[str, Any]]:
    array = np.asarray(logits)
    if array.shape != (1, 1000):
        raise InvalidTritonOutput(f"invalid logits shape: {array.shape}")
    if array.dtype != np.float32:
        raise InvalidTritonOutput(f"invalid logits datatype: {array.dtype}")
    if not np.isfinite(array).all():
        raise InvalidTritonOutput("logits must be finite")
    if len(labels) != 1000:
        raise InvalidTritonOutput("labels must contain 1000 entries")
    values = array[0].astype(np.float64)
    exp = np.exp(values - values.max())
    probabilities = exp / exp.sum()
    indices = np.lexsort((np.arange(1000), -probabilities))[:top_k]
    return [
        {"rank": rank, "class_id": int(index), "label": labels[index], "probability": float(probabilities[index])}
        for rank, index in enumerate(indices, 1)
    ]


class TritonClient:
    def __init__(self, http: httpx.AsyncClient, model_name: str = "mobilenet_v2", version: str = "1") -> None:
        self.http = http
        self.model_name = model_name
        self.version = version

    async def infer(self, tensor: np.ndarray, request_id: str) -> np.ndarray:
        array = np.asarray(tensor)
        if array.shape != (1, 3, 224, 224) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("input tensor must be finite FP32 [1,3,224,224]")
        binary = np.ascontiguousarray(array, dtype="<f4").tobytes()
        request_header = json.dumps(
            {"id": request_id,
             "inputs": [{"name": "input", "shape": [1, 3, 224, 224], "datatype": "FP32", "parameters": {"binary_data_size": len(binary)}}],
             "outputs": [{"name": "logits", "parameters": {"binary_data": True}}]},
            separators=(",", ":"),
        ).encode("utf-8")
        response = await self.http.post(
            f"/v2/models/{self.model_name}/versions/{self.version}/infer",
            content=request_header + binary,
            headers={"Content-Type": "application/octet-stream", "Inference-Header-Content-Length": str(len(request_header))},
        )
        response.raise_for_status()
        response_header_length = response.headers.get("Inference-Header-Content-Length")
        binary_output: bytes | None = None
        if response_header_length is None:
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise InvalidTritonOutput("invalid Triton JSON response") from exc
        else:
            try:
                length = int(response_header_length)
                payload = json.loads(response.content[:length])
                binary_output = response.content[length:]
            except (ValueError, json.JSONDecodeError) as exc:
                raise InvalidTritonOutput("invalid Triton binary response header") from exc
        if not isinstance(payload, dict):
            raise InvalidTritonOutput("Triton response must be an object")
        if payload.get("id") != request_id:
            raise InvalidTritonOutput("Triton response id does not match request")
        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            raise InvalidTritonOutput("Triton outputs are missing")
        output = next((item for item in outputs if isinstance(item, dict) and item.get("name") == "logits"), None)
        if not output or output.get("datatype") != "FP32" or output.get("shape") != [1, 1000]:
            raise InvalidTritonOutput("invalid logits output contract")
        if binary_output is not None:
            size = output.get("parameters", {}).get("binary_data_size")
            if size != 4000 or len(binary_output) != size:
                raise InvalidTritonOutput("invalid logits binary data length")
            result = np.frombuffer(binary_output, dtype="<f4").reshape(1, 1000).copy()
        else:
            data = output.get("data")
            if not isinstance(data, list) or len(data) != 1000:
                raise InvalidTritonOutput("invalid logits data length")
            try:
                result = np.asarray(data, dtype=np.float32).reshape(1, 1000)
            except (TypeError, ValueError) as exc:
                raise InvalidTritonOutput("invalid logits data") from exc
        if not np.isfinite(result).all():
            raise InvalidTritonOutput("logits must be finite")
        return result

    async def ready(self) -> bool:
        response = await self.http.get(f"/v2/models/{self.model_name}/versions/{self.version}/ready")
        return response.status_code == 200

    async def metadata(self) -> dict[str, Any]:
        response = await self.http.get(f"/v2/models/{self.model_name}/versions/{self.version}")
        response.raise_for_status()
        return response.json()

    async def load(self) -> None:
        response = await self.http.post(f"/v2/repository/models/{self.model_name}/load")
        response.raise_for_status()

    async def unload(self) -> None:
        response = await self.http.post(f"/v2/repository/models/{self.model_name}/unload")
        response.raise_for_status()
