import json

import httpx
import numpy as np
import pytest

from app.inference import InvalidTritonOutput, TritonClient, predictions


def test_equal_large_logits_have_stable_index_tie_break() -> None:
    logits = np.full((1, 1000), 10_000.0, dtype=np.float32)
    labels = [f"label-{index}" for index in range(1000)]

    result = predictions(logits, labels, top_k=2)

    assert result == [
        {"rank": 1, "class_id": 0, "label": "label-0", "probability": pytest.approx(0.001)},
        {"rank": 2, "class_id": 1, "label": "label-1", "probability": pytest.approx(0.001)},
    ]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_logits_are_rejected(bad: float) -> None:
    logits = np.zeros((1, 1000), dtype=np.float32)
    logits[0, 7] = bad
    with pytest.raises(InvalidTritonOutput, match="finite"):
        predictions(logits, [str(i) for i in range(1000)], 5)


def test_wrong_logits_shape_is_rejected() -> None:
    with pytest.raises(InvalidTritonOutput, match="shape"):
        predictions(np.zeros((1000,), dtype=np.float32), [str(i) for i in range(1000)], 5)


@pytest.mark.asyncio
async def test_triton_client_sends_v2_tensor_contract_and_request_id() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        header_length = int(request.headers["Inference-Header-Content-Length"])
        body = json.loads(request.content[:header_length])
        assert request.url.path == "/v2/models/mobilenet_v2/versions/1/infer"
        assert body["id"] == "request-123"
        assert body["inputs"][0]["name"] == "input"
        assert body["inputs"][0]["shape"] == [1, 3, 224, 224]
        assert body["inputs"][0]["datatype"] == "FP32"
        assert body["inputs"][0]["parameters"]["binary_data_size"] == 1 * 3 * 224 * 224 * 4
        assert len(request.content) - header_length == 1 * 3 * 224 * 224 * 4
        response_header = json.dumps({"id": "request-123", "outputs": [{"name": "logits", "datatype": "FP32", "shape": [1, 1000], "parameters": {"binary_data_size": 4000}}]}).encode()
        response_data = np.zeros((1, 1000), dtype="<f4").tobytes()
        return httpx.Response(200, headers={"Inference-Header-Content-Length": str(len(response_header))}, content=response_header + response_data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://triton") as http:
        logits = await TritonClient(http).infer(np.zeros((1, 3, 224, 224), dtype=np.float32), "request-123")
    assert logits.shape == (1, 1000)
