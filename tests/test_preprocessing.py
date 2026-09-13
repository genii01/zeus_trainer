from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.preprocessing import ImageValidationError, preprocess_image


def image_bytes(format_: str, size=(8, 8), color=(255, 0, 0)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=format_)
    return output.getvalue()


def test_red_png_is_resized_cropped_and_normalized() -> None:
    tensor = preprocess_image(image_bytes("PNG"))

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(
        tensor[0, :, 100, 100],
        np.array([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]),
        rtol=1e-5,
    )


def test_non_image_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="decode"):
        preprocess_image(b"not an image")


def test_image_over_twenty_megapixels_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="20 megapixels"):
        preprocess_image(image_bytes("PNG", size=(5000, 4001)))


def test_sample_matches_official_torchvision_transform_fixture() -> None:
    expected = np.load("artifacts/sample_input.npy")
    actual = preprocess_image(Path("artifacts/sample.jpg").read_bytes())
    np.testing.assert_array_equal(actual, expected)


def test_resize_expansion_over_pixel_budget_is_rejected() -> None:
    skinny = image_bytes("PNG", size=(1, 10))

    with pytest.raises(ImageValidationError, match="resized image exceeds"):
        preprocess_image(skinny, max_pixels=1_000)


def test_pillow_decompression_bomb_is_an_image_validation_error(monkeypatch) -> None:
    def bomb(*args, **kwargs):
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(Image, "open", bomb)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        preprocess_image(b"irrelevant")
