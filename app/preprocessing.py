from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


class UnsupportedImageFormat(ImageValidationError):
    pass


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def preprocess_image(data: bytes, max_pixels: int = 20_000_000) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as source:
            source.verify()
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise UnsupportedImageFormat("supported decoded formats are JPEG and PNG")
            if source.width * source.height > max_pixels:
                raise ImageValidationError("image exceeds 20 megapixels")
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            scale = 232.0 / min(width, height)
            resized_size = (int(width * scale), int(height * scale))
            if resized_size[0] * resized_size[1] > max_pixels:
                raise ImageValidationError("resized image exceeds pixel limit")
            resized = image.resize(resized_size, Image.Resampling.BILINEAR)
            left = int(round((resized.width - 224) / 2.0))
            top = int(round((resized.height - 224) / 2.0))
            cropped = resized.crop((left, top, left + 224, top + 224))
            chw = np.asarray(cropped, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return np.ascontiguousarray(((chw - MEAN) / STD)[None, ...], dtype=np.float32)
    except ImageValidationError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("image exceeds pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("unable to decode image") from exc
