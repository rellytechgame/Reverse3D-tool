"""Safe image I/O.

Images are size-checked from their header with Pillow *before* OpenCV decodes
the full pixel buffer, which prevents decompression bombs from exhausting
memory. Decoding goes through ``np.fromfile`` + ``cv2.imdecode`` so non-ASCII
paths work on Windows (``cv2.imread`` does not support them).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, ProcessingError
from reverse3d.security import IMAGE_EXTENSIONS, prepare_output_file, validate_input_file


def check_image_header(path: Path, max_pixels: int) -> tuple[int, int]:
    """Return ``(width, height)`` from the image header, enforcing ``max_pixels``."""
    # Pillow's own bomb guard would raise a warning/error at ~2x its default;
    # we apply our explicit, configurable limit instead.
    try:
        with Image.open(path) as img:
            width, height = img.size
    except (OSError, Image.DecompressionBombError) as exc:
        raise InputValidationError(f"Unreadable or unsafe image: {path.name}") from exc
    if width * height > max_pixels:
        raise InputValidationError(
            f"Image too large ({width}x{height} px).",
            hint="Raise REVERSE3D_MAX_IMAGE_PIXELS if this is expected.",
        )
    return width, height


def load_image(
    path: str | Path,
    settings: Settings,
    *,
    flags: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """Validate and decode an image into a NumPy array (BGR / grayscale / BGRA).

    Raises:
        InputValidationError: invalid file, type, or size.
    """
    resolved = validate_input_file(
        path, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=settings.max_input_bytes
    )
    check_image_header(resolved, settings.max_image_pixels)
    buffer = np.fromfile(resolved, dtype=np.uint8)
    image = cv2.imdecode(buffer, flags)
    if image is None:
        raise InputValidationError(f"OpenCV could not decode image: {resolved.name}")
    return image


def save_image(image: np.ndarray, path: str | Path, *, overwrite: bool = False) -> Path:
    """Encode and write ``image`` to ``path`` (format inferred from extension)."""
    out = prepare_output_file(path, allowed_extensions=IMAGE_EXTENSIONS, overwrite=overwrite)
    ok, encoded = cv2.imencode(out.suffix, image)
    if not ok:
        raise ProcessingError(f"Could not encode image as {out.suffix}")
    encoded.tofile(out)
    return out


def downscale(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is ``max_dim``. Returns ``(image, scale)``."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale
