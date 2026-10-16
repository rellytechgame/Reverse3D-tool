"""Safe image I/O.

Images are size-checked from their header with Pillow *before* OpenCV decodes
the full pixel buffer, which prevents decompression bombs from exhausting
memory. Decoding goes through ``np.fromfile`` + ``cv2.imdecode`` so non-ASCII
paths work on Windows (``cv2.imread`` does not support them).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, ProcessingError
from reverse3d.security import (
    HEIC_EXTENSIONS,
    IMAGE_EXTENSIONS,
    prepare_output_file,
    validate_input_file,
)

logger = logging.getLogger(__name__)


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


def _heic_to_jpeg(src: Path, dst: Path) -> Path:
    """Decode one HEIC/HEIF photo and save it as JPEG, keeping its EXIF (focal length)."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    try:
        with Image.open(src) as image:
            exif = image.info.get("exif")
            rgb = image.convert("RGB")
            kwargs: dict[str, object] = {"quality": 95}
            if exif:
                kwargs["exif"] = exif
            rgb.save(dst, "JPEG", **kwargs)
    except Exception as exc:  # pillow-heif raises several error types
        raise InputValidationError(f"Could not read the HEIC photo {src.name}: {exc}") from exc
    return dst


def prepare_heic_input(path: str | Path, work_dir: str | Path) -> Path:
    """Return ``path`` ready for OpenCV and COLMAP, converting HEIC/HEIF photos to JPEG.

    A HEIC file becomes ``work_dir/photos_jpg/<name>.jpg``; a folder with HEIC
    photos becomes ``work_dir/photos_jpg`` holding the converted photos plus the
    other images. Inputs without HEIC are returned unchanged.
    """
    src = Path(path).expanduser()
    if src.is_file() and src.suffix.lower() not in HEIC_EXTENSIONS:
        return src
    if src.is_dir():
        files = [p for p in sorted(src.iterdir()) if p.is_file() and not p.name.startswith(".")]
        if not any(p.suffix.lower() in HEIC_EXTENSIONS for p in files):
            return src
    elif not src.is_file():
        return src
    target = Path(work_dir).expanduser() / "photos_jpg"
    target.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        return _heic_to_jpeg(src, target / f"{src.stem}.jpg")
    converted = 0
    for item in files:
        ext = item.suffix.lower()
        if ext in HEIC_EXTENSIONS:
            out = target / f"{item.stem}.jpg"
            if not out.exists():
                _heic_to_jpeg(item, out)
            converted += 1
        elif ext in IMAGE_EXTENSIONS and not (target / item.name).exists():
            shutil.copy2(item, target / item.name)
    logger.info("Converted %d HEIC photo(s) to JPEG", converted)
    return target
