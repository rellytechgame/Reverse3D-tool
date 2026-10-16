from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reverse3d.imaging import prepare_heic_input

pillow_heif = pytest.importorskip("pillow_heif")


def _heic(path: Path, seed: int) -> Path:
    from PIL import Image

    pillow_heif.register_heif_opener()
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, format="HEIF", quality=90)
    return path


def test_heic_folder_is_converted_with_the_other_photos(tmp_path: Path) -> None:
    src = tmp_path / "fotos"
    _heic(src / "IMG_1.HEIC", 1)
    _heic(src / "IMG_2.heic", 2)
    cv2.imwrite(str(src / "extra.png"), np.zeros((8, 8, 3), np.uint8))
    out = prepare_heic_input(src, tmp_path / "work")
    assert out == tmp_path / "work" / "photos_jpg"
    assert sorted(p.name for p in out.iterdir()) == ["IMG_1.jpg", "IMG_2.jpg", "extra.png"]
    image = cv2.imread(str(out / "IMG_1.jpg"))
    assert image is not None and image.shape == (64, 96, 3)


def test_single_heic_photo_and_plain_inputs(tmp_path: Path) -> None:
    jpg = prepare_heic_input(_heic(tmp_path / "pieza.heif", 3), tmp_path / "work")
    assert jpg.suffix == ".jpg" and jpg.is_file()
    plain = tmp_path / "plain"
    plain.mkdir()
    cv2.imwrite(str(plain / "a.png"), np.zeros((8, 8, 3), np.uint8))
    assert prepare_heic_input(plain, tmp_path / "work2") == plain
    assert not (tmp_path / "work2").exists()
