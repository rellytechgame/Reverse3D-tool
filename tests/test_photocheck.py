"""Optiverse checks the photos before reconstructing: names, problems and gaps."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from reverse3d import photocheck
from reverse3d.assistant import actions
from reverse3d.config import Settings


def _scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big = (rng.random((900, 2400)) * 255).astype(np.uint8)
    big = cv2.GaussianBlur(big, (0, 0), 1.2)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def _shoot(folder: Path) -> list[str]:
    """Two walks around different subjects: each walk overlaps, the walks do not."""
    folder.mkdir()
    names = []
    for walk, seed in enumerate((1, 2)):
        world = _scene(seed)
        for k in range(6):
            x = 120 + k * 160  # each photo shares most of the previous one
            name = f"IMG_{walk}{k:03d}.jpg"
            cv2.imwrite(str(folder / name), world[100:700, x : x + 800])
            names.append(name)
    return names


@pytest.fixture(autouse=True)
def _no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actions, "_own_model", lambda: None)


def test_a_clean_overlapping_walk_is_ready(tmp_path: Path) -> None:
    folder = tmp_path / "fotos"
    folder.mkdir()
    world = _scene(3)
    for k in range(8):
        cv2.imwrite(str(folder / f"IMG_{k:03d}.jpg"), world[100:700, 120 + k * 150 : 920 + k * 150])
    check = photocheck.check_photos(folder, Settings(), scene="object")
    check.min_photos = 5
    assert check.groups == 1 and not check.gaps
    assert check.to_fix == [] and check.verdict == "ready"


def test_it_names_where_photos_are_missing_and_which_to_fix(tmp_path: Path) -> None:
    folder = tmp_path / "fotos"
    names = _shoot(folder)
    blurred = folder / names[2]
    cv2.imwrite(str(blurred), cv2.GaussianBlur(cv2.imread(str(blurred)), (0, 0), 7))
    dark = folder / names[8]
    cv2.imwrite(str(dark), (cv2.imread(str(dark)) * 0.08).astype(np.uint8))
    repeat = folder / "IMG_1005b.jpg"
    repeat.write_bytes((folder / names[11]).read_bytes())  # the same shot twice

    check = photocheck.check_photos(folder, Settings(), scene="object")
    assert check.groups == 2
    assert check.gaps == [("IMG_0005", "IMG_1000")]  # the exact names around the gap
    assert "blurry" in check.issues[names[2]]
    assert "dark" in check.issues[names[8]]
    assert "repeated" in check.issues["IMG_1005b.jpg"]
    assert names[2] in check.to_fix and "IMG_1005b.jpg" not in check.to_fix  # repeated: optional
    assert check.verdict == "retake"  # a split set needs photos, not just dropping some
    out = check.to_dict("es")
    assert any("Faltan fotos entre IMG_0005 y IMG_1000" in line for line in out["summary"])
    fixes = {p["name"]: p["fix"] for p in out["photos_with_issues"]}
    assert "repítela" in fixes[names[2]][0]


def test_keep_good_copies_only_photos_that_need_no_fixing(tmp_path: Path) -> None:
    folder = tmp_path / "fotos"
    folder.mkdir()
    world = _scene(4)
    for k in range(6):
        cv2.imwrite(str(folder / f"IMG_{k:03d}.jpg"), world[100:700, 120 + k * 150 : 920 + k * 150])
    bad = folder / "IMG_003.jpg"
    cv2.imwrite(str(bad), cv2.GaussianBlur(cv2.imread(str(bad)), (0, 0), 7))
    check = photocheck.check_photos(folder, Settings(), scene="object")
    out = photocheck.keep_good(folder, check, tmp_path / "buenas")
    kept = sorted(p.name for p in out.iterdir())
    assert "IMG_003.jpg" not in kept and len(kept) == 5
    assert (folder / "IMG_003.jpg").is_file()  # the originals are never touched
