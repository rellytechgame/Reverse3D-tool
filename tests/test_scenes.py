from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from reverse3d.config import Settings
from reverse3d.scenes import (
    SCENE_PRESETS,
    SceneSignals,
    _photo_metadata,
    _rectilinear,
    _spread_m,
    detect_scene,
    has_gps,
    scene_advice,
    score_scene,
)


def _winner(**signals: object) -> str:
    scores, _ = score_scene(SceneSignals(photos=50, sampled=8, **signals))  # type: ignore[arg-type]
    return max(scores, key=scores.__getitem__)


@pytest.mark.parametrize(
    ("signals", "kind"),
    [
        ({"isolation": 0.7, "rectilinear": 0.05}, "object"),
        ({"isolation": 0.55, "people_share": 0.8}, "character"),
        (
            {"isolation": 0.1, "rectilinear": 0.6, "sky_share": 0.4, "gps_spread_m": 60.0},
            "architecture",
        ),
        (
            {"drone": True, "nadir_share": 0.9, "gps_spread_m": 400.0, "isolation": 0.05},
            "landscape",
        ),
    ],
)
def test_each_scene_wins_with_its_signals(signals: dict[str, object], kind: str) -> None:
    assert _winner(**signals) == kind


def test_reasons_explain_the_prediction() -> None:
    _, reasons = score_scene(
        SceneSignals(photos=200, sampled=8, drone=True, nadir_share=0.9, gps_spread_m=300.0)
    )
    assert {"drone", "nadir", "gps_wide"} <= set(reasons)


def _facade() -> np.ndarray:
    image = np.full((480, 640), 120, np.uint8)
    for x in range(40, 640, 80):
        cv2.line(image, (x, 60), (x, 470), 255, 3)
    for y in range(60, 480, 60):
        cv2.line(image, (40, y), (600, y), 30, 3)
    return image


def test_straight_lines_are_measured() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (480, 640), dtype=np.uint8)
    assert _rectilinear(_facade()) > 0.25 > _rectilinear(cv2.GaussianBlur(noise, (9, 9), 0))


def _dms(value: float) -> tuple[float, float, float]:
    d = int(value)
    m = int((value - d) * 60)
    return (float(d), float(m), round(((value - d) * 60 - m) * 60, 4))


def _gps_jpeg(path: Path, lat: float, lon: float, make: str = "DJI", model: str = "FC3582") -> Path:
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (64, 64, 3), np.uint8))
    exif = Image.Exif()
    exif[271], exif[272] = make, model
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2] = ("N" if lat >= 0 else "S"), _dms(abs(lat))
    gps[3], gps[4] = ("E" if lon >= 0 else "W"), _dms(abs(lon))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", exif=exif)
    return path


def test_exif_gps_and_drone_camera(tmp_path: Path) -> None:
    a = _gps_jpeg(tmp_path / "a.jpg", 8.98, -79.52)
    camera, position, _ = _photo_metadata(a)
    assert camera == "DJI FC3582"
    assert position is not None and position[0] == pytest.approx(8.98, abs=1e-4)
    assert position[1] == pytest.approx(-79.52, abs=1e-4)
    for i in range(1, 3):
        _gps_jpeg(tmp_path / f"{i}.jpg", 8.98 + i * 0.001, -79.52)
    assert has_gps(tmp_path)
    assert _spread_m([(8.98, -79.52), (8.981, -79.52)]) == pytest.approx(111.2, rel=0.02)


def test_detect_scene_on_a_drone_set(tmp_path: Path, settings: Settings) -> None:
    for i in range(6):
        _gps_jpeg(tmp_path / "vuelo" / f"DJI_{i:04d}.jpg", 8.98 + i * 0.0005, -79.52 + i * 0.0005)
    detection = detect_scene(tmp_path / "vuelo", settings)
    assert detection.kind == "landscape"
    assert "drone" in detection.reasons
    assert not detection.enough_photos and "few_photos" in detection.reasons
    assert detection.advice[0].startswith("Solo hay 6 fotos")


def test_presets_and_advice_cover_every_scene() -> None:
    for kind, preset in SCENE_PRESETS.items():
        assert preset.min_photos <= preset.recommended[0] < preset.recommended[1]
        assert scene_advice(kind, preset.recommended[0])[0]
    assert not SCENE_PRESETS["architecture"].fill_holes
    assert SCENE_PRESETS["object"].fill_holes


def _fake_subject(monkeypatch: pytest.MonkeyPatch, subject: str) -> None:
    from reverse3d import scenes

    monkeypatch.setattr(
        scenes, "_ai_subject", lambda *_a: (subject, {subject: 0.9, "product": 0.1})
    )


@pytest.mark.parametrize(
    ("subject", "kind", "render"),
    [
        ("person", "character", "portrait"),
        ("animal", "character", "portrait"),
        ("vehicle", "object", "showroom"),
        ("building", "architecture", "architecture"),
        ("forest", "landscape", "aerial"),
    ],
)
def test_ai_subject_decides_scene_and_render(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    subject: str,
    kind: str,
    render: str,
) -> None:
    _fake_subject(monkeypatch, subject)
    (tmp_path / "set").mkdir()
    for i in range(4):
        cv2.imwrite(str(tmp_path / "set" / f"{i}.jpg"), _facade())
    detection = detect_scene(tmp_path / "set", settings)
    assert (detection.kind, detection.render, detection.subject) == (kind, render, subject)
    assert detection.reasons[0] == f"ai_{subject}" and detection.subject_label


def test_a_drone_flight_is_terrain_whatever_the_ai_says(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_subject(monkeypatch, "building")
    for i in range(6):
        _gps_jpeg(tmp_path / "vuelo" / f"DJI_{i:04d}.jpg", 8.98 + i * 0.0005, -79.52)
        # Gimbal pointing straight down, as DJI writes it in the XMP block.
        path = tmp_path / "vuelo" / f"DJI_{i:04d}.jpg"
        path.write_bytes(path.read_bytes() + b'<x drone-dji:GimbalPitchDegree="-90.0"/>')
    detection = detect_scene(tmp_path / "vuelo", settings)
    assert (detection.kind, detection.subject, detection.render) == (
        "landscape",
        "terrain",
        "aerial",
    )
