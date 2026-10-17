from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from reverse3d.config import Settings


@pytest.fixture(autouse=True)
def _isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real ~/.reverse3d during tests."""
    import os

    for var in list(os.environ):
        if var.startswith("REVERSE3D_"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("REVERSE3D_HOME", str(tmp_path / "_r3d_home"))


@pytest.fixture(autouse=True)
def _no_network_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never depend on the real network: the connectivity probe says "online"."""
    from reverse3d import offline

    monkeypatch.setattr(offline, "_probe", lambda: True)
    monkeypatch.setattr(offline, "_cache", None)


@pytest.fixture(autouse=True)
def _no_ai_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never download or run the real AI models (classifier, segmenter)."""
    from reverse3d.vision import classifier, segmenter

    monkeypatch.setattr(classifier, "classifier_available", lambda: False)
    monkeypatch.setattr(segmenter, "segmenter_available", lambda: False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace=tmp_path.resolve(), subprocess_timeout_s=30)


@pytest.fixture
def plan_image(tmp_path: Path) -> Path:
    """White sheet with a dark 200x120 px plate containing a round hole."""
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 90), (299, 209), (20, 20, 20), thickness=-1)
    cv2.circle(img, (200, 150), 30, (255, 255, 255), thickness=-1)
    path = tmp_path / "plan.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def textured_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two overlapping views of a random texture (horizontal shift of 40 px)."""
    rng = np.random.default_rng(42)
    base = (rng.random((480, 720)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    base = cv2.cvtColor(cv2.equalizeHist(base), cv2.COLOR_GRAY2BGR)
    a, b = base[:, :640], base[:, 40:680]
    pa, pb = tmp_path / "views" / "a.png", tmp_path / "views" / "b.png"
    pa.parent.mkdir()
    cv2.imwrite(str(pa), a)
    cv2.imwrite(str(pb), b)
    return pa, pb


@pytest.fixture
def box_mesh_path(tmp_path: Path) -> Path:
    path = tmp_path / "box.stl"
    trimesh.creation.box(extents=(40.0, 20.0, 10.0)).export(path)
    return path
