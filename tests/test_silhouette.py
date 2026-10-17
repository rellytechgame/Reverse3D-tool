from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

pytest.importorskip("skimage")

from reverse3d.errors import ProcessingError
from reverse3d.reconstruction.silhouette import convergence_point, read_colmap_text, visual_hull

SIZE = 400
F = 420.0
RADIUS, HEIGHT = 0.35, 1.2  # the object: an upright cylinder standing on z = 0


def _look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World→camera rotation and translation (COLMAP convention: +Z forward, +Y down)."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rot = np.stack([right, down, forward])
    return rot, -rot @ eye


def _quat(rot: np.ndarray) -> list[float]:
    return list(
        trimesh.transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rot, [[0], [0], [0]]]), [0, 0, 0, 1]])
        )
    )


def _scene(folder: Path) -> tuple[Path, Path]:
    images = folder / "images"
    text = folder / "txt"
    images.mkdir(parents=True)
    text.mkdir()
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rim = np.concatenate(
        [
            np.stack([RADIUS * np.cos(theta), RADIUS * np.sin(theta), np.full_like(theta, z)], 1)
            for z in (0.0, HEIGHT)
        ]
    )
    lines = []
    k = 0
    for height, count in ((0.6, 12), (2.2, 12), (3.4, 6)):
        for i in range(count):
            a = 2 * np.pi * i / count
            eye = np.array([3.2 * np.cos(a), 3.2 * np.sin(a), height])
            rot, t = _look_at(eye, np.array([0.0, 0.0, 0.6]))
            # Textured background, then the white cylinder's silhouette.
            image = rng.integers(0, 120, (SIZE, SIZE, 3), dtype=np.uint8)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            cam = rim @ rot.T + t
            uv = (cam[:, :2] / cam[:, 2:3]) * F + SIZE / 2
            cv2.fillConvexPoly(image, cv2.convexHull(uv.astype(np.int32)), (235, 235, 235))
            name = f"img_{k:02d}.jpg"
            cv2.imwrite(str(images / name), image)
            q = _quat(rot)
            lines.append(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
            k += 1
    (text / "cameras.txt").write_text(f"1 PINHOLE {SIZE} {SIZE} {F} {F} {SIZE / 2} {SIZE / 2}\n")
    (text / "images.txt").write_text("".join(lines))
    table = np.column_stack([rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400), np.zeros(400)])
    table = table[np.hypot(table[:, 0], table[:, 1]) > RADIUS + 0.05]
    (text / "points3D.txt").write_text(
        "".join(f"{i + 1} {x} {y} {z} 0 0 0 0\n" for i, (x, y, z) in enumerate(table))
    )
    return images, text


def test_cameras_converge_on_the_object(tmp_path: Path) -> None:
    _, text = _scene(tmp_path)
    views, points = read_colmap_text(text)
    center, distance = convergence_point(views)
    assert len(views) == 30 and len(points) > 100
    assert np.linalg.norm(center - [0, 0, 0.6]) < 0.05
    assert 3.0 < distance < 3.8


def test_visual_hull_rebuilds_an_upright_cylinder(tmp_path: Path) -> None:
    images, text = _scene(tmp_path)
    result = visual_hull(images, text, tmp_path / "hull.ply", resolution=80, refinements=1)
    mesh = trimesh.load(result.mesh_path)
    assert result.support_plane
    assert mesh.is_watertight
    width = (mesh.extents[0] + mesh.extents[1]) / 2
    assert mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    # Height / diameter of the real cylinder is 1.2 / 0.7 ≈ 1.7.
    assert 1.3 < mesh.extents[2] / width < 2.2


def test_too_few_photos(tmp_path: Path) -> None:
    images, text = _scene(tmp_path)
    rows = (text / "images.txt").read_text().split("\n\n")
    (text / "images.txt").write_text("\n\n".join(rows[:5]) + "\n\n")
    with pytest.raises(ProcessingError, match="at least"):
        visual_hull(images, text, tmp_path / "hull.ply", resolution=48, refinements=0)


class _FakeSegmenter:
    """Stands in for Segment Anything: the object is the bright part inside the box."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        return {"rgb": rgb}

    def segment(
        self, state: dict[str, np.ndarray], box: tuple[int, int, int, int], point: tuple[int, int]
    ) -> np.ndarray:
        self.calls += 1
        gray = state["rgb"].mean(axis=2)
        mask = np.zeros(gray.shape, np.uint8)
        x0, y0, x1, y1 = box
        mask[y0 : y1 + 1, x0 : x1 + 1] = np.where(gray[y0 : y1 + 1, x0 : x1 + 1] > 200, 255, 0)
        return mask


def test_ai_segmenter_is_loaded_lazily_and_used(tmp_path: Path) -> None:
    images, text = _scene(tmp_path)
    fake = _FakeSegmenter()
    loads: list[int] = []

    def factory() -> _FakeSegmenter:
        loads.append(1)
        return fake

    result = visual_hull(
        images,
        text,
        tmp_path / "hull.ply",
        resolution=64,
        refinements=1,
        segmenter=factory,  # type: ignore[arg-type]
    )
    assert loads == [1] and fake.calls == 2 * 30  # two AI rounds over the 30 photos
    assert result.ai_segmentation
    mesh = trimesh.load(result.mesh_path)
    width = (mesh.extents[0] + mesh.extents[1]) / 2
    assert mesh.is_watertight and 1.3 < mesh.extents[2] / width < 2.2


def test_without_ai_segmenter_grabcut_is_used(tmp_path: Path) -> None:
    images, text = _scene(tmp_path)
    result = visual_hull(
        images, text, tmp_path / "hull.ply", resolution=48, refinements=1, segmenter=lambda: None
    )
    assert not result.ai_segmentation
