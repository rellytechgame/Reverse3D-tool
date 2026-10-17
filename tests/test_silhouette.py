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


def _cup() -> trimesh.Trimesh:
    """An open-top box: its silhouette is a solid block from every side."""
    parts = [
        trimesh.creation.box(
            extents=(0.8, 0.8, 0.12),
            transform=trimesh.transformations.translation_matrix((0, 0, 0.06)),
        ),
        trimesh.creation.box(
            extents=(0.12, 0.8, 1.0),
            transform=trimesh.transformations.translation_matrix((0.34, 0, 0.5)),
        ),
        trimesh.creation.box(
            extents=(0.12, 0.8, 1.0),
            transform=trimesh.transformations.translation_matrix((-0.34, 0, 0.5)),
        ),
        trimesh.creation.box(
            extents=(0.8, 0.12, 1.0),
            transform=trimesh.transformations.translation_matrix((0, 0.34, 0.5)),
        ),
        trimesh.creation.box(
            extents=(0.8, 0.12, 1.0),
            transform=trimesh.transformations.translation_matrix((0, -0.34, 0.5)),
        ),
    ]
    return trimesh.util.concatenate(parts)


def _ray_boxes(origin: np.ndarray, dirs: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
    """Distance along each ray to the nearest axis-aligned box (inf when missed)."""
    best = np.full(len(dirs), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        for lo, hi in boxes:
            t1, t2 = (lo - origin) * inv, (hi - origin) * inv
            near = np.nanmax(np.minimum(t1, t2), axis=1)
            far = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (near <= far) & (far > 0)
            best = np.where(hit & (near < best), np.maximum(near, 0), best)
    return best


def _cup_scene(
    folder: Path, size: int = 120, focal: float = 130.0
) -> tuple[Path, Path, list[tuple[np.ndarray, np.ndarray]]]:
    """Photos of the cup with known cameras, plus the true AI-like disparity of each."""
    images, text = folder / "images", folder / "txt"
    images.mkdir(parents=True)
    text.mkdir()
    boxes = [part.bounds for part in _cup().split(only_watertight=False)]
    rng = np.random.default_rng(1)
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    rays_cam = np.stack(
        [(jj - size / 2) / focal, (ii - size / 2) / focal, np.ones_like(jj, float)], -1
    ).reshape(-1, 3)
    lines: list[str] = []
    depth_by_image: list[tuple[np.ndarray, np.ndarray]] = []
    k = 0
    for height, count in ((0.7, 10), (2.6, 10), (3.6, 6)):
        for i in range(count):
            a = 2 * np.pi * i / count
            eye = np.array([2.6 * np.cos(a), 2.6 * np.sin(a), height])
            rot, t = _look_at(eye, np.array([0.0, 0.0, 0.5]))
            dirs = rays_cam @ rot  # camera → world
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            depth = _ray_boxes(eye, dirs, boxes) * (dirs @ rot[2])
            depth = depth.reshape(size, size)
            image = cv2.GaussianBlur(
                rng.integers(0, 110, (size, size, 3), dtype=np.uint8), (3, 3), 0
            )
            image[np.isfinite(depth)] = (235, 235, 235)
            name = f"img_{k:02d}.png"
            cv2.imwrite(str(images / name), image)
            far = float(np.nanmax(np.where(np.isfinite(depth), depth, np.nan))) * 2
            disparity = 1.0 / np.where(np.isfinite(depth), depth, far)
            depth_by_image.append((cv2.imread(str(images / name)).astype(np.float32), disparity))
            q = _quat(rot)
            lines.append(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
            k += 1
    (text / "cameras.txt").write_text(
        f"1 PINHOLE {size} {size} {focal} {focal} {size / 2} {size / 2}\n"
    )
    (text / "images.txt").write_text("".join(lines))
    table = np.column_stack([rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400), np.zeros(400)])
    table = table[(np.abs(table[:, 0]) > 0.45) | (np.abs(table[:, 1]) > 0.45)]
    rows = (f"{i + 1} {x} {y} {z} 0 0 0 0\n" for i, (x, y, z) in enumerate(table))
    (text / "points3D.txt").write_text("".join(rows))
    return images, text, depth_by_image


def test_ai_depth_carves_the_inside_of_a_cup(tmp_path: Path) -> None:
    images, text, truth = _cup_scene(tmp_path)

    def estimator(rgb: np.ndarray) -> np.ndarray:
        # The hull hands over its (resized) copy of a photo: find which one it is
        # and return that photo's true disparity, like a perfect depth AI would.
        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), truth[0][0].shape[1::-1])
        best = min(truth, key=lambda item: float(np.abs(item[0] - small).mean()))
        return cv2.resize(best[1], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    plain = visual_hull(images, text, tmp_path / "plain.ply", resolution=64, refinements=1)
    carved = visual_hull(
        images, text, tmp_path / "carved.ply", resolution=64, refinements=1, depth=lambda: estimator
    )
    assert carved.depth_carved_voxels > 0
    solid = trimesh.load(plain.mesh_path)
    cup = trimesh.load(carved.mesh_path)
    # The silhouettes alone fill the cup; the AI depth empties a good part of it.
    assert cup.volume < 0.8 * solid.volume
    assert cup.is_watertight


def test_any_mesh_is_coloured_from_the_photos(tmp_path: Path) -> None:
    from reverse3d.reconstruction.silhouette import colorize_from_photos

    images, text = _scene(tmp_path)
    mesh = trimesh.creation.cylinder(radius=RADIUS * 0.98, height=HEIGHT * 0.98, sections=48)
    mesh = mesh.subdivide().subdivide()  # vertices along the sides, not only at the caps
    mesh.apply_translation([0, 0, HEIGHT / 2])
    plain = tmp_path / "cylinder.ply"
    mesh.export(plain)
    out = colorize_from_photos(plain, images, text, tmp_path / "colour.ply")
    colours = np.asarray(trimesh.load(out, process=False).visual.vertex_colors)[:, :3]
    side = np.abs(mesh.vertices[:, 2] - HEIGHT / 2) < HEIGHT * 0.4
    assert colours[side].mean() > 200  # the photographed cylinder is white


class _FakeCore:
    """Stands in for reverse3d-core: records its calls, defers to a plain carve."""

    __version__ = "9.9.9"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    @staticmethod
    def Camera(*args: object) -> tuple[object, ...]:  # noqa: N802 - mirrors the core API
        return args

    def fuse_cameras(self, cameras, masks, points):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        keep = np.ones(len(points), bool)
        for cam, mask in zip(cameras, masks, strict=True):
            _name, w, h, fx, fy, cx, cy, _k, rot, t = cam
            xc = points @ rot.T + t
            u = np.round(fx * xc[:, 0] / xc[:, 2] + cx).astype(int)
            v = np.round(fy * xc[:, 1] / xc[:, 2] + cy).astype(int)
            inside = (xc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
            ok = np.ones(len(points), bool)
            ok[inside] = mask[v[inside], u[inside]] > 127
            keep &= ok
        return keep.astype(np.float32), np.ones(len(cameras))


def test_the_proprietary_core_fuses_the_silhouettes_when_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverse3d.reconstruction import silhouette

    core = _FakeCore()
    monkeypatch.setattr(silhouette, "core_engine", lambda: core)
    images, text = _scene(tmp_path)
    result = visual_hull(images, text, tmp_path / "hull.ply", resolution=48, refinements=1)
    assert core.calls >= 1
    assert result.engine == "reverse3d-core 9.9.9"
    assert result.voxels > 500


def test_a_failing_core_falls_back_to_the_open_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverse3d.reconstruction import silhouette

    core = _FakeCore(fail=True)
    monkeypatch.setattr(silhouette, "core_engine", lambda: core)
    images, text = _scene(tmp_path)
    result = visual_hull(images, text, tmp_path / "hull.ply", resolution=48, refinements=1)
    assert core.calls == 1
    assert result.engine == "reverse3d"
