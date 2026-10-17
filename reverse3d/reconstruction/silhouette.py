"""Shape from silhouettes (visual hull) for objects photographed all around.

Classic multi-view stereo needs texture on the object itself, so plain,
white, glossy or dark objects (a thermos, a bottle, a vase) come out as holes
or blobs, while the textured room around them reconstructs fine. A visual
hull does not need texture on the object: COLMAP only has to place the
cameras (the room helps with that), then the object is carved as the volume
that falls inside its silhouette in every photo.

Pipeline:

1. read the COLMAP model (cameras, poses, sparse points) exported as text;
2. the object is where the cameras' optical axes converge;
3. per photo, segment the object (GrabCut around its projected position);
4. carve a voxel grid; re-segment each photo guided by the projected hull
   and carve again (a few rounds tighten the silhouettes);
5. cut the support (table) plane, keep the main part, mesh it with marching
   cubes, smooth it and stand it upright on its base.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import trimesh

from reverse3d.errors import DependencyMissingError, ProcessingError

if TYPE_CHECKING:
    from collections.abc import Callable

    from reverse3d.vision.segmenter import ObjectSegmenter

logger = logging.getLogger(__name__)

_WORK_SIDE = 640
_MIN_VIEWS = 8


@dataclass
class _Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: float


@dataclass
class _View:
    name: str
    rotation: np.ndarray
    translation: np.ndarray
    camera: _Camera


def core_engine() -> Any | None:
    """reverse3d-core, Voxel Enterprise's proprietary engine, when it is installed.

    It replaces the silhouette fusion with its own robust model (photos that
    disagree with the rest lose weight instead of erasing the object). Without
    it, the open method below is used.
    """
    try:
        return importlib.import_module("reverse3d_core")
    except ImportError:
        return None


@dataclass
class HullResult:
    mesh_path: str
    views: int
    voxels: int
    faces: int
    support_plane: bool
    ai_segmentation: bool = False
    depth_carved_voxels: int = 0
    #: 4x4 matrix from the model's coordinates back to the COLMAP world (textures).
    to_world: list[list[float]] | None = None
    #: engine that fused the silhouettes: "reverse3d" (open) or "reverse3d-core X.Y"
    engine: str = "reverse3d"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_colmap_text(model_dir: Path) -> tuple[list[_View], np.ndarray]:
    """Views (intrinsics + world→camera pose) and sparse points of a TXT model."""
    cameras: dict[int, _Camera] = {}
    for line in (model_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        model, params = p[1], [float(v) for v in p[4:]]
        if model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
            f, cx, cy = params[:3]
            k = params[3] if model in ("SIMPLE_RADIAL", "RADIAL") else 0.0
            cam = _Camera(int(p[2]), int(p[3]), f, f, cx, cy, k)
        else:  # PINHOLE, OPENCV...: fx fy cx cy [distortion ignored]
            cam = _Camera(int(p[2]), int(p[3]), params[0], params[1], params[2], params[3], 0.0)
        cameras[int(p[0])] = cam
    # Each image takes two lines (header, then its 2D points, which may be empty).
    rows = [
        line
        for line in (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    views = []
    i = 0
    while i < len(rows):
        p = rows[i].split()
        if len(p) < 10:
            i += 1
            continue
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        views.append(_View(" ".join(p[9:]), _rotation(q), t, cameras[int(p[8])]))
        i += 2
    points = []
    for line in (model_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            points.append([float(v) for v in line.split()[1:4]])
    return views, np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _project(points: np.ndarray, view: _View) -> tuple[np.ndarray, np.ndarray]:
    cam = view.camera
    xc = points @ view.rotation.T + view.translation
    z = xc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x, y = xc[:, 0] / safe, xc[:, 1] / safe
    s = 1 + cam.k * (x * x + y * y)
    return np.stack([cam.fx * x * s + cam.cx, cam.fy * y * s + cam.cy], axis=1), z


def convergence_point(views: list[_View]) -> tuple[np.ndarray, float]:
    """Point closest to every camera's optical axis, and the median camera distance."""
    a = np.zeros((3, 3))
    b = np.zeros(3)
    centers = []
    for view in views:
        c = -view.rotation.T @ view.translation
        d = view.rotation.T @ np.array([0.0, 0.0, 1.0])
        d /= np.linalg.norm(d)
        m = np.eye(3) - np.outer(d, d)
        a += m
        b += m @ c
        centers.append(c)
    point = np.linalg.solve(a, b)
    return point, float(np.median(np.linalg.norm(np.asarray(centers) - point, axis=1)))


def _grabcut(
    image: np.ndarray, init: np.ndarray, rect: tuple[int, int, int, int] | None
) -> np.ndarray:
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    mode = cv2.GC_INIT_WITH_RECT if rect is not None else cv2.GC_INIT_WITH_MASK
    try:
        cv2.grabCut(image, init, rect or (0, 0, 0, 0), bgd, fgd, 4, mode)
    except cv2.error:
        return np.zeros(image.shape[:2], np.uint8)
    return np.where((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _support_plane(
    points: np.ndarray, center: np.ndarray, radius: float, cameras: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """The table under the object: RANSAC plane through nearby sparse points, normal up."""
    near = points[np.linalg.norm(points - center, axis=1) < 1.2 * radius]
    if len(near) < 30:
        return None
    rng = np.random.default_rng(0)
    best: tuple[np.ndarray, float, int] | None = None
    for _ in range(400):
        s = near[rng.choice(len(near), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -float(n @ s[0])
        count = int((np.abs(near @ n + d) < 0.01 * radius).sum())
        if best is None or count > best[2]:
            best = (n, d, count)
    if best is None or best[2] < 20:
        return None
    n, d, _ = best
    if np.median(cameras @ n + d) < 0:  # cameras are above the table
        n, d = -n, -d
    return n, d


def _vertex_colors(
    mesh: trimesh.Trimesh,
    views: list[_View],
    images: list[tuple[np.ndarray, float]],
    masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Real colour of every vertex from the photos that see it head-on.

    Each photo votes with weight cos(angle)^2 for the vertices that face it and
    are not hidden (coarse depth buffer); vertices no photo sees get the median.
    """
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    total = np.zeros((len(verts), 3))
    weight = np.zeros(len(verts))
    for i, ((small, scale), view) in enumerate(zip(images, views, strict=True)):
        # Only pixels of the object itself give colour (the carved hull is a bit
        # larger than the object, so its rim would otherwise pick up background).
        inner = (
            cv2.erode(masks[i], np.ones((5, 5), np.uint8)) > 0
            if masks is not None and masks[i].shape == small.shape[:2]
            else None
        )
        center = -view.rotation.T @ view.translation
        to_cam = center - verts
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, to_cam)
        uv, z = _project(verts, view)
        u = uv[:, 0] * scale
        v = uv[:, 1] * scale
        h, w = small.shape[:2]
        ok = (facing > 0.2) & (z > 0) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        if not ok.any():
            continue
        # Coarse z-buffer: a vertex counts only if it is (about) the nearest in its cell.
        cell = 4
        cu = (u[ok] / cell).astype(np.int64)
        cv_ = (v[ok] / cell).astype(np.int64)
        zbuf = np.full(((h // cell) + 1, (w // cell) + 1), np.inf)
        np.minimum.at(zbuf, (cv_, cu), z[ok])
        depth_span = float(np.ptp(z[ok])) or 1.0
        visible = z[ok] <= zbuf[cv_, cu] + 0.03 * depth_span
        idx = np.flatnonzero(ok)[visible]
        if inner is not None:
            idx = idx[inner[v[idx].astype(np.int64), u[idx].astype(np.int64)]]
        px = small[v[idx].astype(np.int64), u[idx].astype(np.int64)][:, ::-1].astype(np.float64)
        wgt = facing[idx] ** 2
        total[idx] += px * wgt[:, None]
        weight[idx] += wgt
    colors = np.zeros((len(verts), 4), np.uint8)
    seen = weight > 0
    rgb = np.zeros((len(verts), 3))
    rgb[seen] = total[seen] / weight[seen, None]
    rgb[~seen] = np.median(rgb[seen], axis=0) if seen.any() else 180
    colors[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    colors[:, 3] = 255
    return colors


def _fit_inverse_depth(disparity: np.ndarray, depth: np.ndarray) -> tuple[float, float] | None:
    """Robust fit 1/depth ≈ s·disparity + t (relative AI disparity → metric-like depth)."""
    inv = 1.0 / depth
    keep = np.ones(len(inv), bool)
    for _ in range(3):
        if keep.sum() < 50:
            return None
        a = np.column_stack([disparity[keep], np.ones(int(keep.sum()))])
        (s, t), *_ = np.linalg.lstsq(a, inv[keep], rcond=None)
        residual = np.abs(s * disparity + t - inv)
        mad = float(np.median(residual[keep])) or 1e-9
        keep = residual < 3 * 1.4826 * mad
    return (float(s), float(t)) if s > 0 else None


# Share of the photos seeing a voxel that may wrongly call it background.
_DISSENT_SHARE = 0.1

# Depth carving may hollow concavities (a cup changes ~20 % of its silhouettes),
# never wipe out what the photos show.
_MAX_SILHOUETTE_LOSS = 0.35


def _depth_carve(
    keep: np.ndarray,
    grid: np.ndarray,
    views: list[_View],
    images: list[tuple[np.ndarray, float]],
    projections: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    masks: list[np.ndarray],
    estimator: Callable[[np.ndarray], np.ndarray],
    *,
    tolerance: float = 0.05,
    min_votes: int = 3,
    min_share: float = 0.5,
) -> np.ndarray:
    """Empty the voxels that sit clearly in front of the surface the AI depth sees,
    in at least ``min_votes`` photos *and* in most (``min_share``) of the photos
    that actually see them. A photo where the voxel lies on the surface votes
    solid; one where it is hidden behind the surface abstains. A few wrong depth
    maps therefore cannot carve real material.

    The relative AI depth is calibrated per photo against the carved hull itself
    (a robust fit on the object's pixels), so only real disagreements carve.
    """
    votes = np.zeros(len(grid), np.int32)
    seen = np.zeros(len(grid), np.int32)
    for (small, _), view, (u, v, inside), mask in zip(
        images, views, projections, masks, strict=True
    ):
        sel = keep & inside
        if sel.sum() < 100:
            continue
        h, w = small.shape[:2]
        z = (grid[sel] @ view.rotation.T + view.translation)[:, 2]
        us, vs = u[sel], v[sel]
        # Depth of the hull's visible surface per pixel (z-buffer of kept voxels).
        hull_depth = np.full((h, w), np.inf)
        np.minimum.at(hull_depth, (vs, us), z)
        object_px = (mask > 0) & np.isfinite(hull_depth)
        if object_px.sum() < 100:
            continue
        disparity = np.asarray(estimator(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), np.float64)
        if disparity.shape != (h, w):
            disparity = cv2.resize(disparity, (w, h), interpolation=cv2.INTER_LINEAR)
        fit = _fit_inverse_depth(disparity[object_px], hull_depth[object_px])
        if fit is None:
            continue
        s, t = fit
        with np.errstate(divide="ignore", invalid="ignore"):
            observed = 1.0 / np.maximum(s * disparity + t, 1e-9)
        # A voxel strictly in front of the observed surface, on an object pixel, is air.
        in_mask = mask[vs, us] > 0
        front = z < observed[vs, us] * (1.0 - tolerance)
        on_surface = np.abs(z - observed[vs, us]) <= tolerance * observed[vs, us]
        idx = np.flatnonzero(sel)
        seen[idx[in_mask & (front | on_surface)]] += 1  # hidden voxels abstain
        votes[idx[in_mask & front]] += 1
    air = (votes >= min_votes) & (votes >= min_share * np.maximum(seen, 1))
    return keep & ~air


def _silhouette_coverage(
    keep: np.ndarray,
    projections: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    masks: list[np.ndarray],
    cell: int,
) -> float:
    """Share of the photos' object pixels that the kept voxels still cover.

    Measured on ``cell``-pixel blocks so that projected voxel centres fill the
    silhouette even when a voxel spans several pixels.
    """
    covered = total = 0
    for (u, v, inside), mask in zip(projections, masks, strict=True):
        h, w = mask.shape[:2]
        sel = keep & inside
        drawn = np.zeros((h // cell + 1, w // cell + 1), bool)
        drawn[v[sel] // cell, u[sel] // cell] = True
        obj = np.zeros_like(drawn)
        ys, xs = np.nonzero(mask > 0)
        obj[ys // cell, xs // cell] = True
        covered += int((drawn & obj).sum())
        total += int(obj.sum())
    return covered / total if total else 1.0


def _silhouette_loss(
    before: np.ndarray,
    after: np.ndarray,
    projections: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    masks: list[np.ndarray],
) -> float:
    """How much of the photos' silhouettes ``after`` stops covering, versus ``before``."""
    for cell in (1, 2, 4, 8, 16):  # the finest blocks the original hull fills
        reference = _silhouette_coverage(before, projections, masks, cell)
        if reference >= 0.9:
            break
    return reference - _silhouette_coverage(after, projections, masks, cell)


def colorize_from_photos(
    mesh_path: Path, image_dir: Path, model_text_dir: Path, output_path: Path
) -> Path:
    """Paint any mesh in COLMAP coordinates with the photos that see it (dense scenes)."""
    views, _ = read_colmap_text(model_text_dir)
    images: list[tuple[np.ndarray, float]] = []
    for view in views:
        bgr = cv2.imread(str(image_dir / view.name))
        if bgr is None:
            raise ProcessingError(f"Could not read {view.name} for colouring.")
        h, w = bgr.shape[:2]
        scale = 1280 / max(h, w)
        size = (round(w * scale), round(h * scale))
        images.append((cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), scale))
    mesh: trimesh.Trimesh = trimesh.load(mesh_path, process=False, force="mesh")  # type: ignore[assignment]
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=_vertex_colors(mesh, views, images)
    )
    mesh.export(output_path)
    return output_path


def visual_hull(
    image_dir: Path,
    model_text_dir: Path,
    output_path: Path,
    *,
    resolution: int = 176,
    refinements: int = 3,
    segmenter: Callable[[], ObjectSegmenter | None] | None = None,
    depth: Callable[[], Callable[[np.ndarray], np.ndarray] | None] | None = None,
) -> HullResult:
    """Carve the object seen by every photo; writes a watertight, upright mesh.

    Raises:
        ProcessingError: too few placed photos, or no object could be isolated.
    """
    try:
        from scipy import ndimage
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - core dependencies
        raise DependencyMissingError(
            "scikit-image and scipy are needed for silhouette reconstruction.",
            hint="pip install scikit-image scipy",
        ) from exc

    views, points = read_colmap_text(model_text_dir)
    if len(views) < _MIN_VIEWS:
        raise ProcessingError(
            f"Only {len(views)} photos were placed; at least {_MIN_VIEWS} are needed."
        )
    center, distance = convergence_point(views)
    radius = 0.45 * distance
    cameras = np.asarray([-v.rotation.T @ v.translation for v in views])

    images: list[tuple[np.ndarray, float]] = []
    for view in views:
        bgr = cv2.imread(str(image_dir / view.name))
        if bgr is None:
            raise ProcessingError(f"Could not read {view.name} for silhouette carving.")
        h, w = bgr.shape[:2]
        scale = _WORK_SIDE / max(h, w)
        images.append(
            (
                cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA),
                scale,
            )
        )

    # First silhouettes: GrabCut in a box around where the object projects.
    masks: list[np.ndarray] = []
    for (small, scale), view in zip(images, views, strict=True):
        uv, z = _project(center[None], view)
        if z[0] <= 0:
            masks.append(np.zeros(small.shape[:2], np.uint8))
            continue
        cu, cv_ = uv[0] * scale
        half = view.camera.fx * scale * 0.6 * radius / z[0]
        x0, y0 = max(0, int(cu - half)), max(0, int(cv_ - 1.6 * half))
        x1 = min(small.shape[1] - 1, int(cu + half))
        y1 = min(small.shape[0] - 1, int(cv_ + 1.6 * half))
        init = np.zeros(small.shape[:2], np.uint8)
        rect = (x0, y0, max(2, x1 - x0), max(2, y1 - y0))
        fg = _grabcut(small, init, rect)
        masks.append(cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)))

    n = resolution
    axis = np.linspace(-radius, radius, n)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) + center
    projections = []
    for (small, scale), view in zip(images, views, strict=True):
        uv, z = _project(grid, view)
        u = np.round(uv[:, 0] * scale).astype(np.int64)
        v = np.round(uv[:, 1] * scale).astype(np.int64)
        inside = (z > 0) & (u >= 0) & (v >= 0) & (u < small.shape[1]) & (v < small.shape[0])
        projections.append((u, v, inside))

    core = core_engine()
    core_cameras = (
        [
            core.Camera(
                view.name,
                small.shape[1],
                small.shape[0],
                view.camera.fx * scale,
                view.camera.fy * scale,
                view.camera.cx * scale,
                view.camera.cy * scale,
                view.camera.k,
                view.rotation,
                view.translation,
            )
            for (small, scale), view in zip(images, views, strict=True)
        ]
        if core is not None
        else None
    )
    engine = f"reverse3d-core {core.__version__}" if core is not None else "reverse3d"

    def carve(mask_list: list[np.ndarray], tolerance: int) -> np.ndarray:
        nonlocal core_cameras, engine
        if core is not None and core_cameras is not None:
            try:
                binary = [np.where(m > 0, 255, 0).astype(np.uint8) for m in mask_list]
                prob, _ = core.fuse_cameras(core_cameras, binary, grid)
                return np.asarray(prob) > 0.5
            except Exception as exc:  # never fail a job because of the optional engine
                logger.warning("reverse3d-core failed (%s); using the open method", exc)
                core_cameras, engine = None, "reverse3d"
        votes = np.zeros(len(grid), np.int32)
        seen = np.zeros(len(grid), np.int32)
        for fg, (u, v, inside) in zip(mask_list, projections, strict=True):
            seen += inside
            hit = np.zeros(len(grid), bool)
            hit[inside] = fg[v[inside], u[inside]] > 0
            votes += hit
        # A voxel survives unless a clear share of the photos that see it call it
        # background. A fixed count does not scale: with 130+ photos, three slightly
        # wrong cut-outs (a glare, the handle in front) would erase the whole object.
        allowed = np.maximum(tolerance, np.ceil(_DISSENT_SHARE * seen))
        return (seen >= 3) & (votes >= seen - allowed)

    keep = carve(masks, 2)
    # The AI segmenter is only loaded (or downloaded) once the photos proved usable.
    ai = segmenter() if segmenter is not None else None
    if ai is not None:
        # AI segmentation: a box around the projected hull (taller upwards, where caps
        # and handles get lost) and a point on it. Two rounds, each from the new hull.
        states = [ai.embed(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)) for small, _ in images]
        for _ in range(2):
            ai_masks: list[np.ndarray] = []
            for state, (small, _), (u, v, inside), old in zip(
                states, images, projections, masks, strict=True
            ):
                sel = keep & inside
                if not sel.any():
                    ai_masks.append(old)
                    continue
                us, vs = u[sel], v[sel]
                x0, x1 = int(us.min()), int(us.max())
                y0, y1 = int(vs.min()), int(vs.max())
                pad_x, pad_y = 0.15 * (x1 - x0 + 1), 0.15 * (y1 - y0 + 1)
                box = (
                    max(0, int(x0 - pad_x)),
                    max(0, int(y0 - 2.5 * pad_y)),
                    min(small.shape[1] - 1, int(x1 + pad_x)),
                    min(small.shape[0] - 1, int(y1 + pad_y)),
                )
                point = (int(np.median(us)), int(np.median(vs)))
                ai_masks.append(ai.segment(state, box, point))
            masks = ai_masks
            keep = carve(masks, 1)
        refinements = 0
    for _ in range(refinements):
        refined: list[np.ndarray] = []
        for (small, _), (u, v, inside), old in zip(images, projections, masks, strict=True):
            hull: np.ndarray = np.zeros(small.shape[:2], np.uint8)
            sel = keep & inside
            hull[v[sel], u[sel]] = 255
            hull = cv2.morphologyEx(hull, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            if not hull.any():
                refined.append(old)
                continue
            init = np.full(small.shape[:2], cv2.GC_BGD, np.uint8)
            init[cv2.dilate(hull, np.ones((25, 25), np.uint8)) > 0] = cv2.GC_PR_BGD
            init[hull > 0] = cv2.GC_PR_FGD
            init[cv2.erode(hull, np.ones((21, 21), np.uint8)) > 0] = cv2.GC_FGD
            fg = _grabcut(small, init, None)
            refined.append(cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))
        masks = refined
        keep = carve(masks, 1)

    # Concavities and through-holes (a handle loop, the inside of a cup) never
    # show in a silhouette: the AI depth of each photo reveals them.
    estimator = depth() if depth is not None else None
    carved_by_depth = 0
    if estimator is not None:
        before = int(keep.sum())
        carved = _depth_carve(keep, grid, views, images, projections, masks, estimator)
        # Hollowing a real concavity barely changes what the photos show; if the
        # silhouettes stop matching, the depth is eating real material: discard it.
        lost = _silhouette_loss(keep, carved, projections, masks)
        if lost > _MAX_SILHOUETTE_LOSS:
            logger.info(
                "AI depth carving discarded: it would erase %.0f%% of the silhouettes", lost * 100
            )
        else:
            keep = carved
            carved_by_depth = before - int(keep.sum())
            logger.info(
                "AI depth carved %d voxel(s) of concavities (silhouette loss %.1f%%)",
                carved_by_depth,
                lost * 100,
            )

    volume = keep.reshape(n, n, n).astype(np.float32)
    plane = _support_plane(points, center, radius, cameras)
    if plane is not None:
        normal, offset = plane
        height = (grid @ normal + offset).reshape(n, n, n)
        volume[height < 0.015 * radius] = 0
    # Drop structures thinner than a couple of voxels (stray spikes from a few views).
    volume = ndimage.binary_opening(volume > 0.5, iterations=2).astype(np.float32)
    labels, count = ndimage.label(volume > 0.5)
    if count == 0:
        raise ProcessingError("No object could be isolated in the photos.")
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
    volume = (labels == 1 + int(np.argmax(sizes))).astype(np.float32)
    voxels = int(volume.sum())
    if voxels < 500:
        raise ProcessingError("The isolated object is too small to build a model.")
    touches_border = bool(
        volume[0].any()
        or volume[-1].any()
        or volume[:, 0].any()
        or volume[:, -1].any()
        or volume[:, :, 0].any()
        or volume[:, :, -1].any()
    )
    if touches_border:
        logger.info("The object reaches the edge of the carving volume; it may be cropped.")
    # One empty voxel of margin all around: marching cubes then yields a closed surface.
    volume = np.pad(ndimage.gaussian_filter(volume, 1.4), 2)
    verts, faces, _, _ = marching_cubes(volume, 0.5)
    verts = (verts - 2) / (n - 1) * 2 * radius - radius + center
    mesh = trimesh.Trimesh(verts, faces, process=True)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=15)
    mesh.fix_normals()
    # Real colours from the photos (kept by PLY export; the model gets them as GLB).
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, vertex_colors=_vertex_colors(mesh, views, images, masks)
    )

    # Stand it upright: +Z along the table normal (or the mean camera "up").
    up = plane[0] if plane is not None else np.mean(cameras - center, axis=0)
    up = up / np.linalg.norm(up)
    side = np.cross([0.0, 1.0, 0.0] if abs(up[1]) < 0.9 else [1.0, 0.0, 0.0], up)
    side /= np.linalg.norm(side)
    basis = np.stack([side, np.cross(up, side), up])
    transform = np.eye(4)
    transform[:3, :3] = basis
    mesh.apply_transform(transform)
    shift = np.array([-mesh.centroid[0], -mesh.centroid[1], -mesh.bounds[0][2]])
    mesh.apply_translation(shift)
    world_to_model = trimesh.transformations.translation_matrix(shift) @ transform

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    logger.info("Silhouette model: %d views, %d faces", len(views), len(mesh.faces))
    return HullResult(
        str(output_path),
        len(views),
        voxels,
        len(mesh.faces),
        plane is not None,
        ai is not None,
        carved_by_depth,
        np.linalg.inv(world_to_model).tolist(),
        engine,
    )
