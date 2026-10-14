"""Primitive fitting (planes, spheres, cylinders) on point clouds with RANSAC.

Pure NumPy implementation — no Open3D required — so it is deterministic (seeded),
portable and testable. Each fit is a RANSAC hypothesis search followed by a
least-squares refinement on the inliers. :func:`detect_primitives` extracts
several primitives sequentially, removing the inliers of each accepted one.

Normals, when available (sampled from a mesh), are used both to generate
cylinder hypotheses and to reject inliers whose orientation disagrees.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError
from reverse3d.mesh_io import load_geometry, sample_points

logger = logging.getLogger(__name__)

PrimitiveKind = Literal["plane", "sphere", "cylinder"]
ALL_KINDS: tuple[PrimitiveKind, ...] = ("plane", "cylinder", "sphere")
_EPS = 1e-12


@dataclass
class PrimitiveFit:
    """A fitted primitive. ``params`` depend on ``kind``:

    * plane: ``normal``, ``d`` (n·x + d = 0), ``centroid``, ``extent_2d``
    * sphere: ``center``, ``radius``
    * cylinder: ``axis_point``, ``axis_direction``, ``radius``, ``height``
    """

    kind: PrimitiveKind
    params: dict[str, object]
    inliers: int
    inlier_ratio: float
    rmse: float
    inlier_indices: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, int))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("inlier_indices", None)
        return data


@dataclass
class FitReport:
    """Result of :func:`detect_primitives` / :func:`fit_primitives_file`."""

    point_count: int
    threshold: float
    primitives: list[PrimitiveFit]
    unexplained_ratio: float

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "threshold": self.threshold,
            "primitives": [p.to_dict() for p in self.primitives],
            "unexplained_ratio": self.unexplained_ratio,
        }


def _r(v: np.ndarray | float, nd: int = 6) -> list[float] | float:
    if np.ndim(v) == 0:
        return round(float(v), nd)
    return [round(float(x), nd) for x in np.ravel(v)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _normal_ok(
    normals: np.ndarray | None, expected: np.ndarray, cos_min: float
) -> np.ndarray | bool:
    if normals is None:
        return True
    return np.abs(np.einsum("ij,ij->i", normals, expected)) >= cos_min


# ─── plane ────────────────────────────────────────────────────────────────────


def _refine_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    return normal, -float(normal @ centroid), centroid


def fit_plane(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    threshold: float,
    iterations: int = 500,
    normal_angle_deg: float = 30.0,
    rng: np.random.Generator | None = None,
) -> PrimitiveFit | None:
    """RANSAC plane fit. Returns ``None`` if fewer than 3 points."""
    n = len(points)
    if n < 3:
        return None
    rng = rng or np.random.default_rng(0)
    cos_min = float(np.cos(np.radians(normal_angle_deg)))
    best_mask: np.ndarray | None = None
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(normal) < _EPS:
            continue
        normal = _unit(normal)
        dist = np.abs((points - p0) @ normal)
        mask = dist < threshold
        mask &= _normal_ok(normals, np.broadcast_to(normal, points.shape), cos_min)
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask is None or best_mask.sum() < 3:
        return None
    normal, d, centroid = _refine_plane(points[best_mask])
    dist = np.abs(points @ normal + d)
    mask = dist < threshold
    mask &= _normal_ok(normals, np.broadcast_to(normal, points.shape), cos_min)
    if mask.sum() < 3:
        return None

    inl = points[mask]
    # In-plane extent: project on two orthonormal in-plane axes.
    u, v = plane_basis(normal)
    local = np.stack([(inl - centroid) @ u, (inl - centroid) @ v], axis=1)
    extent = local.max(axis=0) - local.min(axis=0)
    return PrimitiveFit(
        kind="plane",
        params={
            "normal": _r(normal),
            "d": _r(d),
            "centroid": _r(centroid),
            "extent_2d": _r(extent),
        },
        inliers=int(mask.sum()),
        inlier_ratio=round(float(mask.mean()), 4),
        rmse=round(float(np.sqrt(np.mean(dist[mask] ** 2))), 6),
        inlier_indices=np.flatnonzero(mask),
    )


# ─── sphere ───────────────────────────────────────────────────────────────────


def _algebraic_sphere(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Least-squares sphere: x²+y²+z² + Dx + Ey + Fz + G = 0."""
    a = np.hstack([points, np.ones((len(points), 1))])
    b = -(points**2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = -0.5 * sol[:3]
    r2 = float(center @ center - sol[3])
    if r2 <= 0:
        return None
    return center, float(np.sqrt(r2))


def fit_sphere(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    threshold: float,
    iterations: int = 500,
    max_radius: float | None = None,
    normal_angle_deg: float = 30.0,
    rng: np.random.Generator | None = None,
) -> PrimitiveFit | None:
    """RANSAC sphere fit (4-point minimal sample)."""
    n = len(points)
    if n < 4:
        return None
    rng = rng or np.random.default_rng(0)
    cos_min = float(np.cos(np.radians(normal_angle_deg)))

    def evaluate(center: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        offsets = points - center
        norms = np.linalg.norm(offsets, axis=1)
        dist = np.abs(norms - radius)
        mask = dist < threshold
        if normals is not None:
            radial = offsets / np.maximum(norms, _EPS)[:, None]
            mask &= _normal_ok(normals, radial, cos_min)
        return mask, dist

    best: tuple[np.ndarray, float] | None = None
    best_count = 0
    for _ in range(iterations):
        sample = _algebraic_sphere(points[rng.choice(n, 4, replace=False)])
        if sample is None:
            continue
        center, radius = sample
        if max_radius is not None and radius > max_radius:
            continue
        mask, _ = evaluate(center, radius)
        if mask.sum() > best_count:
            best, best_count = sample, int(mask.sum())
    if best is None or best_count < 4:
        return None
    mask, _ = evaluate(*best)
    refined = _algebraic_sphere(points[mask])
    if refined is not None:
        mask, dist = evaluate(*refined)
        best = refined
    else:
        _, dist = evaluate(*best)
    if mask.sum() < 4:
        return None
    center, radius = best
    return PrimitiveFit(
        kind="sphere",
        params={"center": _r(center), "radius": _r(radius)},
        inliers=int(mask.sum()),
        inlier_ratio=round(float(mask.mean()), 4),
        rmse=round(float(np.sqrt(np.mean(dist[mask] ** 2))), 6),
        inlier_indices=np.flatnonzero(mask),
    )


# ─── cylinder ─────────────────────────────────────────────────────────────────


def plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal ``(u, v)`` spanning the plane orthogonal to ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, helper))
    return u, np.cross(axis, u)


def _fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Kåsa algebraic circle fit."""
    a = np.hstack([xy, np.ones((len(xy), 1))])
    b = -(xy**2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = -0.5 * sol[:2]
    r2 = float(center @ center - sol[2])
    return (center, float(np.sqrt(r2))) if r2 > 0 else None


def fit_cylinder(
    points: np.ndarray,
    normals: np.ndarray | None,
    *,
    threshold: float,
    iterations: int = 500,
    max_radius: float | None = None,
    normal_angle_deg: float = 30.0,
    rng: np.random.Generator | None = None,
) -> PrimitiveFit | None:
    """RANSAC cylinder fit. Requires normals (axis = n1 × n2 hypothesis)."""
    if normals is None:
        return None
    n = len(points)
    if n < 6:
        return None
    rng = rng or np.random.default_rng(0)
    cos_min = float(np.cos(np.radians(normal_angle_deg)))

    def evaluate(
        axis: np.ndarray, point: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        rel = points - point
        along = rel @ axis
        radial_vec = rel - np.outer(along, axis)
        radial = np.linalg.norm(radial_vec, axis=1)
        dist = np.abs(radial - radius)
        mask = dist < threshold
        mask &= _normal_ok(normals, radial_vec / np.maximum(radial, _EPS)[:, None], cos_min)
        return mask, dist

    def hypothesis(idx: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float] | None:
        u, v = plane_basis(axis)
        xy = np.stack([points[idx] @ u, points[idx] @ v], axis=1)
        circle = _fit_circle_2d(xy)
        if circle is None:
            return None
        (cx, cy), radius = circle
        return cx * u + cy * v, radius

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    best_count = 0
    for _ in range(iterations):
        i, j, k = rng.choice(n, 3, replace=False)
        axis = np.cross(normals[i], normals[j])
        if np.linalg.norm(axis) < 0.05:  # nearly parallel normals: no axis information
            continue
        axis = _unit(axis)
        hyp = hypothesis(np.array([i, j, k]), axis)
        if hyp is None:
            continue
        point, radius = hyp
        if max_radius is not None and radius > max_radius:
            continue
        mask, _ = evaluate(axis, point, radius)
        if mask.sum() > best_count:
            best, best_count = (axis, point, radius), int(mask.sum())
    if best is None or best_count < 6:
        return None

    # Refinement: axis = direction least represented in the inlier normals.
    axis, point, radius = best
    mask, _ = evaluate(axis, point, radius)
    scatter = normals[mask].T @ normals[mask]
    _, eigvecs = np.linalg.eigh(scatter)
    refined_axis = _unit(eigvecs[:, 0])
    hyp = hypothesis(np.flatnonzero(mask), refined_axis)
    if hyp is not None:
        new_mask, _ = evaluate(refined_axis, *hyp)
        if new_mask.sum() >= mask.sum():
            axis, (point, radius) = refined_axis, hyp
    mask, dist = evaluate(axis, point, radius)
    if mask.sum() < 6:
        return None

    along = (points[mask] - point) @ axis
    lo, hi = float(along.min()), float(along.max())
    base = point + lo * axis
    return PrimitiveFit(
        kind="cylinder",
        params={
            "axis_point": _r(base),
            "axis_direction": _r(axis),
            "radius": _r(radius),
            "height": _r(hi - lo),
        },
        inliers=int(mask.sum()),
        inlier_ratio=round(float(mask.mean()), 4),
        rmse=round(float(np.sqrt(np.mean(dist[mask] ** 2))), 6),
        inlier_indices=np.flatnonzero(mask),
    )


# ─── sequential extraction ────────────────────────────────────────────────────


def detect_primitives(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    *,
    kinds: tuple[PrimitiveKind, ...] = ALL_KINDS,
    threshold: float | None = None,
    min_inlier_ratio: float = 0.05,
    max_primitives: int = 8,
    iterations: int = 500,
    seed: int = 0,
) -> FitReport:
    """Extract up to ``max_primitives`` primitives, best-first.

    Each round fits every requested kind on the remaining points and keeps the
    one explaining the most points (ties favour the simpler model: plane <
    cylinder < sphere). Its inliers are then removed.

    Args:
        threshold: inlier distance; defaults to 1% of the cloud's bbox diagonal.
        min_inlier_ratio: stop when the best candidate explains less than this
            fraction of the *original* points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10:
        raise InputValidationError("points must be an (N>=10, 3) array.")
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape != points.shape:
            raise InputValidationError("normals must have the same shape as points.")
    unknown = set(kinds) - set(ALL_KINDS)
    if unknown or not kinds:
        raise InputValidationError(f"kinds must be a non-empty subset of {ALL_KINDS}.")
    if not 1 <= max_primitives <= 50:
        raise InputValidationError("max_primitives must be in [1, 50].")
    if not 10 <= iterations <= 20_000:
        raise InputValidationError("iterations must be in [10, 20000].")

    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    threshold = threshold if threshold and threshold > 0 else max(diagonal * 0.01, 1e-9)
    max_radius = diagonal * 2
    rng = np.random.default_rng(seed)
    total = len(points)
    remaining = np.arange(total)
    found: list[PrimitiveFit] = []
    preference = {"plane": 0, "cylinder": 1, "sphere": 2}

    while len(found) < max_primitives and len(remaining) >= 10:
        pts = points[remaining]
        nrm = normals[remaining] if normals is not None else None
        candidates: list[PrimitiveFit] = []
        for kind in kinds:
            fit: PrimitiveFit | None
            if kind == "plane":
                fit = fit_plane(pts, nrm, threshold=threshold, iterations=iterations, rng=rng)
            elif kind == "sphere":
                fit = fit_sphere(
                    pts,
                    nrm,
                    threshold=threshold,
                    iterations=iterations,
                    max_radius=max_radius,
                    rng=rng,
                )
            else:
                fit = fit_cylinder(
                    pts,
                    nrm,
                    threshold=threshold,
                    iterations=iterations,
                    max_radius=max_radius,
                    rng=rng,
                )
            if fit is not None:
                candidates.append(fit)
        if not candidates:
            break
        best = max(candidates, key=lambda f: (f.inliers, -preference[f.kind]))
        if best.inliers < min_inlier_ratio * total:
            break
        # Report inliers relative to the full cloud and map indices back.
        best.inlier_indices = remaining[best.inlier_indices]
        best.inlier_ratio = round(best.inliers / total, 4)
        found.append(best)
        keep = np.ones(len(remaining), dtype=bool)
        keep[np.searchsorted(remaining, best.inlier_indices)] = False
        remaining = remaining[keep]

    explained = total - len(remaining)
    return FitReport(
        point_count=total,
        threshold=round(threshold, 9),
        primitives=found,
        unexplained_ratio=round(1 - explained / total, 4),
    )


def fit_primitives_file(
    path: str | Path,
    settings: Settings,
    *,
    kinds: tuple[PrimitiveKind, ...] = ALL_KINDS,
    sample_count: int = 20_000,
    threshold: float | None = None,
    max_primitives: int = 8,
    min_inlier_ratio: float = 0.05,
    seed: int = 0,
) -> FitReport:
    """Load a mesh / point cloud, sample points and run :func:`detect_primitives`."""
    if not 100 <= sample_count <= 500_000:
        raise InputValidationError("sample_count must be in [100, 500000].")
    geometry = load_geometry(path, settings)
    points, normals = sample_points(geometry, sample_count, seed=seed)
    report = detect_primitives(
        points,
        normals,
        kinds=kinds,
        threshold=threshold,
        max_primitives=max_primitives,
        min_inlier_ratio=min_inlier_ratio,
        seed=seed,
    )
    logger.info("Detected %d primitive(s) in %s", len(report.primitives), Path(path).name)
    return report
