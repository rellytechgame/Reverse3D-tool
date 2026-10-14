"""Mesh repair, remeshing and alignment for scanned / reconstructed geometry.

Pipeline (each step optional, executed in this order)::

    sanitize (NaN/inf, degenerate & duplicate faces, merge vertices)
      → component filtering (drop floating debris)
      → normal/winding repair
      → hole filling (small holes by trimesh, larger loops by centroid fan)
      → Taubin smoothing (volume-preserving, unlike Laplacian)
      → quadric decimation to a target face count
      → alignment (OBB / PCA / center) + optional ground placement
      → uniform scale (unit conversion)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import trimesh

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError, ProcessingError
from reverse3d.mesh_io import load_mesh, save_mesh
from reverse3d.reverse_eng.measure import MeshStats, compute_stats

logger = logging.getLogger(__name__)

AlignMode = Literal["none", "center", "obb", "pca"]


@dataclass
class CleanOptions:
    """Options for :func:`clean_mesh`. Defaults are safe for most scans."""

    keep_largest_only: bool = False
    min_component_ratio: float = 0.01
    fix_normals: bool = True
    fill_holes: bool = True
    max_hole_edges: int = 200
    smooth_iterations: int = 0
    target_faces: int | None = None
    align: AlignMode = "none"
    ground: bool = False
    scale: float = 1.0

    def validate(self) -> None:
        if not 0 <= self.min_component_ratio < 1:
            raise InputValidationError("min_component_ratio must be in [0, 1).")
        if not 3 <= self.max_hole_edges <= 100_000:
            raise InputValidationError("max_hole_edges must be in [3, 100000].")
        if not 0 <= self.smooth_iterations <= 200:
            raise InputValidationError("smooth_iterations must be in [0, 200].")
        if self.target_faces is not None and self.target_faces < 4:
            raise InputValidationError("target_faces must be >= 4.")
        if self.align not in ("none", "center", "obb", "pca"):
            raise InputValidationError("align must be none, center, obb or pca.")
        if not 1e-9 < self.scale < 1e9:
            raise InputValidationError("scale must be a positive, finite factor.")


@dataclass
class CleanReport:
    """What :func:`clean_mesh` did, with before/after statistics."""

    before: MeshStats
    after: MeshStats
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ─── individual operations ────────────────────────────────────────────────────


def sanitize(mesh: trimesh.Trimesh) -> int:
    """Remove NaN/inf vertices, degenerate and duplicate faces, merge coincident
    vertices. Returns the number of faces removed."""
    start = len(mesh.faces)
    mesh.remove_infinite_values()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return start - len(mesh.faces)


def filter_components(
    mesh: trimesh.Trimesh, *, keep_largest_only: bool, min_ratio: float
) -> tuple[trimesh.Trimesh, int]:
    """Drop small disconnected pieces. Returns ``(mesh, removed_count)``.

    Size is measured by surface area, not face count: scan debris is often
    densely tessellated while the real part may be a low-poly CAD body.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh, 0
    parts = sorted(parts, key=lambda m: m.area, reverse=True)
    if keep_largest_only:
        kept = parts[:1]
    else:
        threshold = min_ratio * parts[0].area
        kept = [p for p in parts if p.area >= threshold]
    return cast(trimesh.Trimesh, trimesh.util.concatenate(kept)), len(parts) - len(kept)


def _boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    """Return closed boundary loops as ordered vertex lists, oriented like the
    adjacent faces' edges. Loops through non-manifold vertices are skipped."""
    faces = mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    single = np.asarray(
        trimesh.grouping.group_rows(np.sort(directed, axis=1), require_count=1)
    ).reshape(-1)
    if single.size == 0:
        return []
    boundary = directed[single]

    successors: dict[int, int] = {}
    ambiguous: set[int] = set()
    for a, b in boundary:
        if a in successors:
            ambiguous.add(int(a))
        successors[int(a)] = int(b)

    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in successors:
        if start in visited or start in ambiguous:
            continue
        loop = [start]
        visited.add(start)
        current = successors[start]
        valid = True
        while current != start:
            if current in visited or current in ambiguous or current not in successors:
                valid = False
                break
            loop.append(current)
            visited.add(current)
            current = successors[current]
        if valid and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_boundary_loops(mesh: trimesh.Trimesh, max_hole_edges: int) -> tuple[trimesh.Trimesh, int]:
    """Close every boundary loop up to ``max_hole_edges`` edges with a fan of
    triangles around the loop centroid. Returns ``(mesh, holes_filled)``.

    Suitable for roughly planar/convex holes typical of scan occlusions; very
    large or saddle-shaped holes need a dedicated surface reconstruction.
    """
    loops = [loop for loop in _boundary_loops(mesh) if len(loop) <= max_hole_edges]
    if not loops:
        return mesh, 0
    vertices: list[np.ndarray] = [np.asarray(mesh.vertices)]
    new_faces: list[np.ndarray] = [np.asarray(mesh.faces)]
    next_index = len(mesh.vertices)
    for loop in loops:
        # Existing faces traverse a→b; the patch must traverse b→a to keep winding consistent.
        if len(loop) == 3:
            new_faces.append(np.array([[loop[2], loop[1], loop[0]]]))
            continue
        centroid = mesh.vertices[loop].mean(axis=0)
        vertices.append(centroid[None, :])
        ring = np.array(loop)
        fan = np.stack([np.roll(ring, -1), ring, np.full(len(ring), next_index)], axis=1)
        new_faces.append(fan)
        next_index += 1
    filled = trimesh.Trimesh(
        vertices=np.vstack(vertices), faces=np.vstack(new_faces), process=False
    )
    return filled, len(loops)


def align_mesh(mesh: trimesh.Trimesh, mode: AlignMode) -> None:
    """Align in place. ``obb``: minimum oriented box → axes (largest extent on X);
    ``pca``: principal axes of the vertices; ``center``: bbox center to origin."""
    if mode == "none":
        return
    if mode == "center":
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
        return
    if mode == "obb":
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        mesh.apply_transform(to_origin)
        order = np.argsort(extents)[::-1]
        permute = np.eye(4)
        permute[:3, :3] = np.eye(3)[order]
        if np.linalg.det(permute[:3, :3]) < 0:
            permute[2, :3] *= -1
        mesh.apply_transform(permute)
        return
    # PCA
    centered = mesh.vertices - mesh.vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rotation = vt.copy()
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ mesh.vertices.mean(axis=0)
    mesh.apply_transform(transform)


# ─── public API ───────────────────────────────────────────────────────────────


def clean_mesh(
    mesh: trimesh.Trimesh, options: CleanOptions | None = None
) -> tuple[trimesh.Trimesh, CleanReport]:
    """Repair ``mesh`` and return ``(clean_mesh, report)``. The input is not modified.

    Raises:
        InputValidationError: invalid options.
        ProcessingError: the mesh became empty.
        DependencyMissingError: decimation requested without ``fast-simplification``.
    """
    options = options or CleanOptions()
    options.validate()
    mesh = mesh.copy()
    before = compute_stats(mesh)
    steps: list[str] = []
    warnings: list[str] = []

    removed = sanitize(mesh)
    steps.append(f"sanitize: removed {removed} degenerate/duplicate face(s)")

    mesh, dropped = filter_components(
        mesh, keep_largest_only=options.keep_largest_only, min_ratio=options.min_component_ratio
    )
    if dropped:
        steps.append(f"components: dropped {dropped} small piece(s)")

    if options.fill_holes:
        trimesh.repair.fill_holes(mesh)
        mesh, filled = fill_boundary_loops(mesh, options.max_hole_edges)
        mesh.merge_vertices()
        steps.append(f"holes: filled {filled} boundary loop(s)")

    if options.fix_normals:
        trimesh.repair.fix_normals(mesh)
        steps.append("normals: winding and orientation fixed")

    if options.smooth_iterations:
        trimesh.smoothing.filter_taubin(mesh, iterations=options.smooth_iterations)
        steps.append(f"smooth: Taubin x{options.smooth_iterations}")

    if options.target_faces and len(mesh.faces) > options.target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=options.target_faces)
        except ImportError as exc:
            raise DependencyMissingError(
                "Decimation requires the 'fast-simplification' package.",
                hint="pip install fast-simplification",
            ) from exc
        steps.append(f"decimate: {len(mesh.faces)} face(s)")

    if len(mesh.faces) == 0:
        raise ProcessingError("Mesh is empty after cleaning; relax the options.")

    align_mesh(mesh, options.align)
    if options.align != "none":
        steps.append(f"align: {options.align}")
    if options.ground:
        mesh.apply_translation([0.0, 0.0, -mesh.bounds[0][2]])
        steps.append("ground: min Z moved to 0")
    if options.scale != 1.0:
        mesh.apply_scale(options.scale)
        steps.append(f"scale: x{options.scale}")

    after = compute_stats(mesh)
    if not after.watertight:
        warnings.append(
            f"Mesh is still open ({after.boundary_edges} boundary edges); "
            "raise max_hole_edges or re-scan the missing areas."
        )
    return mesh, CleanReport(before=before, after=after, steps=steps, warnings=warnings)


def clean_mesh_file(
    input_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    options: CleanOptions | None = None,
    *,
    overwrite: bool = False,
) -> CleanReport:
    """Load, clean and save a mesh file. Returns the :class:`CleanReport`."""
    mesh = load_mesh(input_path, settings)
    cleaned, report = clean_mesh(mesh, options)
    report.output_path = str(save_mesh(cleaned, output_path, overwrite=overwrite))
    logger.info(
        "Cleaned %s: %d → %d faces", Path(input_path).name, report.before.faces, report.after.faces
    )
    return report
