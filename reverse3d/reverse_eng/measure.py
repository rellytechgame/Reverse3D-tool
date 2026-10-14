"""Mesh statistics and principal dimensions (cotas)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh

from reverse3d.config import Settings
from reverse3d.mesh_io import load_mesh


def _r(values: np.ndarray | list[float], nd: int = 6) -> list[float]:
    return [round(float(v), nd) for v in values]


def boundary_edge_count(mesh: trimesh.Trimesh) -> int:
    """Number of edges used by a single face (open boundary edges)."""
    if len(mesh.faces) == 0:
        return 0
    groups = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return len(groups)


@dataclass
class MeshStats:
    """Topology and dimension summary of a triangle mesh.

    Lengths are in the mesh's own units (whatever the source file used).
    ``obb_extents`` are the sorted (largest first) side lengths of the minimum
    oriented bounding box: the most robust "overall dimensions" of a part.
    """

    vertices: int
    faces: int
    components: int
    watertight: bool
    winding_consistent: bool
    boundary_edges: int
    euler_number: int
    aabb_min: list[float]
    aabb_max: list[float]
    aabb_extents: list[float]
    obb_extents: list[float]
    surface_area: float
    volume: float | None
    center_mass: list[float] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_stats(mesh: trimesh.Trimesh) -> MeshStats:
    """Compute :class:`MeshStats` for ``mesh`` (no file I/O)."""
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces.")
    watertight = bool(mesh.is_watertight)
    try:
        _, obb_extents = trimesh.bounds.oriented_bounds(mesh)
    except Exception:  # degenerate (flat/collinear) input
        obb_extents = mesh.extents
    components = len(
        trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))
    )
    return MeshStats(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        components=components,
        watertight=watertight,
        winding_consistent=bool(mesh.is_winding_consistent),
        boundary_edges=boundary_edge_count(mesh),
        euler_number=int(mesh.euler_number),
        aabb_min=_r(mesh.bounds[0]),
        aabb_max=_r(mesh.bounds[1]),
        aabb_extents=_r(mesh.extents),
        obb_extents=_r(sorted(obb_extents, reverse=True)),
        surface_area=round(float(mesh.area), 6),
        volume=round(abs(float(mesh.volume)), 6) if watertight else None,
        center_mass=_r(mesh.center_mass) if watertight else None,
    )


def measure_mesh_file(path: str | Path, settings: Settings) -> MeshStats:
    """Load a mesh file and return its :class:`MeshStats`."""
    return compute_stats(load_mesh(path, settings))
