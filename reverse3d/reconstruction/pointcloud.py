"""Point cloud → surface meshing with Open3D (optional ``[recon]`` extra)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError, ProcessingError
from reverse3d.security import (
    MESH_EXTENSIONS,
    POINT_CLOUD_EXTENSIONS,
    prepare_output_file,
    validate_input_file,
)

logger = logging.getLogger(__name__)


def require_open3d() -> Any:
    """Import Open3D or raise a helpful :class:`DependencyMissingError`."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise DependencyMissingError(
            "Open3D is required for point-cloud meshing.",
            hint='pip install "reverse3d[recon]" (Open3D wheels exist for Python 3.10-3.12).',
        ) from exc
    return o3d


@dataclass
class MeshingResult:
    """Outputs of :func:`poisson_mesh`."""

    mesh_path: str
    input_points: int
    filtered_points: int
    vertices: int
    faces: int
    poisson_depth: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def poisson_mesh(
    cloud_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    *,
    depth: int = 9,
    voxel_size: float | None = None,
    density_quantile: float = 0.02,
    outlier_std_ratio: float = 2.0,
    overwrite: bool = False,
) -> MeshingResult:
    """Reconstruct a watertight-ish surface from a point cloud (screened Poisson).

    Steps: optional voxel downsampling → statistical outlier removal → normal
    estimation and consistent orientation → Poisson → removal of low-density
    vertices (the "balloon" Poisson extrapolates into empty space) → crop to the
    input bounding box.

    Raises:
        DependencyMissingError: Open3D not installed.
        ProcessingError: too few points or empty result.
    """
    if not 5 <= depth <= 12:
        raise InputValidationError("depth must be in [5, 12].")
    if not 0 <= density_quantile < 0.5:
        raise InputValidationError("density_quantile must be in [0, 0.5).")
    o3d = require_open3d()

    src = validate_input_file(
        cloud_path, allowed_extensions=POINT_CLOUD_EXTENSIONS, max_bytes=settings.max_input_bytes
    )
    out = prepare_output_file(output_path, allowed_extensions=MESH_EXTENSIONS, overwrite=overwrite)

    pcd = o3d.io.read_point_cloud(str(src))
    n_input = len(pcd.points)
    if n_input < 100:
        raise ProcessingError(f"Point cloud too small for meshing ({n_input} points).")

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=outlier_std_ratio)
    extent = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
    radius = max(extent * 0.01, 1e-6)
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    if density_quantile > 0 and len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, density_quantile))
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ProcessingError("Poisson reconstruction produced an empty mesh.")
    if len(mesh.triangles) > settings.max_mesh_faces:
        mesh = mesh.simplify_quadric_decimation(settings.max_mesh_faces)

    if not o3d.io.write_triangle_mesh(str(out), mesh):
        raise ProcessingError(f"Open3D could not write {out.name}")
    logger.info("Poisson mesh %s: %d faces", out.name, len(mesh.triangles))
    return MeshingResult(
        mesh_path=str(out),
        input_points=n_input,
        filtered_points=len(pcd.points),
        vertices=len(mesh.vertices),
        faces=len(mesh.triangles),
        poisson_depth=depth,
    )
