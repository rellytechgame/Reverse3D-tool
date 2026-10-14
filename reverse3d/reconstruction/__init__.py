"""3D reconstruction: COLMAP photogrammetry, monocular depth and point-cloud meshing.

Heavy optional imports (Open3D, PyTorch) happen inside the functions that need
them, so importing this package is always cheap.
"""

from reverse3d.reconstruction.depth_estimation import depth_to_mesh, estimate_depth
from reverse3d.reconstruction.photogrammetry import (
    ColmapPipeline,
    PhotogrammetryOptions,
    run_photogrammetry,
)
from reverse3d.reconstruction.pointcloud import poisson_mesh

__all__ = [
    "ColmapPipeline",
    "PhotogrammetryOptions",
    "depth_to_mesh",
    "estimate_depth",
    "poisson_mesh",
    "run_photogrammetry",
]
