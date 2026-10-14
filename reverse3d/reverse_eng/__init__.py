"""Reverse engineering: mesh repair, measurement, primitive fitting and CAD export."""

from reverse3d.reverse_eng.geometry_fit import (
    FitReport,
    PrimitiveFit,
    detect_primitives,
    fit_primitives_file,
)
from reverse3d.reverse_eng.measure import MeshStats, compute_stats, measure_mesh_file
from reverse3d.reverse_eng.mesh_cleaner import (
    CleanOptions,
    CleanReport,
    clean_mesh,
    clean_mesh_file,
)
from reverse3d.reverse_eng.parametric import export_parametric, generate_cadquery, generate_openscad

__all__ = [
    "CleanOptions",
    "CleanReport",
    "FitReport",
    "MeshStats",
    "PrimitiveFit",
    "clean_mesh",
    "clean_mesh_file",
    "compute_stats",
    "detect_primitives",
    "export_parametric",
    "fit_primitives_file",
    "generate_cadquery",
    "generate_openscad",
    "measure_mesh_file",
]
