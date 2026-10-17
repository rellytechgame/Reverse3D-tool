from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from reverse3d.config import Settings
from reverse3d.recipes import _clean, _color_from_cloud, _has_vertex_colors, _Run


def _two_tone_cloud(path: Path) -> Path:
    """Points on a sphere: red on top, blue at the bottom (like a COLMAP fused.ply)."""
    pts = trimesh.creation.icosphere(subdivisions=5).vertices
    rgb = np.where(pts[:, 2:3] > 0, [220, 30, 30], [30, 30, 220])
    trimesh.PointCloud(pts, colors=np.column_stack([rgb, np.full(len(pts), 255)])).export(path)
    return path


def test_mesh_takes_the_colours_of_its_point_cloud(tmp_path: Path) -> None:
    plain = tmp_path / "mesh.ply"
    trimesh.creation.icosphere(subdivisions=3).export(plain)
    assert not _has_vertex_colors(plain)
    colored = _color_from_cloud(plain, _two_tone_cloud(tmp_path / "cloud.ply"), tmp_path / "c.ply")
    assert colored is not None and _has_vertex_colors(colored)
    mesh = trimesh.load(colored, process=False)
    top = mesh.vertices[:, 2] > 0.3
    assert mesh.visual.vertex_colors[top, 0].mean() > 150  # red on top
    assert mesh.visual.vertex_colors[~top & (mesh.vertices[:, 2] < -0.3), 2].mean() > 150


def test_cleaning_a_coloured_mesh_also_writes_glb(tmp_path: Path, settings: Settings) -> None:
    plain = tmp_path / "mesh.ply"
    trimesh.creation.icosphere(subdivisions=3).export(plain)
    colored = _color_from_cloud(plain, _two_tone_cloud(tmp_path / "cloud.ply"), tmp_path / "c.ply")
    assert colored is not None
    run = _Run(recipe="photos_to_model", settings=settings, out=tmp_path)
    _clean(run, colored, tmp_path / "model.stl", scale=1.0, overwrite=True)
    assert {"model", "model_color", "model_glb"} <= set(run.outputs)
    glb = trimesh.load(run.outputs["model_glb"], force="mesh")
    assert len(np.unique(np.asarray(glb.visual.vertex_colors)[:, :3], axis=0)) > 1


def test_plain_meshes_get_no_colour_outputs(tmp_path: Path, settings: Settings) -> None:
    plain = tmp_path / "mesh.ply"
    trimesh.creation.icosphere(subdivisions=3).export(plain)
    run = _Run(recipe="mesh_report", settings=settings, out=tmp_path)
    _clean(run, plain, tmp_path / "model.stl", scale=1.0, overwrite=True)
    assert "model_glb" not in run.outputs


def test_glb_stands_up_in_y_up_viewers(tmp_path: Path) -> None:
    from reverse3d.recipes import _export_glb

    tall = trimesh.creation.box(extents=(1.0, 1.0, 4.0))  # Z-up, like our models
    tall.apply_translation((0, 0, 2.0))
    _export_glb(tall, tmp_path / "m.glb")
    scene = trimesh.load(tmp_path / "m.glb")
    extents = scene.to_geometry().extents
    assert extents[1] == pytest.approx(4.0)  # glTF up axis is Y
    # Viewers that read only the geometry (Studio) still get the Z-up mesh.
    raw = next(iter(scene.geometry.values()))
    assert raw.extents[2] == pytest.approx(4.0)
