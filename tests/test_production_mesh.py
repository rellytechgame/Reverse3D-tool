from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh

from reverse3d.config import Settings
from reverse3d.recipes import _production_mesh, _Run


def test_the_core_makes_the_textured_model_light(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverse3d.reconstruction import silhouette

    dense = trimesh.creation.icosphere(subdivisions=5)
    light = trimesh.creation.icosphere(subdivisions=3)
    report = {
        "faces_before": len(dense.faces),
        "faces_after": len(light.faces),
        "watertight": True,
        "deviation": 0.001,
        "removed_pieces": 0,
    }
    fake = SimpleNamespace(optimize_mesh=lambda mesh, target: (light, report))
    monkeypatch.setattr(silhouette, "core_engine", lambda: fake)
    run = _Run(recipe="photos_to_model", settings=Settings(workspace=tmp_path), out=tmp_path)
    assert _production_mesh(run, dense) is light
    assert run.details["production_mesh"]["faces_after"] == len(light.faces)


def test_without_the_core_the_mesh_is_kept(tmp_path: Path) -> None:
    dense = trimesh.creation.icosphere(subdivisions=3)
    run = _Run(recipe="photos_to_model", settings=Settings(workspace=tmp_path), out=tmp_path)
    assert _production_mesh(run, dense) is dense
    assert "production_mesh" not in run.details


def test_the_real_height_turns_the_model_into_millimetres(tmp_path: Path) -> None:
    from reverse3d.recipes import RecipeOptions, _real_scale

    box = trimesh.creation.box(extents=(1.0, 1.0, 2.5))
    box.apply_translation((0, 0, 1.25))
    path = tmp_path / "hull.ply"
    box.export(path)
    run = _Run(recipe="photos_to_model", settings=Settings(workspace=tmp_path), out=tmp_path)
    factor = _real_scale(run, path, RecipeOptions(real_height_mm=300.0), upright=True)
    assert factor == pytest.approx(120.0)  # 2.5 units -> 300 mm
    assert run.details["real_scale"]["units"] == "mm"
    other = _Run(recipe="photos_to_model", settings=Settings(workspace=tmp_path), out=tmp_path)
    assert _real_scale(other, path, RecipeOptions(real_height_mm=300.0), upright=False) == 1.0
    assert other.warnings
