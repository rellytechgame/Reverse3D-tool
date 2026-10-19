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


def test_colmap_depth_maps_are_read_in_image_layout(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    from reverse3d.reconstruction.silhouette import read_depth_map

    depth = np.arange(12, dtype=np.float32).reshape(3, 4)  # 3 rows, 4 columns
    path = tmp_path / "a.jpg.geometric.bin"
    path.write_bytes(b"4&3&1&" + depth.tobytes())  # COLMAP: rows one after another
    assert np.array_equal(read_depth_map(path), depth)


def test_colmap_bat_launcher_is_replaced_by_the_real_exe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from reverse3d.reconstruction.photogrammetry import _real_executable

    bat = tmp_path / "COLMAP.bat"
    bat.write_text("@echo off")
    assert _real_executable(bat) == bat  # no exe next to it: keep the launcher
    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / "colmap.exe"
    exe.write_bytes(b"")
    assert _real_executable(bat) == exe  # paths with spaces break cmd.exe
    assert _real_executable(exe) == exe


def test_models_commands_need_the_core_and_a_pack_file(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from reverse3d import ai_models
    from reverse3d.errors import DependencyMissingError, InputValidationError

    monkeypatch.setattr("reverse3d.reconstruction.silhouette.core_engine", lambda: None)
    with pytest.raises(DependencyMissingError):
        ai_models.list_models()
    monkeypatch.setattr(ai_models, "_core_models", lambda: object())
    with pytest.raises(InputValidationError):
        ai_models.install_model(tmp_path / "not-a-pack.zip")
