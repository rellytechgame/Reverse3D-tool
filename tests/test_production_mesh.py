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


def test_agents_can_be_switched_off_for_benchmarks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from reverse3d.reconstruction.silhouette import agent_skipped, core_engine

    monkeypatch.setenv("R3D_SKIP", "fidelity, depth")
    assert agent_skipped("fidelity") and agent_skipped("depth")
    assert not agent_skipped("photo")
    monkeypatch.setenv("R3D_SKIP", "core")
    assert core_engine() is None


def test_own_vision_model_maps_outputs_back_to_the_photo() -> None:
    from typing import ClassVar

    import numpy as np

    from reverse3d.ai_models import VisionModel

    class Info:
        name, version = "optiverse-1", "9.9.9"
        profiles = ("object", "character", "architecture", "landscape")
        manifest: ClassVar = {"input": {"size": 28}, "profiles": list(profiles)}

    class Session:
        def run(self, _outputs, feeds):  # type: ignore[no-untyped-def]
            x = feeds["pixels"]
            assert x.shape == (1, 3, 28, 28)
            n = x.shape[2]
            depth = np.tile(np.arange(n, dtype=np.float32), (n, 1))[None]
            return depth, np.full((1, n, n), 5.0, np.float32), np.array([[0.0, 4.0, 0.0, 0.0]])

    model = VisionModel(Session(), Info())
    disp, mask, scene = model.run(np.zeros((40, 20, 3), np.uint8))  # portrait photo
    assert disp.shape == (40, 20) and mask.shape == (40, 20)
    assert max(scene, key=scene.__getitem__) == "character"
    assert float(mask.min()) > 0.99  # sigmoid of the logits
    assert model.label == "optiverse-1 9.9.9"


def test_own_segmenter_keeps_only_the_hinted_object() -> None:
    import numpy as np

    from reverse3d.ai_models import OwnSegmenter

    class Model:
        def run(self, rgb):  # type: ignore[no-untyped-def]
            prob = np.zeros(rgb.shape[:2], np.float32)
            prob[10:30, 10:30] = 0.9  # the object
            prob[50:60, 50:60] = 0.9  # a chair in the background
            return None, prob, {}

    seg = OwnSegmenter(Model())  # type: ignore[arg-type]
    state = seg.embed(np.zeros((80, 80, 3), np.uint8))
    mask = seg.segment(state, (0, 0, 79, 79), (20, 20))
    assert mask[20, 20] == 255 and mask[55, 55] == 0
    # Outside the box nothing is kept, even the object.
    assert seg.segment(state, (40, 40, 79, 79), (55, 55))[20, 20] == 0
