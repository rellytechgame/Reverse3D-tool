from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, SecurityViolationError
from reverse3d.mesh_io import SandboxedResolver, load_geometry, load_mesh, save_mesh
from reverse3d.reverse_eng.measure import compute_stats, measure_mesh_file
from reverse3d.reverse_eng.mesh_cleaner import (
    CleanOptions,
    clean_mesh,
    clean_mesh_file,
    fill_boundary_loops,
)


def _open_box() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(2, 2, 2))
    # Remove the two triangles of the +Z face to create a 4-edge hole.
    top = np.flatnonzero(box.face_normals[:, 2] > 0.9)
    mask = np.ones(len(box.faces), bool)
    mask[top] = False
    box.update_faces(mask)
    box.remove_unreferenced_vertices()
    return box


class TestMeshIO:
    @pytest.mark.parametrize("ext", [".stl", ".obj", ".ply", ".glb", ".gltf", ".off"])
    def test_roundtrip(self, tmp_path: Path, settings: Settings, ext: str) -> None:
        box = trimesh.creation.box(extents=(1, 2, 3))
        out = save_mesh(box, tmp_path / f"m{ext}")
        loaded = load_mesh(out, settings)
        assert np.allclose(sorted(loaded.extents), [1, 2, 3], atol=1e-5)

    def test_face_limit(self, box_mesh_path: Path, tmp_path: Path) -> None:
        with pytest.raises(InputValidationError, match="too many faces"):
            load_mesh(box_mesh_path, Settings(workspace=tmp_path, max_mesh_faces=4))

    def test_point_cloud_ply(self, tmp_path: Path, settings: Settings) -> None:
        path = tmp_path / "cloud.ply"
        trimesh.PointCloud(np.random.default_rng(0).random((50, 3))).export(path)
        geometry = load_geometry(path, settings)
        assert isinstance(geometry, trimesh.PointCloud)
        with pytest.raises(InputValidationError):
            load_mesh(path, settings)

    def test_garbage_rejected(self, tmp_path: Path, settings: Settings) -> None:
        bad = tmp_path / "bad.stl"
        bad.write_bytes(b"\x00\x01garbage" * 3)
        with pytest.raises(InputValidationError):
            load_mesh(bad, settings)

    def test_resolver_blocks_parent_traversal(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (tmp_path / "secret.bin").write_bytes(b"secret")
        (model_dir / "ok.bin").write_bytes(b"ok")
        resolver = SandboxedResolver(model_dir / "scene.gltf")
        assert resolver.get("ok.bin") == b"ok"
        for evil in ("../secret.bin", "..%2Fsecret.bin", str(tmp_path / "secret.bin")):
            with pytest.raises(SecurityViolationError):
                resolver.get(evil)


class TestMeasure:
    def test_box_dimensions(self, box_mesh_path: Path, settings: Settings) -> None:
        stats = measure_mesh_file(box_mesh_path, settings)
        assert stats.watertight
        assert stats.obb_extents == pytest.approx([40, 20, 10], abs=1e-4)
        assert stats.volume == pytest.approx(8000)
        assert stats.surface_area == pytest.approx(2 * (800 + 400 + 200))
        assert stats.boundary_edges == 0
        assert stats.components == 1

    def test_open_mesh(self) -> None:
        stats = compute_stats(_open_box())
        assert not stats.watertight
        assert stats.volume is None
        assert stats.boundary_edges == 4


class TestCleaner:
    def test_fill_square_hole(self) -> None:
        filled, holes = fill_boundary_loops(_open_box(), max_hole_edges=10)
        assert holes == 1
        assert filled.is_watertight
        assert filled.is_winding_consistent
        assert abs(filled.volume) == pytest.approx(8.0)

    def test_hole_too_large_is_left_open(self) -> None:
        _, holes = fill_boundary_loops(_open_box(), max_hole_edges=3)
        assert holes == 0

    def test_full_pipeline_removes_debris_and_aligns(self) -> None:
        part = trimesh.creation.box(extents=(30, 10, 5))
        part.apply_transform(trimesh.transformations.rotation_matrix(0.7, [0.3, 0.5, 1.0]))
        part.apply_translation([5, -3, 8])
        debris = trimesh.creation.icosphere(subdivisions=0, radius=0.1)
        debris.apply_translation([100, 100, 100])
        mesh = trimesh.util.concatenate([part, debris])

        cleaned, report = clean_mesh(mesh, CleanOptions(align="obb", ground=True))
        assert report.before.components == 2
        assert report.after.components == 1
        assert report.after.aabb_extents == pytest.approx([30, 10, 5], abs=1e-3)
        assert cleaned.bounds[0][2] == pytest.approx(0, abs=1e-6)
        assert report.after.watertight

    def test_decimation(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=4)
        cleaned, report = clean_mesh(sphere, CleanOptions(target_faces=500))
        assert len(cleaned.faces) <= 600
        assert any(s.startswith("decimate") for s in report.steps)

    def test_scale_and_input_untouched(self) -> None:
        box = trimesh.creation.box(extents=(1, 1, 1))
        cleaned, _ = clean_mesh(box, CleanOptions(scale=1000))
        assert cleaned.extents == pytest.approx([1000, 1000, 1000])
        assert box.extents == pytest.approx([1, 1, 1])

    @pytest.mark.parametrize(
        "options",
        [
            CleanOptions(min_component_ratio=1.5),
            CleanOptions(max_hole_edges=1),
            CleanOptions(smooth_iterations=1000),
            CleanOptions(scale=0),
            CleanOptions(align="bogus"),  # type: ignore[arg-type]
        ],
    )
    def test_invalid_options(self, options: CleanOptions) -> None:
        with pytest.raises(InputValidationError):
            clean_mesh(trimesh.creation.box(), options)

    def test_file_api(self, tmp_path: Path, settings: Settings) -> None:
        src = save_mesh(_open_box(), tmp_path / "open.ply")
        report = clean_mesh_file(src, tmp_path / "closed.stl", settings)
        assert report.after.watertight
        assert Path(report.output_path or "").is_file()
