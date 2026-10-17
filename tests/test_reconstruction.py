from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from reverse3d.config import Settings
from reverse3d.errors import ExternalToolError, InputValidationError, ProcessingError
from reverse3d.reconstruction import photogrammetry as pg
from reverse3d.reconstruction.depth_estimation import depth_to_mesh, predict_depth


class TestDepthToMesh:
    def test_plane_ramp(self) -> None:
        depth = np.tile(np.linspace(0, 1, 64), (48, 1)).astype(np.float32)
        mesh = depth_to_mesh(depth, mode="depth", near=1, far=2, stride=1, max_depth_jump=0.5)
        assert len(mesh.faces) == 2 * 63 * 47
        z = -mesh.vertices[:, 2]
        assert z.min() == pytest.approx(1, abs=0.05) and z.max() == pytest.approx(2, abs=0.05)

    def test_discontinuity_is_cut(self) -> None:
        depth = np.zeros((40, 40), np.float32)
        depth[:, 20:] = 1.0
        mesh = depth_to_mesh(depth, mode="depth", stride=1, max_depth_jump=0.1)
        # The column of quads bridging the step is removed.
        assert len(mesh.faces) == 2 * 39 * 38

    def test_mask_and_colors(self) -> None:
        depth = np.ones((20, 20), np.float32)
        mask = np.zeros((20, 20), bool)
        mask[5:15, 5:15] = True
        rgb = np.full((20, 20, 3), 200, np.uint8)
        mesh = depth_to_mesh(depth, rgb, stride=1, mask=mask)
        assert len(mesh.faces) == 2 * 9 * 9
        assert mesh.visual.vertex_colors.shape[1] == 4

    @pytest.mark.parametrize("kwargs", [{"fov_deg": 1}, {"near": 2, "far": 1}, {"stride": 0}])
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(InputValidationError):
            depth_to_mesh(np.ones((10, 10)), **kwargs)

    def test_model_allowlist_enforced_before_download(self) -> None:
        with pytest.raises(InputValidationError, match="allowlist"):
            predict_depth(np.zeros((8, 8, 3), np.uint8), model="evil/remote-code-model")


class FakeColmap:
    """Stand-in for ``run_command`` that emulates COLMAP's side effects."""

    def __init__(
        self, help_text: str, fail: set[str] | None = None, points_per_image: int = 1000
    ) -> None:
        self.calls: list[list[str]] = []
        self.help_text = help_text
        self.fail = fail or set()
        self.points_per_image = points_per_image

    def __call__(
        self, executable: Path, args: list[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        args = [str(a) for a in args]
        self.calls.append(args)
        command = args[0]
        if len(args) > 1 and args[1] == "-h":
            return subprocess.CompletedProcess(args, 0, self.help_text, "")
        if command in self.fail:
            raise ExternalToolError(f"colmap {command} failed")

        def opt(name: str) -> str:
            return args[args.index(name) + 1]

        if command == "mapper":
            (Path(opt("--output_path")) / "0").mkdir(parents=True)
            (Path(opt("--output_path")) / "1").mkdir(parents=True)
        elif command == "model_analyzer":
            n = 5 if opt("--path").endswith("1") else 2
            return subprocess.CompletedProcess(
                args, 0, "", f"Registered images: {n}\nPoints: {n * self.points_per_image}"
            )
        elif command == "model_converter":
            if opt("--output_type") == "TXT":
                Path(opt("--output_path")).mkdir(parents=True, exist_ok=True)
            else:
                Path(opt("--output_path")).write_text("ply\n")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def photo_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(5):
        cv2.imwrite(str(folder / f"img_{i}.jpg"), np.full((32, 32, 3), i * 40, np.uint8))
    (folder / "notes.txt").write_text("ignored")
    return folder


class TestColmapPipeline:
    def _pipeline(
        self, settings: Settings, fake: FakeColmap, monkeypatch: pytest.MonkeyPatch
    ) -> pg.ColmapPipeline:
        monkeypatch.setattr(pg, "run_command", fake)
        return pg.ColmapPipeline(settings, executable=Path("colmap"))

    def test_sparse_pipeline_new_option_names(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap(
            "--FeatureExtraction.use_gpu --FeatureMatching.use_gpu "
            "--SiftExtraction.max_num_features"
        )
        pipeline = self._pipeline(settings, fake, monkeypatch)
        options = pg.PhotogrammetryOptions(build_mesh=False)
        result = pipeline.run(photo_dir, tmp_path / "recon", options)

        assert result.total_images == 5
        assert result.registered_images == 5  # model "1" is the largest
        assert result.sparse_model_dir.endswith("1")
        assert Path(result.sparse_points_path).is_file()
        assert sorted(p.name for p in (tmp_path / "recon" / "images").iterdir()) == [
            f"img_{i}.jpg" for i in range(5)
        ]
        extractor = next(c for c in fake.calls if c[0] == "feature_extractor" and "-h" not in c)
        assert "--FeatureExtraction.use_gpu" in extractor
        assert "--SiftExtraction.use_gpu" not in extractor
        assert any(c[0] == "exhaustive_matcher" for c in fake.calls)
        assert any("Dense MVS skipped" in w for w in result.warnings)

    def test_legacy_option_names(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap(
            "--SiftExtraction.use_gpu --SiftMatching.use_gpu --SiftExtraction.max_image_size"
        )
        pipeline = self._pipeline(settings, fake, monkeypatch)
        pipeline.run(
            photo_dir,
            tmp_path / "r",
            pg.PhotogrammetryOptions(build_mesh=False, matcher="sequential"),
        )
        extractor = next(c for c in fake.calls if c[0] == "feature_extractor" and "-h" not in c)
        assert (
            "--SiftExtraction.use_gpu" in extractor
            and "--SiftExtraction.max_image_size" in extractor
        )
        assert any(c[0] == "sequential_matcher" for c in fake.calls)

    def test_thin_sparse_model_stops_before_dense(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap("", points_per_image=100)
        pipeline = self._pipeline(settings, fake, monkeypatch)
        with pytest.raises(ProcessingError, match="too little"):
            pipeline.run(photo_dir, tmp_path / "r", pg.PhotogrammetryOptions(use_gpu=True))
        assert not any(call[0] == "patch_match_stereo" for call in fake.calls)

    def test_dense_failure_degrades_in_auto(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap("", fail={"patch_match_stereo"})
        pipeline = self._pipeline(settings, fake, monkeypatch)
        result = pipeline.run(
            photo_dir, tmp_path / "r", pg.PhotogrammetryOptions(use_gpu=True, build_mesh=False)
        )
        assert result.dense_points_path is None
        assert any("Dense stage skipped" in w for w in result.warnings)

    def test_dense_failure_raises_when_forced(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap("", fail={"patch_match_stereo"})
        pipeline = self._pipeline(settings, fake, monkeypatch)
        with pytest.raises(ExternalToolError):
            pipeline.run(
                photo_dir, tmp_path / "r", pg.PhotogrammetryOptions(dense="on", build_mesh=False)
            )

    def test_no_model_registered(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeColmap("")
        original = fake.__call__

        def no_models(
            executable: Path, args: list[str], **kw: Any
        ) -> subprocess.CompletedProcess[str]:
            if args[0] == "mapper":
                return subprocess.CompletedProcess(args, 0, "", "")
            return original(executable, args, **kw)

        monkeypatch.setattr(pg, "run_command", no_models)
        with pytest.raises(ProcessingError):
            pg.ColmapPipeline(settings, executable=Path("colmap")).run(
                photo_dir, tmp_path / "r", pg.PhotogrammetryOptions(build_mesh=False)
            )

    def test_existing_output_requires_overwrite(
        self, settings: Settings, photo_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = self._pipeline(settings, FakeColmap(""), monkeypatch)
        opts = pg.PhotogrammetryOptions(build_mesh=False)
        pipeline.run(photo_dir, tmp_path / "r", opts)
        with pytest.raises(InputValidationError):
            pipeline.run(photo_dir, tmp_path / "r", opts)
        pipeline.run(photo_dir, tmp_path / "r", opts, overwrite=True)

    def test_source_inside_artifacts_is_refused(
        self, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "r"
        src = out / "images"
        src.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(src / f"{i}.png"), np.zeros((8, 8, 3), np.uint8))
        pipeline = self._pipeline(settings, FakeColmap(""), monkeypatch)
        with pytest.raises(InputValidationError):
            pipeline.run(src, out, overwrite=True)
        assert len(list(src.iterdir())) == 3

    @pytest.mark.parametrize(
        "options",
        [
            pg.PhotogrammetryOptions(quality="ultra"),  # type: ignore[arg-type]
            pg.PhotogrammetryOptions(camera_model="EVIL; rm -rf"),
            pg.PhotogrammetryOptions(dense="maybe"),  # type: ignore[arg-type]
        ],
    )
    def test_invalid_options(self, options: pg.PhotogrammetryOptions) -> None:
        with pytest.raises(InputValidationError):
            options.validate()


def test_object_solid_is_closed_and_stands_on_its_base() -> None:
    from reverse3d.reconstruction.depth_estimation import object_solid

    yy, xx = np.mgrid[0:200, 0:200]
    r2 = (xx - 100) ** 2 + (yy - 100) ** 2
    mask = np.where(r2 < 60**2, 255, 0).astype(np.uint8)
    # Disparity: a tilted desk plane plus a dome where the object is.
    disparity = 0.002 * yy + np.where(mask > 0, np.sqrt(np.clip(60**2 - r2, 0, None)) / 60, 0)
    solid = object_solid(disparity.astype(np.float32), mask, stride=2)
    assert solid.is_watertight
    assert solid.volume > 0
    assert solid.bounds[0][2] == pytest.approx(0.0)
    width = solid.extents[0]
    assert 0.2 * width < solid.extents[2] < 0.5 * width


def test_object_solid_rejects_empty_mask() -> None:
    from reverse3d.reconstruction.depth_estimation import object_solid

    with pytest.raises(ProcessingError):
        object_solid(np.zeros((50, 50), np.float32), np.zeros((50, 50), np.uint8))
