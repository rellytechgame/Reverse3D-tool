"""Photogrammetry pipeline built on the COLMAP command-line interface.

Stages::

    stage images → feature_extractor → (exhaustive|sequential)_matcher → mapper
      → model_analyzer (pick largest model) → model_converter (sparse PLY)
      → [dense: image_undistorter → patch_match_stereo → stereo_fusion]  (CUDA)
      → [mesh: Poisson via Open3D]                                        ([recon])

Design notes:

* COLMAP option names changed across releases (e.g. ``SiftExtraction.use_gpu``
  became ``FeatureExtraction.use_gpu`` in 3.12). Instead of pinning a version,
  each stage probes ``<command> -h`` once and uses whichever option exists.
* Input images are validated and **copied** into ``<output>/images`` so COLMAP
  only ever reads files that passed the allowlist and size checks.
* Dense MVS requires CUDA. With ``dense="auto"`` a dense failure degrades
  gracefully to the sparse cloud and is reported as a warning.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from reverse3d.config import Settings
from reverse3d.errors import (
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    ProcessingError,
)
from reverse3d.security import (
    list_images,
    prepare_output_dir,
    resolve_executable,
    run_command,
    validate_input_dir,
)

logger = logging.getLogger(__name__)

Quality = Literal["low", "medium", "high"]
MatcherKind = Literal["auto", "exhaustive", "sequential", "spatial"]
DenseMode = Literal["auto", "on", "off"]

CAMERA_MODELS = frozenset(
    {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}
)

_PRESETS: dict[str, dict[str, int]] = {
    "low": {"max_image_size": 1000, "max_features": 4096, "poisson_depth": 8},
    "medium": {"max_image_size": 1600, "max_features": 8192, "poisson_depth": 9},
    "high": {"max_image_size": 3200, "max_features": 16384, "poisson_depth": 10},
}

# Below these, the sparse model is too thin for a usable surface: stop before the
# (slow) dense stage so the caller can fall back to something better.
_MIN_REGISTERED = 5
_MIN_SPARSE_POINTS = 1000

# Above this many images exhaustive matching (O(n²)) becomes impractical.
_EXHAUSTIVE_LIMIT = 150

# Names of the artifacts this pipeline creates; only these are removed on overwrite.
_ARTIFACTS = (
    "database.db",
    "images",
    "sparse",
    "sparse_txt",
    "dense",
    "sparse_points.ply",
    "mesh.ply",
)


@dataclass
class PhotogrammetryOptions:
    """User-facing options of :meth:`ColmapPipeline.run`."""

    quality: Quality = "medium"
    matcher: MatcherKind = "auto"
    dense: DenseMode = "auto"
    use_gpu: bool = False
    camera_model: str = "SIMPLE_RADIAL"
    single_camera: bool = True
    build_mesh: bool = True
    #: Poisson octree depth; None uses the quality preset (scene profiles raise it).
    poisson_depth: int | None = None
    #: share of the least supported surface trimmed after Poisson.
    density_quantile: float = 0.02
    #: "sift" (fast, GPU) or "aliked": learned ALIKED features + LightGlue matching,
    #: which places photos of hard sets SIFT cannot (CPU, much slower).
    features: Literal["sift", "aliked"] = "sift"

    def validate(self) -> None:
        if self.quality not in _PRESETS:
            raise InputValidationError(f"quality must be one of {sorted(_PRESETS)}")
        if self.matcher not in ("auto", "exhaustive", "sequential", "spatial"):
            raise InputValidationError("matcher must be auto, exhaustive, sequential or spatial")
        if self.poisson_depth is not None and not 5 <= self.poisson_depth <= 13:
            raise InputValidationError("poisson_depth must be in [5, 13]")
        if not 0 <= self.density_quantile < 0.5:
            raise InputValidationError("density_quantile must be in [0, 0.5)")
        if self.dense not in ("auto", "on", "off"):
            raise InputValidationError("dense must be auto, on or off")
        if self.camera_model not in CAMERA_MODELS:
            raise InputValidationError(f"camera_model must be one of {sorted(CAMERA_MODELS)}")


@dataclass
class StageTiming:
    name: str
    seconds: float


@dataclass
class PhotogrammetryResult:
    """Outputs of a photogrammetry run (paths are absolute strings)."""

    output_dir: str
    sparse_model_dir: str
    sparse_points_path: str
    dense_points_path: str | None
    mesh_path: str | None
    total_images: int
    registered_images: int
    sparse_points: int
    stages: list[StageTiming] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: the photos as COLMAP saw them, and the chosen model as text (silhouette carving).
    images_dir: str | None = None
    sparse_text_dir: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ColmapPipeline:
    """Thin, safe wrapper around the ``colmap`` executable."""

    def __init__(self, settings: Settings, executable: Path | None = None) -> None:
        self.settings = settings
        self.executable = executable or resolve_executable(
            ["colmap", "COLMAP", "colmap.exe", "COLMAP.bat"], settings.colmap_path, tool="colmap"
        )
        self._help_cache: dict[str, str] = {}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _help(self, command: str) -> str:
        if command not in self._help_cache:
            proc = run_command(
                self.executable, [command, "-h"], timeout_s=60, tool="colmap", allow_nonzero=True
            )
            self._help_cache[command] = (proc.stdout or "") + (proc.stderr or "")
        return self._help_cache[command]

    def _pick_option(self, command: str, candidates: tuple[str, ...]) -> str | None:
        help_text = self._help(command)
        for option in candidates:
            if f"--{option}" in help_text:
                return option
        return None

    def _with_options(
        self, command: str, args: list[str], options: dict[tuple[str, ...], str]
    ) -> list[str]:
        for candidates, value in options.items():
            name = self._pick_option(command, candidates)
            if name:
                args += [f"--{name}", value]
            else:
                logger.debug("COLMAP %s: none of %s supported, skipping", command, candidates)
        return args

    def _stage(self, result: PhotogrammetryResult, name: str, args: list[str]) -> str:
        start = time.perf_counter()
        logger.info("COLMAP stage: %s", name)
        proc = run_command(
            self.executable,
            args,
            timeout_s=self.settings.subprocess_timeout_s,
            tool=f"colmap {name}",
        )
        result.stages.append(StageTiming(name, round(time.perf_counter() - start, 2)))
        return (proc.stdout or "") + (proc.stderr or "")

    def version(self) -> str:
        """Return the COLMAP version banner (first line of ``colmap help``)."""
        proc = run_command(
            self.executable, ["help"], timeout_s=60, tool="colmap", allow_nonzero=True
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return text.splitlines()[0] if text else "unknown"

    def has_cuda(self) -> bool:
        """False only when COLMAP's banner states it was built ``without CUDA``."""
        proc = run_command(
            self.executable, ["help"], timeout_s=60, tool="colmap", allow_nonzero=True
        )
        return "without cuda" not in ((proc.stdout or "") + (proc.stderr or "")).lower()

    # ── pipeline ──────────────────────────────────────────────────────────────

    def _prepare_workspace(self, image_dir: Path, output_dir: Path, overwrite: bool) -> list[Path]:
        images = list_images(
            image_dir, max_images=self.settings.max_images, max_bytes=self.settings.max_input_bytes
        )
        if len(images) < 3:
            raise InputValidationError("Photogrammetry needs at least 3 images.")
        existing = [name for name in _ARTIFACTS if (output_dir / name).exists()]
        if existing and not overwrite:
            raise InputValidationError(
                "Output directory already contains a reconstruction.",
                hint="Use another output_dir or pass overwrite=true.",
            )
        for name in existing:
            target = output_dir / name
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        staged = output_dir / "images"
        staged.mkdir()
        for img in images:
            shutil.copy2(img, staged / img.name)
        return images

    @staticmethod
    def _parse_model_stats(text: str) -> tuple[int, int]:
        def grab(label: str) -> int:
            match = re.search(rf"{label}\s*:\s*(\d+)", text)
            return int(match.group(1)) if match else 0

        return grab("Registered images"), grab("Points")

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        options: PhotogrammetryOptions | None = None,
        *,
        overwrite: bool = False,
    ) -> PhotogrammetryResult:
        """Run the full pipeline and return paths and statistics.

        Raises:
            InputValidationError: invalid options or image folder.
            ExternalToolError: a COLMAP stage failed.
            ProcessingError: no model could be reconstructed.
        """
        options = options or PhotogrammetryOptions()
        options.validate()
        preset = _PRESETS[options.quality]
        src = validate_input_dir(image_dir)
        out = prepare_output_dir(output_dir)
        # Overwrite deletes our artifact folders; never let that touch the source photos.
        if any(src.is_relative_to(out / name) for name in _ARTIFACTS):
            raise InputValidationError(
                "image_dir must not be inside the reconstruction artifacts of output_dir."
            )
        images = self._prepare_workspace(src, out, overwrite)

        db = out / "database.db"
        staged = out / "images"
        sparse_root = out / "sparse"
        sparse_root.mkdir()
        result = PhotogrammetryResult(
            output_dir=str(out),
            sparse_model_dir="",
            sparse_points_path="",
            dense_points_path=None,
            mesh_path=None,
            total_images=len(images),
            registered_images=0,
            sparse_points=0,
        )
        if options.use_gpu and not self.has_cuda():
            # A CPU-only COLMAP build fails (or uses headless-incompatible OpenGL)
            # with use_gpu=1, so degrade instead of failing the whole job.
            options = replace(options, use_gpu=False)
            result.warnings.append("COLMAP was built without CUDA; running on CPU.")
        gpu = "1" if options.use_gpu else "0"

        # 1. Features
        args = [
            "feature_extractor",
            "--database_path",
            str(db),
            "--image_path",
            str(staged),
            "--ImageReader.camera_model",
            options.camera_model,
            "--ImageReader.single_camera",
            "1" if options.single_camera else "0",
        ]
        learned = options.features == "aliked"
        if learned:
            from reverse3d.offline import colmap_model_path, download_colmap_models

            if not colmap_model_path("aliked-n16rot.onnx"):
                download_colmap_models()
            aliked = colmap_model_path("aliked-n16rot.onnx")
            if aliked is None:
                raise DependencyMissingError(
                    "The AI feature model (ALIKED) is not on this computer.",
                    hint="Connect to the internet once and run: reverse3d offline prepare",
                )
            # ONNX on the CPU: the CUDA runtime it needs is not bundled with COLMAP.
            args += [
                "--FeatureExtraction.type",
                "ALIKED_N16ROT",
                "--FeatureExtraction.use_gpu",
                "0",
                "--AlikedExtraction.n16rot_model_path",
                str(aliked),
            ]
        self._stage(
            result,
            "feature_extractor",
            self._with_options(
                "feature_extractor",
                args,
                {}
                if learned
                else {
                    ("FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"): gpu,
                    ("FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"): str(
                        preset["max_image_size"]
                    ),
                    ("SiftExtraction.max_num_features",): str(preset["max_features"]),
                },
            ),
        )

        # 2. Matching
        matcher = options.matcher
        if matcher == "spatial":
            from reverse3d.scenes import has_gps

            # GPS-guided matching (drone / large outdoor sets); without GPS, sequential.
            matcher = "spatial" if has_gps(staged) else "sequential"
        if matcher == "auto":
            matcher = "exhaustive" if len(images) <= _EXHAUSTIVE_LIMIT else "sequential"
        command = f"{matcher}_matcher"
        extra: dict[tuple[str, ...], str] = {
            ("FeatureMatching.use_gpu", "SiftMatching.use_gpu"): "0" if learned else gpu
        }
        if learned:
            from reverse3d.offline import colmap_model_path

            glue = colmap_model_path("aliked-lightglue.onnx")
            if glue is None:
                raise DependencyMissingError(
                    "The AI matching model (LightGlue) is not on this computer.",
                    hint="Connect to the internet once and run: reverse3d offline prepare",
                )
            extra[("FeatureMatching.type",)] = "ALIKED_LIGHTGLUE"
            extra[("AlikedMatching.lightglue_model_path",)] = str(glue)
        if matcher == "spatial":
            extra[("SpatialMatching.is_gps",)] = "1"
        try:
            self._stage(
                result,
                command,
                self._with_options(command, [command, "--database_path", str(db)], extra),
            )
        except ExternalToolError:
            if matcher != "spatial":
                raise
            result.warnings.append("GPS matching failed; used sequential matching instead.")
            self._stage(
                result,
                "sequential_matcher",
                self._with_options(
                    "sequential_matcher",
                    ["sequential_matcher", "--database_path", str(db)],
                    {("FeatureMatching.use_gpu", "SiftMatching.use_gpu"): gpu},
                ),
            )

        # 3. Sparse reconstruction (SfM)
        self._stage(
            result,
            "mapper",
            [
                "mapper",
                "--database_path",
                str(db),
                "--image_path",
                str(staged),
                "--output_path",
                str(sparse_root),
            ],
        )
        models = sorted(p for p in sparse_root.iterdir() if p.is_dir())
        if not models:
            raise ProcessingError(
                "COLMAP could not register the images into a 3D model.",
                hint="Run the image analysis tool: more overlap, texture and sharpness are needed.",
            )
        best, best_stats = models[0], (0, 0)
        for model in models:
            stats = self._parse_model_stats(
                self._stage(
                    result, f"model_analyzer:{model.name}", ["model_analyzer", "--path", str(model)]
                )
            )
            if stats[0] > best_stats[0]:
                best, best_stats = model, stats
        result.sparse_model_dir = str(best)
        result.registered_images, result.sparse_points = best_stats
        if result.registered_images < _MIN_REGISTERED or result.sparse_points < _MIN_SPARSE_POINTS:
            raise ProcessingError(
                f"COLMAP placed {result.registered_images}/{len(images)} photos with only "
                f"{result.sparse_points} 3D points: too little for a reliable model.",
                hint="More overlap, a still object, even light and matte surfaces are needed.",
            )
        if result.registered_images and result.registered_images < len(images) * 0.5:
            result.warnings.append(
                f"Only {result.registered_images}/{len(images)} images were registered."
            )

        sparse_ply = out / "sparse_points.ply"
        self._stage(
            result,
            "model_converter",
            [
                "model_converter",
                "--input_path",
                str(best),
                "--output_path",
                str(sparse_ply),
                "--output_type",
                "PLY",
            ],
        )
        result.sparse_points_path = str(sparse_ply)
        result.images_dir = str(staged)
        text_dir = out / "sparse_txt"
        text_dir.mkdir(parents=True, exist_ok=True)
        self._stage(
            result,
            "model_converter_txt",
            [
                "model_converter",
                "--input_path",
                str(best),
                "--output_path",
                str(text_dir),
                "--output_type",
                "TXT",
            ],
        )
        result.sparse_text_dir = str(text_dir)

        # 4. Dense reconstruction (MVS) — CUDA only
        run_dense = options.dense == "on" or (options.dense == "auto" and options.use_gpu)
        if run_dense:
            try:
                result.dense_points_path = str(self._dense(result, staged, best, out, preset))
            except ExternalToolError as exc:
                if options.dense == "on":
                    raise
                result.warnings.append(f"Dense stage skipped ({exc.message}); using sparse cloud.")
        elif options.dense == "auto":
            result.warnings.append("Dense MVS skipped (requires use_gpu=true and CUDA COLMAP).")

        # 5. Surface
        if options.build_mesh:
            from reverse3d.reconstruction.pointcloud import poisson_mesh

            cloud = result.dense_points_path or result.sparse_points_path
            try:
                meshing = poisson_mesh(
                    cloud,
                    out / "mesh.ply",
                    self.settings,
                    depth=options.poisson_depth or preset["poisson_depth"],
                    density_quantile=options.density_quantile,
                    overwrite=True,
                )
                result.mesh_path = meshing.mesh_path
            except (DependencyMissingError, ProcessingError) as exc:
                result.warnings.append(f"Meshing skipped: {exc.message}")

        logger.info(
            "Photogrammetry done: %d/%d images registered", result.registered_images, len(images)
        )
        return result

    def _dense(
        self,
        result: PhotogrammetryResult,
        staged: Path,
        model: Path,
        out: Path,
        preset: dict[str, int],
    ) -> Path:
        dense = out / "dense"
        self._stage(
            result,
            "image_undistorter",
            [
                "image_undistorter",
                "--image_path",
                str(staged),
                "--input_path",
                str(model),
                "--output_path",
                str(dense),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(preset["max_image_size"]),
            ],
        )
        self._stage(
            result,
            "patch_match_stereo",
            [
                "patch_match_stereo",
                "--workspace_path",
                str(dense),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "true",
            ],
        )
        fused = dense / "fused.ply"
        self._stage(
            result,
            "stereo_fusion",
            [
                "stereo_fusion",
                "--workspace_path",
                str(dense),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_path",
                str(fused),
            ],
        )
        return fused


def run_photogrammetry(
    image_dir: str | Path,
    output_dir: str | Path,
    settings: Settings,
    options: PhotogrammetryOptions | None = None,
    *,
    overwrite: bool = False,
) -> PhotogrammetryResult:
    """Convenience wrapper: locate COLMAP and run :class:`ColmapPipeline`."""
    return ColmapPipeline(settings).run(image_dir, output_dir, options, overwrite=overwrite)
