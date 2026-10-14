from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, ProcessingError
from reverse3d.imaging import load_image
from reverse3d.vision.feature_matcher import analyze_image_set, match_pair
from reverse3d.vision.mask_extractor import extract_mask, silhouette_to_mesh


class TestMaskExtraction:
    def test_plate_with_hole(self, plan_image: Path, tmp_path: Path, settings: Settings) -> None:
        result = extract_mask(plan_image, tmp_path / "out", settings, units_per_pixel=0.5)
        assert result.polygon_count == 1
        doc = json.loads(Path(result.contours_path).read_text())
        polygon = doc["polygons"][0]
        assert len(polygon["holes"]) == 1
        expected = 200 * 120 - np.pi * 30**2
        assert polygon["area_px"] == pytest.approx(expected, rel=0.05)
        assert Path(result.svg_path).read_text().startswith("<svg")
        assert Path(result.mask_path).is_file()

    def test_refuses_to_overwrite(
        self, plan_image: Path, tmp_path: Path, settings: Settings
    ) -> None:
        extract_mask(plan_image, tmp_path / "out", settings)
        with pytest.raises(InputValidationError):
            extract_mask(plan_image, tmp_path / "out", settings)
        extract_mask(plan_image, tmp_path / "out", settings, overwrite=True)

    def test_blank_image_reports_no_region(self, tmp_path: Path, settings: Settings) -> None:
        blank = tmp_path / "blank.png"
        cv2.imwrite(str(blank), np.full((100, 100, 3), 255, np.uint8))
        with pytest.raises(ProcessingError):
            extract_mask(blank, tmp_path / "out", settings)

    def test_alpha_requires_rgba(
        self, plan_image: Path, tmp_path: Path, settings: Settings
    ) -> None:
        with pytest.raises(InputValidationError):
            extract_mask(plan_image, tmp_path / "out", settings, method="alpha")

    def test_rejects_bad_arguments(
        self, plan_image: Path, tmp_path: Path, settings: Settings
    ) -> None:
        with pytest.raises(InputValidationError):
            extract_mask(plan_image, tmp_path / "o", settings, min_area_ratio=2)

    def test_extrude_to_scaled_solid(
        self, plan_image: Path, tmp_path: Path, settings: Settings
    ) -> None:
        mask = extract_mask(plan_image, tmp_path / "out", settings, units_per_pixel=0.5)
        info = silhouette_to_mesh(mask.contours_path, tmp_path / "plate.stl", settings, thickness=3)
        assert info["watertight"] is True
        ext = sorted(info["extents"], reverse=True)
        assert ext[0] == pytest.approx(100, rel=0.03)  # 200 px * 0.5
        assert ext[1] == pytest.approx(60, rel=0.03)
        assert ext[2] == pytest.approx(3)
        plate = 100 * 60 - np.pi * 15**2
        assert info["volume"] == pytest.approx(plate * 3, rel=0.06)

    def test_malformed_contours_rejected(self, tmp_path: Path, settings: Settings) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema_version": 1, "polygons": [{"exterior": "x"}]}))
        with pytest.raises(InputValidationError):
            silhouette_to_mesh(bad, tmp_path / "x.stl", settings)
        bad.write_text("{not json")
        with pytest.raises(InputValidationError):
            silhouette_to_mesh(bad, tmp_path / "x.stl", settings)


class TestImaging:
    def test_pixel_limit(self, plan_image: Path, tmp_path: Path) -> None:
        tiny = Settings(workspace=tmp_path, max_image_pixels=1000)
        with pytest.raises(InputValidationError, match="too large"):
            load_image(plan_image, tiny)

    def test_fake_image_rejected(self, tmp_path: Path, settings: Settings) -> None:
        fake = tmp_path / "fake.png"
        fake.write_bytes(b"not an image at all")
        with pytest.raises(InputValidationError):
            load_image(fake, settings)


class TestFeatureMatching:
    @pytest.mark.parametrize("detector", ["sift", "orb"])
    def test_overlapping_views_match(
        self, textured_pair: tuple[Path, Path], settings: Settings, detector: str
    ) -> None:
        result = match_pair(*textured_pair, settings, detector=detector)  # type: ignore[arg-type]
        assert result.inliers >= 50
        assert result.inlier_ratio > 0.5

    def test_visualization(
        self, textured_pair: tuple[Path, Path], tmp_path: Path, settings: Settings
    ) -> None:
        out = tmp_path / "viz" / "m.jpg"
        result = match_pair(*textured_pair, settings, visualization_path=out)
        assert result.visualization_path and Path(result.visualization_path).is_file()

    def test_unrelated_images_do_not_match(self, tmp_path: Path, settings: Settings) -> None:
        rng = np.random.default_rng(1)
        paths = []
        for i in range(2):
            p = tmp_path / f"n{i}.png"
            cv2.imwrite(str(p), (rng.random((300, 300)) * 255).astype(np.uint8))
            paths.append(p)
        result = match_pair(paths[0], paths[1], settings)
        assert result.inliers < 50

    def test_analyze_set(self, textured_pair: tuple[Path, Path], settings: Settings) -> None:
        report = analyze_image_set(textured_pair[0].parent, settings)
        assert report.image_count == 2
        assert len(report.pairs) == 1
        assert report.suitable_for_photogrammetry is False  # only 2 images
        assert any("20-40" in r for r in report.recommendations)
