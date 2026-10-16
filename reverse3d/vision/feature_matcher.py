"""Image quality analysis and feature matching for multi-view capture sets.

Before spending minutes (or hours) in COLMAP, an agent can call
:func:`analyze_image_set` to learn whether the photos are sharp, well exposed,
textured enough and overlapping enough, and get actionable recommendations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError
from reverse3d.imaging import downscale, load_image, save_image
from reverse3d.security import list_images, validate_input_dir

logger = logging.getLogger(__name__)

# AKAZE is intentionally not offered: OpenCV 5 moved it out of the base package.
Detector = Literal["sift", "orb"]

_ANALYSIS_MAX_DIM = 1600
# Empirical thresholds on images downscaled to 1600 px; documented in docs/architecture.md.
_BLUR_THRESHOLD = 100.0
_MIN_KEYPOINTS = 500
_MIN_PAIR_INLIERS = 40
# Set analysis: features per image, and how many images still get all-pairs matching.
_SET_FEATURES = 3000
_ALL_PAIRS_LIMIT = 80
# Share of the photos that must form one overlapping group.
_MIN_CONNECTED_SHARE = 0.7


@dataclass
class ImageQuality:
    """Per-image quality metrics."""

    name: str
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float
    keypoints: int
    issues: list[str] = field(default_factory=list)


@dataclass
class PairMatch:
    """Feature matching statistics for an image pair."""

    image_a: str
    image_b: str
    detector: str
    keypoints_a: int
    keypoints_b: int
    tentative_matches: int
    inliers: int
    inlier_ratio: float
    visualization_path: str | None = None


@dataclass
class ImageSetReport:
    """Aggregate report returned by :func:`analyze_image_set`."""

    image_count: int
    images: list[ImageQuality]
    pairs: list[PairMatch]
    weak_links: list[str]
    suitable_for_photogrammetry: bool
    recommendations: list[str]
    #: photos in the largest group that overlaps well (what a reconstruction can use).
    connected_images: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _create_detector(detector: Detector, max_features: int) -> cv2.Feature2D:
    if detector == "sift":
        return cv2.SIFT.create(nfeatures=max_features)
    if detector == "orb":
        return cv2.ORB.create(nfeatures=max_features)
    raise InputValidationError(f"Unknown detector: {detector}")


def _norm_type(detector: Detector) -> int:
    return cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def assess_image(path: Path, settings: Settings, detector: Detector = "sift") -> ImageQuality:
    """Compute sharpness (variance of Laplacian), exposure and texture metrics."""
    image = load_image(path, settings)
    h, w = image.shape[:2]
    small, _ = downscale(image, _ANALYSIS_MAX_DIM)
    gray = _gray(small)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    keypoints = len(_create_detector(detector, 8000).detect(gray, None))

    issues: list[str] = []
    if sharpness < _BLUR_THRESHOLD:
        issues.append("blurry")
    if brightness < 40:
        issues.append("underexposed")
    elif brightness > 220:
        issues.append("overexposed")
    if contrast < 20:
        issues.append("low_contrast")
    if keypoints < _MIN_KEYPOINTS:
        issues.append("low_texture")
    return ImageQuality(
        name=path.name,
        width=w,
        height=h,
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        keypoints=keypoints,
        issues=issues,
    )


def _verified_matches(
    kp_a: Sequence[cv2.KeyPoint],
    des_a: np.ndarray | None,
    kp_b: Sequence[cv2.KeyPoint],
    des_b: np.ndarray | None,
    detector: Detector,
    ratio: float = 0.75,
) -> tuple[list[cv2.DMatch], list[cv2.DMatch]]:
    """Lowe's ratio test, then RANSAC on the fundamental matrix: (tentative, inliers)."""
    tentative: list[cv2.DMatch] = []
    if des_a is not None and des_b is not None and len(des_a) >= 2 and len(des_b) >= 2:
        matcher = cv2.BFMatcher(_norm_type(detector))
        for pair in matcher.knnMatch(des_a, des_b, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                tentative.append(pair[0])
    inliers: list[cv2.DMatch] = []
    if len(tentative) >= 8:
        pts_a = np.asarray([kp_a[m.queryIdx].pt for m in tentative], dtype=np.float32)
        pts_b = np.asarray([kp_b[m.trainIdx].pt for m in tentative], dtype=np.float32)
        _, inlier_mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.5, 0.999)
        if inlier_mask is not None:
            inliers = [m for m, ok in zip(tentative, inlier_mask.ravel(), strict=True) if ok]
    return tentative, inliers


def _largest_group(n: int, edges: list[tuple[int, int]]) -> set[int]:
    """Largest set of images linked (directly or through others) by good pairs."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return max(groups.values(), key=len) if groups else set()


def match_pair(
    image_a: str | Path,
    image_b: str | Path,
    settings: Settings,
    *,
    detector: Detector = "sift",
    ratio: float = 0.75,
    max_features: int = 8000,
    visualization_path: str | Path | None = None,
    overwrite: bool = False,
) -> PairMatch:
    """Detect and match features between two images and verify them geometrically.

    Uses Lowe's ratio test followed by a RANSAC fundamental-matrix estimation, so
    ``inliers`` approximates how many correspondences a reconstruction could use.

    Args:
        visualization_path: optional PNG/JPG showing the inlier matches.

    Raises:
        InputValidationError: bad arguments or unreadable images.
    """
    if not 0.5 <= ratio <= 0.95:
        raise InputValidationError("ratio must be in [0.5, 0.95].")
    if not 100 <= max_features <= 50_000:
        raise InputValidationError("max_features must be in [100, 50000].")

    img_a, _ = downscale(load_image(image_a, settings), _ANALYSIS_MAX_DIM)
    img_b, _ = downscale(load_image(image_b, settings), _ANALYSIS_MAX_DIM)
    feat = _create_detector(detector, max_features)
    kp_a, des_a = feat.detectAndCompute(_gray(img_a), None)
    kp_b, des_b = feat.detectAndCompute(_gray(img_b), None)

    tentative, inlier_matches = _verified_matches(kp_a, des_a, kp_b, des_b, detector, ratio)

    vis_out: str | None = None
    if visualization_path is not None:
        # outImg=None is valid at runtime (OpenCV allocates it); the stubs disagree.
        canvas = cv2.drawMatches(  # type: ignore[call-overload]
            img_a,
            kp_a,
            img_b,
            kp_b,
            inlier_matches[:300],
            None,
            matchColor=(0, 220, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        vis_out = str(save_image(canvas, visualization_path, overwrite=overwrite))

    return PairMatch(
        image_a=Path(image_a).name,
        image_b=Path(image_b).name,
        detector=detector,
        keypoints_a=len(kp_a),
        keypoints_b=len(kp_b),
        tentative_matches=len(tentative),
        inliers=len(inlier_matches),
        inlier_ratio=round(len(inlier_matches) / len(tentative), 3) if tentative else 0.0,
        visualization_path=vis_out,
    )


def analyze_image_set(
    image_dir: str | Path,
    settings: Settings,
    *,
    detector: Detector = "sift",
    max_pairs: int = 60,
) -> ImageSetReport:
    """Assess every image in ``image_dir`` and check how well the set overlaps.

    The photos do not need to be in capture order: up to ``_ALL_PAIRS_LIMIT``
    images every pair is matched (features are computed once per image); larger
    sets match each image with its next neighbours. The set is suitable when
    most photos form one overlapping group, which is what COLMAP needs.
    ``max_pairs`` limits how many pairs are listed in the report.
    """
    if not 1 <= max_pairs <= 1000:
        raise InputValidationError("max_pairs must be in [1, 1000].")
    directory = validate_input_dir(image_dir)
    images = list_images(
        directory, max_images=settings.max_images, max_bytes=settings.max_input_bytes
    )
    qualities = [assess_image(p, settings, detector) for p in images]
    n = len(images)

    feat = _create_detector(detector, _SET_FEATURES)
    features = []
    for path in images:
        small, _ = downscale(load_image(path, settings), _ANALYSIS_MAX_DIM)
        features.append(feat.detectAndCompute(_gray(small), None))
    if n <= _ALL_PAIRS_LIMIT:
        candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        candidates = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 4))]
        candidates.append((n - 1, 0))

    best: dict[int, PairMatch] = {}
    good: list[tuple[int, int]] = []
    for i, j in candidates:
        (kp_a, des_a), (kp_b, des_b) = features[i], features[j]
        tentative, inliers = _verified_matches(kp_a, des_a, kp_b, des_b, detector)
        pair = PairMatch(
            image_a=images[i].name,
            image_b=images[j].name,
            detector=detector,
            keypoints_a=len(kp_a),
            keypoints_b=len(kp_b),
            tentative_matches=len(tentative),
            inliers=len(inliers),
            inlier_ratio=round(len(inliers) / len(tentative), 3) if tentative else 0.0,
        )
        if len(inliers) >= _MIN_PAIR_INLIERS:
            good.append((i, j))
        for k in (i, j):
            if k not in best or pair.inliers > best[k].inliers:
                best[k] = pair
    group = _largest_group(n, good) if n >= 2 else set(range(n))
    # Each photo's strongest partner, without duplicates.
    pairs: list[PairMatch] = []
    for k in sorted(best):
        if best[k] not in pairs:
            pairs.append(best[k])
    pairs = pairs[:max_pairs]
    connected = len(group)

    weak = [images[k].name for k in range(n) if k not in group]
    recommendations: list[str] = []
    if n < 3:
        recommendations.append(
            "Capture at least 20-40 photos around the object for photogrammetry."
        )
    elif n < 15:
        recommendations.append("Few images: add more viewpoints (aim for 60-80% overlap).")
    blurry = [q.name for q in qualities if "blurry" in q.issues]
    if blurry:
        recommendations.append(
            f"Retake or drop {len(blurry)} blurry image(s): use a tripod or faster shutter."
        )
    if any("low_texture" in q.issues for q in qualities):
        recommendations.append(
            "Low texture detected: add matte spray, markers or a patterned background."
        )
    if any(i in q.issues for q in qualities for i in ("underexposed", "overexposed")):
        recommendations.append("Fix exposure: use diffuse, even lighting and avoid glare.")
    if weak:
        recommendations.append(
            f"{len(weak)} of {n} photo(s) do not overlap with the rest: take intermediate "
            "shots, keep the object still and the lighting constant."
        )

    bad_ratio = sum(1 for q in qualities if q.issues) / max(n, 1)
    suitable = n >= 3 and bad_ratio <= 0.3 and connected >= _MIN_CONNECTED_SHARE * n
    if suitable and not recommendations:
        recommendations.append("Image set looks suitable for photogrammetry.")

    logger.info("Analyzed %d image(s), %d pair(s)", n, len(pairs))
    return ImageSetReport(
        image_count=n,
        images=qualities,
        pairs=pairs,
        weak_links=weak,
        suitable_for_photogrammetry=suitable,
        recommendations=recommendations,
        connected_images=connected,
    )
