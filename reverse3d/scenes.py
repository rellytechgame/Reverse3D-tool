"""Scene profiles: what kind of capture a photo set is, and how to process it.

reverse3d handles very different captures: a part on a table, a person for a
game character, a building, or a landscape flown with a drone. They need
different photogrammetry settings, cleaning and advice. :func:`detect_scene`
predicts the kind from a quick look at a sample of the photos (EXIF, GPS,
drone metadata, how isolated the subject is, people, straight lines, sky), and
:data:`SCENE_PRESETS` holds the settings for each kind.

The prediction is deterministic (plain image statistics, no AI model), so it
works offline and gives the same answer on every machine. It always returns
its reasons, so the user can confirm or change it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from reverse3d.config import Settings
from reverse3d.imaging import downscale, load_image
from reverse3d.security import list_images, validate_input_dir

logger = logging.getLogger(__name__)

SceneKind = Literal["object", "character", "architecture", "landscape"]
SCENES: tuple[SceneKind, ...] = ("object", "character", "architecture", "landscape")

SCENE_LABEL_ES: dict[str, str] = {
    "object": "Objeto o pieza",
    "character": "Personaje o persona",
    "architecture": "Arquitectura",
    "landscape": "Paisaje o terreno",
}

_SAMPLE = 8
_SIDE = 640
_DRONE_MAKES = ("dji", "parrot", "autel", "skydio", "yuneec")


@dataclass(frozen=True)
class ScenePreset:
    """How a scene kind is captured and processed."""

    min_photos: int
    recommended: tuple[int, int]
    #: photogrammetry quality; None keeps the hardware profile's choice.
    quality: Literal["low", "medium", "high"] | None
    #: "spatial" uses GPS when the photos have it (falls back to sequential).
    matcher: Literal["auto", "exhaustive", "sequential", "spatial"]
    poisson_depth: int | None
    #: share of the least supported surface removed after Poisson (open scenes need more).
    density_quantile: float
    #: closed solid (objects) or open surface (buildings, terrain).
    fill_holes: bool
    keep_largest: bool


SCENE_PRESETS: dict[str, ScenePreset] = {
    "object": ScenePreset(20, (40, 60), None, "auto", None, 0.02, True, True),
    "character": ScenePreset(40, (60, 100), "high", "auto", 10, 0.03, True, True),
    "architecture": ScenePreset(50, (80, 200), None, "auto", 11, 0.05, False, False),
    "landscape": ScenePreset(60, (100, 400), None, "spatial", 11, 0.06, False, False),
}


@dataclass
class SceneSignals:
    """What the photos look like (all shares are 0..1)."""

    photos: int
    sampled: int
    gps_photos: int = 0
    gps_spread_m: float | None = None
    drone: bool = False
    nadir_share: float = 0.0
    isolation: float = 0.0
    people_share: float = 0.0
    rectilinear: float = 0.0
    sky_share: float = 0.0
    camera: str | None = None


@dataclass
class SceneDetection:
    """Result of :func:`detect_scene`."""

    kind: SceneKind
    confidence: float
    #: short codes the apps translate (drone, gps_wide, nadir, people, isolated_object,
    #: straight_lines, sky, few_photos).
    reasons: list[str]
    scores: dict[str, float]
    signals: SceneSignals
    min_photos: int
    recommended: tuple[int, int]
    enough_photos: bool
    advice: list[str] = field(default_factory=list)
    #: what the AI classifier saw (product, person, animal, vehicle, building...), if available.
    subject: str | None = None
    subject_label: str | None = None
    subject_scores: dict[str, float] = field(default_factory=dict)
    #: render look: studio, showroom, portrait, architecture or aerial.
    render: str = "studio"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ─── EXIF / XMP ───────────────────────────────────────────────────────────────


def _rational(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _gps_degrees(dms: object, ref: object) -> float | None:
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        return None
    d, m, s = (_rational(x) for x in dms)
    if any(math.isnan(x) for x in (d, m, s)):
        return None
    value = d + m / 60 + s / 3600
    return -value if str(ref).upper() in ("S", "W") else value


def _photo_metadata(path: Path) -> tuple[str | None, tuple[float, float] | None, float | None]:
    """(camera "make model", (lat, lon), gimbal pitch in degrees) of one photo."""
    from PIL import Image

    camera: str | None = None
    position: tuple[float, float] | None = None
    pitch: float | None = None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            make, model = str(exif.get(271, "")).strip(), str(exif.get(272, "")).strip()
            camera = f"{make} {model}".strip() or None
            gps = exif.get_ifd(0x8825)
            lat = _gps_degrees(gps.get(2), gps.get(1))
            lon = _gps_degrees(gps.get(4), gps.get(3))
            if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
                position = (lat, lon)
    except Exception as exc:  # unreadable metadata is not an error: it is just missing
        logger.debug("No EXIF in %s: %s", path.name, exc)
    try:
        head = path.read_bytes()[:262_144]
        marker = b'GimbalPitchDegree="'
        k = head.find(marker)
        if k >= 0:
            end = head.find(b'"', k + len(marker))
            pitch = float(head[k + len(marker) : end].decode("ascii", "ignore"))
    except (OSError, ValueError):
        pass
    return camera, position, pitch


def _spread_m(points: list[tuple[float, float]]) -> float | None:
    """Largest distance between GPS positions, in metres (equirectangular)."""
    if len(points) < 2:
        return None
    lat0 = math.radians(sum(p[0] for p in points) / len(points))
    xy = np.array([(math.radians(lon) * math.cos(lat0), math.radians(lat)) for lat, lon in points])
    xy *= 6_371_000.0
    span = xy.max(axis=0) - xy.min(axis=0)
    return float(np.hypot(*span))


# ─── image statistics ─────────────────────────────────────────────────────────


def _rectilinear(gray: np.ndarray) -> float:
    """How much of the picture is long straight vertical/horizontal edges (buildings)."""
    edges = cv2.Canny(gray, 60, 160)
    diag = math.hypot(*gray.shape[:2])
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=diag * 0.08, maxLineGap=6)
    if lines is None:
        return 0.0
    total = 0.0
    straight = 0.0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        total += length
        if min(angle, 180 - angle) < 8 or abs(angle - 90) < 8:
            straight += length
    # ~6 picture diagonals of straight lines is already a very rectilinear scene.
    return float(min(straight / (6 * diag), 1.0)) * (straight / total if total else 0.0)


def _sky(bgr: np.ndarray) -> float:
    """Share of the top third that looks like sky (bright and grey, or blue)."""
    top = bgr[: bgr.shape[0] // 3]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grey_sky = (v > 175) & (s < 45)
    blue_sky = (h >= 90) & (h <= 130) & (s > 40) & (v > 110)
    return float((grey_sky | blue_sky).mean())


def _looks_like_person(bgr: np.ndarray, mask: np.ndarray) -> bool:
    """A standing person: an isolated subject clearly taller than wide, with skin.

    A plain-statistics heuristic (no detector model, so it works offline); the
    apps always let the user confirm or change the predicted scene.
    """
    inside = mask > 0
    if inside.mean() < 0.04:
        return False
    ys, xs = np.nonzero(inside)
    tall = (np.ptp(ys) + 1) / (np.ptp(xs) + 1)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 130)
    skin_share = float(skin[inside].mean())
    return bool(tall >= 1.5 and 0.04 <= skin_share <= 0.6)


# ─── prediction ───────────────────────────────────────────────────────────────


def _signals(images: list[Path], settings: Settings) -> SceneSignals:
    from reverse3d.vision.mask_extractor import foreground_mask, mask_score

    n = len(images)
    step = max(1, n // _SAMPLE)
    sample = images[::step][:_SAMPLE]
    cameras: list[str] = []
    positions: list[tuple[float, float]] = []
    pitches: list[float] = []
    for path in images:
        camera, position, pitch = _photo_metadata(path)
        if camera:
            cameras.append(camera)
        if position:
            positions.append(position)
        if pitch is not None:
            pitches.append(pitch)

    isolation, people, rect, sky = [], 0, [], []
    for path in sample:
        small, _ = downscale(load_image(path, settings), _SIDE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        try:
            mask = foreground_mask(small, work_side=320)
            isolation.append(mask_score(mask))
            people += _looks_like_person(small, mask)
        except cv2.error:
            isolation.append(0.0)
        rect.append(_rectilinear(gray))
        sky.append(_sky(small))

    camera = max(set(cameras), key=cameras.count) if cameras else None
    drone = bool(camera and camera.lower().startswith(_DRONE_MAKES)) or bool(pitches)
    return SceneSignals(
        photos=n,
        sampled=len(sample),
        gps_photos=len(positions),
        gps_spread_m=_spread_m(positions),
        drone=drone,
        nadir_share=sum(1 for p in pitches if p <= -60) / len(pitches) if pitches else 0.0,
        isolation=float(np.mean(isolation)) if isolation else 0.0,
        people_share=people / len(sample) if sample else 0.0,
        rectilinear=float(np.mean(rect)) if rect else 0.0,
        sky_share=float(np.mean(sky)) if sky else 0.0,
        camera=camera,
    )


def score_scene(signals: SceneSignals) -> tuple[dict[str, float], list[str]]:
    """Scores per scene kind (higher wins) and the reasons that drove them."""
    s = signals
    wide = s.gps_spread_m is not None and s.gps_spread_m > 80
    building_range = s.gps_spread_m is not None and 8 < s.gps_spread_m < 400
    reasons: list[str] = []
    if s.drone:
        reasons.append("drone")
    if wide:
        reasons.append("gps_wide")
    if s.nadir_share > 0.3:
        reasons.append("nadir")
    if s.people_share >= 0.4:
        reasons.append("people")
    if s.isolation >= 0.45:
        reasons.append("isolated_object")
    if s.rectilinear >= 0.25:
        reasons.append("straight_lines")
    if s.sky_share >= 0.25:
        reasons.append("sky")
    scores = {
        "object": 0.15 + 0.7 * s.isolation + 0.15 * (1 - s.rectilinear) - 0.4 * s.people_share,
        "character": 0.9 * s.people_share + 0.2 * s.isolation,
        "architecture": 0.65 * s.rectilinear
        + 0.25 * s.sky_share
        + 0.15 * building_range
        + 0.2 * (1 - s.isolation)
        - 0.3 * s.nadir_share,
        "landscape": 0.45 * s.drone
        + 0.3 * wide
        + 0.35 * s.nadir_share
        + 0.25 * s.sky_share * (1 - s.rectilinear)
        + 0.1 * (1 - s.isolation),
    }
    return {k: round(max(v, 0.0), 3) for k, v in scores.items()}, reasons


def _ai_subject(images: list[Path], settings: Settings) -> tuple[str | None, dict[str, float]]:
    """The AI classifier's subject for a sample of the photos (None if unavailable)."""
    from reverse3d.errors import Reverse3DError
    from reverse3d.vision.classifier import classify_photos

    step = max(1, len(images) // _SAMPLE)
    try:
        scores = classify_photos(images[::step][:_SAMPLE], settings)
    except Reverse3DError as exc:
        logger.info("AI subject classifier unavailable: %s", exc.message)
        return None, {}
    return max(scores, key=scores.__getitem__), scores


def _subject_label(subject: str | None) -> str | None:
    from reverse3d.vision.classifier import SUBJECT_LABEL_ES

    return SUBJECT_LABEL_ES.get(subject) if subject else None


def detect_scene(image_dir: str | Path, settings: Settings) -> SceneDetection:
    """Predict the scene kind of a photo folder (takes a few seconds)."""
    directory = validate_input_dir(image_dir)
    images = list_images(
        directory, max_images=settings.max_images, max_bytes=settings.max_input_bytes
    )
    signals = _signals(images, settings)
    scores, reasons = score_scene(signals)
    kind: SceneKind = max(SCENES, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    confidence = round(min(0.97, 0.5 + margin * 1.5), 2)
    subject, subject_scores = _ai_subject(images, settings)
    render = (
        "aerial" if kind == "landscape" else "architecture" if kind == "architecture" else "studio"
    )
    if subject is not None:
        from reverse3d.vision.classifier import SUBJECT_RENDER, SUBJECT_SCENE

        flight = signals.drone and signals.nadir_share > 0.3
        if flight:  # a drone looking down is terrain whatever the AI thinks it sees
            subject = "terrain" if subject not in ("forest", "terrain") else subject
        kind = SUBJECT_SCENE[subject]  # type: ignore[assignment]
        render = SUBJECT_RENDER[subject]
        confidence = round(min(0.98, max(subject_scores.get(subject, 0.0), 0.5)), 2)
        reasons.insert(0, f"ai_{subject}")
    preset = SCENE_PRESETS[kind]
    enough = signals.photos >= preset.min_photos
    if not enough:
        reasons.append("few_photos")
    advice = scene_advice(kind, signals.photos)
    logger.info(
        "Scene: %s (%.0f%%) from %d photo(s): %s", kind, confidence * 100, signals.photos, reasons
    )
    return SceneDetection(
        kind=kind,
        confidence=confidence,
        reasons=reasons,
        scores=scores,
        signals=signals,
        min_photos=preset.min_photos,
        recommended=preset.recommended,
        enough_photos=enough,
        advice=advice,
        subject=subject,
        subject_label=_subject_label(subject),
        subject_scores={k: round(v, 3) for k, v in subject_scores.items()},
        render=render,
    )


_ADVICE_ES: dict[str, str] = {
    "object": (
        "Objeto: 40-60 fotos (mínimo 20) en 4 perfiles: anillo bajo a 10-20° (16), medio a "
        "35-45° (16), alto a 60-70° (12) y 4 desde arriba, sin mover el objeto, luz pareja y "
        "sin reflejos."
    ),
    "character": (
        "Personaje: 60-100 fotos (mínimo 40) en menos de 2 minutos, con la persona quieta en "
        "pose A (brazos separados), mirando a un punto fijo. Anillos a la altura de las rodillas "
        "(24), del pecho (24) y por encima de la cabeza (16), más 10-12 de cerca de la cara. "
        "Ropa mate, sin pelo suelto al viento."
    ),
    "architecture": (
        "Arquitectura: 80-200 fotos (mínimo 50). Rodea el edificio a distancia constante con "
        "una foto cada 2-3 pasos (70 % de solape), a la altura de los ojos y otra vuelta más "
        "alejada para ver el techo; fachadas de frente y esquinas a 45°. Interiores: desde las "
        "esquinas en diagonal, una foto por metro. Día nublado, sin gente ni autos moviéndose."
    ),
    "landscape": (
        "Paisaje (dron): 100-400 fotos (mínimo 60). Vuelo en cuadrícula a altura constante "
        "(30-120 m) con 75-80 % de solape frontal y 65-70 % lateral, cámara vertical (90°), más "
        "una pasada oblicua a 45-60° para taludes y fachadas. GPS activado, misma exposición, "
        "sin sombras duras del mediodía."
    ),
}


def scene_advice(kind: str, photos: int) -> list[str]:
    """Capture advice (Spanish) for a scene kind, plus a note if photos are few."""
    preset = SCENE_PRESETS[kind]
    advice = [_ADVICE_ES[kind]]
    if photos < preset.min_photos:
        advice.insert(
            0,
            f"Solo hay {photos} fotos: para {SCENE_LABEL_ES[kind].lower()} se necesitan al "
            f"menos {preset.min_photos} ({preset.recommended[0]}-{preset.recommended[1]} "
            "recomendadas).",
        )
    return advice


def has_gps(image_dir: Path, minimum: int = 3) -> bool:
    """True when at least ``minimum`` photos carry GPS positions (spatial matching)."""
    count = 0
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and _photo_metadata(path)[1] is not None:
            count += 1
            if count >= minimum:
                return True
    return False
