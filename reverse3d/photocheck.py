"""Optiverse checks the photos before reconstructing (reverse3d photos check).

    reverse3d photos check fotos/            # which photos to fix, by name, and why
    reverse3d --json photos check fotos/

Per photo: blurred (against its siblings), dark, overexposed, the subject cut by
the edge or too small (Optiverse's cut-out), and repeated (no new viewpoint).
For the whole set: photos no other photo connects to (the reconstruction would
drop them), places where the set splits in groups that do not join (photos are
missing in between), and whether there are enough photos for the scene.

Connections are measured as geometric matches between photos (ORB features
checked with RANSAC). On real captures, neighbouring photos share a median of
~300 such matches and photos five shots apart ~20: 30 is the line.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reverse3d.config import Settings

logger = logging.getLogger(__name__)

LINK = 30  # geometric matches for two photos to count as connected
SAME = 0.985  # thumbnail correlation above which a photo repeats the previous one
WINDOW = 12  # with many photos, each one is compared with its neighbours in capture order

FIX = {
    "blurry": (
        "movida o desenfocada: repítela con el móvil quieto y el objeto enfocado",
        "blurred or out of focus: retake it holding still, with the object in focus",
    ),
    "dark": (
        "muy oscura: repítela con más luz (sin flash directo)",
        "too dark: retake it with more light (no direct flash)",
    ),
    "bright": (
        "quemada: repítela con menos luz o sin reflejos directos",
        "overexposed: retake it with less light or without direct glare",
    ),
    "cut": (
        "el objeto sale cortado: aléjate un poco para que entre completo",
        "the object is cut off: step back so it fits completely",
    ),
    "small": (
        "el objeto sale muy pequeño: acércate para que ocupe más la foto",
        "the object is too small: get closer so it fills more of the photo",
    ),
    "isolated": (
        "no comparte nada con las demás: la reconstrucción la descartaría; toma fotos intermedias desde ese ángulo",
        "shares nothing with the others: the reconstruction would drop it; add photos in between from that angle",
    ),
    "repeated": (
        "repetida: es casi igual a la anterior y no aporta un ángulo nuevo (puedes quitarla)",
        "repeated: almost the same as the previous one, no new angle (you can remove it)",
    ),
    "unreadable": ("no se puede leer el archivo", "the file cannot be read"),
}
#: problems that spoil the model (repeated only wastes time)
SERIOUS = {"blurry", "dark", "bright", "cut", "small", "isolated", "unreadable"}


@dataclass
class PhotoCheck:
    photos: int
    issues: dict[str, list[str]]  # photo name -> problems
    gaps: list[tuple[str, str]] = field(
        default_factory=list
    )  # (last of a group, first of the next)
    groups: int = 1
    min_photos: int = 0
    recommended: tuple[int, int] = (0, 0)
    scene: str | None = None
    used_optiverse: bool = False

    @property
    def to_fix(self) -> list[str]:
        return [n for n, p in self.issues.items() if set(p) & SERIOUS]

    @property
    def usable(self) -> int:
        return self.photos - len(self.to_fix)

    @property
    def verdict(self) -> str:
        """ready: go; fixable: drop the bad ones and go; retake: take photos first."""
        if self.groups > 1 or self.usable < self.min_photos:
            return "retake"
        return "fixable" if self.to_fix else "ready"

    def to_dict(self, lang: str = "es") -> dict[str, Any]:
        k = 0 if lang == "es" else 1
        return {
            "verdict": self.verdict,
            "photos": self.photos,
            "usable": self.usable,
            "min_photos": self.min_photos,
            "recommended": list(self.recommended),
            "scene": self.scene,
            "used_optiverse": self.used_optiverse,
            "groups": self.groups,
            "gaps": [{"after": a, "before": b} for a, b in self.gaps],
            "to_fix": self.to_fix,
            "photos_with_issues": [
                {"name": name, "problems": problems, "fix": [FIX[p][k] for p in problems]}
                for name, problems in self.issues.items()
                if problems
            ],
            "summary": summary(self, lang),
        }


def summary(check: PhotoCheck, lang: str = "es") -> list[str]:
    es = lang == "es"
    lines = []
    if check.verdict == "ready":
        lines.append(
            f"Las {check.photos} fotos están listas: puedes reconstruir."
            if es
            else f"All {check.photos} photos are ready: go ahead and reconstruct."
        )
    for a, b in check.gaps:
        lines.append(
            f"Faltan fotos entre {a} y {b}: no comparten lo suficiente y el modelo se partiría. Toma 2 o 3 fotos intermedias rodeando el objeto desde {a} hasta {b}."
            if es
            else f"Photos are missing between {a} and {b}: they share too little and the model would split. Take 2 or 3 photos in between, moving around the object from {a} to {b}."
        )
    fix = check.to_fix
    if fix:
        lines.append(
            f"Corrige {len(fix)} fotos: {', '.join(fix)}."
            if es
            else f"Fix {len(fix)} photos: {', '.join(fix)}."
        )
    if check.usable < check.min_photos:
        lines.append(
            f"Quedan {check.usable} fotos buenas y se necesitan al menos {check.min_photos} (ideal {check.recommended[0]}–{check.recommended[1]})."
            if es
            else f"{check.usable} good photos left; at least {check.min_photos} are needed (ideally {check.recommended[0]}–{check.recommended[1]})."
        )
    elif check.verdict == "fixable":
        lines.append(
            f"Sin ellas te quedan {check.usable} fotos buenas: puedo reconstruir solo con esas."
            if es
            else f"Without them you have {check.usable} good photos: I can reconstruct with those only."
        )
    return lines


def _thumb(rgb: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 48)).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def _features(rgb: np.ndarray, mask: np.ndarray | None) -> tuple[Any, Any]:
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keep = None
    if mask is not None and mask.mean() > 0.03:
        # Features on and around the subject: a shared background is not a shared view.
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8))
        keep = cv2.resize(grown, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    orb: Any = cv2.ORB_create(3000)  # type: ignore[attr-defined]
    return orb.detectAndCompute(gray, keep)


def _links(a: tuple[Any, Any], b: tuple[Any, Any]) -> int:
    """Geometric matches between two photos (RANSAC inliers of the fundamental matrix)."""
    import cv2

    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return len(good)
    pa = np.array([ka[m.queryIdx].pt for m in good], dtype=np.float32)
    pb = np.array([kb[m.trainIdx].pt for m in good], dtype=np.float32)
    _, inliers = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 3.0, 0.99)
    return int(inliers.sum()) if inliers is not None else 0


def _groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def check_photos(
    folder: str | Path,
    settings: Settings | None = None,
    scene: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PhotoCheck:
    from reverse3d.assistant.actions import _own_model, _photo_issues, read_rgb
    from reverse3d.scenes import SCENE_PRESETS, detect_scene
    from reverse3d.security import HEIC_EXTENSIONS, IMAGE_EXTENSIONS, validate_input_dir

    settings = settings or Settings.load()
    directory = validate_input_dir(folder)
    # iPhone photos (HEIC) included: they are read directly, nothing is converted.
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | HEIC_EXTENSIONS
    )[: settings.max_images]
    model = _own_model()
    issues: dict[str, list[str]] = {}
    sharp: dict[str, float] = {}
    feats: list[tuple[Any, Any]] = []
    thumbs: list[np.ndarray | None] = []
    for i, f in enumerate(files):
        try:
            info = _photo_issues(f, settings, model)
            rgb = read_rgb(f, 960)
            mask = None
            if model is not None:
                mask = model.run(rgb)[1] > 0.5
            feats.append(_features(rgb, mask))
            thumbs.append(_thumb(rgb))
            issues[f.name] = list(info["issues"])
            sharp[f.name] = float(info["sharpness"])
        except Exception as exc:  # one unreadable photo must not stop the check
            logger.debug("photo check: %s: %s", f.name, exc)
            issues[f.name] = ["unreadable"]
            feats.append((None, None))
            thumbs.append(None)
        if progress:
            progress(i + 1, len(files))
    if len(sharp) >= 3:  # blurred against its siblings, never one absolute number
        limit = max(15.0, 0.45 * float(np.median(list(sharp.values()))))
        for name, value in sharp.items():
            if value < limit:
                issues[name].insert(0, "blurry")
    for i in range(1, len(files)):
        a, b = thumbs[i - 1], thumbs[i]
        if a is not None and b is not None and float((a * b).mean()) > SAME:
            issues[files[i].name].append("repeated")
    n = len(files)
    pairs = [
        (i, j) for i in range(n) for j in range(i + 1, n) if n <= 2 * WINDOW + 40 or j - i <= WINDOW
    ]
    edges = [(i, j) for i, j in pairs if _links(feats[i], feats[j]) >= LINK]
    connected = {i for e in edges for i in e}
    for i, f in enumerate(files):
        if i not in connected and "unreadable" not in issues[f.name]:
            issues[f.name].append("isolated")
    real = [g for g in _groups(n, edges) if len(g) > 1]
    gaps = []
    for g, h in itertools.pairwise(real):
        gaps.append((files[max(g)].stem, files[min(h)].stem))
    if scene is None:  # the scene decides how many photos are needed
        try:
            scene = str(detect_scene(directory, settings).kind)
        except Exception as exc:  # e.g. HEIC only: Studio passes the scene it detected
            logger.debug("photo check: scene not detected: %s", exc)
    preset = SCENE_PRESETS.get(scene or "object", SCENE_PRESETS["object"])
    return PhotoCheck(
        photos=n,
        issues=issues,
        gaps=gaps,
        groups=max(1, len(real)),
        min_photos=preset.min_photos,
        recommended=preset.recommended,
        scene=scene,
        used_optiverse=model is not None,
    )


def keep_good(folder: str | Path, check: PhotoCheck, out: Path) -> Path:
    """Copies the photos that need no fixing to a new folder (the originals stay)."""
    import shutil

    out.mkdir(parents=True, exist_ok=True)
    bad = set(check.to_fix)
    for f in sorted(Path(folder).iterdir()):
        if f.is_file() and f.name in check.issues and f.name not in bad:
            shutil.copy2(f, out / f.name)
    return out
