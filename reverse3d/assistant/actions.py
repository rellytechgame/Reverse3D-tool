"""Actions of the assistant: photos before processing, the 3D result, capture advice.

Originals are never modified: photo changes go to ``<work>/fotos-preparadas``
and every model edit is a new file in ``<work>/modelos`` (undo goes back).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError

logger = logging.getLogger(__name__)

PHOTO_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}
MAX_PHOTOS = 400
Lang = Literal["es", "en"]


def t(lang: str, es: str, en: str) -> str:
    return es if lang == "es" else en


SCENES_ES = {
    "object": "objeto",
    "character": "personaje",
    "architecture": "arquitectura",
    "landscape": "paisaje",
}


def scene_name(lang: str, scene: str) -> str:
    return SCENES_ES.get(scene, scene) if lang == "es" else scene


def num(lang: str, value: int) -> str:
    """4.093 in Spanish, 4,093 in English."""
    text = f"{value:,}"
    return text.replace(",", ".") if lang == "es" else text


# ─── session ──────────────────────────────────────────────────────────────────


class Session:
    """Where the assistant works: the photos, the job result and its own folder."""

    def __init__(
        self,
        work: Path,
        photos: Path | None = None,
        job: Path | None = None,
        settings: Settings | None = None,
        lang: str = "es",
    ) -> None:
        self.work = Path(work).expanduser().resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.photos = Path(photos).expanduser().resolve() if photos else None
        self.job = Path(job).expanduser().resolve() if job else None
        self.settings = settings or Settings.load()
        self.lang = "es" if lang.startswith("es") else "en"
        self._state_file = self.work / "state.json"
        try:
            self.state: dict[str, Any] = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}
        self.state.setdefault("excluded", [])
        self.state.setdefault("enhanced", [])
        self.state.setdefault("meshes", [])
        self.state.setdefault("review", {})

    def save(self) -> None:
        self._state_file.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def photo_files(self) -> list[Path]:
        if not self.photos or not self.photos.is_dir():
            return []
        files = sorted(p for p in self.photos.iterdir() if p.suffix.lower() in PHOTO_TYPES)
        return files[:MAX_PHOTOS]

    def find_photos(self, names: list[str]) -> list[Path]:
        """Photos by name, with or without extension, case-insensitive."""
        by_key = {p.stem.lower(): p for p in self.photo_files()}
        by_key.update({p.name.lower(): p for p in self.photo_files()})
        found = []
        for name in names:
            key = Path(str(name)).name.lower()
            hit = by_key.get(key) or by_key.get(Path(key).stem)
            if hit is None and key.isdigit():  # "4950" -> IMG_4950.HEIC
                hit = next((p for p in self.photo_files() if p.stem.lower().endswith(key)), None)
            if hit and hit not in found:
                found.append(hit)
        return found

    def source_mesh(self) -> Path | None:
        if not self.job:
            return None
        for name in ("model.glb", "model.stl", "model_color.ply", "model.obj"):
            if (self.job / name).is_file():
                return self.job / name
        return None

    def current_mesh(self) -> Path | None:
        for item in reversed(self.state["meshes"]):
            if Path(item).is_file():
                return Path(item)
        return self.source_mesh()


# ─── helpers ──────────────────────────────────────────────────────────────────


def read_rgb(path: Path, max_side: int | None = None) -> np.ndarray:
    import cv2
    from PIL import Image, ImageOps

    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    if max_side and max(rgb.shape[:2]) > max_side:
        f = max_side / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def auto_enhance(rgb: np.ndarray) -> np.ndarray:
    """Even exposure and local contrast (CLAHE on lightness), colours kept."""
    import cv2

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0].astype(np.float32) / 255
    mean = float(lightness.mean())
    if 0.05 < mean < 0.95:  # bring the average towards a mid tone
        gamma = np.log(0.48) / np.log(mean)
        lightness = lightness ** np.clip(gamma, 0.5, 2.0)
    lab[..., 0] = np.clip(lightness * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _own_model() -> Any:
    from reverse3d.ai_models import own_model

    return own_model()


def _photo_issues(path: Path, settings: Settings, model: Any) -> dict[str, Any]:
    import cv2

    rgb = read_rgb(path, 1280)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean()) / 255
    # Sharpness on a contrast-normalised copy: a dark photo is not a blurred one.
    small = cv2.resize(gray, (960, int(960 * gray.shape[0] / gray.shape[1]))).astype(np.float32)
    norm = (small - small.mean()) / (small.std() + 1.0) * 50 + 128
    sharp = float(cv2.Laplacian(norm, cv2.CV_32F).var())
    clipped = float(((gray < 6) | (gray > 249)).mean())
    issues: list[str] = []  # "blurry" is decided against the whole set (review_photos)
    if mean < 0.22:
        issues.append("dark")
    elif mean > 0.82 or clipped > 0.2:
        issues.append("bright")
    info: dict[str, Any] = {"sharpness": round(sharp, 1), "brightness": round(mean, 3)}
    if model is not None:
        _, subject, scene = model.run(rgb)
        mask = subject > 0.5
        coverage = float(mask.mean())
        border = np.zeros_like(mask)
        b = max(2, int(0.01 * max(mask.shape)))
        border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
        cut = float((mask & border).sum()) / max(float(mask.sum()), 1.0)
        info.update(subject=round(coverage, 4), scene=max(scene, key=scene.get))
        if coverage < 0.03:
            issues.append("small")
        if cut > 0.02:
            issues.append("cut")
    info["issues"] = issues
    return info


ISSUE_TEXT = {
    "blurry": ("movida o desenfocada", "blurred or out of focus"),
    "dark": ("muy oscura", "too dark"),
    "bright": ("quemada o con mucha luz", "overexposed"),
    "small": ("el objeto sale muy pequeño", "the object is too small"),
    "cut": ("el objeto sale cortado por el borde", "the object is cut by the edge"),
}


def _issue(lang: str, key: str) -> str:
    es, en = ISSUE_TEXT[key]
    return t(lang, es, en)


# ─── actions ──────────────────────────────────────────────────────────────────


class NoArgs(BaseModel):
    pass


class PhotosArgs(BaseModel):
    names: list[str] = Field(default_factory=list, max_length=MAX_PHOTOS)
    issue: Literal["blurry", "dark", "bright", "small", "cut", "any"] | None = None


class EnhanceArgs(BaseModel):
    names: list[str] = Field(default_factory=list, max_length=MAX_PHOTOS)
    all: bool = False


class PreviewArgs(BaseModel):
    name: str = ""


class HolesArgs(BaseModel):
    max_hole_edges: int = Field(1000, ge=3, le=100_000)


class SmoothArgs(BaseModel):
    strength: int = Field(2, ge=1, le=3)


class SimplifyArgs(BaseModel):
    percent: int = Field(50, ge=5, le=95, description="faces to keep, in %")


class HeightArgs(BaseModel):
    mm: float = Field(gt=0.1, lt=1_000_000)


@dataclass
class Result:
    ok: bool
    lines: list[str]
    data: dict[str, Any] | None = None
    images: list[str] | None = None
    prepared_photos: str | None = None
    model: str | None = None


def review_photos(s: Session, _: NoArgs) -> Result:
    files = s.photo_files()
    if not files:
        return Result(
            False, [t(s.lang, "No hay fotos que revisar.", "There are no photos to review.")]
        )
    model = _own_model()
    review = {}
    for f in files:
        try:
            review[f.name] = _photo_issues(f, s.settings, model)
        except Exception as exc:  # one unreadable photo must not stop the review
            review[f.name] = {"issues": ["unreadable"], "error": str(exc)[:120]}
    # Blurred = far less sharp than its siblings (measured on real captures: sharp
    # photos 120-240, shaken or out of focus 3-80), never by one absolute number.
    values = [v["sharpness"] for v in review.values() if "sharpness" in v]
    if values:
        limit = max(15.0, 0.45 * float(np.median(values))) if len(values) >= 3 else 60.0
        for item in review.values():
            if item.get("sharpness", limit) < limit:
                item["issues"].insert(0, "blurry")
    s.state["review"] = review
    s.save()
    counts: dict[str, int] = {}
    for item in review.values():
        for issue in item["issues"]:
            counts[issue] = counts.get(issue, 0) + 1
    scenes = [str(v["scene"]) for v in review.values() if v.get("scene")]
    lines = [
        t(
            s.lang,
            f"Revisé {len(files)} fotos con Optiverse.",
            f"I reviewed {len(files)} photos with Optiverse.",
        )
        if model is not None
        else t(
            s.lang,
            f"Revisé {len(files)} fotos (sin Optiverse: solo nitidez y luz).",
            f"I reviewed {len(files)} photos (no Optiverse: sharpness and light only).",
        )
    ]
    if scenes:
        top = max(set(scenes), key=scenes.count)
        lines.append(
            t(s.lang, f"Escena detectada: {scene_name(s.lang, top)}.", f"Detected scene: {top}.")
        )
    bad = {k: v for k, v in counts.items() if k in ISSUE_TEXT}
    if not bad:
        lines.append(
            t(
                s.lang,
                "Todas se ven bien para reconstruir.",
                "They all look fine for reconstruction.",
            )
        )
    for key, n in sorted(bad.items(), key=lambda kv: -kv[1]):
        names = [name for name, v in review.items() if key in v["issues"]][:4]
        more = "…" if counts[key] > 4 else ""
        lines.append(f"• {n} · {_issue(s.lang, key)}: {', '.join(names)}{more}")
    if bad:
        lines.append(
            t(
                s.lang,
                "Puedo descartarlas («descarta las fotos movidas») o mejorar su luz («mejora las fotos oscuras»).",
                "I can drop them (“drop the blurred photos”) or fix their light (“enhance the dark photos”).",
            )
        )
    return Result(True, lines, data={"counts": counts})


def _select(s: Session, a: PhotosArgs) -> list[Path]:
    chosen = s.find_photos(a.names)
    if a.issue:
        review = s.state.get("review") or {}
        if not review:
            review_photos(s, NoArgs())
            review = s.state["review"]
        for name, item in review.items():
            issues = [i for i in item.get("issues", []) if i in ISSUE_TEXT]
            if (a.issue == "any" and issues) or a.issue in issues:
                chosen += [p for p in s.find_photos([name]) if p not in chosen]
    return chosen


def exclude_photos(s: Session, a: PhotosArgs) -> Result:
    chosen = _select(s, a)
    if not chosen:
        return Result(
            False, [t(s.lang, "No encontré esas fotos.", "I couldn't find those photos.")]
        )
    for p in chosen:
        if p.name not in s.state["excluded"]:
            s.state["excluded"].append(p.name)
    s.save()
    left = len(s.photo_files()) - len(s.state["excluded"])
    names = ", ".join(p.name for p in chosen[:6]) + ("…" if len(chosen) > 6 else "")
    lines = [
        t(
            s.lang,
            f"Descarté {len(chosen)} fotos: {names}.",
            f"Dropped {len(chosen)} photos: {names}.",
        ),
        t(s.lang, f"Quedan {left} para reconstruir.", f"{left} are left for reconstruction."),
    ]
    if left < 20:
        lines.append(
            t(
                s.lang,
                "Ojo: con menos de 20 fotos el modelo pierde detalle.",
                "Careful: under 20 photos the model loses detail.",
            )
        )
    return Result(True, lines, data={"excluded": s.state["excluded"]})


def restore_photos(s: Session, a: PhotosArgs) -> Result:
    names = {p.name for p in _select(s, a)} if (a.names or a.issue) else set(s.state["excluded"])
    before = len(s.state["excluded"])
    s.state["excluded"] = [n for n in s.state["excluded"] if n not in names]
    s.save()
    back = before - len(s.state["excluded"])
    return Result(True, [t(s.lang, f"Recuperé {back} fotos.", f"Brought back {back} photos.")])


def enhance_photos(s: Session, a: EnhanceArgs) -> Result:
    if a.all:
        chosen = s.photo_files()
    else:
        chosen = s.find_photos(a.names)
        if not chosen:
            chosen = _select(s, PhotosArgs(issue="dark")) + _select(s, PhotosArgs(issue="bright"))
    if not chosen:
        return Result(
            False,
            [t(s.lang, "No hay fotos con problemas de luz.", "No photos have light problems.")],
        )
    for p in chosen:
        if p.name not in s.state["enhanced"]:
            s.state["enhanced"].append(p.name)
    s.save()
    return Result(
        True,
        [
            t(
                s.lang,
                f"Ajustaré la luz y el contraste de {len(chosen)} fotos al prepararlas.",
                f"I'll even out light and contrast of {len(chosen)} photos when preparing them.",
            ),
            t(s.lang, "Tus originales no se tocan.", "Your originals are not touched."),
        ],
    )


def prepare_photos(s: Session, _: NoArgs) -> Result:
    files = [p for p in s.photo_files() if p.name not in s.state["excluded"]]
    if not files:
        return Result(False, [t(s.lang, "No queda ninguna foto.", "No photos are left.")])
    from PIL import Image

    out = s.work / "fotos-preparadas"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    enhanced = set(s.state["enhanced"])
    for p in files:
        heic = p.suffix.lower() in {".heic", ".heif"}
        if p.name in enhanced or heic:
            rgb = read_rgb(p)
            if p.name in enhanced:
                rgb = auto_enhance(rgb)
            exif = None
            try:
                with Image.open(p) as im:
                    exif = im.info.get("exif")
            except Exception:  # EXIF is a bonus (focal length helps COLMAP)
                exif = None
            target = out / f"{p.stem}.jpg"
            kwargs: dict[str, Any] = {"quality": 95}
            if exif:
                kwargs["exif"] = exif
            Image.fromarray(rgb).save(target, **kwargs)
        else:
            shutil.copy2(p, out / p.name)
    return Result(
        True,
        [
            t(
                s.lang,
                f"Preparé {len(files)} fotos ({len(enhanced & {p.name for p in files})} con la luz mejorada).",
                f"Prepared {len(files)} photos ({len(enhanced & {p.name for p in files})} with better light).",
            ),
            t(
                s.lang,
                "Pulsa «Reconstruir con las fotos preparadas».",
                "Press “Rebuild with the prepared photos”.",
            ),
        ],
        prepared_photos=str(out),
    )


def preview_cutout(s: Session, a: PreviewArgs) -> Result:
    files = s.find_photos([a.name]) if a.name else s.photo_files()[:1]
    if not files:
        return Result(False, [t(s.lang, "No encontré esa foto.", "I couldn't find that photo.")])
    model = _own_model()
    if model is None:
        return Result(
            False, [t(s.lang, "Optiverse no está instalado.", "Optiverse is not installed.")]
        )
    import cv2

    rgb = read_rgb(files[0], 1280)
    _, subject, scene = model.run(rgb)
    alpha = ((subject > 0.5) * 255).astype(np.uint8)
    out = s.work / "vistas"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{files[0].stem}_recorte.png"
    cv2.imwrite(str(target), np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), alpha]))
    return Result(
        True,
        [
            t(
                s.lang,
                f"Así recorta Optiverse {files[0].name} (escena: {scene_name(s.lang, max(scene, key=scene.get))}).",
                f"This is Optiverse's cut-out of {files[0].name} (scene: {max(scene, key=scene.get)}).",
            )
        ],
        images=[str(target)],
    )


def _load_mesh(path: Path) -> Any:
    import trimesh

    return trimesh.load(path, force="mesh", process=False)


def _stats(mesh: Any) -> dict[str, Any]:
    return {
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "parts": len(mesh.split(only_watertight=False)) if len(mesh.faces) < 400_000 else None,
        "size": [round(float(v), 4) for v in mesh.extents],
    }


def _vertex_colors(mesh: Any) -> np.ndarray | None:
    if getattr(mesh.visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(mesh.visual.vertex_colors)
    return colors if len(colors) == len(mesh.vertices) else None


def _edit(
    s: Session, verb_es: str, verb_en: str, change: Callable[[Any], Any], keeps_topology: bool
) -> Result:
    src = s.current_mesh()
    if src is None:
        return Result(
            False,
            [
                t(
                    s.lang,
                    "Todavía no hay modelo 3D: reconstruye primero.",
                    "There is no 3D model yet: reconstruct first.",
                )
            ],
        )
    mesh = _load_mesh(src)
    before = _stats(mesh)
    texture_lost = False
    if not keeps_topology and getattr(mesh.visual, "kind", None) == "texture":
        mesh.visual = mesh.visual.to_color()  # new faces have no UVs: keep the colours per vertex
        texture_lost = True
    colors = _vertex_colors(mesh)
    original = np.asarray(mesh.vertices).copy()
    mesh = change(mesh)
    if colors is not None:
        # Edits add faces and may reorder vertices: rebuild the colours from the
        # nearest original vertex so the visual always matches the new mesh.
        import trimesh
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(original).query(np.asarray(mesh.vertices))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors[nearest])
    out = s.work / "modelos"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"modelo-{len(s.state['meshes']) + 1:02d}.glb"
    mesh.export(target)
    s.state["meshes"].append(str(target))
    s.save()
    after = _stats(mesh)
    lines = [t(s.lang, f"Listo: {verb_es}.", f"Done: {verb_en}.")]
    if after["faces"] != before["faces"]:
        lines.append(
            t(
                s.lang,
                f"Caras: {num(s.lang, before['faces'])} → {num(s.lang, after['faces'])}.",
                f"Faces: {num(s.lang, before['faces'])} → {num(s.lang, after['faces'])}.",
            )
        )
    if after["watertight"] and not before["watertight"]:
        lines.append(
            t(
                s.lang,
                "Ahora el modelo está cerrado, listo para imprimir.",
                "The model is now watertight, ready to print.",
            )
        )
    if texture_lost:
        lines.append(
            t(
                s.lang,
                "La textura pasó a color por vértice (esta operación cambia la malla).",
                "The texture became per-vertex colour (this edit changes the mesh).",
            )
        )
    lines.append(t(s.lang, "Si no te gusta, di «deshacer».", "Say “undo” if you don't like it."))
    return Result(True, lines, data={"before": before, "after": after}, model=str(target))


def fill_holes(s: Session, a: HolesArgs) -> Result:
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh

    def change(mesh: Any) -> Any:
        options = CleanOptions(
            min_component_ratio=0.0, fill_holes=True, max_hole_edges=a.max_hole_edges
        )
        return clean_mesh(mesh, options)[0]

    return _edit(s, "rellené los huecos", "holes filled", change, keeps_topology=False)


def smooth_model(s: Session, a: SmoothArgs) -> Result:
    import trimesh

    def change(mesh: Any) -> Any:
        trimesh.smoothing.filter_taubin(mesh, iterations={1: 5, 2: 12, 3: 25}[a.strength])
        return mesh

    return _edit(s, "suavicé la superficie", "surface smoothed", change, keeps_topology=True)


def remove_loose_parts(s: Session, _: NoArgs) -> Result:
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh

    def change(mesh: Any) -> Any:
        options = CleanOptions(keep_largest_only=True, fill_holes=False)
        return clean_mesh(mesh, options)[0]

    return _edit(
        s, "quité los restos sueltos", "loose pieces removed", change, keeps_topology=False
    )


def simplify_model(s: Session, a: SimplifyArgs) -> Result:
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh

    def change(mesh: Any) -> Any:
        target = max(4, int(len(mesh.faces) * a.percent / 100))
        options = CleanOptions(min_component_ratio=0.0, fill_holes=False, target_faces=target)
        return clean_mesh(mesh, options)[0]

    return _edit(
        s,
        f"simplifiqué al {a.percent} %",
        f"simplified to {a.percent} %",
        change,
        keeps_topology=False,
    )


def set_height(s: Session, a: HeightArgs) -> Result:
    src = s.current_mesh()
    up = 1 if src is not None and src.suffix.lower() in {".glb", ".gltf"} else 2  # glTF is Y-up

    def change(mesh: Any) -> Any:
        height = float(mesh.extents[up])
        if height <= 0:
            raise InputValidationError("The model has no height.")
        mesh.apply_scale(a.mm / height)
        return mesh

    return _edit(
        s,
        f"el modelo ahora mide {a.mm:g} mm de alto",
        f"the model is now {a.mm:g} mm tall",
        change,
        keeps_topology=True,
    )


class ScaleArgs(BaseModel):
    percent: float = Field(gt=-90, lt=1000, description="+20 = 20 % bigger, -20 = 20 % smaller")


def scale_model(s: Session, a: ScaleArgs) -> Result:
    factor = 1 + a.percent / 100

    def change(mesh: Any) -> Any:
        mesh.apply_scale(factor)
        return mesh

    es = f"lo hice un {abs(a.percent):g} % {'más grande' if a.percent > 0 else 'más pequeño'}"
    en = f"made it {abs(a.percent):g} % {'bigger' if a.percent > 0 else 'smaller'}"
    return _edit(s, es, en, change, keeps_topology=True)


def ground_model(s: Session, _: NoArgs) -> Result:
    src = s.current_mesh()
    up = 1 if src is not None and src.suffix.lower() in {".glb", ".gltf"} else 2

    def change(mesh: Any) -> Any:
        low, high = mesh.bounds
        shift = -(low + high) / 2
        shift[up] = -low[up]  # base on the ground, centred on the other axes
        mesh.apply_translation(shift)
        return mesh

    return _edit(
        s,
        "lo centré y lo apoyé sobre su base",
        "centred it and set it on its base",
        change,
        keeps_topology=True,
    )


def undo(s: Session, _: NoArgs) -> Result:
    if not s.state["meshes"]:
        return Result(
            False, [t(s.lang, "No hay cambios que deshacer.", "There is nothing to undo.")]
        )
    s.state["meshes"].pop()
    s.save()
    current = s.current_mesh()
    return Result(
        True,
        [
            t(
                s.lang,
                "Deshecho: volví a la versión anterior.",
                "Undone: back to the previous version.",
            )
        ],
        model=str(current) if current else None,
    )


def model_info(s: Session, _: NoArgs) -> Result:
    src = s.current_mesh()
    if src is None:
        return Result(False, [t(s.lang, "Todavía no hay modelo 3D.", "There is no 3D model yet.")])
    st = _stats(_load_mesh(src))
    lines = [
        t(
            s.lang,
            f"{num(s.lang, st['faces'])} caras · {'cerrado' if st['watertight'] else 'con huecos'}.",
            f"{st['faces']:,} faces · {'watertight' if st['watertight'] else 'has holes'}.",
        ),
    ]
    if st["parts"] and st["parts"] > 1:
        lines.append(
            t(
                s.lang,
                f"Tiene {st['parts']} piezas sueltas: puedo quitar los restos.",
                f"It has {st['parts']} loose pieces: I can remove the leftovers.",
            )
        )
    return Result(True, lines, data=st)


def capture_advice(s: Session, _: NoArgs) -> Result:
    lines: list[str] = []
    report = s.job / "report.json" if s.job else None
    if report and report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
        for w in (data.get("warnings") or [])[:4]:
            lines.append(f"• {w}")
        quality = ((data.get("details") or {}).get("silhouette") or {}).get("quality") or {}
        if quality.get("covered") is not None:
            lines.insert(
                0,
                t(
                    s.lang,
                    f"Tus fotos cubren bien el {round(100 * quality['covered'])} % del modelo.",
                    f"Your photos support {round(100 * quality['covered'])} % of the model well.",
                ),
            )
    review = s.state.get("review") or {}
    weak = [k for k, v in review.items() if set(v.get("issues", [])) & {"blurry", "cut", "small"}]
    if weak:
        lines.append(
            t(
                s.lang,
                f"Repite {len(weak)} fotos con problemas (movidas, cortadas o lejanas).",
                f"Retake {len(weak)} photos with problems (blurred, cut or far).",
            )
        )
    if not lines:
        lines = [
            t(
                s.lang,
                "Consejos: 40–60 fotos en 3 vueltas (baja, media y alta) más 4 desde arriba, el objeto quieto, luz pareja y sin reflejos. Pídeme «revisa las fotos» para ver cuáles fallan.",
                "Tips: 40–60 photos in 3 rings (low, mid, high) plus 4 from above, the object still, even light, no glare. Ask me to “review the photos” to see which fail.",
            )
        ]
    return Result(True, lines)


def show_help(s: Session, _: NoArgs) -> Result:
    return Result(
        True,
        [
            t(s.lang, "Puedo ayudarte con:", "I can help with:"),
            t(
                s.lang,
                "• Fotos: «revisa las fotos», «descarta las movidas», «mejora la luz», «muéstrame el recorte de IMG_4923», «prepara las fotos».",
                "• Photos: “review the photos”, “drop the blurred ones”, “fix the light”, “show the cut-out of IMG_4923”, “prepare the photos”.",
            ),
            t(
                s.lang,
                "• Modelo 3D: «rellena los huecos», «suavízalo», «quita los restos», «simplifícalo a la mitad», «mide 24 cm de alto», «deshacer».",
                "• 3D model: “fill the holes”, “smooth it”, “remove the leftovers”, “halve the faces”, “make it 24 cm tall”, “undo”.",
            ),
            t(
                s.lang,
                "• Captura: «¿qué fotos me faltan?».",
                "• Capture: “which photos am I missing?”.",
            ),
        ],
    )


@dataclass(frozen=True)
class Action:
    run: Callable[[Session, Any], Result]
    args: type[BaseModel]
    doc: str  # what the language model reads


ACTIONS: dict[str, Action] = {
    "review_photos": Action(
        review_photos, NoArgs, "check every photo (blur, light, subject cut or small, scene)"
    ),
    "exclude_photos": Action(
        exclude_photos,
        PhotosArgs,
        'drop photos: {"names": [...]} or {"issue": "blurry|dark|bright|small|cut|any"}',
    ),
    "restore_photos": Action(
        restore_photos, PhotosArgs, 'bring dropped photos back: {"names": [...]} or {} for all'
    ),
    "enhance_photos": Action(
        enhance_photos,
        EnhanceArgs,
        'fix light/contrast: {"names": [...]} or {"all": true}; {} = the dark/bright ones',
    ),
    "prepare_photos": Action(
        prepare_photos, NoArgs, "build the photo set to reconstruct with (after dropping/enhancing)"
    ),
    "preview_cutout": Action(
        preview_cutout, PreviewArgs, 'show the subject cut-out of a photo: {"name": "IMG_4923"}'
    ),
    "fill_holes": Action(fill_holes, HolesArgs, "close holes in the 3D model"),
    "smooth_model": Action(smooth_model, SmoothArgs, 'smooth the surface: {"strength": 1|2|3}'),
    "remove_loose_parts": Action(
        remove_loose_parts, NoArgs, "keep only the main piece of the 3D model"
    ),
    "simplify_model": Action(
        simplify_model, SimplifyArgs, 'fewer faces: {"percent": 5..95 to keep}'
    ),
    "set_height": Action(set_height, HeightArgs, 'real size: {"mm": height in millimetres}'),
    "undo": Action(undo, NoArgs, "undo the last 3D model edit"),
    "model_info": Action(model_info, NoArgs, "faces, holes and loose pieces of the 3D model"),
    "capture_advice": Action(
        capture_advice, NoArgs, "which photos are missing and how to capture better"
    ),
    "scale_model": Action(scale_model, ScaleArgs, 'bigger/smaller: {"percent": +20 or -20}'),
    "ground_model": Action(ground_model, NoArgs, "centre the model and set it on its base"),
    "help": Action(show_help, NoArgs, "what the assistant can do"),
}


def run_action(s: Session, name: str, args: dict[str, Any] | None) -> Result:
    action = ACTIONS.get(name)
    if action is None:
        return Result(False, [t(s.lang, "No sé hacer eso todavía.", "I can't do that yet.")])
    try:
        parsed = action.args.model_validate(args or {})
    except ValidationError as exc:
        logger.info("Invalid arguments for %s: %s", name, exc)
        return Result(
            False,
            [
                t(
                    s.lang,
                    "No entendí bien los datos de esa orden.",
                    "I didn't get the details of that request.",
                )
            ],
        )
    try:
        return action.run(s, parsed)
    except InputValidationError as exc:
        return Result(False, [exc.message])
    except Exception as exc:  # report, never crash the chat
        logger.warning("Assistant action %s failed: %s", name, exc)
        return Result(False, [t(s.lang, f"No pude hacerlo: {exc}", f"I couldn't do it: {exc}")])
