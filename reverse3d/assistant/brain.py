"""Understanding the request: a local multilingual language model plus exact rules.

The model (paraphrase-multilingual-MiniLM-L12-v2, Apache-2.0, 471 MB, pinned
revision) does not write text: it measures which action the request means by
comparing it with example phrases in Spanish and English. It acts only when it
is sure; otherwise the assistant asks. Numbers, units and photo names are read
with exact rules, so a "24 cm" never becomes 24 mm.

A small generative model (Qwen2.5-0.5B) was tried first and rejected: it chose
wrong actions ("hola" -> undo) and took 7-31 s per message on a laptop CPU.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from reverse3d.assistant.talk import CHAT_EXAMPLES
from reverse3d.config import Settings

logger = logging.getLogger(__name__)

ASSISTANT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#: one model loads at a time: two threads importing transformers at once break it
LOAD_LOCK = threading.Lock()
#: model id → (kind, licence); every entry must be commercial-friendly.
ALLOWED_ASSISTANTS: dict[str, tuple[str, str]] = {ASSISTANT_MODEL: ("assistant", "Apache-2.0")}


@dataclass
class Plan:
    actions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    reply: str = ""
    source: str = "rules"  # model | rules | none
    ask: str = ""  # what to ask when an action needs more detail


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_PHOTO = re.compile(
    r"\b([a-z]{2,6}[_-]?\d{2,6}(?:\.(?:jpe?g|png|heic|heif|tiff?|webp|bmp))?)\b", re.I
)


_PHOTO_WORD = re.compile(r"(foto|photo|imagen|image|picture|img)", re.I)
_NUMBER = re.compile(r"(?<![\d.,])(\d{3,6})(?![\d.,%]|\s*(?:mm|cm|m|%))")


def _photo_names(text: str) -> list[str]:
    names = [m.group(1) for m in _PHOTO.finditer(text)]
    if not names and _PHOTO_WORD.search(text):  # "las fotos 4950 y 4951"
        names = [m.group(1) for m in _NUMBER.finditer(text)]
    return names


def _height_mm(text: str) -> float | None:
    m = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m|metros?|centimetros?|milimetros?|in|inch(?:es)?|pulgadas?)\b",
        _plain(text),
    )
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    if unit.startswith(("mm", "mili")):
        return value
    if unit.startswith(("cm", "centi")):
        return value * 10
    if unit.startswith(("in", "pulg")):
        return value * 25.4
    return value * 1000


def _percent(text: str) -> int | None:
    plain = _plain(text)
    m = re.search(r"(\d{1,2})\s*%", plain)
    if m:
        return int(m.group(1))
    if re.search(r"\b(mitad|half)\b", plain):
        return 50
    if re.search(r"\b(tercio|third)\b", plain):
        return 33
    if re.search(r"\b(cuarto|quarter)\b", plain):
        return 25
    return None


_ISSUES = [
    ("blurry", r"movid|borros|desenfoc|blur|shak|fuera de foco"),
    ("dark", r"oscur|dark|poca luz"),
    ("bright", r"quemad|brillant|sobreexpu|bright|overexpo|mucha luz"),
    ("cut", r"cortad|\bcut\b|cropped"),
    ("small", r"pequen|lejos|small|far away"),
]


def _issue(text: str) -> str | None:
    plain = _plain(text)
    for key, pattern in _ISSUES:
        if re.search(pattern, plain):
            return key
    if re.search(r"\b(malas|feas|problemas|bad|wrong|fallan)\b", plain):
        return "any"
    return None


_RULES: list[tuple[str, str]] = [
    (
        "scale_model",
        r"(mas grande|mas pequen|agranda|achica|encoge|el tamano|su tamano|bigger|smaller|enlarge|shrink|scale it up|scale it down)",
    ),
    (
        "ground_model",
        r"(apoya|apoyalo|ponlo derecho|en el suelo|sobre la mesa|centralo|centra el|center it|on the ground|on its base)",
    ),
    (
        "chat_history",
        r"(que has hecho|que hiciste|que cambios|que le hiciste|what did you do|what changed)",
    ),
    ("chat_who", r"\b(quien eres|que eres|como te llamas|who are you|what are you)\b"),
    (
        "chat_status",
        r"(listo para imprimir|cuantas fotos|como va|en que estado|ready to print|how many photos)",
    ),
    ("chat_bye", r"\b(adios|hasta luego|nos vemos|chao|bye|see you)\b"),
    ("undo", r"\b(desha[cz]|undo|revert|reviert|vuelve atras|como estaba)"),
    (
        "restore_photos",
        r"\b(recupera|restaura|devuelve|restore|bring back|vuelve a (?:usar|poner))",
    ),
    (
        "exclude_photos",
        r"\b(descart|quita(?:r)? (?:la|las|esa|esas) foto|elimina(?:r)? (?:la|las) foto|borra(?:r)? (?:la|las) foto|no (?:la |las |lo |los )?uses|drop|exclude|remove the photo|discard)",
    ),
    (
        "enhance_photos",
        r"\b(mejora(?:r)? (?:la )?luz|ilumina|aclara|brillo|contraste|enhance|brighten|fix the light|exposure|exposicion)",
    ),
    ("prepare_photos", r"\b(prepara|prepare|listas para reconstruir)"),
    ("preview_cutout", r"\b(recorte|cut-?out|mascara|mask|silueta)"),
    (
        "review_photos",
        r"\b(revisa|analiza|comprueba|check|review|analy[sz]e|como estan|que fotos fallan|(?:que tal|como) (?:salieron|quedaron|estan|se ven) (?:mis |las )?fotos)",
    ),
    ("fill_holes", r"\b(hueco|agujero|cierra|rellena|hole|fill|watertight|estanco)"),
    ("smooth_model", r"\b(suavi|lisa|liso|alis|puli|arruga|pico|smooth|wrinkl|noise|ruido|bump)"),
    (
        "remove_loose_parts",
        r"\b(resto|suelt|flota|basur|pedaz|loose|leftover|debris|floating|island)",
    ),
    (
        "simplify_model",
        r"\b(simplific|aligera|menos caras|reduce las caras|reduce el peso|lighter|decimat|fewer faces|optimiza|pesad|pesa|heavy|weigh|file size)",
    ),
    ("set_height", r"\b(mid[aeo]n?|medir|altura|alto|tamano|escala|tall|height|size|scale to)"),
    (
        "capture_advice",
        r"\b(consejo|falta|que fotos tomo|como (?:tomo|saco|hago) las fotos|advice|missing|tips|how to capture)",
    ),
    ("model_info", r"\b(info|estado del modelo|cuantas caras|stats|faces)"),
    ("help", r"\b(ayuda|que puedes|help|what can you|hola|buenas|hello|hi|hey)\b"),
]


_DROP_VERBS = r"((?:quita|saca|elimina|borra|descarta)(?:melas|mela|me|las|la|los|lo)?|no (?:la |las |lo |los )?uses|remove|drop|delete|exclude|discard)"
_SEE_VERBS = r"(ver|mira|muestra|muestrame|ensena|ensename|show|see|look)"


def rules(message: str, whole: str | None = None) -> Plan:
    """Exact rules for Spanish and English requests (also the model's backup).

    whole: the full request when message is one of its clauses ("…, no las uses"
    takes "las" from the rest).
    """
    plain = _plain(message)
    plan = Plan(source="rules")
    names = _photo_names(message)
    if names and re.search(r"(?<![a-z])" + _DROP_VERBS + r"(?![a-z])", plain):
        plan.actions.append(("exclude_photos", {"names": names}))
        return plan
    if names and re.search(r"(?<![a-z])" + _SEE_VERBS + r"(?![a-z])", plain):
        plan.actions.append(("preview_cutout", {"name": names[0]}))
        return plan
    for name, pattern in _RULES:
        if not re.search(pattern, plain):
            continue
        args: dict[str, Any] = {}
        if name in ("exclude_photos", "restore_photos"):
            names, issue = (
                _photo_names(message),
                _issue(message) or (_issue(whole) if whole else None),
            )
            if names:
                args["names"] = names
            elif issue:
                args["issue"] = issue
            elif name == "exclude_photos":
                continue  # "drop" alone: nothing to drop
        elif name == "enhance_photos":
            names = _photo_names(message)
            if names:
                args["names"] = names
            elif re.search(r"\b(todas|all)\b", plain):
                args["all"] = True
        elif name == "preview_cutout":
            names = _photo_names(message)
            args["name"] = names[0] if names else ""
        elif name == "scale_model":
            args["percent"] = _scale_percent(message)
        elif name == "set_height":
            mm = _height_mm(message)
            if mm is None:
                continue
            args["mm"] = mm
        elif name == "simplify_model":
            pct = _percent(message)
            if pct:
                args["percent"] = pct
        elif name == "smooth_model":
            if re.search(r"\b(mucho|bastante|fuerte|more|strong|a lot)\b", plain):
                args["strength"] = 3
            elif re.search(r"\b(poco|leve|suave|light|bit|slightly)\b", plain):
                args["strength"] = 1
        plan.actions.append((name, args))
        if name == "undo":
            break
    if len(plan.actions) > 1:  # "hola, …": the greeting is not a request
        plan.actions = [a for a in plan.actions if a[0] != "help"]
    found = {n for n, _ in plan.actions}
    if "set_height" in found:  # "que tenga 18 cm de alto" is a size, not a rescale
        plan.actions = [a for a in plan.actions if a[0] != "scale_model"]
    if "scale_model" in found:  # "reduce el tamaño" changes size, not faces
        plan.actions = [a for a in plan.actions if a[0] != "simplify_model"]
    return plan


#: example requests per action (Spanish and English); the model compares with these
EXAMPLES: dict[str, list[str]] = {
    "scale_model": [
        "hazlo un 20 % más grande",
        "agrándalo un poco",
        "hazlo más pequeño",
        "reduce su tamaño a la mitad",
        "make it 20 % bigger",
        "make it smaller",
        "enlarge it",
        "shrink it",
    ],
    "ground_model": [
        "apóyalo en la base",
        "ponlo derecho sobre el suelo",
        "céntralo",
        "que quede apoyado en la mesa",
        "put it on the ground",
        "center the model",
        "set it on its base",
    ],
    "review_photos": [
        "revisa mis fotos",
        "mira si alguna foto está mal",
        "¿están bien mis fotos?",
        "analiza las fotos",
        "qué fotos fallan",
        "comprueba la calidad de las fotos",
        "review my photos",
        "are my photos ok",
        "check the photos",
        "which photos are bad",
        "¿alguna foto no sirve?",
        "hay fotos que no valen",
        "is any picture useless",
    ],
    "exclude_photos": [
        "descarta las fotos movidas",
        "saca de la lista las fotos oscuras",
        "no uses las fotos borrosas",
        "elimina las fotos malas",
        "quita esas fotos del proceso",
        "deja fuera las fotos cortadas",
        "drop the blurred photos",
        "remove the dark photos",
        "do not use the bad photos",
        "exclude those photos",
    ],
    "restore_photos": [
        "recupera las fotos",
        "vuelve a usar todas las fotos",
        "devuelve las fotos que quitaste",
        "restaura las fotos",
        "bring the photos back",
        "use all the photos again",
        "restore the photos",
    ],
    "enhance_photos": [
        "mejora la luz de las fotos",
        "aclara las fotos oscuras",
        "arregla el brillo",
        "sube el contraste",
        "corrige la exposición",
        "las fotos están muy oscuras, mejóralas",
        "fix the light",
        "brighten the dark photos",
        "improve the exposure",
        "enhance the photos",
    ],
    "prepare_photos": [
        "prepara las fotos",
        "deja listas las fotos para reconstruir",
        "aplica los cambios a las fotos",
        "prepare the photos",
        "get the photos ready",
        "apply the photo changes",
    ],
    "preview_cutout": [
        "muéstrame el recorte",
        "enséñame cómo recorta la foto",
        "cómo ve Optiverse esta foto",
        "ver la máscara",
        "qué parte de la foto es el objeto",
        "show me the cut-out",
        "how does Optiverse see this photo",
        "show the mask",
    ],
    "fill_holes": [
        "rellena los huecos",
        "el modelo tiene agujeros, arréglalo",
        "cierra el modelo",
        "tapa los agujeros de la base",
        "que quede cerrado para imprimir",
        "fill the holes",
        "close the gaps",
        "make it watertight",
        "the model has holes",
    ],
    "smooth_model": [
        "suavízalo",
        "que quede más suave",
        "hazlo más liso",
        "quita las arrugas",
        "elimina el ruido de la superficie",
        "smooth it",
        "make it smoother",
        "remove the bumps",
        "the surface is noisy",
    ],
    "remove_loose_parts": [
        "quita los restos sueltos",
        "elimina los pedacitos que flotan",
        "borra la basura alrededor del modelo",
        "deja solo la pieza principal",
        "remove the leftovers",
        "delete the floating bits",
        "keep only the main object",
        "clean the debris",
        "hay mucha suciedad alrededor del objeto",
        "está lleno de cosas sueltas",
        "there is junk around the model",
    ],
    "simplify_model": [
        "simplifícalo",
        "hazlo más ligero",
        "reduce las caras",
        "que pese menos el archivo",
        "optimízalo para la web",
        "make it lighter",
        "reduce the polygon count",
        "simplify the mesh",
        "smaller file for the web",
        "el archivo pesa demasiado",
        "the file is too big",
        "too many polygons",
    ],
    "set_height": [
        "que mida 24 cm de alto",
        "ponlo a tamaño real",
        "la altura real es 18 centímetros",
        "escálalo a 10 cm",
        "quiero imprimirlo a 20 cm",
        "make it 24 cm tall",
        "set the real height",
        "scale it to 10 cm",
    ],
    "undo": [
        "deshacer",
        "vuelve atrás",
        "déjalo como estaba",
        "no me gusta, quita el cambio",
        "undo",
        "go back",
        "revert the last change",
        "deshaz el último cambio",
        "quita lo último que hiciste",
    ],
    "model_info": [
        "¿cómo está el modelo?",
        "cuántas caras tiene",
        "información del modelo",
        "model info",
        "how many faces",
        "is the model closed",
    ],
    "capture_advice": [
        "¿qué fotos me faltan?",
        "cómo debería tomar las fotos la próxima vez",
        "consejos para capturar mejor",
        "desde dónde tomo más fotos",
        "qué ángulos tengo que repetir",
        "what angles should I retake",
        "where should I take more pictures",
        "which photos am I missing",
        "how should I take the photos",
        "capture tips",
    ],
    "help": [
        "hola",
        "¿qué puedes hacer?",
        "ayuda",
        "buenas",
        "¿cómo funcionas?",
        "hello",
        "what can you do",
        "help",
    ],
}

_POLITE = re.compile(r"(?i)(por favor|porfa|porfavor|please|gracias de antemano)")
#: below this similarity the assistant asks instead of acting (measured in tests)
CONFIDENT = 0.55
_JOINERS = ("y luego", "y despues", "y después", "luego", "and then", "then", "y", "and")
_SPLIT = re.compile(r"\s*[,;]\s*|(?<!\d)\s+(?:" + "|".join(_JOINERS) + r")\s+(?!\d)", re.I)


_SHORT = [
    (
        "chat_yes",
        r"^(si|vale|dale|ok|okay|claro|hazlo|de acuerdo|adelante|perfecto|yes|sure|yep|go ahead|do it)[.! ]*$",
    ),
    ("chat_no", r"^(no|mejor no|nop|dejalo asi|asi esta bien|no gracias|no thanks|nope)[.! ]*$"),
    ("chat_more", r"^(mas|un poco mas|aun mas|otra vez|de nuevo|again|more|once more)[.! ]*$"),
    ("chat_less", r"^(menos|un poco menos|te pasaste|demasiado|less|too much)[.! ]*$"),
    ("chat_thanks", r"^(gracias|muchas gracias|mil gracias|thanks|thank you)[.! ]*$"),
]


def short_reply(message: str) -> str | None:
    """«sí», «no», «más», «menos», «gracias»: exact, before any model."""
    plain = _plain(message).strip()
    for name, pattern in _SHORT:
        if re.match(pattern, plain):
            return name
    pieces = [p.strip(" .!¡?¿") for p in re.split(r"[,;]", plain) if p.strip(" .!¡?¿")]
    if len(pieces) > 1:
        found = {next((n for n, pat in _SHORT if re.match(pat, piece)), None) for piece in pieces}
        if len(found) == 1 and None not in found:
            return found.pop()
    return None


def _scale_percent(message: str) -> float:
    """+20 for "20 % bigger", -20 for "20 % smaller"; "double" is +100."""
    plain = _plain(message)
    smaller = re.search(r"(pequen|chico|reduce|achica|encoge|smaller|shrink)", plain) is not None
    if not smaller and re.search(r"(doble|double|twice)", plain):
        return 100.0
    pct = _percent(message) or 20
    return float(-min(pct, 89) if smaller else pct)


def _with_args(name: str, message: str) -> tuple[str, dict[str, Any]] | str:
    """The action with its details from the user's own words, or what to ask."""
    if name.startswith("chat_"):
        return name, {}
    if name == "scale_model":
        return name, {"percent": _scale_percent(message)}
    for rule_name, args in rules(message).actions:
        if rule_name == name:
            return name, args
    plain = _plain(message)
    if name in ("exclude_photos", "restore_photos", "enhance_photos"):
        names, issue = _photo_names(message), _issue(message)
        if names:
            return name, {"names": names}
        if issue and name != "enhance_photos":
            return name, {"issue": issue}
        if name == "exclude_photos":
            return "Dime cuáles: por ejemplo «las movidas», «las oscuras» o «IMG_4930»."
        return name, ({"all": True} if re.search(r"(?<![a-z])(todas|all)(?![a-z])", plain) else {})
    if name == "set_height":
        mm = _height_mm(message)
        return (name, {"mm": mm}) if mm else "¿Cuánto debe medir de alto? Por ejemplo «24 cm»."
    if name == "simplify_model":
        pct = _percent(message)
        return name, ({"percent": pct} if pct else {})
    if name == "preview_cutout":
        names = _photo_names(message)
        return name, {"name": names[0] if names else ""}
    return name, {}


class Brain:
    """Loads the language model once; the exact rules work even without it."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._bank: Any = None
        self._labels: list[str] = []
        self._failed = False
        self._lock = LOAD_LOCK

    @property
    def available(self) -> bool:
        from reverse3d.offline import local_model_path

        return local_model_path(ASSISTANT_MODEL) is not None

    def _embed(self, texts: list[str]) -> Any:
        import torch

        batch = self._tokenizer(
            texts, padding=True, truncation=True, max_length=96, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = self._encoder(**batch).last_hidden_state
        weights = batch["attention_mask"].unsqueeze(-1).float()
        vectors = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(vectors, dim=1)

    def _load(self) -> bool:
        if self._encoder is not None:
            return True
        if self._failed:
            return False
        try:
            from transformers import AutoModel, AutoTokenizer

            from reverse3d.offline import resolve_model

            path = resolve_model(ASSISTANT_MODEL, self.settings)
            # Local files of the pinned revision only (resolve_model checked them).
            self._tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)  # nosec B615
            encoder: Any = AutoModel.from_pretrained(str(path), local_files_only=True)  # nosec B615
            self._encoder = encoder.eval()
            bank = {**EXAMPLES, **CHAT_EXAMPLES}
            self._labels = [name for name, phrases in bank.items() for _ in phrases]
            self._bank = self._embed([p for phrases in bank.values() for p in phrases])
            return True
        except Exception as exc:  # no model / no torch: the rules still work
            logger.info("Assistant language model unavailable: %s", exc)
            self._failed = True
            return False

    def warm_up(self) -> None:
        """Load the model in the background: until it is ready, the rules answer."""

        def load() -> None:
            with self._lock:
                self._load()

        threading.Thread(target=load, name="assistant-model", daemon=True).start()

    def ready(self) -> bool:
        """Whether the model can answer now (never waits for a background load)."""
        if self._encoder is not None:
            return True
        if self._lock.locked() or self._failed:
            return False
        with self._lock:
            return self._load()

    def classify(self, text: str) -> tuple[str, float]:
        """The action the text most likely means and how sure the model is (0-1)."""
        sims = (self._embed([text]) @ self._bank.T)[0]
        best: dict[str, float] = {}
        for label, sim in zip(self._labels, sims.tolist(), strict=True):
            best[label] = max(best.get(label, -1.0), sim)
        name = max(best, key=lambda k: best[k])
        return name, best[name]

    def understand(self, message: str, context: str = "") -> Plan:
        quick = short_reply(message)
        if quick:
            return Plan(actions=[(quick, {})], source="rules")
        exact = rules(message)
        if not self.ready():
            return exact
        if (
            exact.actions
            and exact.actions[0][0] in ("exclude_photos", "preview_cutout")
            and _photo_names(message)
        ):
            return exact
        plan = Plan(source="model")
        clauses = [c for c in _SPLIT.split(message) if c and len(c.strip()) > 1] or [message]
        for clause in clauses:
            clause = _POLITE.sub(" ", clause).strip(" .,!?¡¿")
            if not clause or (
                len(clauses) > 1 and len(clause.split()) < 2 and not rules(clause).actions
            ):
                continue  # "ajústalo", "oye", "por favor": not a request on their own
            exact_steps = [a for a in rules(clause, message).actions if a[0] != "help"]
            if exact_steps:
                # The user's exact words ("huecos", "suavízalo", a photo name) win:
                # the model decides only when the rules recognise nothing.
                steps: list[tuple[str, dict[str, Any]] | str] = list(exact_steps)
            else:
                name, sure = self.classify(clause)
                logger.debug("assistant: %r -> %s (%.2f)", clause, name, sure)
                needed = 0.75 if name.startswith("chat_") else CONFIDENT
                if sure < needed:
                    continue
                steps = [_with_args(name, clause if name != "set_height" else message)]
            for step in steps:
                if isinstance(step, str):
                    plan.ask = plan.ask or step
                elif step not in plan.actions:
                    plan.actions.append(step)
        if len(plan.actions) > 1:  # "oye, …": a greeting next to a request is not a request
            plan.actions = [a for a in plan.actions if a[0] != "help"]
        if any(not n.startswith("chat_") for n, _ in plan.actions):
            plan.actions = [a for a in plan.actions if not a[0].startswith("chat_")]
        if any(n == "exclude_photos" and a.get("issue") for n, a in plan.actions):
            plan.actions = [
                a for a in plan.actions if not (a[0] == "enhance_photos" and not a[1].get("names"))
            ]
        if not plan.actions and exact.actions:
            return exact
        if not plan.actions and not plan.ask:
            plan.source = "none"
        return plan
