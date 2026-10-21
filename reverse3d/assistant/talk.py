"""Optiverse talking: natural first-person replies, follow-ups and offers.

Everything said is built from what the actions really did (their result
lines, the session history and the files on disk): nothing is invented. What
makes it feel like a conversation:

* varied, first-person wording ("¡Listo! Cerré los huecos…");
* it remembers the last change: "más", "otra vez", "un poco menos";
* it offers the natural next step and acts when you say "sí";
* it answers about itself, its history and the state of your work.
"""

from __future__ import annotations

import random
import time
from typing import Any

from reverse3d.assistant.actions import ACTIONS, Result, Session, num, run_action, t

#: conversation intents the language model recognises next to the actions
CHAT_EXAMPLES: dict[str, list[str]] = {
    "chat_yes": [
        "sí",
        "vale",
        "dale",
        "ok",
        "hazlo",
        "de acuerdo",
        "claro que sí",
        "perfecto, hazlo",
        "yes",
        "sure",
        "go ahead",
        "do it",
    ],
    "chat_no": [
        "no",
        "mejor no",
        "déjalo así",
        "no gracias",
        "así está bien",
        "no thanks",
        "leave it",
        "it's fine as it is",
    ],
    "chat_thanks": [
        "gracias",
        "muchas gracias",
        "genial, gracias",
        "te lo agradezco",
        "thank you",
        "thanks a lot",
    ],
    "chat_bye": ["adiós", "hasta luego", "nos vemos", "chao", "bye", "see you"],
    "chat_who": [
        "quién eres",
        "qué eres",
        "cómo te llamas",
        "qué sabes hacer tú",
        "who are you",
        "what are you",
    ],
    "chat_history": [
        "qué has hecho",
        "qué cambios hiciste",
        "resúmeme los cambios",
        "qué le hiciste al modelo",
        "what did you do",
        "what changed",
    ],
    "chat_status": [
        "está listo para imprimir",
        "cómo va el modelo",
        "cuántas fotos quedan",
        "en qué estado está",
        "is it ready to print",
        "how many photos are left",
    ],
    "chat_more": [
        "más",
        "un poco más",
        "otra vez",
        "repítelo",
        "aún más",
        "dale más",
        "again",
        "more",
        "do it again",
    ],
    "chat_less": ["menos", "te pasaste", "demasiado", "un poco menos", "too much", "less"],
}

_OPEN = {
    "es": ["¡Listo!", "Hecho.", "Ya está.", "¡Perfecto!", "Listo."],
    "en": ["Done!", "All set.", "There you go.", "Done.", "Finished."],
}

#: the natural next step after each action: (action, args, question es, question en)
_OFFERS: dict[str, tuple[str, dict[str, Any], str, str]] = {
    "exclude_photos": (
        "prepare_photos",
        {},
        "¿Preparo el resto de las fotos para reconstruir?",
        "Shall I prepare the remaining photos for rebuilding?",
    ),
    "enhance_photos": (
        "prepare_photos",
        {},
        "¿Preparo las fotos con la luz mejorada?",
        "Shall I prepare the photos with the better light?",
    ),
    "fill_holes": (
        "smooth_model",
        {"strength": 1},
        "¿Lo suavizo un poco para que no se noten los parches?",
        "Shall I smooth it a little so the patches don't show?",
    ),
    "remove_loose_parts": (
        "fill_holes",
        {},
        "¿Reviso también si quedan huecos y los cierro?",
        "Shall I also close any remaining holes?",
    ),
    "set_height": (
        "model_info",
        {},
        "¿Quieres que te diga cómo quedó el modelo?",
        "Want me to tell you how the model looks now?",
    ),
}


def _clean(lines: list[str], lang: str) -> list[str]:
    """First person, one opener, the undo hint only once."""
    hint = t(lang, "Si no te gusta, di «deshacer».", "Say “undo” if you don't like it.")
    out: list[str] = []
    for line in lines:
        if line == hint:
            continue
        for prefix in ("Listo: ", "Done: "):
            if line.startswith(prefix):
                body = line[len(prefix) :]
                line = f"{random.choice(_OPEN[lang])} {body[:1].upper()}{body[1:]}"
        out.append(line)
    if hint in lines:
        out.append(hint)
    return out


def offer_after(
    s: Session, done: list[tuple[str, Result]]
) -> tuple[str, dict[str, Any], str] | None:
    """The next step worth offering after these results, if any (never one just done)."""
    just_done = {name for name, result in done if result.ok}
    for name, result in reversed(done):
        if not result.ok:
            continue
        if name == "review_photos" and (result.data or {}).get("counts"):
            bad = [
                k
                for k in (result.data or {})["counts"]
                if k in ("blurry", "dark", "bright", "small", "cut")
            ]
            if bad:
                return (
                    "exclude_photos",
                    {"issue": "any"},
                    t(s.lang, "¿Descarto esas fotos?", "Shall I drop those photos?"),
                )
        if name == "model_info":
            data = result.data or {}
            if data.get("parts") and data["parts"] > 1:
                return (
                    "remove_loose_parts",
                    {},
                    t(s.lang, "¿Quito las piezas sueltas?", "Shall I remove the loose pieces?"),
                )
            if not data.get("watertight"):
                return (
                    "fill_holes",
                    {},
                    t(s.lang, "¿Cierro los huecos?", "Shall I close the holes?"),
                )
        if name in _OFFERS:
            action, args, es, en = _OFFERS[name]
            if action in just_done:
                continue
            if name == "remove_loose_parts" and (result.data or {}).get("after", {}).get(
                "watertight"
            ):
                continue
            return (action, args, t(s.lang, es, en))
    return None


def remember(
    s: Session, done: list[tuple[str, Result]], steps: list[tuple[str, dict[str, Any]]]
) -> None:
    history = s.state.setdefault("history", [])
    for (name, result), (_, args) in zip(done, steps, strict=False):
        history.append({"action": name, "args": args, "ok": result.ok, "at": int(time.time())})
    del history[:-40]
    s.save()


def _last_edit(s: Session) -> dict[str, Any] | None:
    for item in reversed(s.state.get("history", [])):
        if (
            item["ok"]
            and item["action"] in ACTIONS
            and item["action"]
            not in (
                "undo",
                "model_info",
                "help",
                "capture_advice",
                "review_photos",
                "preview_cutout",
            )
        ):
            return item
    return None


def more_or_less(s: Session, more: bool) -> list[tuple[str, dict[str, Any]]] | str:
    """«más» repeats the last change stronger; «menos» undoes it and redoes it gentler."""
    last = _last_edit(s)
    if last is None:
        return t(
            s.lang,
            "Todavía no he hecho ningún cambio que repetir.",
            "I haven't made any change to repeat yet.",
        )
    name, args = last["action"], dict(last.get("args") or {})
    if name == "smooth_model":
        strength = int(args.get("strength", 2))
        if more:
            return [(name, {"strength": min(3, strength + 1)})]
        return [("undo", {})] + ([(name, {"strength": strength - 1})] if strength > 1 else [])
    if name == "simplify_model":
        pct = int(args.get("percent", 50))
        return [(name, {"percent": max(5, pct // 2)})] if more else [("undo", {})]
    if name == "scale_model":
        step = float(args.get("percent", 10))
        return (
            [(name, {"percent": step})] if more else [("undo", {}), (name, {"percent": step / 2})]
        )
    if more:
        return [(name, args)]
    return [("undo", {})]


def chat(s: Session, intent: str) -> tuple[list[str], list[tuple[str, dict[str, Any]]]]:
    """Reply to a conversation intent; may return actions to run (yes / more / less)."""
    lang = s.lang
    if intent == "chat_yes":
        offer = s.state.get("offer")
        if offer:
            s.state["offer"] = None
            s.save()
            return [], [(offer[0], offer[1])]
        return [t(lang, "¡Claro! Dime qué quieres que haga.", "Sure! Tell me what to do.")], []
    if intent == "chat_no":
        s.state["offer"] = None
        s.save()
        return [
            t(
                lang,
                "Perfecto, lo dejo así. Aquí estoy si necesitas algo más.",
                "All right, I'll leave it. I'm here if you need anything else.",
            )
        ], []
    if intent == "chat_thanks":
        return [
            random.choice(
                [
                    t(
                        lang,
                        "¡De nada! Me encanta ayudarte con tus modelos.",
                        "You're welcome! I love helping with your models.",
                    ),
                    t(
                        lang,
                        "¡Un placer! ¿Seguimos mejorándolo?",
                        "My pleasure! Shall we keep improving it?",
                    ),
                ]
            )
        ], []
    if intent == "chat_bye":
        return [
            t(
                lang,
                "¡Hasta luego! Tus cambios quedan guardados en el proyecto.",
                "See you! Your changes stay saved in the project.",
            )
        ], []
    if intent == "chat_who":
        return [
            t(
                lang,
                "Soy Optiverse, el modelo de IA de Voxel Enterprise. Miro tus fotos —profundidad, recorte del objeto y tipo de escena— y, junto con Reverse3D Core, retoco tus modelos 3D. Todo pasa en tu equipo: nada sale de él.",
                "I'm Optiverse, Voxel Enterprise's AI model. I look at your photos —depth, the subject cut-out and the scene type— and, together with Reverse3D Core, I retouch your 3D models. Everything happens on your computer: nothing leaves it.",
            )
        ], []
    if intent == "chat_history":
        items = [h for h in s.state.get("history", []) if h["ok"]][-6:]
        if not items:
            return [
                t(
                    lang,
                    "Aún no he hecho cambios en este proyecto.",
                    "I haven't changed anything in this project yet.",
                )
            ], []
        names = {
            "es": {
                "fill_holes": "cerré huecos",
                "smooth_model": "suavicé",
                "remove_loose_parts": "quité restos",
                "simplify_model": "simplifiqué",
                "set_height": "ajusté el tamaño",
                "scale_model": "cambié la escala",
                "ground_model": "lo apoyé en la base",
                "undo": "deshice un cambio",
                "exclude_photos": "descarté fotos",
                "restore_photos": "recuperé fotos",
                "enhance_photos": "mejoré la luz",
                "prepare_photos": "preparé las fotos",
                "review_photos": "revisé las fotos",
                "preview_cutout": "te mostré un recorte",
                "model_info": "revisé el modelo",
                "capture_advice": "te di consejos",
                "help": "te expliqué qué hago",
            },
            "en": {
                "fill_holes": "closed holes",
                "smooth_model": "smoothed it",
                "remove_loose_parts": "removed leftovers",
                "simplify_model": "simplified it",
                "set_height": "set the size",
                "scale_model": "rescaled it",
                "ground_model": "set it on its base",
                "undo": "undid a change",
                "exclude_photos": "dropped photos",
                "restore_photos": "restored photos",
                "enhance_photos": "fixed the light",
                "prepare_photos": "prepared the photos",
                "review_photos": "reviewed the photos",
                "preview_cutout": "showed you a cut-out",
                "model_info": "checked the model",
                "capture_advice": "gave you tips",
                "help": "explained what I do",
            },
        }[lang]
        said = ", ".join(names.get(h["action"], h["action"]) for h in items)
        return [t(lang, f"Hasta ahora: {said}.", f"So far I {said}.")], []
    if intent == "chat_status":
        lines: list[str] = []
        photos = s.photo_files()
        if photos:
            left = len(photos) - len(s.state["excluded"])
            lines.append(
                t(
                    lang,
                    f"Tienes {left} de {len(photos)} fotos en uso.",
                    f"{left} of {len(photos)} photos are in use.",
                )
            )
        info = run_action(s, "model_info", {})
        if info.ok:
            data = info.data or {}
            lines.append(
                t(
                    lang,
                    f"El modelo tiene {num(lang, data['faces'])} caras",
                    f"The model has {num(lang, data['faces'])} faces",
                )
                + (
                    t(
                        lang,
                        ", está cerrado y listo para imprimir.",
                        ", is watertight and ready to print.",
                    )
                    if data.get("watertight")
                    else t(
                        lang,
                        " y todavía tiene huecos: puedo cerrarlos.",
                        " and still has holes: I can close them.",
                    )
                )
            )
        return lines or [
            t(
                lang,
                "Todavía no hay fotos ni modelo en este proyecto.",
                "There are no photos or model in this project yet.",
            )
        ], []
    if intent in ("chat_more", "chat_less"):
        steps = more_or_less(s, intent == "chat_more")
        if isinstance(steps, str):
            return [steps], []
        return [], steps
    return [], []


def humanize(
    s: Session, done: list[tuple[str, Result]], offer: tuple[str, dict[str, Any], str] | None
) -> list[str]:
    lines: list[str] = []
    for _, result in done:
        lines += result.lines
    lines = _clean(lines, s.lang)
    if offer:
        lines.append(offer[2])
    return lines
