"""Conversation loop for Studio (``reverse3d assistant serve``) and one-shot answers.

Studio writes one JSON request per line on stdin and reads one JSON answer per
line on stdout, so the language model is loaded only once per session.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from reverse3d.assistant.actions import ACTIONS, Result, Session, run_action, t
from reverse3d.assistant.brain import Brain
from reverse3d.assistant.ears import Ears
from reverse3d.assistant.persona import Memory
from reverse3d.assistant.talk import chat, humanize, offer_after, remember
from reverse3d.assistant.voice import Voice
from reverse3d.config import Settings

logger = logging.getLogger(__name__)

#: chips Studio shows under each answer
SUGGESTIONS = {
    "es": {
        "photos": [
            "Revisa las fotos",
            "Descarta las fotos movidas",
            "Mejora la luz",
            "Prepara las fotos",
        ],
        "model": ["Rellena los huecos", "Suavízalo", "Quita los restos", "Deshacer"],
        "none": ["¿Qué puedes hacer?", "¿Qué fotos me faltan?"],
    },
    "en": {
        "photos": [
            "Review the photos",
            "Drop the blurred photos",
            "Fix the light",
            "Prepare the photos",
        ],
        "model": ["Fill the holes", "Smooth it", "Remove the leftovers", "Undo"],
        "none": ["What can you do?", "Which photos am I missing?"],
    },
}


def _context(s: Session) -> str:
    files = s.photo_files()
    names = ", ".join(p.stem for p in files[:12]) + ("…" if len(files) > 12 else "")
    mesh = s.current_mesh()
    return (
        f"Language: {s.lang}. Photos: {len(files)} ({names}). "
        f"Dropped: {len(s.state['excluded'])}. 3D model: {'yes' if mesh else 'not yet'}."
    )


def _speak(
    s: Session,
    voice: Voice | None,
    message: str,
    facts: list[str],
    offer: tuple[str, dict[str, Any], str] | None,
    written: list[str],
) -> list[str]:
    """Optiverse words the reply itself when its voice is ready; else the written reply."""
    if voice is None or not message.strip():
        return written
    if not facts:  # it must never claim a change it did not make
        facts = [t(s.lang, "no hice ningún cambio", "I did not change anything")]
    history = [(str(r), str(x)) for r, x in s.state.get("talk", [])]
    try:
        text = voice.say(s.lang, facts, history, message)
    except Exception as exc:  # the written reply is always there
        logger.warning("Optiverse voice failed: %s", exc)
        text = None
    if not text:
        return written
    lines = [text] + ([offer[2]] if offer and offer[2] not in text else [])
    s.state["talk"] = [*s.state.get("talk", []), ["user", message], ["assistant", " ".join(lines)]][
        -12:
    ]
    s.save()
    return lines


def answer(
    request: dict[str, Any],
    brain: Brain,
    settings: Settings,
    ears: Ears | None = None,
    voice: Voice | None = None,
) -> dict[str, Any]:
    work = request.get("work")
    if not work:
        return {"id": request.get("id"), "ok": False, "reply": "Missing work folder."}
    s = Session(
        Path(work), request.get("photos"), request.get("job"), settings, request.get("lang", "es")
    )
    message = str(request.get("message", ""))[:2000]
    heard: str | None = None
    if request.get("audio"):  # a spoken request: Whisper turns it into text, locally
        try:
            heard = (ears or Ears(settings)).transcribe(Path(str(request["audio"])), s.lang)
        except Exception as exc:  # report it in the chat, never crash
            return _reply(
                request,
                s,
                t(s.lang, f"No pude escuchar el audio: {exc}", f"I couldn't hear the audio: {exc}"),
                heard="",
            )
        if not heard:
            return _reply(
                request,
                s,
                t(
                    s.lang,
                    "No te escuché bien, ¿lo repites?",
                    "I didn't catch that, could you say it again?",
                ),
                heard="",
            )
        message = heard
    if request.get("action"):  # a suggestion chip or a button: no language model needed
        steps = [(str(request["action"]), request.get("args") or {})]
        source, ask = "direct", ""
    else:
        plan = brain.understand(message, _context(s))
        steps, source, ask = list(plan.actions), plan.source, plan.ask
    said: list[str] = []
    done: list[tuple[str, Result]] = []
    executed: list[tuple[str, dict[str, Any]]] = []
    queue = list(steps)
    while queue and len(executed) < 8:
        name, args = queue.pop(0)
        if name.startswith("chat_"):
            lines, more = chat(s, name, brain.memory)
            said += lines
            queue = more + queue
            continue
        done.append((name, run_action(s, name, args)))
        executed.append((name, args))
    remember(s, done, executed)
    if brain.memory is not None:
        learnt = brain.memory.observe(
            message, [n for n, _ in steps], source, direct=bool(request.get("action"))
        )
        if learnt:
            brain.learn(*learnt)
            if request.get("action"):  # it had not understood: say what it learnt
                said.append(
                    t(
                        s.lang,
                        f"Anotado: cuando digas «{learnt[0]}», sabré que quieres esto.",
                        f"Noted: when you say “{learnt[0]}”, I'll know you mean this.",
                    )
                )
    offer = offer_after(s, done)
    s.state["offer"] = [offer[0], offer[1]] if offer else None
    s.save()
    lines = said + humanize(s, done, offer)
    lines = _speak(s, voice, message, said + humanize(s, done, None), offer, lines)
    if not lines:
        lines = (
            [ask]
            if ask
            else [
                t(
                    s.lang,
                    "No entendí qué hacer. Prueba con una de estas:",
                    "I didn't get what to do. Try one of these:",
                )
            ]
        )
    out: dict[str, Any] = {"images": [], "prepared_photos": None, "model": None}
    for _, r in done:
        out["images"] += r.images or []
        out["prepared_photos"] = r.prepared_photos or out["prepared_photos"]
        out["model"] = r.model or out["model"]
    return _reply(
        request,
        s,
        "\n".join(lines),
        ok=all(r.ok for _, r in done) if done else bool(said),
        asks=bool(ask and not done and not said) or offer is not None,
        actions=[{"name": n, "ok": r.ok} for n, r in done],
        source=source,
        heard=heard,
        **out,
    )


def _reply(request: dict[str, Any], s: Session, text: str, **extra: Any) -> dict[str, Any]:
    kind = "model" if s.current_mesh() else ("photos" if s.photo_files() else "none")
    offer = s.state.get("offer")
    chips = list(SUGGESTIONS[s.lang][kind])
    if offer:  # answering the question is one click away
        chips = [t(s.lang, "Sí", "Yes"), t(s.lang, "No", "No"), *chips[:3]]
    return {
        "id": request.get("id"),
        "ok": extra.pop("ok", False),
        "reply": text,
        "asks": extra.pop("asks", False),
        "actions": extra.pop("actions", []),
        "understood_by": extra.pop("source", "none"),
        "heard": extra.pop("heard", None),
        "suggestions": chips,
        "images": extra.pop("images", []),
        "prepared_photos": extra.pop("prepared_photos", None),
        "model": extra.pop("model", None),
    }


def serve(settings: Settings | None = None) -> int:
    settings = settings or Settings.load()
    brain = Brain(settings)
    brain.memory = Memory()  # it grows with this person: their words, their time together
    brain.warm_up()  # the first messages are answered by the rules while it loads
    ears = Ears(settings)
    voice = Voice()
    voice.warm_up()  # Optiverse's own voice: written replies answer until it is ready
    # JSON lines are UTF-8 whatever the console code page («sí», «¿qué…?»).
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    out = sys.stdout
    out.write(
        json.dumps({"ready": True, "model": brain.available, "actions": list(ACTIONS)}) + "\n"
    )
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = answer(request, brain, settings, ears, voice)
        except Exception as exc:  # one bad request must not end the conversation
            response = {"ok": False, "reply": f"Error: {exc}"}
        out.write(json.dumps(response, ensure_ascii=False) + "\n")
        out.flush()
    return 0
