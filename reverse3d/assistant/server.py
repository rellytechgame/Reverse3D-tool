"""Conversation loop for Studio (``reverse3d assistant serve``) and one-shot answers.

Studio writes one JSON request per line on stdin and reads one JSON answer per
line on stdout, so the language model is loaded only once per session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reverse3d.assistant.actions import ACTIONS, Result, Session, run_action, t
from reverse3d.assistant.brain import Brain
from reverse3d.assistant.ears import Ears
from reverse3d.assistant.talk import chat, humanize, offer_after, remember
from reverse3d.config import Settings

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


def answer(
    request: dict[str, Any], brain: Brain, settings: Settings, ears: Ears | None = None
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
            lines, more = chat(s, name)
            said += lines
            queue = more + queue
            continue
        done.append((name, run_action(s, name, args)))
        executed.append((name, args))
    remember(s, done, executed)
    offer = offer_after(s, done)
    s.state["offer"] = [offer[0], offer[1]] if offer else None
    s.save()
    lines = said + humanize(s, done, offer)
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
    brain.warm_up()  # the first messages are answered by the rules while it loads
    ears = Ears(settings)
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
            response = answer(request, brain, settings, ears)
        except Exception as exc:  # one bad request must not end the conversation
            response = {"ok": False, "reply": f"Error: {exc}"}
        out.write(json.dumps(response, ensure_ascii=False) + "\n")
        out.flush()
    return 0
