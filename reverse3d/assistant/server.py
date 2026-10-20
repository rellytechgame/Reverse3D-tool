"""Conversation loop for Studio (``reverse3d assistant serve``) and one-shot answers.

Studio writes one JSON request per line on stdin and reads one JSON answer per
line on stdout, so the language model is loaded only once per session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from reverse3d.assistant.actions import ACTIONS, Session, run_action, t
from reverse3d.assistant.brain import Brain
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


def answer(request: dict[str, Any], brain: Brain, settings: Settings) -> dict[str, Any]:
    work = request.get("work")
    if not work:
        return {"id": request.get("id"), "ok": False, "reply": "Missing work folder."}
    s = Session(
        Path(work), request.get("photos"), request.get("job"), settings, request.get("lang", "es")
    )
    if request.get("action"):  # a suggestion chip or a button: no language model needed
        steps = [(str(request["action"]), request.get("args") or {})]
        source, intro = "direct", ""
    else:
        plan = brain.understand(str(request.get("message", ""))[:2000], _context(s))
        steps, source, intro = plan.actions, plan.source, plan.ask
    results = []
    lines: list[str] = []
    out: dict[str, Any] = {"images": [], "prepared_photos": None, "model": None}
    for name, args in steps[:6]:
        r = run_action(s, name, args)
        results.append({"name": name, "ok": r.ok})
        lines += r.lines
        out["images"] += r.images or []
        out["prepared_photos"] = r.prepared_photos or out["prepared_photos"]
        out["model"] = r.model or out["model"]
    if not steps:
        lines = (
            [intro]
            if intro
            else [
                t(
                    s.lang,
                    "No entendí qué hacer. Prueba con una de estas:",
                    "I didn't get what to do. Try one of these:",
                )
            ]
        )
    kind = "model" if s.current_mesh() else ("photos" if s.photo_files() else "none")
    return {
        "id": request.get("id"),
        "ok": all(r["ok"] for r in results) if results else False,
        "reply": "\n".join(lines),
        "asks": bool(intro and not steps),
        "actions": results,
        "understood_by": source,
        "suggestions": SUGGESTIONS[s.lang][kind],
        **out,
    }


def serve(settings: Settings | None = None) -> int:
    settings = settings or Settings.load()
    brain = Brain(settings)
    brain.warm_up()  # the first messages are answered by the rules while it loads
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
            response = answer(request, brain, settings)
        except Exception as exc:  # one bad request must not end the conversation
            response = {"ok": False, "reply": f"Error: {exc}"}
        out.write(json.dumps(response, ensure_ascii=False) + "\n")
        out.flush()
    return 0
