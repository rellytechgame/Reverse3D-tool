"""Two ways to work with a model in a folder: the terminal, and JSON lines (Studio, cloud).

JSON lines (`reverse3d work run --json`), one object per line:
  out: {"type": "text"|"call"|"result"|"denied"|"done"|"error", ...}
       {"type": "permission", "id": 3, "tool": ..., "args": ..., "preview": ..., "risky": ...}
  in:  {"id": 3, "answer": "yes" | "session" | "folder" | "no"}
Any interface that speaks this (Studio's window, a web page over a socket in the
cloud) shows the same questions and lets the person answer them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

ANSWERS = {"s": "yes", "y": "yes", "t": "session", "c": "folder", "n": "no"}
KIND_ES = {"write": "cambiar archivos", "run": "ejecutar comandos"}


def terminal(console: Console) -> tuple[Any, Any]:
    """(ask, on_event) for a person at a terminal."""

    def on_event(e: dict[str, Any]) -> None:
        kind = e["type"]
        if kind == "text":
            console.print(e["text"])
        elif kind == "call":
            args = ", ".join(
                f"{k}={str(v)[:60]!r}"
                for k, v in e["args"].items()
                if k not in ("content", "old", "new")
            )
            console.print(f"[dim]→ {e['tool']}({args})[/dim]")
        elif kind == "result":
            text = str(e["result"])
            console.print(f"[dim]{text[:600]}{'…' if len(text) > 600 else ''}[/dim]")
        elif kind == "denied":
            console.print(f"[yellow]✕ {e['tool']}: {e['reason']}[/yellow]")
        elif kind == "error":
            console.print(f"[red]{e['text']}[/red]")

    def ask(req: dict[str, Any]) -> str:
        title = f"¿Permites {req['tool']}?"
        if req.get("risky"):
            title += "  [bold red](delicado: se pregunta siempre)[/bold red]"
        preview = req.get("preview") or json.dumps(req["args"], ensure_ascii=False)[:2000]
        lexer = "diff" if preview.startswith("---") else "bash"
        console.print(
            Panel(Syntax(preview, lexer, word_wrap=True), title=title, border_style="yellow")
        )
        if req.get("risky"):
            options = "[s] sí, esta vez   [n] no"
        else:
            what = KIND_ES.get(req["kind"], req["kind"])
            options = (
                f"[s] sí   [t] {what} sin preguntar en esta sesión   "
                f"[c] {what} siempre en esta carpeta   [n] no"
            )
        while True:
            choice = console.input(escape(options) + "\n> ").strip().lower()[:1]
            if choice in ANSWERS and not (req.get("risky") and choice in "tc"):
                return ANSWERS[choice]

    return ask, on_event


def json_lines() -> tuple[Any, Any]:
    """(ask, on_event) speaking JSON lines on stdin/stdout."""
    counter = {"id": 0}

    def send(obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()

    def ask(req: dict[str, Any]) -> str:
        counter["id"] += 1
        send({**req, "id": counter["id"]})
        for line in sys.stdin:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == counter["id"]:
                answer = msg.get("answer", "no")
                return answer if answer in ("yes", "session", "folder", "no") else "no"
        return "no"  # the interface went away: nothing changes

    return ask, send


def find_model(explicit: Path | None) -> Path | None:
    """The GGUF to work with: --model, else the newest in ~/.reverse3d/models/llm/."""
    if explicit:
        return explicit
    folder = Path.home() / ".reverse3d" / "models" / "llm"
    found = (
        sorted(folder.glob("*.gguf"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []
    )
    first = [p for p in found if "-00001-of-" in p.name or "-of-" not in p.name]
    return first[-1] if first else None
