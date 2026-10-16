"""Watched inbox: drop files in a folder, get results without typing anything.

``reverse3d watch`` keeps an eye on ``<workspace>/bandeja``. Whatever you drop
there (a folder of photos, a drawing, a photo, an STL...) is processed with the
autopilot (:func:`reverse3d.autopilot.auto_run`) once it has finished copying:

* results go to ``<workspace>/resultados/<name>/`` together with ``RESUMEN.md``;
* the original moves to ``bandeja/_procesado/``;
* if it cannot be processed it moves to ``bandeja/_revisar/`` with a
  ``<name>.LEEME.txt`` explaining what to change.

It needs no internet and no AI model, so it keeps working offline. Polling is
used instead of OS file events: it is portable and has no extra dependency.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reverse3d.config import Settings
from reverse3d.errors import (
    CaptureQualityError,
    DependencyMissingError,
    InputValidationError,
    Reverse3DError,
)

logger = logging.getLogger(__name__)

INBOX_NAME = "bandeja"
OUTBOX_NAME = "resultados"
DONE_DIR = "_procesado"
REVIEW_DIR = "_revisar"
#: Partial downloads and editor temp files are never picked up.
_TEMP_SUFFIXES = (".part", ".crdownload", ".tmp", ".download", ".partial", ".lock")
_MAX_SCAN_FILES = 2000

Event = dict[str, Any]


@dataclass
class WatchFolders:
    inbox: Path
    outbox: Path

    @classmethod
    def for_workspace(cls, settings: Settings) -> WatchFolders:
        return cls(settings.workspace / INBOX_NAME, settings.workspace / OUTBOX_NAME)

    def ensure(self) -> None:
        for folder in (self.inbox, self.outbox, self.inbox / DONE_DIR, self.inbox / REVIEW_DIR):
            folder.mkdir(parents=True, exist_ok=True)


@dataclass
class _Tracker:
    """Remembers each item's fingerprint to know when it has finished copying."""

    seen: dict[str, tuple[int, int, int]] = field(default_factory=dict)


def _is_candidate(path: Path) -> bool:
    name = path.name
    if name.startswith((".", "_", "~$")) or name.lower().endswith(_TEMP_SUFFIXES):
        return False
    return path.is_dir() or path.is_file()


def _fingerprint(path: Path) -> tuple[int, int, int]:
    if path.is_file():
        stat = path.stat()
        return (1, stat.st_size, stat.st_mtime_ns)
    count = size = newest = 0
    for i, file in enumerate(path.rglob("*")):
        if i >= _MAX_SCAN_FILES:
            break
        if file.is_file():
            stat = file.stat()
            count += 1
            size += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
    return (count, size, newest)


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 10_000):
        candidate = (
            path.with_name(f"{path.stem}-{n}{path.suffix}")
            if path.suffix
            else path.with_name(f"{path.name}-{n}")
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many items named {path.name}")  # pragma: no cover


def advice_es(error: Reverse3DError) -> str:
    """Plain Spanish advice for the user, by kind of problem."""
    if isinstance(error, CaptureQualityError):
        return (
            "Las fotos no sirven todavía para reconstruir el objeto. Repítelas con buena luz, "
            "sin desenfoque, rodeando el objeto y con mucho solape entre una foto y la siguiente "
            "(20-60 fotos)."
        )
    if isinstance(error, DependencyMissingError):
        return f"Falta un componente en este equipo. {error.hint or ''}".strip()
    if isinstance(error, InputValidationError):
        return (
            "Deja una carpeta con 20-60 fotos del objeto, una imagen (plano, silueta o foto) "
            "o un archivo 3D: STL, OBJ, PLY, GLB, XYZ o PCD."
        )
    return "Revisa el detalle técnico de abajo; si el problema sigue, abre un issue en GitHub."


def _review_note(item: Path, error: Reverse3DError) -> str:
    lines = [
        f"reverse3d no pudo procesar «{item.name}».",
        "",
        f"Qué hacer: {advice_es(error)}",
        "",
        "Cuando lo corrijas, vuelve a dejarlo en la carpeta «bandeja»: se procesará solo.",
        "",
        "Detalle técnico:",
        f"  {error.message}",
    ]
    if error.hint:
        lines.append(f"  {error.hint}")
    return "\n".join(lines) + "\n"


def process_item(item: Path, folders: WatchFolders, settings: Settings) -> Event:
    """Process one inbox item and move it to _procesado or _revisar."""
    from reverse3d.autopilot import auto_run, detect_input

    detection = detect_input(item)
    stem = item.stem if item.is_file() else item.name
    out = _unique(folders.outbox / stem)
    try:
        summary = auto_run(item, out, settings)
    except Reverse3DError as exc:
        target = _unique(folders.inbox / REVIEW_DIR / item.name)
        shutil.move(str(item), str(target))
        note = target.with_name(f"{target.name}.LEEME.txt")
        note.write_text(_review_note(item, exc), encoding="utf-8")
        if out.exists() and not any(out.iterdir()):
            out.rmdir()
        return {
            "event": "failed",
            "item": item.name,
            "kind": detection.kind,
            "error": exc.to_dict(),
            "advice": advice_es(exc),
            "moved_to": str(target),
            "note": str(note),
        }
    target = _unique(folders.inbox / DONE_DIR / item.name)
    shutil.move(str(item), str(target))
    return {
        "event": "done",
        "item": item.name,
        "kind": detection.kind,
        "recipe": summary.get("recipe"),
        "output_dir": str(out),
        "summary": summary["outputs"].get("summary"),
        "dimensions": summary.get("dimensions", {}),
        "warnings": summary.get("warnings", []),
        "moved_to": str(target),
    }


def scan(
    folders: WatchFolders,
    settings: Settings,
    tracker: _Tracker,
    *,
    settle: bool = True,
    on_event: Callable[[Event], None] | None = None,
) -> list[Event]:
    """Look at the inbox once and process every item that has finished copying.

    Args:
        settle: when True an item is processed only after two scans with the
            same size and modification time (it is still being copied otherwise).
    """
    emit = on_event or (lambda _e: None)
    events: list[Event] = []
    for item in sorted(p for p in folders.inbox.iterdir() if _is_candidate(p)):
        key = str(item)
        try:
            fingerprint = _fingerprint(item)
        except OSError:
            continue
        if settle and tracker.seen.get(key) != fingerprint:
            if key not in tracker.seen:
                emit({"event": "detected", "item": item.name})
            tracker.seen[key] = fingerprint
            continue
        tracker.seen.pop(key, None)
        emit({"event": "processing", "item": item.name})
        try:
            event = process_item(item, folders, settings)
        except OSError as exc:  # e.g. the file is still locked by another program
            event = {"event": "retry", "item": item.name, "error": str(exc)}
        events.append(event)
        emit(event)
    return events


def watch(
    folders: WatchFolders,
    settings: Settings,
    *,
    interval_s: float = 2.0,
    once: bool = False,
    on_event: Callable[[Event], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> list[Event]:
    """Watch the inbox until ``stop()`` returns True (or Ctrl+C).

    With ``once=True`` it processes what is already in the inbox and returns.
    """
    folders.ensure()
    tracker = _Tracker()
    if once:
        return scan(folders, settings, tracker, settle=False, on_event=on_event)
    events: list[Event] = []
    while not (stop and stop()):
        events.extend(scan(folders, settings, tracker, on_event=on_event))
        time.sleep(interval_s)
    return events
