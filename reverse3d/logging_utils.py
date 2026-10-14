"""Logging setup.

Logs always go to **stderr**: stdout is reserved for command results (``--json``)
and for the MCP stdio transport, which would be corrupted by stray log lines.
Logged messages never include file contents, credentials or full absolute paths
supplied by agents; modules log file *names* only.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str | None] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_lines: bool = False) -> None:
    """Configure the ``reverse3d`` logger hierarchy (idempotent)."""
    root = logging.getLogger("reverse3d")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        _JsonFormatter() if json_lines else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False
