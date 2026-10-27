"""What a model may do on this machine, and when it must ask first.

The person chooses a mode, and can widen it while working:

    ask    asks before every change and every command        (default)
    edits  changes files in the folder freely; asks before commands
    free   changes files and runs commands freely

and to each question they can answer: yes (this once), always this session,
always in this folder (saved in <folder>/.reverse3d/permissions.json), or no.

Some things are never a free choice, whatever the mode:
  - reading is free, but only inside the working folder (and folders the
    person adds); everything else is outside the model's reach
  - secrets are never read or written: signing keys (~/.voxel-keys), SSH and
    cloud credentials, .env files, private keys
  - risky commands always ask, even in "free": deleting recursively, pushing
    to git, publishing packages, changing system settings, downloading and
    running scripts
  - destroying a disk or the system (format, rm -rf /, shutdown...) is refused
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Mode = Literal["ask", "edits", "free"]
Kind = Literal["read", "write", "run"]
Answer = Literal["yes", "session", "folder", "no"]
MODES: tuple[Mode, ...] = ("ask", "edits", "free")

SECRET_NAMES = re.compile(
    r"(^|[\\/])(\.voxel-keys|\.ssh|\.aws|\.azure|\.gnupg|\.kube|\.docker)([\\/]|$)"
    r"|(^|[\\/])\.env(\.[^\\/]*)?$"
    r"|\.(pem|key|p12|pfx|keystore|jks|r3dkey)$"
    r"|(^|[\\/])(id_rsa|id_ed25519|credentials|\.netrc|\.pypirc|\.npmrc)$",
    re.IGNORECASE,
)

REFUSED = [
    r"\bformat(\.com)?\s+[a-z]:",
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*f?[a-z]*\s+(/|~|\$HOME|[a-z]:\\?)\s*$",
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*\s+/\*?(\s|$)",
    r"\b(shutdown|reboot|halt|poweroff)\b",
    r"\b(mkfs|diskpart|fdisk|dd\s+if=.*\s+of=/dev/)",
    r"\bdel\s+(/[a-z]\s+)*[a-z]:\\\*",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
]
RISKY = [
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*r",
    r"\b(rmdir|rd)\s+/s\b",
    r"\bdel\s+(/[a-z]\s+)*/[sq]\b",
    r"\bRemove-Item\b.*-Recurse",
    r"\bgit\s+(push|reset\s+--hard|clean\s+-[a-z]*f|filter-branch|rebase)",
    r"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b",
    r"\b(curl|wget|iwr|Invoke-WebRequest)\b.*\|\s*(sh|bash|iex|python)",
    r"\bInvoke-Expression\b|\biex\b",
    r"\b(reg|regedit)\s+(add|delete)\b|\bSet-ItemProperty\b.*HKLM",
    r"\b(sudo|runas)\b",
    r"\b(chmod|chown)\s+-R\b",
    r"\bdocker\s+(system\s+prune|rm|rmi)\b",
    r"\b(pip|npm|winget|choco|apt|brew)\s+(install|uninstall|remove)\b",
]


def is_secret(path: Path) -> bool:
    return bool(SECRET_NAMES.search(str(path)))


def command_risk(command: str) -> Literal["refused", "risky", "normal"]:
    text = command.strip()
    if any(re.search(p, text, re.IGNORECASE) for p in REFUSED):
        return "refused"
    if any(re.search(p, text, re.IGNORECASE) for p in RISKY):
        return "risky"
    return "normal"


@dataclass
class Decision:
    verdict: Literal["allow", "ask", "deny"]
    reason: str = ""


@dataclass
class Permissions:
    """The person's choices for one working folder."""

    folder: Path
    mode: Mode = "ask"
    session: set[str] = field(default_factory=set)  # "write", "run", "write:<tool>"...
    saved: set[str] = field(default_factory=set)
    extra_folders: list[Path] = field(default_factory=list)

    @property
    def file(self) -> Path:
        return self.folder / ".reverse3d" / "permissions.json"

    @classmethod
    def load(cls, folder: Path, mode: Mode = "ask") -> Permissions:
        perms = cls(folder.resolve(), mode)
        try:
            data = json.loads(perms.file.read_text(encoding="utf-8"))
            perms.saved = set(data.get("always", []))
            perms.extra_folders = [Path(p) for p in data.get("folders", [])]
        except (OSError, ValueError):
            pass
        return perms

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(
            json.dumps(
                {"always": sorted(self.saved), "folders": [str(p) for p in self.extra_folders]},
                indent=1,
            ),
            encoding="utf-8",
        )

    def roots(self) -> list[Path]:
        return [self.folder, *(p.resolve() for p in self.extra_folders)]

    def decide(self, kind: Kind, tool: str, command: str = "") -> Decision:
        """allow, ask or deny one action (paths are checked by the tools themselves)."""
        if kind == "read":
            return Decision("allow")
        if kind == "run":
            risk = command_risk(command)
            if risk == "refused":
                return Decision("deny", "este comando puede destruir el disco o el sistema")
            if risk == "risky":  # always asks, whatever the mode or earlier answers
                return Decision("ask", "comando delicado: siempre se pregunta")
        granted = self.session | self.saved
        if kind in granted or f"{kind}:{tool}" in granted:
            return Decision("allow")
        if self.mode == "free" or (self.mode == "edits" and kind == "write"):
            return Decision("allow")
        return Decision("ask")

    def answer(self, kind: Kind, answer: Answer, risky: bool = False) -> bool:
        """Apply the person's answer; True if the action may go ahead."""
        if answer == "no":
            return False
        if risky:  # a risky command is allowed once, never "always"
            return True
        if answer == "session":
            self.session.add(kind)
        elif answer == "folder":
            self.saved.add(kind)
            self.save()
        return True
