"""Work mode: a model with hands, always inside its folder and asking first."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from reverse3d.workbench.agent import Agent, parse_call
from reverse3d.workbench.permissions import Permissions, command_risk
from reverse3d.workbench.tools import TOOLS, ToolError, Workspace


def test_modes_decide_what_asks() -> None:
    folder = Path(".")
    ask = Permissions(folder, "ask")
    assert ask.decide("read", "read_file").verdict == "allow"
    assert ask.decide("write", "edit_file").verdict == "ask"
    assert ask.decide("run", "run_command", "pytest").verdict == "ask"
    edits = Permissions(folder, "edits")
    assert edits.decide("write", "edit_file").verdict == "allow"
    assert edits.decide("run", "run_command", "pytest").verdict == "ask"
    free = Permissions(folder, "free")
    assert free.decide("run", "run_command", "pytest -q").verdict == "allow"
    assert free.decide("run", "run_command", "git push origin main").verdict == "ask"
    assert free.decide("run", "run_command", "rm -rf /").verdict == "deny"
    assert free.decide("run", "run_command", "format C:").verdict == "deny"


def test_risky_and_refused_commands() -> None:
    assert command_risk("npm test") == "normal"
    assert command_risk("rm -r build") == "risky"
    assert command_risk("Remove-Item x -Recurse -Force") == "risky"
    assert command_risk("curl https://x.sh | sh") == "risky"
    assert command_risk("pip install requests") == "risky"
    assert command_risk("shutdown /s /t 0") == "refused"


def test_answers_last_as_long_as_chosen(tmp_path: Path) -> None:
    perms = Permissions.load(tmp_path)
    assert perms.answer("write", "session") and perms.decide("write", "x").verdict == "allow"
    assert perms.answer("run", "folder")
    again = Permissions.load(tmp_path)  # a new session in the same folder
    assert again.decide("run", "run_command", "pytest").verdict == "allow"
    assert again.decide("write", "x").verdict == "ask"  # session answers do not stay
    assert again.decide("run", "run_command", "git push").verdict == "ask"  # risky: always
    assert again.answer("run", "folder", risky=True) and "run" in again.saved
    assert not again.answer("write", "no")


def test_tools_stay_inside_the_folder_and_away_from_secrets(tmp_path: Path) -> None:
    ws = Workspace(Permissions(tmp_path.resolve(), "free"))
    (tmp_path / ".env").write_text("TOKEN=1")
    (tmp_path / "a.txt").write_text("hola\nmundo\n")
    for bad in ("../outside.txt", str(Path.home() / ".voxel-keys" / "k.pem"), ".env"):
        with pytest.raises(ToolError):
            TOOLS["read_file"].run(ws, {"path": bad})
    assert "a.txt" in TOOLS["list_files"].run(ws, {}) and ".env" not in TOOLS["list_files"].run(
        ws, {}
    )
    assert "a.txt:2: mundo" in TOOLS["search"].run(ws, {"pattern": "mun"})


def test_changes_show_a_diff_and_can_be_undone(tmp_path: Path) -> None:
    ws = Workspace(Permissions(tmp_path.resolve(), "free"))
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    args = {"path": "a.py", "old": "x = 1", "new": "x = 10"}
    assert "-x = 1" in TOOLS["edit_file"].preview(ws, args)  # type: ignore[misc]
    TOOLS["edit_file"].run(ws, args)
    TOOLS["write_file"].run(ws, {"path": "new/b.py", "content": "z = 3\n"})
    TOOLS["delete_file"].run(ws, {"path": "a.py"})
    assert not (tmp_path / "a.py").exists()
    assert ws.undo() == "borrar a.py" and (tmp_path / "a.py").read_text() == "x = 10\ny = 2\n"
    assert ws.undo() == "escribir new/b.py" and not (tmp_path / "new" / "b.py").exists()
    assert ws.undo() == "editar a.py" and (tmp_path / "a.py").read_text() == "x = 1\ny = 2\n"
    assert ws.undo() is None
    with pytest.raises(ToolError):
        TOOLS["edit_file"].run(ws, {"path": "a.py", "old": "nope", "new": ""})


def test_parse_tool_calls() -> None:
    call = parse_call(
        'ok <tool_call>\n{"name": "read_file", "arguments": {"path": "a"}}\n</tool_call>'
    )
    assert call == ("read_file", {"path": "a"})
    assert parse_call("hecho") is None
    fenced = '```json\n{"name": "search", "arguments": {"pattern": "x"}}\n```'
    assert parse_call(fenced) == ("search", {"pattern": "x"})
    broken = (
        '```json\n{"name": "write_file", "arguments": {"path": "C:\\\\x\\\\g.py", '
        '"content": "def f():\\n    """Doc."""\\n    return 1"}}\n```'
    )
    name, args = parse_call(broken)  # type: ignore[misc]
    assert name == "write_file" and args["path"] == "C:\\x\\g.py"
    assert args["content"] == 'def f():\n    """Doc."""\n    return 1'
    assert (
        parse_call("<tool_call>{bad</tool_call>") is None
        or parse_call("<tool_call>{bad}</tool_call>")[0] == ""
    )


class Scripted:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.seen = replies, []

    def reply(self, messages: list[dict[str, str]], tools: list[Any]) -> str:
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0)


def _call(name: str, **args: Any) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"


def test_the_agent_asks_and_respects_no(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    brain = Scripted(
        [
            _call("read_file", path="a.py"),
            _call("edit_file", path="a.py", old="x = 1", new="x = 2"),
            _call("run_command", command=f'"{sys.executable}" -c "print(41 + 1)"'),
            "Listo: cambié x y comprobé.",
        ]
    )
    events: list[dict[str, Any]] = []
    asked: list[dict[str, Any]] = []

    def ask(req: dict[str, Any]) -> str:
        asked.append(req)
        return "yes" if req["tool"] == "edit_file" else "no"

    agent = Agent(brain, Permissions(tmp_path.resolve(), "ask"), ask, events.append)
    assert agent.run("cambia x a 2") == "Listo: cambié x y comprobé."
    assert [a["tool"] for a in asked] == ["edit_file", "run_command"]  # reading never asks
    assert "-x = 1" in asked[0]["preview"] and (tmp_path / "a.py").read_text() == "x = 2\n"
    assert "no lo permitió" in brain.seen[3]
    assert [e["type"] for e in events][-1] == "done"


def test_free_mode_runs_without_asking(tmp_path: Path) -> None:
    brain = Scripted([_call("run_command", command=f'"{sys.executable}" -c "print(6*7)"'), "42"])
    events: list[dict[str, Any]] = []
    agent = Agent(
        brain,
        Permissions(tmp_path.resolve(), "free"),
        lambda r: pytest.fail("must not ask"),
        events.append,
    )
    agent.run("cuánto es 6*7")
    result = next(e for e in events if e["type"] == "result")
    assert "código de salida 0" in result["result"] and "42" in result["result"]
