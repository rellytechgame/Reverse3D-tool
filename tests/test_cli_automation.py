from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reverse3d import setup_agents
from reverse3d.cli import app

runner = CliRunner()


def _json(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])


def test_config_set_show_unset() -> None:
    assert runner.invoke(app, ["--json", "config", "set", "device", "cpu"]).exit_code == 0
    shown = _json(runner.invoke(app, ["--json", "config", "show"]).stdout)["result"]
    assert shown["effective"]["device"] == "cpu"
    assert shown["stored"] == {"device": "cpu"}
    bad = runner.invoke(app, ["--json", "config", "set", "device", "warp-drive"])
    assert bad.exit_code == 2
    assert runner.invoke(app, ["--json", "config", "unset", "device"]).exit_code == 0


def test_global_device_override(box_mesh_path: Path) -> None:
    runner.invoke(app, ["config", "set", "device", "gpu"])
    shown = _json(runner.invoke(app, ["--json", "--device", "cpu", "config", "show"]).stdout)
    assert shown["result"]["effective"]["device"] == "cpu"


def test_run_recipe_cli(plan_image: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--profile",
            "fast",
            "run",
            "plan_to_part",
            str(plan_image),
            str(tmp_path / "res"),
            "--thickness",
            "2",
            "--units-per-pixel",
            "0.5",
            "--no-render",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = _json(result.stdout)["result"]
    assert summary["profile"] == "fast"
    assert summary["dimensions"]["size_obb"][2] == pytest.approx(2)


def test_setup_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    (home / ".cursor").mkdir(parents=True)
    monkeypatch.setattr(setup_agents, "_home", lambda: home)
    monkeypatch.setattr(setup_agents, "_appdata", lambda: home / "AppData")
    monkeypatch.setattr(setup_agents, "_xdg_config", lambda: home / ".config")
    monkeypatch.setattr(setup_agents, "_claude_cli", lambda: None)

    ws = tmp_path / "ws"
    dry = _json(runner.invoke(app, ["--json", "setup", "--workspace", str(ws), "--dry-run"]).stdout)
    assert "Dry run" in dry["result"]["next_step"]
    assert not (home / ".cursor" / "mcp.json").exists()

    done = runner.invoke(
        app, ["--json", "setup", "--workspace", str(ws), "--device", "gpu", "--toolset", "lean"]
    )
    assert done.exit_code == 0, done.output
    data = _json(done.stdout)["result"]
    assert {a["agent"]: a["status"] for a in data["agents"]}["cursor"] == "configured"
    assert data["config"]["device"] == "gpu" and data["config"]["toolset"] == "lean"
    assert ws.is_dir()

    snippets = _json(runner.invoke(app, ["--json", "setup", "--snippets"]).stdout)["result"]
    assert "mcpServers" in snippets and "codex_toml" in snippets


def test_agent_tools_lean() -> None:
    result = runner.invoke(app, ["agent", "tools", "--toolset", "lean"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 5
