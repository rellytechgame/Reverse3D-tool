from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reverse3d.cli import app

runner = CliRunner()


def _json(output: str) -> dict:
    return json.loads(output.strip().splitlines()[-1])


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "reverse3d" in result.stdout


def test_doctor_json() -> None:
    result = runner.invoke(app, ["--json", "doctor"])
    assert result.exit_code == 0
    assert _json(result.stdout)["result"]["capabilities"]["vision"] is True


def test_measure_and_clean(box_mesh_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["--json", "mesh", "measure", str(box_mesh_path)])
    assert result.exit_code == 0
    assert _json(result.stdout)["result"]["obb_extents"] == pytest.approx([40, 20, 10], abs=1e-4)

    out = tmp_path / "clean.obj"
    args = ["--json", "mesh", "clean", str(box_mesh_path), str(out), "--align", "obb"]
    assert runner.invoke(app, args).exit_code == 0
    again = runner.invoke(app, args)
    assert again.exit_code == 2
    assert _json(again.stdout)["error"]["code"] == "invalid_input"
    assert runner.invoke(app, [*args, "--force"]).exit_code == 0


def test_invalid_choice_rejected(box_mesh_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["mesh", "clean", str(box_mesh_path), str(tmp_path / "o.stl"), "--align", "evil"]
    )
    assert result.exit_code != 0


def test_vision_pipeline(plan_image: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "mask"
    result = runner.invoke(
        app, ["--json", "vision", "mask", str(plan_image), str(out_dir), "--units-per-pixel", "0.5"]
    )
    assert result.exit_code == 0, result.output
    contours = _json(result.stdout)["result"]["contours_path"]
    result = runner.invoke(
        app,
        ["--json", "vision", "extrude", contours, str(tmp_path / "plate.stl"), "--thickness", "4"],
    )
    assert result.exit_code == 0
    assert _json(result.stdout)["result"]["watertight"] is True


def test_fit_bad_kind(box_mesh_path: Path) -> None:
    result = runner.invoke(app, ["mesh", "fit", str(box_mesh_path), "--kinds", "cone"])
    assert result.exit_code != 0


def test_agent_tools_formats(tmp_path: Path) -> None:
    for fmt in ("anthropic", "openai", "gemini"):
        result = runner.invoke(app, ["agent", "tools", "--format", fmt])
        assert result.exit_code == 0
        assert len(json.loads(result.stdout)) >= 10


def test_agent_call_is_sandboxed(box_mesh_path: Path, tmp_path: Path) -> None:
    ws = ["--json", "--workspace", str(tmp_path)]
    ok = runner.invoke(
        app, [*ws, "agent", "call", "reverse3d_measure_mesh", "--args", '{"input_path": "box.stl"}']
    )
    assert ok.exit_code == 0 and _json(ok.stdout)["ok"] is True

    escape = runner.invoke(
        app,
        [*ws, "agent", "call", "reverse3d_measure_mesh", "--args", '{"input_path": "../box.stl"}'],
    )
    assert escape.exit_code == 1
    assert _json(escape.stdout)["error"]["code"] == "security_violation"

    bad_json = runner.invoke(
        app, [*ws, "agent", "call", "reverse3d_measure_mesh", "--args", "{oops"]
    )
    assert bad_json.exit_code != 0
