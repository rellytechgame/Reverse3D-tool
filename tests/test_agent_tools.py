from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverse3d.agent.tools import TOOLS, execute_tool, tool_definitions
from reverse3d.config import Settings

DOCS_TOOLS = Path(__file__).resolve().parents[1] / "docs" / "tools"


@pytest.mark.parametrize("dialect", ["anthropic", "openai", "gemini"])
def test_definitions_are_complete(dialect: str) -> None:
    tools = tool_definitions(dialect)  # type: ignore[arg-type]
    assert len(tools) == len(TOOLS)
    for tool in tools:
        fn = tool.get("function", tool)
        assert fn["name"].startswith("reverse3d_")
        assert len(fn["description"]) > 40
        schema = fn.get("input_schema") or fn.get("parameters")
        assert schema["type"] == "object"
        json.dumps(schema)
        if dialect == "gemini":
            assert "additionalProperties" not in json.dumps(schema)
            assert "anyOf" not in json.dumps(schema)
        else:
            assert schema["additionalProperties"] is False


@pytest.mark.parametrize("dialect", ["anthropic", "openai", "gemini"])
def test_published_schemas_are_in_sync(dialect: str) -> None:
    """docs/tools/*.json must be regenerated when tools change:
    reverse3d agent tools --format <dialect> -o docs/tools/<dialect>.json"""
    published = json.loads((DOCS_TOOLS / f"{dialect}.json").read_text(encoding="utf-8"))
    assert published == tool_definitions(dialect)  # type: ignore[arg-type]


def test_unknown_tool(settings: Settings) -> None:
    result = execute_tool("rm_rf", {}, settings)
    assert result["ok"] is False and result["error"]["code"] == "unknown_tool"


def test_extra_arguments_rejected(box_mesh_path: Path, settings: Settings) -> None:
    result = execute_tool(
        "reverse3d_measure_mesh", {"input_path": box_mesh_path.name, "shell": "rm -rf /"}, settings
    )
    assert result["error"]["code"] == "invalid_input"
    assert "shell" in result["error"]["message"]


def test_bounds_enforced(settings: Settings) -> None:
    result = execute_tool(
        "reverse3d_render",
        {"mesh_path": "a.stl", "output_path": "a.png", "width": 100000},
        settings,
    )
    assert result["error"]["code"] == "invalid_input"


def test_non_object_arguments(settings: Settings) -> None:
    result = execute_tool("reverse3d_measure_mesh", ["x"], settings)  # type: ignore[arg-type]
    assert result["error"]["code"] == "invalid_input"


def test_path_traversal_blocked(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = Settings(workspace=ws)
    for evil in ("../box.stl", str(tmp_path / "box.stl")):
        result = execute_tool("reverse3d_measure_mesh", {"input_path": evil}, settings)
        assert result["error"]["code"] == "security_violation"


def test_output_path_traversal_blocked(box_mesh_path: Path, tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    result = execute_tool(
        "reverse3d_clean_mesh",
        {"input_path": box_mesh_path.name, "output_path": "../../escape.stl"},
        settings,
    )
    assert result["error"]["code"] == "security_violation"
    assert not (tmp_path.parent.parent / "escape.stl").exists()


def test_happy_path_measure_and_clean(box_mesh_path: Path, settings: Settings) -> None:
    measured = execute_tool("reverse3d_measure_mesh", {"input_path": "box.stl"}, settings)
    assert measured["ok"] is True
    assert measured["result"]["volume"] == pytest.approx(8000)

    cleaned = execute_tool(
        "reverse3d_clean_mesh",
        {"input_path": "box.stl", "output_path": "out/box_clean.glb", "align": "obb"},
        settings,
    )
    assert cleaned["ok"] is True
    again = execute_tool(
        "reverse3d_clean_mesh",
        {"input_path": "box.stl", "output_path": "out/box_clean.glb"},
        settings,
    )
    assert again["ok"] is False and again["error"]["code"] == "invalid_input"


def test_fit_with_parametric(box_mesh_path: Path, settings: Settings) -> None:
    result = execute_tool(
        "reverse3d_fit_primitives",
        {"input_path": "box.stl", "kinds": ["plane"], "parametric_output": "cad/box.scad"},
        settings,
    )
    assert result["ok"] is True
    assert len(result["result"]["primitives"]) == 6
    # compact mode returns workspace-relative paths
    assert result["result"]["parametric_path"] == "cad/box.scad"
    assert (settings.workspace / "cad" / "box.scad").is_file()


def test_missing_dependency_is_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    (tmp_path / "photos").mkdir()
    result = execute_tool(
        "reverse3d_photogrammetry",
        {"image_dir": "photos", "output_dir": "out"},
        Settings(workspace=tmp_path),
    )
    assert result["error"]["code"] == "dependency_missing"
    assert "hint" in result["error"]


def test_environment_check(settings: Settings) -> None:
    result = execute_tool("reverse3d_environment_check", {}, settings)
    assert result["ok"] and result["result"]["capabilities"]["mesh_repair"] is True


def test_unexpected_errors_are_not_leaked(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reverse3d.agent import tools as tools_mod

    def boom(*_: object) -> dict[str, object]:
        raise RuntimeError("internal detail /secret/path")

    spec = tools_mod.TOOLS["reverse3d_environment_check"]
    monkeypatch.setitem(
        tools_mod.TOOLS,
        spec.name,
        tools_mod.ToolSpec(spec.name, spec.description, spec.model, boom),
    )
    result = execute_tool(spec.name, {}, settings)
    assert result["error"]["code"] == "internal_error"
    assert "secret" not in json.dumps(result)
