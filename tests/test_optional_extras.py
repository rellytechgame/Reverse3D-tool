"""Integration tests for optional extras; skipped when the extra is not installed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from reverse3d.config import Settings


def test_pyvista_render(box_mesh_path: Path, tmp_path: Path, settings: Settings) -> None:
    pytest.importorskip("pyvista")
    from reverse3d.rendering.preview import render_with_pyvista

    result = render_with_pyvista(box_mesh_path, tmp_path / "r.png", settings)
    assert result.backend == "pyvista"
    assert Path(result.files[0]).stat().st_size > 1000


def test_mcp_server_end_to_end(box_mesh_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def scenario() -> tuple[int, dict, dict, list[str], str]:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "reverse3d", "agent", "mcp"],
            env={"REVERSE3D_WORKSPACE": str(tmp_path), "SYSTEMROOT": "C:\\Windows", "PATH": ""},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            ok = await session.call_tool("reverse3d_measure_mesh", {"input_path": "box.stl"})
            bad = await session.call_tool("reverse3d_measure_mesh", {"input_path": "../box.stl"})
            prompts = await session.list_prompts()
            gpu = await session.get_prompt("usar-gpu", {})
            return (
                len(tools.tools),
                json.loads(ok.content[0].text),
                json.loads(bad.content[0].text),
                [p.name for p in prompts.prompts],
                gpu.messages[0].content.text,
            )

    count, ok, bad, prompt_names, gpu_text = anyio.run(scenario)
    assert count == 14
    assert "fotos-a-3d" in prompt_names and "usar-gpu" in prompt_names
    assert "reverse3d_configure" in gpu_text
    assert ok["ok"] is True and ok["result"]["volume"] == pytest.approx(8000)
    assert bad["error"]["code"] == "security_violation"
