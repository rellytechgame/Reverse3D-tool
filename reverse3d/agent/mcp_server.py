"""Model Context Protocol (MCP) server over stdio (``[mcp]`` extra).

Exposes the reverse3d tools and slash-command prompts to any MCP client
(Claude Desktop, Claude Code, Cursor, VS Code, Windsurf, Gemini CLI, Codex,
OpenCode...).

* Transport is **stdio only**: no network port is opened, so the server is only
  reachable by the local client process that spawned it.
* Settings are re-read on every call, so ``reverse3d_configure`` (e.g. switching
  to GPU) takes effect immediately without restarting the client.
* Tool calls run in a worker thread and are serialised by a semaphore: heavy
  jobs (photogrammetry, rendering) cannot be launched concurrently by a
  runaway agent (OWASP LLM10).
* All logging goes to stderr; stdout carries only MCP frames.
* Works with both MCP Python SDK generations: 1.x registers handlers with
  decorators, 2.x passes them to the ``Server`` constructor.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from reverse3d._version import __version__
from reverse3d.agent.prompts import PROMPTS, SERVER_INSTRUCTIONS, render_prompt
from reverse3d.agent.tools import execute_tool, input_schema, visible_tools
from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError

logger = logging.getLogger(__name__)

SettingsProvider = Callable[[], Settings]


def build_server(settings_provider: SettingsProvider, *, max_concurrent: int = 1) -> Any:
    """Create the MCP ``Server`` with tool and prompt handlers registered.

    Args:
        settings_provider: returns fresh settings for each request.

    Raises:
        DependencyMissingError: the ``mcp`` package is not installed.
    """
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
    except ImportError as exc:
        raise DependencyMissingError(
            "The MCP server requires the 'mcp' package.", hint='pip install "reverse3d[mcp]"'
        ) from exc

    limiter = anyio.Semaphore(max(1, max_concurrent))
    toolset = settings_provider().toolset  # the tool list is fixed per server session

    # camelCase keys are field names in SDK 1.x and aliases in 2.x: model_validate
    # accepts the same payload on both generations.
    def tool_list() -> list[Any]:
        return [
            types.Tool.model_validate(
                {"name": s.name, "description": s.description, "inputSchema": input_schema(s)}
            )
            for s in visible_tools(toolset)
        ]

    def prompt_list() -> list[Any]:
        return [
            types.Prompt.model_validate(
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": [
                        {"name": a.name, "description": a.description, "required": a.required}
                        for a in p.arguments
                    ],
                }
            )
            for p in PROMPTS.values()
        ]

    def prompt_result(name: str, arguments: dict[str, str] | None) -> Any:
        try:
            text = render_prompt(name, arguments)
        except KeyError:
            text = f"Unknown prompt '{name}'. Available: {', '.join(PROMPTS)}"
        except ValueError as exc:
            text = str(exc)
        return types.GetPromptResult.model_validate(
            {
                "description": PROMPTS[name].description if name in PROMPTS else None,
                "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
            }
        )

    async def run_tool(name: str, arguments: dict[str, Any] | None) -> tuple[Any, bool]:
        async with limiter:
            result = await anyio.to_thread.run_sync(
                execute_tool, name, arguments, settings_provider()
            )
        text = types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))
        return text, not result.get("ok", False)

    if hasattr(Server, "list_tools"):  # MCP SDK 1.x: decorator registration
        try:
            server: Any = Server("reverse3d", instructions=SERVER_INSTRUCTIONS)
        except TypeError:  # very old 1.x without instructions support
            server = Server("reverse3d")

        @server.list_tools()
        async def list_tools() -> list[Any]:
            return tool_list()

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
            content, _ = await run_tool(name, arguments)
            return [content]

        @server.list_prompts()
        async def list_prompts() -> list[Any]:
            return prompt_list()

        @server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> Any:
            return prompt_result(name, arguments)

        return server

    # MCP SDK 2.x: constructor registration with (ctx, params) handlers.
    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult.model_validate({"tools": tool_list()})

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        content, is_error = await run_tool(params.name, params.arguments)
        return types.CallToolResult.model_validate({"content": [content], "isError": is_error})

    async def on_list_prompts(ctx: Any, params: Any) -> Any:
        return types.ListPromptsResult.model_validate({"prompts": prompt_list()})

    async def on_get_prompt(ctx: Any, params: Any) -> Any:
        return prompt_result(params.name, params.arguments)

    return Server(
        "reverse3d",
        version=__version__,
        instructions=SERVER_INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
        on_list_prompts=on_list_prompts,
        on_get_prompt=on_get_prompt,
    )


def serve(settings_provider: SettingsProvider, *, max_concurrent: int = 1) -> None:
    """Run the MCP stdio server until the client disconnects."""
    server = build_server(settings_provider, max_concurrent=max_concurrent)
    import anyio
    from mcp.server.stdio import stdio_server

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    logger.info("reverse3d MCP server starting (workspace=%s)", settings_provider().workspace.name)
    anyio.run(_main)
