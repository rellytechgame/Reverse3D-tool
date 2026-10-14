"""Minimal Claude agent that drives reverse3d through tool use.

Usage::

    pip install -e ".[agents]"
    ant auth login                         # or export ANTHROPIC_API_KEY=...
    export REVERSE3D_WORKSPACE=./workspace # sandbox for every file the agent touches
    python examples/claude_agent.py "Mide piezas/soporte.stl, repárala y genera un render"

Safety properties of this loop:

* every tool call goes through ``execute_tool`` (strict validation + workspace sandbox);
* expensive tools (photogrammetry, rendering) require human approval;
* hard caps on turns and tokens (OWASP LLM10);
* refusals are handled explicitly instead of being treated as answers.
"""

from __future__ import annotations

import json
import sys

import anthropic

from reverse3d import Settings
from reverse3d.agent import execute_tool, tool_definitions

MODEL = "claude-opus-5-5"
MAX_TURNS = 20
REQUIRE_APPROVAL = {"reverse3d_photogrammetry", "reverse3d_render", "reverse3d_estimate_depth"}

SYSTEM = (
    "You are a 3D reverse-engineering assistant. Use the reverse3d tools to analyse "
    "images, reconstruct, repair, measure, fit primitives and render. All paths are "
    "relative to the user's workspace. Start with reverse3d_environment_check. When a "
    "tool returns ok=false, read error.hint and adjust. Report dimensions with units "
    "and state clearly which values are estimates."
)


def approve(name: str, arguments: dict[str, object]) -> bool:
    if name not in REQUIRE_APPROVAL:
        return True
    print(f"\n[approval] {name} {json.dumps(arguments, ensure_ascii=False)}")
    return input("Run this tool? [y/N] ").strip().lower() == "y"


def run(task: str) -> int:
    settings = Settings.from_env()
    client = anthropic.Anthropic()
    tools = tool_definitions("anthropic")
    messages: list[dict[str, object]] = [{"role": "user", "content": task}]

    for _ in range(MAX_TURNS):
        # Server-side fallbacks route refusals to an appropriate model automatically.
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            tools=tools,
            messages=messages,
            output_config={"effort": "high"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            print("The model declined this request.", file=sys.stderr)
            return 2

        for block in response.content:
            if block.type == "text":
                print(block.text)

        if response.stop_reason != "tool_use":
            return 0

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if approve(block.name, block.input):
                outcome = execute_tool(block.name, block.input, settings)
            else:
                outcome = {
                    "ok": False,
                    "error": {"code": "user_declined", "message": "The user declined."},
                }
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(outcome, ensure_ascii=False),
                    "is_error": not outcome.get("ok", False),
                }
            )
        # All results of one turn go back in a single user message.
        messages.append({"role": "user", "content": results})

    print(f"Stopped after {MAX_TURNS} turns.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(64)
    try:
        sys.exit(run(" ".join(sys.argv[1:])))
    except anthropic.RateLimitError:
        print("Rate limited by the API; retry later.", file=sys.stderr)
        sys.exit(75)
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}", file=sys.stderr)
        sys.exit(1)
    except anthropic.APIConnectionError:
        print("Could not reach the API.", file=sys.stderr)
        sys.exit(1)
