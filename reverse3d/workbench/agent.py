"""A model working in a folder: it thinks, uses its tools, and asks before changing things.

The model answers in plain text, or calls a tool with

    <tool_call>{"name": "edit_file", "arguments": {...}}</tool_call>

(the format Qwen2.5 and Hermes-style models are trained on). Each call goes
through the permissions (permissions.py): allowed, asked to the person
(through `ask`, which can be a terminal, Studio or a cloud page), or refused.
The result goes back to the model as <tool_response>, until it answers
without calling a tool or reaches max_steps.

Everything that happens is an event (dict) given to `on_event`, so any
interface can show it live: {"type": "text" | "call" | "result" | "denied" |
"done" | "error", ...}.
"""

from __future__ import annotations

import json
import platform
import re
from collections.abc import Callable
from typing import Any, Protocol

from reverse3d.workbench.permissions import Permissions, command_risk
from reverse3d.workbench.tools import TOOLS, ToolError, Workspace

Event = dict[str, Any]
# How models write a call: Qwen/Hermes <tool_call>, or a JSON block (small models often do).
CALL = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    r"|```(?:json)?\s*(\{\s*\"name\".*?\})\s*```"
    r"|^\s*(\{\s*\"name\"\s*:.*\})\s*$",
    re.DOTALL,
)


class Brain(Protocol):
    def reply(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str: ...


def tools_text(tools: list[dict[str, Any]]) -> str:
    """The tools in words, for models whose chat template does not take them."""
    listing = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return (
        "\n\n# Tools\n\nYou may call these functions:\n<tools>\n"
        f"{listing}\n</tools>\n\nTo call one, reply ONLY with:\n"
        '<tool_call>\n{"name": <name>, "arguments": <JSON object>}\n</tool_call>'
    )


class LlamaBrain:
    """A local GGUF model through llama.cpp (MIT)."""

    def __init__(self, model_path: str, n_ctx: int = 16384, gpu_layers: int = -1) -> None:
        import llama_cpp

        self.llm = llama_cpp.Llama(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=gpu_layers, verbose=False
        )
        template = self.llm.metadata.get("tokenizer.chat_template") or ""
        self.native_tools = "tools" in template  # its template places the tools itself

    def reply(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        if not self.native_tools:
            messages = [
                {**messages[0], "content": messages[0]["content"] + tools_text(tools)},
                *messages[1:],
            ]
        out = self.llm.create_chat_completion(
            messages=messages,
            tools=tools if self.native_tools else None,
            temperature=0.2,
            max_tokens=2048,
            stop=["</tool_call>"],
        )
        text = out["choices"][0]["message"]["content"] or ""
        if "<tool_call>" in text and "</tool_call>" not in text:
            text += "</tool_call>"  # the stop word is cut from the reply
        return text


def system_prompt(identity: str, ws: Workspace) -> str:
    # English: small models follow it best; they still answer in the person's language.
    mode = {
        "ask": "they approve every change and command",
        "edits": "you may edit files; commands need their approval",
        "free": "you may edit files and run commands; risky ones need approval",
    }[ws.perms.mode]
    return (
        f"{identity}\n"
        f"You work in the folder {ws.root} on {platform.system()}, through your tools: "
        "call one tool at a time and wait for its result. Read files before changing them, "
        "make small exact edits, and check your work (for example run the tests). "
        f"Permissions: {mode}. If the person refuses something, do not insist. Never try "
        "to read keys or credentials. When the task is done, answer in plain text, in the "
        "person's language, saying what you did."
    )


def parse_call(text: str) -> tuple[str, dict[str, Any]] | None:
    m = CALL.search(text)
    if not m:
        return None
    raw = next(g for g in m.groups() if g)
    try:
        data = json.loads(raw, strict=False)  # raw newlines inside strings are fine
    except ValueError:
        data = lenient(raw)
        if data is None:
            return ("", {"_raw": raw})
    args = data.get("arguments") or data.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return str(data.get("name", "")), args


_FIELD = re.compile(
    r'"(\w+)"\s*:\s*(?:"(.*?)"(?=\s*(?:,\s*"\w+"\s*:|\}))|(-?\d+(?:\.\d+)?|true|false|null))',
    re.DOTALL,
)


def lenient(raw: str) -> dict[str, Any] | None:
    """A call whose JSON is broken the way small models break it (unescaped quotes
    inside code): each field is read up to the quote that closes it."""
    name = re.search(r'"name"\s*:\s*"([\w.-]+)"', raw)
    body = re.search(r'"(?:arguments|parameters)"\s*:\s*\{(.*)\}\s*\}?\s*$', raw, re.DOTALL)
    if not name or not body:
        return None
    args: dict[str, Any] = {}
    for key, text, scalar in _FIELD.findall(body.group(1)):
        if scalar:
            args[key] = json.loads(scalar)
        else:
            args[key] = (
                text.replace("\\\\", "\0")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\0", "\\")
            )
    return {"name": name.group(1), "arguments": args}


Ask = Callable[[Event], str]  # returns "yes" | "session" | "folder" | "no"


class Agent:
    def __init__(
        self,
        brain: Brain,
        perms: Permissions,
        ask: Ask,
        on_event: Callable[[Event], None],
        identity: str = "You are a model by Voxel Enterprise that works with files and code.",
        max_steps: int = 30,
    ) -> None:
        self.brain, self.ask, self.emit = brain, ask, on_event
        self.ws = Workspace(perms)
        self.max_steps = max_steps
        self.messages = [{"role": "system", "content": system_prompt(identity, self.ws)}]
        self.tools = [t.schema() for t in TOOLS.values()]

    def act(self, name: str, args: dict[str, Any]) -> str:
        tool = TOOLS.get(name)
        if tool is None:
            return f"error: no existe la herramienta «{name}». Usa una de: {', '.join(TOOLS)}"
        missing = [k for k in tool.required if k not in args]
        if missing:
            return f"error: faltan argumentos: {', '.join(missing)}"
        command = str(args.get("command", ""))
        decision = self.ws.perms.decide(tool.kind, name, command)
        if decision.verdict == "deny":
            self.emit({"type": "denied", "tool": name, "args": args, "reason": decision.reason})
            return f"rechazado: {decision.reason}"
        if decision.verdict == "ask":
            try:
                preview = tool.preview(self.ws, args) if tool.preview else ""
            except (ToolError, OSError) as exc:
                return f"error: {exc}"
            risky = tool.kind == "run" and command_risk(command) == "risky"
            answer = self.ask(
                {
                    "type": "permission",
                    "tool": name,
                    "kind": tool.kind,
                    "args": args,
                    "preview": preview,
                    "reason": decision.reason,
                    "risky": risky,
                }
            )
            if not self.ws.perms.answer(tool.kind, answer, risky):  # type: ignore[arg-type]
                self.emit(
                    {
                        "type": "denied",
                        "tool": name,
                        "args": args,
                        "reason": "la persona dijo que no",
                    }
                )
                return "la persona no lo permitió: no lo hagas; propón otra cosa o pregunta"
        try:
            result = tool.run(self.ws, args)
        except (ToolError, OSError, UnicodeError) as exc:
            result = f"error: {exc}"
        self.emit({"type": "result", "tool": name, "args": args, "result": result})
        return result

    def run(self, task: str) -> str:
        self.messages.append({"role": "user", "content": task})
        recent: list[str] = []
        for _ in range(self.max_steps):
            text = self.brain.reply(self.messages, self.tools)
            self.messages.append({"role": "assistant", "content": text})
            call = parse_call(text)
            said = CALL.sub("", text).strip()
            if said:
                self.emit({"type": "text", "text": said})
            if call is None:
                self.emit({"type": "done", "text": said})
                return said
            name, args = call
            self.emit({"type": "call", "tool": name, "args": args})
            key = json.dumps([name, args], sort_keys=True)
            recent = [*recent[-5:], key]
            if recent.count(key) >= 3:  # going round in circles: stop and say so
                self.emit({"type": "error", "text": "se detuvo: repetía la misma acción"})
                return ""
            if not name:
                result = "error: la llamada no es JSON válido; repítela bien"
            else:
                result = self.act(name, args)
            self.messages.append(
                {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
            )
        self.emit({"type": "error", "text": f"se detuvo tras {self.max_steps} pasos"})
        return ""
