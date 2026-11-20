"""Task-scoped bridge to the native MAC-Bench application tools."""

from __future__ import annotations

import asyncio
import contextvars
import json
from typing import Any


_active_task: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "macbench_active_task", default=None,
)


def set_active_task(task: Any) -> contextvars.Token:
    """Bind one task handle to the current case's async context."""
    return _active_task.set(task)


def clear_active_task(token: contextvars.Token | None = None) -> None:
    """Clear a task binding without affecting concurrently running cases."""
    if token is None:
        _active_task.set(None)
    else:
        _active_task.reset(token)


SCHEMA = {
    "name": "macbench",
    "description": (
        "Use the current MAC-Bench task's official application tools. First call "
        "action='list_tools' to inspect tool names and descriptions, then call "
        "action='describe' for the JSON schema of a tool, and action='call' to "
        "execute it. The task environment validates every call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_tools", "describe", "call"],
                "description": "List task tools, inspect one schema, or invoke one.",
            },
            "tool_name": {
                "type": "string",
                "description": "Native MAC-Bench task tool name for describe/call.",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments matching the selected task tool's schema.",
                "additionalProperties": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _tool_catalog(task: Any) -> list[Any]:
    return list(task.view.tools)


async def execute(
    action: str = "list_tools",
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    **_kwargs,
) -> str:
    task = _active_task.get()
    if task is None:
        return "[error: no active MAC-Bench task]"

    catalog = _tool_catalog(task)
    if action == "list_tools":
        return json.dumps(
            [
                {"name": tool.name, "description": tool.description}
                for tool in catalog
            ],
            ensure_ascii=False,
            indent=2,
        )

    if action not in {"describe", "call"}:
        return "[error: action must be list_tools, describe, or call]"
    tool = next((candidate for candidate in catalog if candidate.name == tool_name), None)
    if tool is None:
        return f"[error: unknown MAC-Bench task tool {tool_name!r}; call list_tools]"

    if action == "describe":
        return json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
            ensure_ascii=False,
            indent=2,
        )

    try:
        result = await asyncio.to_thread(
            task.environment.call, tool_name, arguments or {},
        )
    except Exception as exc:
        return f"[MAC-Bench tool error: {type(exc).__name__}: {exc}]"
    return json.dumps(result, ensure_ascii=False, default=str)
