from __future__ import annotations

import asyncio
from contextvars import ContextVar
import json
import os
import signal
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from Skill_MAS.bench_adapters.base import NativeBenchmarkAdapter, NativeTask
from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient, normalize_usage_tokens


class NativeToolRuntime:
    """Isolated tool-call loop used by native benchmark adapters.

    The runtime implements the common file/shell tools itself. Benchmark
    adapters can add an ``execute_tool`` method later for MCP or Docker tools;
    unsupported tools are recorded as structured failures instead of silently
    falling back to text-only execution.
    """

    def __init__(self, adapter: NativeBenchmarkAdapter, task: NativeTask, context: dict[str, Any], model: str):
        self.adapter = adapter
        self.task = task
        self.context = context
        self._actor = ContextVar("native_actor", default="")
        self.workspace = Path(str(context["workspace"])).resolve()
        self.trace: list[dict[str, Any]] = []
        self.llm_usage: list[dict[str, Any]] = []
        self._tool_schemas: list[dict[str, Any]] | None = None
        try:
            adapter.setup_runtime(task, context)
            self._tool_schemas = adapter.tool_schemas(task, context)
            self.client = AsyncOpenAIClient(model=model)
        except BaseException:
            adapter.teardown_runtime(task, context)
            raise

    @staticmethod
    def _schema(name: str) -> dict[str, Any]:
        schemas = {
            "read_file": {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file relative to the task workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            "write_file": {"type": "function", "function": {"name": "write_file", "description": "Write UTF-8 text to a file relative to the task workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            "bash": {"type": "function", "function": {"name": "bash", "description": "Run a shell command in the task workspace.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
            "web_search": {"type": "function", "function": {"name": "web_search", "description": "Search the web for factual evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
            "web_fetch": {"type": "function", "function": {"name": "web_fetch", "description": "Fetch a web page URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
        }
        return schemas[name]

    def _safe_path(self, raw: str) -> Path:
        p = (self.workspace / str(raw)).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise ValueError("path escapes task workspace")
        return p

    async def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        started = time.monotonic()
        try:
            if self._tool_schemas is not None:
                allowed = {tool["function"]["name"] for tool in self._tool_schemas}
                if name not in allowed:
                    result = f"[native tool unavailable: {name}]"
                else:
                    if hasattr(self.adapter, "execute_scoped_tool"):
                        adapter_result = await self.adapter.execute_scoped_tool(name, args, self.context, self._actor.get())
                    else:
                        adapter_result = await self.adapter.execute_tool(name, args, self.context)
                    result = adapter_result if adapter_result is not None else f"[native tool unavailable: {name}]"
            elif name == "read_file":
                p = self._safe_path(args.get("path", ""))
                result = p.read_text(encoding="utf-8", errors="replace")[:50000] if p.is_file() else f"[error: file not found: {p}]"
            elif name == "write_file":
                p = self._safe_path(args.get("path", "")); p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(args.get("content", "")), encoding="utf-8")
                result = f"[wrote {len(str(args.get('content', '')))} chars to {p}]"
            elif name == "bash":
                timeout = min(max(int(args.get("timeout", 30) or 30), 1), 120)
                proc = await asyncio.create_subprocess_shell(str(args.get("command", "")), cwd=str(self.workspace), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
                try:
                    out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    result = (out + b"\n" + err).decode(errors="replace")[-30000:]
                except asyncio.TimeoutError:
                    os.killpg(proc.pid, signal.SIGKILL); await proc.wait(); result = f"[timeout after {timeout}s]"
            elif name == "web_search":
                query = urllib.parse.quote_plus(str(args.get("query", "")))
                req = urllib.request.Request(f"https://duckduckgo.com/html/?q={query}", headers={"User-Agent": "Skill-MAS/1.0"})
                with urllib.request.urlopen(req, timeout=20) as response:
                    result = response.read().decode(errors="replace")[:30000]
            elif name == "web_fetch":
                req = urllib.request.Request(str(args.get("url", "")), headers={"User-Agent": "Skill-MAS/1.0"})
                with urllib.request.urlopen(req, timeout=20) as response:
                    result = response.read().decode(errors="replace")[:30000]
            else:
                result = f"[native tool unavailable: {name}]"
        except Exception as exc:
            result = f"[tool error {name}: {type(exc).__name__}: {exc}]"
        self.trace.append({**({"actor": self._actor.get()} if hasattr(self.adapter, "bind_actor") else {}), "tool": name, "arguments": args, "result": result[:30000], "elapsed_sec": round(time.monotonic() - started, 6)})
        return result

    async def run(self, prompt: Any, *, max_rounds: int = 12) -> tuple[str, dict[str, Any]]:
        actor = self.adapter.bind_actor(prompt, self.context) if hasattr(self.adapter, "bind_actor") else ""
        token = self._actor.set(actor)
        try:
            return await self._run_impl(prompt, max_rounds=max_rounds)
        finally:
            self._actor.reset(token)

    async def _run_impl(self, prompt: Any, *, max_rounds: int = 12) -> tuple[str, dict[str, Any]]:
        if isinstance(prompt, dict):
            prompt = str(prompt.get("prompt") or prompt.get("task_input") or "")
        prompt = str(prompt)
        self._tool_schemas = self.adapter.tool_schemas(self.task, self.context)
        if self._tool_schemas is None:
            tools = [self._schema(n) for n in self.adapter.tool_names(self.task) if n in {"read_file", "write_file", "bash", "web_search", "web_fetch"}]
        else:
            tools = list(self._tool_schemas)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        aggregate = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0, "tool_calls": self.trace}
        for _ in range(max(1, int(max_rounds))):
            text, calls, usage = await self.client.generate_with_tools(user_prompt="", tools=tools, messages=messages)
            self.llm_usage.append(dict(usage))
            prompt_tokens, output_tokens, total_tokens = normalize_usage_tokens(usage)
            for key, value in zip(("prompt_tokens", "output_tokens", "total_tokens"), (prompt_tokens, output_tokens, total_tokens)):
                aggregate[key] += value
            aggregate["estimated_cost_usd"] += float(usage.get("estimated_cost_usd", 0.0) or 0.0)
            if not calls:
                return text, aggregate
            assistant = {"role": "assistant", "content": text or None, "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]}
            messages.append(assistant)
            for call in calls:
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self.execute_tool(call["name"], args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return "[native tool loop exhausted]", aggregate

    async def close(self) -> None:
        try:
            self.adapter.teardown_runtime(self.task, self.context)
        finally:
            await self.client.aclose()
