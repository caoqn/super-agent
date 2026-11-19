"""LLM response functions plugged into AgentNet's RouterModule / ExecutorModule.

Routing, splitting, forwarding, experience pools and ability evolution are the
original AgentNet code. The only behavioural change is that the executor's
answer step (EXECUTOR_SYSTEM_PROMPT) may call tools for the current GAIA task.
"""
import json
import os
import re

from config.setting import (EXECUTOR_SYSTEM_PROMPT, ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT,
                            ROUTER_SYSTEM_PROMPT, THOUGHT_SYSTEM_PROMPT)

from .llm import chat, simple_response
from .tracker import TRACKER

CURRENT = {"toolbox": None}   # set by the runner for each task

TOOL_NOTE = """

You have tools for this task (web search, web page fetching, reading the attached file, running Python, ...).
Use them whenever the answer depends on facts you cannot verify, on the attached file, or on calculation.
When you are done, reply WITHOUT a tool call, and put your final output on its own line as:
RESULT: <answer in the Required Format>"""

TEXT_REACT_NOTE = """

You can use tools. To call one, reply with exactly one line and nothing else:
ACTION: {{"tool": "<name>", "args": {{...}}}}
You will then receive an OBSERVATION. Available tools (JSON schemas):
{schemas}
When you are done, reply without ACTION and put your final output on its own line as:
RESULT: <answer in the Required Format>"""


def _role_for(system_prompt):
    if system_prompt == ROUTER_SYSTEM_PROMPT:
        return "router"
    if system_prompt == ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT:
        return "router_next_agent"
    if system_prompt == THOUGHT_SYSTEM_PROMPT:
        return "thought"
    if system_prompt == EXECUTOR_SYSTEM_PROMPT:
        return "executor"
    return "other"


def plain_response(system_prompt, query_prompt):
    """Drop-in replacement for get_gpt_response (router + thought calls)."""
    return simple_response(system_prompt, query_prompt, role=_role_for(system_prompt))


def executor_response(system_prompt, query_prompt):
    """Used for ExecutorModule.get_response: tool loop on the answer step only."""
    box = CURRENT.get("toolbox")
    if system_prompt != EXECUTOR_SYSTEM_PROMPT or box is None or os.environ.get("GAIA_TOOLS", "1") == "0":
        return plain_response(system_prompt, query_prompt)
    max_steps = int(os.environ.get("GAIA_MAX_TOOL_STEPS", 10))
    if os.environ.get("LLM_SUPPORTS_TOOLS", "1") == "1":
        return _native_tool_loop(system_prompt, query_prompt, box, max_steps)
    return _text_react_loop(system_prompt, query_prompt, box, max_steps)


def _native_tool_loop(system_prompt, query_prompt, box, max_steps):
    tools = box.schemas()
    messages = [{"role": "system", "content": system_prompt + TOOL_NOTE},
                {"role": "user", "content": query_prompt}]
    for _ in range(max_steps):
        try:
            msg = chat(messages, tools=tools, role="executor")
        except Exception as e:
            if "HTTP 400" not in str(e):
                raise
            # endpoint rejected the native tool-calling request: redo this step with text-based tools
            print(f"[executor] native tool call rejected ({str(e)[:160]}); falling back to text ReAct for this step")
            return _text_react_loop(system_prompt, query_prompt, box, max_steps)
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip()
        calls = [{"id": c.get("id") or f"call_{i}", "type": "function",
                  "function": {"name": c.get("function", {}).get("name", ""),
                               "arguments": c.get("function", {}).get("arguments") or "{}"}}
                 for i, c in enumerate(calls)]
        messages.append({"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls})
        for c in calls:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            out = box.call(fn.get("name", ""), args) or "(empty)"
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
    # Final answer without tools. A history containing tool messages but no `tools`
    # parameter is rejected with HTTP 400 by most providers, so flatten it to plain text.
    msg = chat(_flatten(messages) + [{"role": "user", "content":
               "Tool budget exhausted. Using the tool results above, give your final answer now, ending with a line 'RESULT: <answer>'."}],
               role="executor")
    return (msg.get("content") or "").strip()


def _flatten(messages):
    sys_msg, convo = messages[0], []
    for m in messages[1:]:
        if m["role"] == "tool":
            convo.append(f"[tool result]\n{m['content']}")
        elif m["role"] == "assistant":
            calls = "; ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in m.get("tool_calls") or [])
            convo.append(f"[assistant]\n{m.get('content') or ''}" + (f"\n[called tools] {calls}" if calls else ""))
        else:
            convo.append(m["content"])
    return [{"role": "system", "content": sys_msg["content"]}, {"role": "user", "content": "\n\n".join(convo)}]


_ACTION = re.compile(r"ACTION:\s*(\{.*\})", re.S)


def _text_react_loop(system_prompt, query_prompt, box, max_steps):
    schemas = json.dumps([t["function"] for t in box.schemas()], ensure_ascii=False)
    messages = [{"role": "system", "content": system_prompt + TEXT_REACT_NOTE.format(schemas=schemas)},
                {"role": "user", "content": query_prompt}]
    for _ in range(max_steps):
        text = (chat(messages, role="executor").get("content") or "").strip()
        m = _ACTION.search(text)
        if not m or "RESULT:" in text.split("ACTION:")[0]:
            return text
        try:
            act = json.loads(m.group(1))
            out = box.call(act.get("tool", ""), act.get("args", {}))
        except json.JSONDecodeError as e:
            out = f"ERROR: could not parse ACTION json: {e}"
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": "OBSERVATION:\n" + out})
    messages.append({"role": "user", "content": "Tool budget exhausted. Give your final answer now, ending with a line 'RESULT: <answer>'."})
    return (chat(messages, role="executor").get("content") or "").strip()
