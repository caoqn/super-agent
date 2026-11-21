"""AgentNet as a MAC-Bench adapter (object with `name` and `run(context)`).

* Every model request goes through context.client (so MAC-Bench's budget and
  recording see all router / thought / executor calls) and every action through
  context.environment.call.
* Routing / splitting / forwarding / experience pools are the original AgentNet code.
  The executor's answer step runs a tool loop over context.task.tools.
* One AgentNet network persists across episodes; after MAC-Bench scores an episode
  the runner calls `feedback(result)` so AgentNet evolves (update_agent_graph).
"""
import json
import os
import time

from config.setting import (EXECUTOR_SYSTEM_PROMPT, ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT,
                            ROUTER_SYSTEM_PROMPT, THOUGHT_SYSTEM_PROMPT)

from .tracker import TRACKER

CTX = {"context": None}
ENGINEERING = {"product_application", "platform_data", "developer_release"}
ABILITIES = ["planning", "data_reconciliation", "record_editing", "coding", "testing"]
# Resource limits can be overridden in .env; record them with experiment settings.
# MAC_TOOL_OUTPUT_CHARS limits characters in one tool observation.
# MAC_MAX_HISTORY_ROUNDS retains recent rounds and replaces older ones with placeholders.
# MAC_BUDGET_RESERVE_TOKENS stops tools and forces finalization below the reserve.
TOOL_OUT_CHARS = int(os.environ.get("MAC_TOOL_OUTPUT_CHARS", 4000))
MAX_HISTORY_ROUNDS = int(os.environ.get("MAC_MAX_HISTORY_ROUNDS", 8))
BUDGET_RESERVE_TOKENS = int(os.environ.get("MAC_BUDGET_RESERVE_TOKENS", 150000))


def register_task_types(categories):
    from config.setting import BASIC_PROBLEM_COMPLEXITY, task_to_ability_map
    for c in categories:
        t = "mac_" + c
        task_to_ability_map[t] = (["planning", "coding", "testing"] if c in ENGINEERING
                                  else ["planning", "data_reconciliation", "record_editing"])
        BASIC_PROBLEM_COMPLEXITY[t] = 1.0


class MACConstraints:
    @staticmethod
    def get_constraint():
        return ("You MUST actually perform the required changes in the task environment using the write tools "
                "(create/patch/update records, make reservations, create calendar events, etc.). Planning or reading "
                "materials is NOT enough: the evaluator checks only the final environment state, not your report. "
                "Do not stop at a plan — keep calling tools until the required records and appointments really exist. "
                "When the work is truly done, give a short handover report of what you changed.")

    @staticmethod
    def get_split_constraint():
        return ("Perform only the part in **Task Description** using the tools, then report concretely what you changed "
                "(record ids, files, values) and what remains for other agents.")

    @staticmethod
    def get_thought_constraint():
        return ("You are in the reasoning phase. Plan which records/files to inspect, which tools to call, the business "
                "or engineering rules to respect, and how to verify the result. Do not claim work is done.")


def _role(system_prompt):
    return {ROUTER_SYSTEM_PROMPT: "router", ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT: "router_next_agent",
            THOUGHT_SYSTEM_PROMPT: "thought", EXECUTOR_SYSTEM_PROMPT: "executor"}.get(system_prompt, "other")


def _default_max_tokens():
    # MAC-Bench uses LiteLLM and needs a larger output budget when thinking is enabled.
    # Use a separate limit so GAIA LLM_MAX_TOKENS=2048 stays independent.
    return int(os.environ.get("MAC_BENCH_LLM_MAX_TOKENS", os.environ.get("LLM_MAX_TOKENS", 4096)))


def _complete(messages, tools=None, role=None, max_tokens=None):
    ctx = CTX["context"]
    t0 = time.time()
    try:
        reply = ctx.client.complete(messages, tools=tools,
                                    max_tokens=max_tokens or _default_max_tokens())
    except Exception as e:
        # LiteLLM raises ModelOutputTruncatedError when finish_reason=length.
        # Retry once without tools, requesting only the conclusion on shorter history.
        # This avoids scoring an entire episode as zero after one truncated response.
        if type(e).__name__ == "ModelOutputTruncatedError" and tools:
            from mac_bench.runtime.llm.client import ChatMessage
            flat = []
            for m in messages:
                if getattr(m, "role", None) == "tool":
                    flat.append(ChatMessage("user", f"[tool result, omitted]\n{(getattr(m, 'content', '') or '')[:2000]}"))
                elif getattr(m, "role", None) == "assistant" and getattr(m, "tool_calls", None):
                    flat.append(ChatMessage("assistant", "(assistant called tools here)"))
                else:
                    flat.append(m)
            flat.append(ChatMessage("user", "Your previous answer was cut off. Give only the final RESULT now, no more tools."))
            reply = ctx.client.complete(flat, max_tokens=_default_max_tokens())
        else:
            raise
    u = dict(reply.usage or {})
    with TRACKER.role_scope(role):
        TRACKER.record("llm", time.time() - t0, u.get("prompt_tokens", u.get("input_tokens", 0)),
                       u.get("completion_tokens", u.get("output_tokens", 0)), extra={"model": reply.model})
    return reply


def plain_response(system_prompt, query_prompt):
    from mac_bench.runtime.llm.client import ChatMessage
    reply = _complete([ChatMessage("system", system_prompt), ChatMessage("user", query_prompt)],
                      role=_role(system_prompt), max_tokens=_default_max_tokens())
    return (reply.content or "").strip()


def _remaining_tokens():
    ctx = CTX["context"]
    budget = getattr(ctx, "budget", None)
    if budget is None or getattr(budget, "limits", None) is None:
        return None
    used = getattr(budget, "input_tokens", 0) + getattr(budget, "output_tokens", 0)
    return budget.limits.max_tokens - used


_DSML_OPEN = "｜｜"  # DeepSeek DSML uses doubled full-width vertical bars (U+FF5C).


def _parse_dsml(text):
    """Parse DSML tool text into name/args dictionaries; return an empty list if absent."""
    import re
    calls = []
    # Split invoke blocks, including the doubled full-width-bar DSML syntax.
    for block in re.findall(r"<\s*｜｜DSML｜｜\s*invoke\s+name=\"([^\"]+)\">(.*?)<\s*/｜｜DSML｜｜\s*invoke\s*>",
                            text, re.S):
        name, body = block[0], block[1]
        args = {}
        for pn, pt, pv in re.findall(
                r"<\s*｜｜DSML｜｜\s*parameter\s+name=\"([^\"]+)\"\s+string=\"(true|false)\">(.*?)<\s*/｜｜DSML｜｜\s*parameter\s*>",
                body, re.S):
            if pt == "true":
                args[pn] = pv
            else:
                v = pv.strip()
                try:
                    args[pn] = json.loads(v) if v else None
                except json.JSONDecodeError:
                    args[pn] = v
        calls.append({"name": name, "args": args})
    return calls


def executor_response(system_prompt, query_prompt):
    if system_prompt != EXECUTOR_SYSTEM_PROMPT:
        return plain_response(system_prompt, query_prompt)
    from mac_bench.runtime.llm.client import ChatMessage, assistant_chat_message
    ctx = CTX["context"]
    functions = [t.as_function() for t in ctx.task.tools]
    sys_p = (system_prompt + "\n\n# Environment instructions\n" + (ctx.task.interaction_instructions or "")
             + "\n\nYou are operating the real task environment. Keep calling tools to actually make the required "
               "changes until the objective is fulfilled — a plan or a summary of what you *would* do does not count "
               "and will score zero.\n\n"
               "# How to make changes (MANDATORY sequence — this is a 3-step indirection):\n"
               "1. First call `list_tools` to discover the registered business operations. The write operations "
               "(create / patch / update / delete records, make reservations, create events, ...) live there, NOT "
               "in `read_resource`/`query_resource`/`compute`, which are all READ-ONLY.\n"
               "2. For the operation you need, call `describe_tool` to read its exact request structure and fields.\n"
               "3. Then call `call_tool` with the correct arguments to actually execute the write. Repeat steps 1-3 "
               "for every required change. Reading materials alone will score zero — you MUST reach `call_tool`.\n\n"
               "Only when the changes really exist in the environment, reply without a tool call and end with "
               "'RESULT: <report of what you changed>'. Always call tools with native function calling only; "
               "never write tool calls as DSML/XML/markdown text.")
    # Keep system/user messages; replace older tool rounds with one-line placeholders.
    head = [ChatMessage("system", sys_p), ChatMessage("user", query_prompt)]
    rounds = []
    max_steps = int(os.environ.get("MAC_MAX_TOOL_STEPS", 80))

    def _run_dsml(content, step):
        """Execute DSML calls from content; return (handled, assistant_msg, tool_msgs)."""
        dsml_calls = _parse_dsml(content)
        if not dsml_calls:
            return False, None, []
        # Create assistant tool_calls so subsequent tool messages have valid call references.
        syn_calls = [{"id": f"dsml_{step}_{i}", "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                     for i, c in enumerate(dsml_calls)]
        assistant = ChatMessage("assistant", content, tool_calls=syn_calls)
        tool_msgs = []
        for i, c in enumerate(dsml_calls):
            t0 = time.time()
            try:
                result = ctx.environment.call(c["name"], c["args"])
                text = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:
                if type(e).__name__ == "BudgetExceeded":
                    raise
                text = json.dumps({"error": f"{type(e).__name__}: {e}"})
            with TRACKER.role_scope("tool"):
                TRACKER.record("tool", time.time() - t0, extra={"tool": c["name"], "dsml": True})
            if len(text) > TOOL_OUT_CHARS:
                text = text[:TOOL_OUT_CHARS] + '... [truncated]'
            tool_msgs.append(ChatMessage("tool", text, tool_call_id=f"dsml_{step}_{i}"))
        return True, assistant, tool_msgs

    def _compact(step):
        if len(rounds) > MAX_HISTORY_ROUNDS:
            del rounds[:-MAX_HISTORY_ROUNDS]
            head.append(ChatMessage(
                "user", f"[... {step + 1 - MAX_HISTORY_ROUNDS} earlier tool-exchange rounds omitted to stay within budget ...]"))

    for step in range(max_steps):
        remaining = _remaining_tokens()
        if remaining is not None and remaining < BUDGET_RESERVE_TOKENS:
            head.append(ChatMessage(
                "user", "Token budget is nearly exhausted. Stop calling tools and give your final RESULT now, "
                        "based on what you have already changed."))
            reply = _complete(head + _flatten_rounds(rounds), role="executor")
            content = (reply.content or "").strip()
            handled, assistant, tool_msgs = _run_dsml(content, step)
            if handled:
                rounds.append((assistant, tool_msgs))
                _compact(step)
                continue
            return content
        reply = _complete(head + _flatten_rounds(rounds), tools=functions, role="executor")
        if not reply.tool_calls:
            content = (reply.content or "").strip()
            handled, assistant, tool_msgs = _run_dsml(content, step)
            if handled:
                # The model emitted DSML text instead of native function calls.
                # Parse and execute it rather than treating tool markup as the final answer.
                rounds.append((assistant, tool_msgs))
                _compact(step)
                continue
            return content
        assistant = assistant_chat_message(reply)
        tool_msgs = []
        for call in reply.tool_calls:
            fn = call["function"]
            t0 = time.time()
            try:
                args = json.loads(fn.get("arguments") or "{}")
                result = ctx.environment.call(fn["name"], args)
                text = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:
                if type(e).__name__ == "BudgetExceeded":
                    raise
                text = json.dumps({"error": f"{type(e).__name__}: {e}"})
            with TRACKER.role_scope("tool"):
                TRACKER.record("tool", time.time() - t0, extra={"tool": fn.get("name")})
            if len(text) > TOOL_OUT_CHARS:
                text = text[:TOOL_OUT_CHARS] + '... [truncated]'
            tool_msgs.append(ChatMessage("tool", text, tool_call_id=call["id"]))
        rounds.append((assistant, tool_msgs))
        # Keep recent rounds intact and replace older rounds to bound history growth.
        _compact(step)
    head.append(ChatMessage("user", "Tool-step limit for this agent reached. Summarise what you changed, ending with 'RESULT: ...'."))
    reply = _complete(head + _flatten_rounds(rounds), role="executor")
    content = (reply.content or "").strip()
    handled, assistant, tool_msgs = _run_dsml(content, max_steps)
    if handled:
        return "Tool budget exhausted and the model kept emitting tool-call text instead of a final answer."
    return content


def _flatten_rounds(rounds):
    out = []
    for assistant, tool_msgs in rounds:
        out.append(assistant)
        out.extend(tool_msgs)
    return out


class AgentNetMACAdapter:
    name = "agentnet"

    def __init__(self, experiment, success_ps=1.0):
        self.experiment = experiment
        self.success_ps = success_ps
        self.last_chain = None
        self.counter = 0
        for a in experiment.agent_graph.agents.values():
            a.router_module.get_llm_response = plain_response
            a.executor_module.get_response = executor_response

    def run(self, context):
        from mac_bench.evaluation.episodes.identity import call_scope
        from src.task import Task
        CTX["context"] = context
        self.last_chain = None
        task = Task(task_id=self.counter, task_type="mac_" + self.category, complexity=0.3,
                    major_problem=context.task.instruction, correct_answer="")
        self.counter += 1
        context.event({"kind": "agent_start", "agent_id": "agentnet"})
        try:
            with call_scope(agent_id="agentnet"):
                chain = self._solve(task)
            return {"final_response": chain.get_final_result() or "",
                    "trace": [{"mode": h.mode, "agent": h.current_agent_id} for h in chain.task_history]}
        finally:
            context.event({"kind": "agent_end", "agent_id": "agentnet"})

    def _solve(self, task):
        # if MAC-Bench stops the episode on budget, no chain is returned and that
        # episode is not used for evolution (feedback() returns None)
        chain = self.experiment.solve_a_single_task(task)
        self.last_chain = chain
        return chain

    def feedback(self, result):
        """Call after run_episode: evolve AgentNet with the MAC-Bench score."""
        if self.last_chain is None:
            return None
        ps = result.get("ps") or 0.0
        success = bool(result.get("rr") == 1 or result.get("rr") == 1.0 or ps >= self.success_ps)
        self.experiment.update_agent_graph(self.last_chain, {
            "task_chain_id": self.last_chain.task_chain_id, "raw_result": self.last_chain.get_final_result(),
            "final_result": "", "correct_answer": "", "success": success})
        return success
