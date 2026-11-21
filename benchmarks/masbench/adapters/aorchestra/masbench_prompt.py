"""Build the MASBench MainAgent prompt.

The MainAgent decomposes each task and delegates subtasks to ReActAgent subagents.
Subagents report with finish, and the MainAgent submits the final answer with complete.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def _format_tools(tools: List[Any]) -> str:
    if not tools:
        return "No tools available."
    parts = []
    for t in tools:
        parts.append(f"- {t.name}: {t.description}")
    return "\n".join(parts)


class MASBenchMainAgentPrompt:
    """Build a shared manager prompt for Reasoning, Scheduling, and Game."""

    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        tools: List[Any] = None,
    ) -> str:
        remaining = max_attempts - attempt_index
        return f"""You are the MainAgent (Orchestrator). Solve the given TASK by decomposing it and delegating subtasks to SubAgents, then submit a final answer.

DECISION PROCESS:
1. READ the TASK carefully — identify what information or actions are needed.
2. Use 'delegate_task' to have a SubAgent gather evidence or perform environment actions. The SubAgent has access to the tools described in the TASK.
3. REVIEW the subtask result, then use 'complete' with the final answer once sufficient.

BUDGET AWARENESS:
- You have LIMITED attempts (see Progress below). Make each delegation count.

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining} attempts

==== TASK ====
{instruction}

==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== AVAILABLE TOOLS (your actions) ====
- delegate_task: Assign a subtask to a SubAgent. The SubAgent can use the tools described in the TASK (e.g. lookup evidence, query free slots, move/pick/put in the grid world).
- complete: Submit your final answer.

==== OUTPUT ====
You MUST reply with ONLY a valid JSON object (no markdown, no prose outside JSON). Exactly one of:

1) Delegate a subtask:
{{"action": "delegate_task", "reasoning": "...", "params": {{"task_instruction": "<specific actionable subtask>", "context": "<findings from previous attempts>", "model": "{sub_models[0] if sub_models else 'deepseek-v4.1-flash'}"}}}}

2) Submit the final answer:
{{"action": "complete", "reasoning": "...", "params": {{"answer": "<final answer>"}}}}

Use 'complete' when you have enough information to answer. Do not delegate forever.
""".strip()
