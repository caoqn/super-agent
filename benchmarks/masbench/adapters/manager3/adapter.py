"""Shared MASBench task loading, role-scoped tools and evaluation."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re

from . import reasoning, scheduling
from .base import NativeEvalResult, NativeTask
from .game import GameRPC

SPLIT_DIR = Path(__file__).resolve().parents[2]
CATEGORIES = ("reasoning", "scheduling", "game")
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK_UP", "PUT_DOWN")
PROMPTS = {
    "scheduling": "Find a shared continuous 20-minute meeting for five members across five days, using their private calendars.",
    "game": "Cooperate to explore, collect and deliver resources in a partially observed grid world, within 50 simultaneous environment rounds.",
}
FINAL_FORMAT = {
    "reasoning": "FINAL ANSWER: <answer only>",
    "scheduling": 'FINAL ANSWER: {"day":1,"start_minute":600,"end_minute":620}',
    "game": "FINAL ANSWER: finished",
}


class MASBenchAdapter:
    name = "masbench"

    def __init__(self, source_root: str | Path | None = None):
        root = source_root or os.getenv("MASBENCH_ROOT")
        if not root:
            raise ValueError("Set MASBENCH_ROOT to the upstream MASBench directory")
        self.source_root = Path(root).expanduser().resolve()

    def load_tasks(self, source: Path | None = None, *, split: str = "test", max_tasks: int = 0) -> list[NativeTask]:
        if split not in {"train", "test"}:
            raise ValueError("MASBench split must be train or test")
        manifest = Path(source or SPLIT_DIR) / f"{split}_manifest.json"
        selected = json.loads(manifest.read_text(encoding="utf-8"))
        reasoning_path = self.source_root / "Reasoning/reasoning.json"
        reasoning_rows = {row["id"]: row for row in json.loads(reasoning_path.read_text(encoding="utf-8"))}
        tasks = []
        for category in CATEGORIES:
            ids = selected[category]["ids"]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {category} IDs in {manifest}")
            for source_id in ids:
                raw = {"category": category, "id": source_id, "source_root": str(self.source_root)}
                if category == "reasoning":
                    raw["row"] = reasoning_rows[source_id]
                    prompt = raw["row"]["question"]
                elif category == "scheduling":
                    layer, case = source_id.split("_")
                    path = self.source_root / "Scheduling/meeting/minutes_10/5days" / layer / case
                    for name in ("task_plans", *(f"Member{i}_final_schedule.json" for i in range(1, 6))):
                        if not (path / name).is_file():
                            raise FileNotFoundError(path / name)
                    raw["path"] = str(path)
                    prompt = PROMPTS[category]
                else:
                    path = self.source_root / "Game/custom_map" / source_id
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    raw["path"] = str(path)
                    prompt = PROMPTS[category]
                tasks.append(NativeTask(f"{category}__{source_id}", prompt, raw, split))
        return tasks[:max_tasks] if max_tasks else tasks

    def setup_runtime(self, task: NativeTask, context: dict) -> None:
        category = task.raw["category"]
        context["mas_task"] = task
        if category == "reasoning":
            context["private"] = reasoning.private_context(self.source_root, task.raw["id"])
        elif category == "scheduling":
            context["private"] = scheduling.private_context(Path(task.raw["path"]))
        else:
            out = Path(context.get("artifact_dir") or context["workspace"]) / "game"
            rpc = GameRPC(self.source_root, Path(task.raw["path"]), out)
            context["game_rpc"] = rpc
            context["private"] = dict.fromkeys(rpc.actors)
        context["inboxes"] = {name: [] for name in context["private"]}
        context["messages"] = []

    def build_prompt(self, task: NativeTask, context: dict) -> str:
        actors = list(context["private"])
        category = task.raw["category"]
        prompt = (
            task.prompt + "\n"
            f"Environment actor identities (use exact names or include exactly one identity as an underscore-delimited token in SubAgentRequest.role_name, e.g. collect_Alice_evidence): {json.dumps(actors)}. "
            "Each actor must run in native_tool mode. observe_private reveals ONLY the caller role's evidence/calendar/local view. "
            "Other roles have no private access. Use send_message and read_messages for explicit communication. "
            "Never read benchmark files or reference answers. Workflow planning and additional coordinator roles are allowed, but they have no private observations. "
        )
        if category == "game":
            prompt += (
                "For Game: repeatedly invoke every actor once per environment round, up to 50 rounds. "
                "Each actor observes its local view, communicates only within range, and submits one game_action with move "
                "UP/DOWN/LEFT/RIGHT/STAY/PICK_UP/PUT_DOWN; return immediately after submitting, do not submit twice within one actor invocation. "
                "Physics advances only after all actors submit. Continue until done=true; do not stop after one pass. "
            )
        return prompt + "Return " + FINAL_FORMAT[category]

    def bind_actor(self, payload: dict, context: dict) -> str:
        role = str(payload.get("role_name", "")) if isinstance(payload, dict) else ""
        matches = [
            name for name in context["private"]
            if re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])", role.lower())
        ]
        return matches[0] if len(matches) == 1 else role

    def tool_schemas(self, task: NativeTask, context: dict | None = None) -> list[dict]:
        def schema(name, description, properties=None, required=()):
            return {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": properties or {},
                               "required": list(required), "additionalProperties": False},
            }}

        tools = [
            schema("observe_private", "Observe only your own private evidence/calendar/local map; caller identity is bound by runtime."),
            schema("read_messages", "Read messages explicitly addressed to your role."),
            schema("send_message", "Send explicit text to a teammate; Game requires communication range.",
                   {"recipient": {"type": "string"}, "message": {"type": "string"}}, ("recipient", "message")),
        ]
        if task.raw["category"] == "game":
            tools.append(schema("game_action", "Submit your one action this round. World advances once all actors have acted.",
                                {"move": {"type": "string", "enum": list(ACTIONS)}}, ("move",)))
        return tools

    async def execute_scoped_tool(self, name: str, args: dict, context: dict, actor: str) -> str:
        if actor not in context["private"]:
            raise PermissionError("Role has no environment actor identity; use exact registered role_name")
        rpc = context.get("game_rpc")
        if name == "observe_private":
            result = await asyncio.to_thread(rpc.call, "observe", actor=actor) if rpc else context["private"][actor]
        elif name == "read_messages":
            result = list(context["inboxes"][actor])
        elif name == "send_message":
            recipient = args["recipient"]
            if recipient not in context["inboxes"]:
                raise ValueError("unknown recipient")
            if rpc and not await asyncio.to_thread(rpc.call, "in_range", actor=actor, recipient=recipient):
                raise PermissionError("recipient outside communication range")
            row = {"sender": actor, "recipient": recipient, "message": str(args["message"])}
            context["inboxes"][recipient].append(row)
            context["messages"].append(row)
            result = {"delivered": True}
        elif name == "game_action" and rpc:
            result = await asyncio.to_thread(rpc.call, "action", actor=actor, action={"move": args["move"]})
        else:
            raise ValueError("unavailable tool")
        return json.dumps(result, ensure_ascii=False, default=str)

    def evaluate(self, task: NativeTask, output: str, context: dict) -> NativeEvalResult:
        category = task.raw["category"]
        answer = re.split(r"(?i)final\s*answer\s*:", output or "")[-1].strip()
        if category == "reasoning":
            score, details = reasoning.evaluate(self.source_root, task.raw["row"]["answer"], answer)
        elif category == "scheduling":
            score, details = scheduling.evaluate(Path(task.raw["path"]), answer)
        else:
            details = context["game_rpc"].call("score")
            score = details["score"]
        return NativeEvalResult(score == 1.0, float(score), f"MASBench {category}", details)

    def teardown_runtime(self, task: NativeTask, context: dict) -> None:
        rpc = context.pop("game_rpc", None)
        if rpc:
            rpc.close()
