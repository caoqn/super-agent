"""Isolated JSON-lines RPC worker for upstream MASBench Game physics."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    source_root, map_path, output_dir, dependency_dir = sys.argv[1:5]
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    sys.path.insert(0, dependency_dir)
    sys.path.insert(0, str(Path(source_root) / "Game"))
    os.chdir(output_dir)

    class Brain:
        usage = {}

        def system(self, *_args, **_kwargs):
            pass

    from masbench.environment import MASEnvironment
    from masbench.agent import MASAgent
    from masbench.logger import MASLogger

    data = json.loads(Path(map_path).read_text(encoding="utf-8"))
    config = data["map"]
    agent_config = config.get("agent_config", {})
    names = [a.get("name", f"Agent{i + 1}") for i, a in enumerate(config["agents"])]
    agents = {
        name: MASAgent(
            name, Brain(), "Cooperate to collect and deliver resources.",
            enable_inventory=True,
            carrying_capacity=agent_config.get("item_number", 4),
            view_memory=0,
            history_memory=0,
        )
        for name in names
    }
    logger = MASLogger("metateam", {}, log_dir=str(Path(output_dir) / "native_logs"))
    env = MASEnvironment(
        "metateam", agents, logger, task="Foraging", max_round=50,
        width=config["width"], height=config["height"], custom_map=data,
        view_size=agent_config.get("view_size", 9),
        communication_size=agent_config.get("communication_size", 15),
        message_phase_enabled=False, messages_enabled=True,
    )
    pending = {}

    def status():
        totals = env.tasks[env.task].count_items_on_tables(env)
        targets = data["task"]["scoring"]
        earned = sum(min(int(totals.get(k, 0)), int(v)) for k, v in targets.items())
        maximum = sum(int(v) for v in targets.values())
        return {"round": env.round, "done": env.is_done(),
                "score": earned / maximum if maximum else 0.0,
                "earned": earned, "maximum": maximum}

    wire.write(json.dumps({"ok": True, "actors": names}) + "\n")
    wire.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            actor = request.get("actor")
            if actor not in agents and operation not in {"score", "close"}:
                raise ValueError(f"unknown actor: {actor}")
            if operation == "observe":
                result = {"round": env.round, "done": env.is_done(),
                          "view": agents[actor].gen_prompt(env.get_round_obs(agents[actor]), {}),
                          "pending": list(pending)}
            elif operation == "action":
                if env.is_done():
                    raise ValueError("episode finished")
                if actor in pending:
                    raise ValueError("one action per actor per round; wait for teammates")
                action = request["action"]
                if action.get("move") not in {"UP", "DOWN", "LEFT", "RIGHT", "STAY", "PICK_UP", "PUT_DOWN"}:
                    raise ValueError("invalid move")
                quantity = action.get("interaction_qty")
                if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1):
                    raise ValueError("interaction_qty must be a positive integer")
                if action.get("interaction_type") not in {None, "wood", "sand", "stone"}:
                    raise ValueError("invalid interaction_type")
                if action.get("interaction_type") is not None and action.get("move") != "PICK_UP":
                    raise ValueError("interaction_type is only valid with PICK_UP")
                pending[actor] = action
                if set(pending) == set(names):
                    for name in names:
                        asyncio.run(env.act(agents[name], pending[name]))
                    env.update()
                    env.clear_round_obs_cache()
                    pending.clear()
                result = status()
            elif operation == "in_range":
                result = env.is_in_communication_range(actor, request["recipient"])
            elif operation == "score":
                result = status()
            elif operation == "close":
                break
            else:
                raise ValueError(f"unknown operation: {operation}")
            wire.write(json.dumps({"ok": True, "result": result}, default=str) + "\n")
            wire.flush()
        except Exception as exc:
            wire.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
            wire.flush()
    env.cleanup_output_logging()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
