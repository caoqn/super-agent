"""Run AOrchestra MainAgent and ReActAgent subagents on 90 MASBench tasks.

Usage:
    python run_masbench.py --task reasoning|scheduling|game|all
                          [--limit N] [--model deepseek-v4.1-flash]
                          [--max-attempts 5] [--concurrency 1]

This runner does not modify official AOrchestra files. Configure the LLM through AUTOENV_* environment variables.
Scoring follows the official MASBench protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Paths.
HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
AORCHESTRA = REPO_ROOT / "AOrchestra-main"
MASBENCH = REPO_ROOT / "MASBench-main"
for p in [str(AORCHESTRA), str(MASBENCH / "Reasoning"), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")


def _load_env() -> dict:
    env = {}
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return env
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.split(" #")[0].strip()
    return env


_ENV = {**_load_env(), **os.environ}
DEFAULT_BASE = _ENV.get("LLM_BASE_URL", "").rstrip("/")
DEFAULT_KEY = _ENV.get("LLM_API_KEY", "")
DEFAULT_MODEL = _ENV.get("LLM_MODEL", "deepseek-v4.1-flash")


def _setup_llm_env(model: str, base: str, key: str):
    """Expose model settings through the environment for LLMsConfig.default()."""
    os.environ.setdefault("AUTOENV_OPENAI_API_KEY", key)
    os.environ.setdefault("AUTOENV_OPENAI_BASE_URL", base)
    os.environ.setdefault("AUTOENV_OPENAI_MODELS", model)
    os.environ.setdefault("AUTOENV_OPENAI_TEMPERATURE", "0")
    os.environ.setdefault("AUTOENV_OPENAI_TOP_P", "1")


# Increase the SDK default five-second connection timeout for slower gateways.
# Set the client default at the adapter entry point; upstream async_llm.py supplies no timeout.
import httpx  # noqa: E402
import openai._base_client as _obc  # noqa: E402
_obc.DEFAULT_TIMEOUT = httpx.Timeout(timeout=600.0, connect=60.0, read=600.0, write=600.0, pool=60.0)

from base.engine.logs import logger, LogLevel  # noqa: E402
from base.engine.cost_monitor import CostMonitor  # noqa: E402
from base.engine.async_llm import LLMsConfig, create_llm_instance  # noqa: E402
from benchmark.common.env import BasicInfo, Environment  # noqa: E402
from benchmark.common.runner import Runner, StepRecord, LevelResult  # noqa: E402
from aorchestra.main_agent import MainAgent  # noqa: E402
from aorchestra.tools.delegate import DelegateTaskTool  # noqa: E402
from aorchestra.tools.complete import CompleteTool  # noqa: E402
from src.metrics import cal_f1_score  # noqa: E402


class SubAgentRunner(Runner):
    """Run the agent-environment loop with current_step and max_steps.

    Do not import aorchestra.runners: its __init__.py also imports swebench_runner
    -> benchmark.benchmark, which fails on Python 3.10 because `Type[Agent]` is given
    a module instead of a type. Inherit directly from benchmark.common.runner.Runner
    and reproduce its logic while passing current_step/max_steps correctly.
    """

    async def run(self, agent, env: Environment) -> LevelResult:
        import inspect
        logger.info("[MASBench SubAgentRunner] Starting SubAgent execution")
        info = env.get_basic_info()
        agent.reset(info)
        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result
        history = []
        total_reward = 0.0
        max_steps = info.max_steps
        for t in range(max_steps):
            current_step = t + 1
            step_result = await agent.step(
                observation=obs, history=history,
                current_step=current_step, max_steps=max_steps,
            )
            if isinstance(step_result, (list, tuple)):
                if len(step_result) == 3:
                    action, raw_response, raw_input = step_result
                else:
                    action, raw_response = step_result
                    raw_input = None
            else:
                raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")
            obs_next, reward, done, step_info = await env.step(action)
            step_record = StepRecord(
                observation=obs, action=action, reward=reward,
                raw_response=raw_response, done=done, info=step_info, raw_input=raw_input,
            )
            history.append(step_record)
            total_reward += reward
            obs = obs_next
            if done:
                break
        usage_summary = agent.llm.get_usage_summary()
        return LevelResult(
            model=usage_summary.get("model", ""),
            total_reward=total_reward,
            steps=len(history),
            done=history[-1].done if history else False,
            trace=history,
            cost=usage_summary.get("total_cost", 0.0),
            input_tokens=usage_summary.get("total_input_tokens", 0),
            output_tokens=usage_summary.get("total_output_tokens", 0),
        )

from masbench_env import ReasoningEnv, SchedulingEnv, GameEnv  # noqa: E402
from masbench_prompt import MASBenchMainAgentPrompt  # noqa: E402

# Load task data.
SEL = MASBENCH / "MASBench_selected_30"


def load_levels(task: str):
    if task == "reasoning":
        data = json.loads((SEL / "Reasoning/reasoning_subset.json").read_text(encoding="utf-8"))
        return [{"task": "reasoning", "item": it} for it in data]
    if task == "scheduling":
        dirs = sorted((SEL / "Scheduling/scenarios").iterdir())
        return [{"task": "scheduling", "scenario": d} for d in dirs]
    if task == "game":
        maps = sorted((SEL / "Game/maps").glob("*.json"))
        return [{"task": "game", "map": m} for m in maps]
    raise ValueError(task)


def make_env(spec):
    t = spec["task"]
    if t == "reasoning":
        return ReasoningEnv(spec["item"])
    if t == "scheduling":
        return SchedulingEnv(spec["scenario"])
    if t == "game":
        return GameEnv(spec["map"])
    raise ValueError(t)


def spec_id(spec) -> str:
    t = spec["task"]
    if t == "reasoning":
        return spec["item"].get("id")
    if t == "scheduling":
        return spec["scenario"].name
    if t == "game":
        return spec["map"].name
    return str(spec)


async def run_one(spec, main_model, sub_models, max_attempts, sub_runner, cost_monitor):
    env = make_env(spec)
    tid = spec_id(spec)
    start = time.time()

    main_llm = create_llm_instance(LLMsConfig.default().get(main_model))

    delegate_tool = DelegateTaskTool(
        env=env,
        runner=sub_runner,
        models=sub_models,
        benchmark_type="gaia",
        alias_to_model={m: m for m in sub_models},
    )
    complete_tool = CompleteTool()

    main_agent = MainAgent(
        llm=main_llm,
        sub_models=sub_models,
        tools=[delegate_tool, complete_tool],
        prompt_builder=MASBenchMainAgentPrompt,
        max_attempts=max_attempts,
        benchmark_type="gaia",
    )

    basic_info = env.get_basic_info()
    main_info = BasicInfo(
        env_id=basic_info.env_id,
        instruction=basic_info.instruction,
        action_space="",
        max_steps=max_attempts,
        meta_data=basic_info.meta_data,
    )
    main_agent.reset(main_info)

    final_answer = None
    attempts_detail = []

    async def _step_retry(agent, retries=2):
        """Retry MainAgent.step when an empty response causes upstream JSON parsing to fail."""
        last = None
        for r in range(retries + 1):
            try:
                return await agent.step(None, [])
            except Exception as e:
                last = e
                logger.warning(f"[MASBench] MainAgent.step error (attempt {r+1}): {type(e).__name__}: {str(e)[:200]}")
                await asyncio.sleep(2)
        raise last

    for attempt_idx in range(max_attempts):
        try:
            action, resp = await _step_retry(main_agent)
        except Exception as e:
            attempts_detail.append({
                "attempt": attempt_idx + 1, "action": "error",
                "params": {}, "result": {"error": str(e)},
            })
            break
        action_name = action.get("action")
        params = action.get("params", {})
        result = action.get("result", {})
        attempts_detail.append({
            "attempt": attempt_idx + 1,
            "action": action_name,
            "params": params,
            "result": {k: v for k, v in result.items() if k != "trace"},
        })
        if action_name == "complete":
            final_answer = params.get("answer")
            break
        if action_name == "error":
            break

    wall = time.time() - start

    # Score the result.
    record = {
        "task": spec["task"],
        "task_id": tid,
        "wall_s": round(wall, 1),
        "attempts": len(attempts_detail),
        "final_answer": final_answer,
        "main_cost": main_agent.get_usage_cost(),
    }
    if spec["task"] == "reasoning":
        gt = spec["item"]["answer"]
        f1 = cal_f1_score(gt, final_answer) if final_answer else None
        record.update({"f1": f1, "ground_truth": gt})
    elif spec["task"] == "scheduling":
        record["status"] = env.evaluate(final_answer)
        record["scenario"] = env.scenario_id
    elif spec["task"] == "game":
        record.update({
            "score": env.best_score,
            "delivered": env.best_delivered,
            "target": env.target,
        })

    # CostMonitor includes manager, subagent, and trace-summary model calls.
    record["total_input_tokens"] = cost_monitor.total_input_tokens
    record["total_output_tokens"] = cost_monitor.total_output_tokens
    record["total_tokens"] = cost_monitor.total_input_tokens + cost_monitor.total_output_tokens
    record["total_cost"] = round(cost_monitor.total_cost, 6)
    record["call_count"] = cost_monitor.call_count
    return record


def load_results(task: str) -> dict:
    p = HERE / "results" / f"{task}.jsonl"
    seen = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen[r.get("task_id")] = r
            except Exception:
                pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["reasoning", "scheduling", "game", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()

    _setup_llm_env(args.model, DEFAULT_BASE, DEFAULT_KEY)
    # Remove proxy variables containing [::1] to avoid httpx parsing errors.
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy", "grpc_proxy"]:
        os.environ.pop(k, None)

    tasks = ["reasoning", "scheduling", "game"] if args.task == "all" else [args.task]

    (HERE / "results").mkdir(exist_ok=True)

    for task in tasks:
        levels = load_levels(task)
        if args.limit:
            levels = levels[: args.limit]
        seen = load_results(task)
        pending = [l for l in levels if spec_id(l) not in seen]
        print(f"\n=== [{task}] {len(levels)} tasks; completed {len(seen)}; pending {len(pending)} ===", flush=True)
        if not pending:
            continue

        out_path = HERE / "results" / f"{task}.jsonl"
        sub_runner = SubAgentRunner()
        # Pass current_step/max_steps; env.get_basic_info().max_steps bounds subagent steps.

        async def run_pending():
            sem = asyncio.Semaphore(max(1, args.concurrency))
            lock = asyncio.Lock()
            done_count = 0

            async def one(spec):
                nonlocal done_count
                async with sem:
                    with CostMonitor() as cm:
                        rec = await run_one(
                            spec, args.model, [args.model],
                            args.max_attempts, sub_runner, cm,
                        )
                        rec["model"] = args.model
                        rec["timestamp"] = datetime.now().isoformat()
                    async with lock:
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done_count += 1
                        metric = rec.get("f1") if task == "reasoning" else (rec.get("status") if task == "scheduling" else rec.get("score"))
                        print(f"[{done_count}/{len(pending)}] {rec['task_id']} -> {metric} "
                              f"tok={rec['total_tokens']} cost={rec['total_cost']} {rec['wall_s']}s", flush=True)

            await asyncio.gather(*(one(spec) for spec in pending))

        asyncio.run(run_pending())

    print("\nCompleted all tasks. Results are under results/.")


if __name__ == "__main__":
    main()
