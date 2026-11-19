"""Run AgentNet on MAC-Bench and record PS / RR, tokens, cost and time.

Prerequisites (see README_BENCH.md): Python 3.11+, `pip install -e '../../MAC_Bench-1.0[modal,agents]'`,
`modal token new`, task data downloaded with `mac-bench data download`, and in .env:
MAC_BENCH_LLM_BASE_URL / MAC_BENCH_LLM_API_KEY / MAC_BENCH_LLM_MODEL (+ LLM_PRICE_* for cost).

  python run_macbench.py --task-ids op26_service_administration_007      # one dev task
  python run_macbench.py --split test                                     # all 48 test tasks
  python run_macbench.py --split test --warmup-dev                        # evolve on 6 dev tasks first
"""
import argparse
import datetime
import json
import os
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FUXIAN = os.path.abspath(os.path.join(HERE, ".."))
# MAC_Bench-1.0 的位置：默认在父目录下，也可用 MAC_BENCH_ROOT 环境变量显式指定
MAC_ROOT = os.environ.get("MAC_BENCH_ROOT", os.path.join(FUXIAN, "MAC_Bench-1.0"))

from gaia.envfile import load_env  # noqa: E402

load_env([os.path.join(HERE, ".env"), os.path.join(FUXIAN, ".env"), os.path.join(MAC_ROOT, ".env")])
for _k in ("BASE_URL", "API_KEY", "MODEL"):  # reuse the GAIA LLM settings if MAC_BENCH_* are not set
    if os.environ.get("LLM_" + _k) and not os.environ.get("MAC_BENCH_LLM_" + _k):
        os.environ["MAC_BENCH_LLM_" + _k] = os.environ["LLM_" + _k]
try:
    import networkx  # noqa: F401
except ImportError:
    sys.modules["networkx"] = types.ModuleType("networkx")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["test", "dev", "all"])
    p.add_argument("--task-ids", default="", help="comma separated; overrides --split")
    p.add_argument("--warmup-dev", action="store_true", help="run (and evolve on) the 6 dev tasks first")
    p.add_argument("--max-tokens", type=int, default=1_000_000)
    p.add_argument("--max-model-calls", type=int, default=80)
    p.add_argument("--max-tool-calls", type=int, default=160)
    p.add_argument("--max-seconds", type=int, default=2400)
    p.add_argument("--success-ps", type=float, default=1.0, help="PS threshold counted as success for evolution")
    p.add_argument("--config", default=os.path.join(HERE, "config/experiment/gaia.yaml"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or os.path.join(HERE, "save", "macbench_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    import logging
    logging.basicConfig(filename=os.path.join(out, "agentnet.log"), level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from mac_bench.evaluation.episodes.runner import run_episode
    from mac_bench.evaluation.tasks import load_catalog, open_task
    from mac_bench.runtime.agents.workflow import AgentBudget
    from mac_bench.runtime.llm.client import LiteLLMChatClient
    from mac_bench.runtime.llm.config import LLMConfig

    from gaia.tracker import TRACKER
    TRACKER.reload_prices()
    TRACKER.set_log_path(os.path.join(out, "calls.jsonl"))
    import src.agent as agent_mod
    from gaia import macbench_adapter as mb
    from src.experiment import Experiment
    from src.utils import read_yaml

    catalog = os.path.join(MAC_ROOT, "data", "tasks.json")
    # pick ids from the raw catalogue first: load_catalog validates bundles, which only
    # exist for tasks already fetched with `mac-bench data download`
    items = json.load(open(catalog))["tasks"]
    mb.register_task_types(sorted({t["category"] for t in items}))
    if args.task_ids:
        ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
    else:
        dev = [t["task_id"] for t in items if t["split"] == "dev"]
        test = [t["task_id"] for t in items if t["split"] == "test"]
        ids = {"test": (dev if args.warmup_dev else []) + test, "dev": dev, "all": dev + test}[args.split]
    by_id = {r.task_id: r for r in load_catalog(catalog, task_ids=ids)}
    records = [by_id[i] for i in ids]

    cfg = read_yaml(args.config)
    for a in cfg["agent_config"]:
        a["config"]["abilities"] = {k: 0.6 for k in mb.ABILITIES}
        a["config"]["LLM"] = "macbench_client"
    cfg["experiment_config"]["global_router_experience"] = False
    cfg["agent_graph_config"]["global_router_experience"] = False
    agent_mod.llm_model_response_map["macbench_client"] = mb.plain_response
    experiment = Experiment(
        cfg["experiment_config"], cfg["agent_config"], cfg["agent_graph_config"], [], [],
        json_file_path_task_result=os.path.join(out, "agentnet_task_result.json"),
        json_file_path_agent_info=os.path.join(out, "agentnet_ability.json"),
        json_file_path_edge_weight=os.path.join(out, "agentnet_edge_weight.json"),
        json_file_path_task_history=os.path.join(out, "agentnet_task_history.json"),
        json_file_path_experience=os.path.join(out, "agentnet_experience.json"),
        constraints=mb.MACConstraints.get_constraint(),
        split_constraints=mb.MACConstraints.get_split_constraint(),
        thought_constraints=mb.MACConstraints.get_thought_constraint())
    adapter = mb.AgentNetMACAdapter(experiment, success_ps=args.success_ps)
    client = LiteLLMChatClient(LLMConfig.from_env(require_api_key=True))
    results_path = os.path.join(out, "results.jsonl")

    for i, rec in enumerate(records):
        adapter.category = rec.category
        budget = AgentBudget(max_tokens=args.max_tokens, max_model_calls=args.max_model_calls,
                             max_tool_calls=args.max_tool_calls, max_env_actions=args.max_tool_calls,
                             max_wall_seconds=args.max_seconds)
        ep_dir = os.path.join(out, rec.task_id)
        t0 = time.time()
        with TRACKER.task_scope(rec.task_id):
            # 本地 Docker 沙盒模式：设 MAC_BENCH_LOCAL_DOCKER=1 时用本地 docker 容器替代 Modal 云
            if os.environ.get("MAC_BENCH_LOCAL_DOCKER") == "1":
                from gaia.docker_backend import DockerSandboxBackend
                task = open_task(rec, os.path.join(ep_dir, "runtime"), backend=DockerSandboxBackend())
            else:
                task = open_task(rec, os.path.join(ep_dir, "runtime"))
            res = run_episode(task, adapter, client, output=os.path.join(ep_dir, "episode"), budget=budget)
            evolved = None
            try:
                evolved = adapter.feedback(res)
            except Exception as e:
                logging.exception("feedback failed: %s", e)
        usage = TRACKER.summarize(rec.task_id)
        row = {"task_id": rec.task_id, "category": rec.category, "split": rec.split,
               "status": res.get("status"), "ps": res.get("ps"), "rr": res.get("rr"),
               "stop_reason": res.get("stop_reason"), "elapsed_seconds": res.get("elapsed_seconds"),
               "wall_incl_setup_s": round(time.time() - t0, 1),
               "input_tokens": res.get("input_tokens"), "output_tokens": res.get("output_tokens"),
               "model_calls": res.get("model_calls"), "tool_calls": res.get("tool_calls"),
               "cost": TRACKER.llm_cost(res.get("input_tokens") or 0, res.get("output_tokens") or 0),
               "evolved_as_success": evolved, "by_role": usage["by_role"]}
        with open(results_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(records)}] {rec.task_id} PS={row['ps']} RR={row['rr']} {row['status']} "
              f"{row['elapsed_seconds'] or 0:.0f}s tok={row['input_tokens']}/{row['output_tokens']} "
              f"cost={row['cost']:.4f} {TRACKER.currency}", flush=True)

    rows = [json.loads(l) for l in open(results_path)]
    test_rows = [r for r in rows if r["split"] == "test"] or rows
    n = len(test_rows)
    summ = {"n": n, "currency": TRACKER.currency,
            "mean_ps": sum((r["ps"] or 0) for r in test_rows) / n, "mean_rr": sum((r["rr"] or 0) for r in test_rows) / n,
            "total_cost": sum(r["cost"] for r in test_rows), "avg_cost": sum(r["cost"] for r in test_rows) / n,
            "total_time_s": sum((r["elapsed_seconds"] or 0) for r in test_rows),
            "avg_time_s": sum((r["elapsed_seconds"] or 0) for r in test_rows) / n,
            "input_tokens": sum((r["input_tokens"] or 0) for r in test_rows),
            "output_tokens": sum((r["output_tokens"] or 0) for r in test_rows)}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
