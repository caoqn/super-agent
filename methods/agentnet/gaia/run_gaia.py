"""Run AgentNet on GAIA and record accuracy, cost and time.

  python run_gaia.py                       # validation, all 165 tasks, online evolution
  python run_gaia.py --limit 3             # smoke test
  python run_gaia.py --resume save/gaia_XXXX   # continue an interrupted run

Modes
  online      (default) tasks are solved in order; after each task AgentNet updates
              experience pools / abilities / edge weights (its `fit` step), scored online.
  train_test  first --train_ratio of tasks are used for evolution (fit), the rest are
              solved with the evolved network without updates (like run_bigbenchhard_train_test.py).
"""
import argparse
import copy
import datetime
import json
import logging
import os
import pickle
import random
import sys
import time
import types

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from gaia.envfile import load_env  # noqa: E402

# 打包后本包即根目录：.env 与 GAIA 数据（2023/validation）都放在本包下
FUXIAN = HERE
load_env([os.path.join(HERE, ".env"), os.path.join(os.path.dirname(HERE), ".env")])

try:  # agentgraph.py imports networkx but never uses it
    import networkx  # noqa: F401
except ImportError:
    sys.modules["networkx"] = types.ModuleType("networkx")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default=FUXIAN, help="folder that contains 2023/validation")
    p.add_argument("--split", default="validation")
    p.add_argument("--data_file", default=None, help="override: a metadata .parquet or .jsonl")
    p.add_argument("--levels", default="1,2,3")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--mode", choices=["online", "train_test"], default="online")
    p.add_argument("--train_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--config", default=os.path.join(HERE, "config/experiment/gaia.yaml"))
    p.add_argument("--no_python", action="store_true", help="disable the python tool")
    p.add_argument("--no_tools", action="store_true", help="pure AgentNet (no tools at all)")
    p.add_argument("--global_router_experience", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", default=None, help="existing output dir to continue")
    return p.parse_args()


def setup_logging(path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    root.addHandler(sh)


def fmt_money(x, cur):
    return f"{x:.4f} {cur}"


def main():
    args = get_args()
    if args.no_tools:
        os.environ["GAIA_TOOLS"] = "0"
    random.seed(args.seed)

    out = args.resume or args.out or os.path.join(
        HERE, "save", "gaia_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    setup_logging(os.path.join(out, "agentnet.log"))
    log = logging.getLogger("run_gaia")

    # ---- imports that need env / sys.path ready (pool.py loads the encoder here)
    from gaia.tracker import TRACKER
    TRACKER.reload_prices()
    TRACKER.set_log_path(os.path.join(out, "calls.jsonl"))
    t_load = time.time()
    from gaia.dataset import load_gaia, register_task_types, to_agentnet_task
    register_task_types()
    import src.agent as agent_mod
    from gaia.executor import CURRENT, executor_response, plain_response
    from gaia.prompts import GAIAPromptSet
    from gaia.scorer import question_scorer
    from gaia.tools import ToolBox
    from src.experiment import Experiment
    from src.utils import read_yaml
    from utils.Result_Extractor import extract_answer
    agent_mod.llm_model_response_map["openai_compat"] = plain_response
    encoder_load_s = time.time() - t_load

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    rows = load_gaia(args.data_root, args.split, levels, args.data_file)
    rows = rows[args.start:]
    if args.limit:
        rows = rows[:args.limit]
    n_train = len(rows) if args.mode == "online" else int(round(len(rows) * args.train_ratio))

    cfg = read_yaml(args.config)
    exp_cfg, agent_cfg, graph_cfg = cfg["experiment_config"], cfg["agent_config"], cfg["agent_graph_config"]
    exp_cfg["global_router_experience"] = args.global_router_experience
    graph_cfg["global_router_experience"] = args.global_router_experience
    assert exp_cfg["agent_num"] == len(agent_cfg)

    ckpt = os.path.join(out, "checkpoint.pkl")
    run_meta_path = os.path.join(out, "run_config.json")
    done = {}
    experiment = Experiment(
        exp_cfg, agent_cfg, graph_cfg, [], [],
        json_file_path_task_result=os.path.join(out, "agentnet_task_result.json"),
        json_file_path_agent_info=os.path.join(out, "agentnet_ability.json"),
        json_file_path_edge_weight=os.path.join(out, "agentnet_edge_weight.json"),
        json_file_path_task_history=os.path.join(out, "agentnet_task_history.json"),
        json_file_path_experience=os.path.join(out, "agentnet_experience.json"),
        constraints=GAIAPromptSet.get_constraint(),
        split_constraints=GAIAPromptSet.get_split_constraint(),
        thought_constraints=GAIAPromptSet.get_thought_constraint(),
    )
    if args.resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        experiment.agent_graph = state["agent_graph"]
        experiment.train_accuracy = state["train_accuracy"]
        experiment.test_accuracy = state["test_accuracy"]
        for k in ("task_result_record", "agent_info_record", "edge_weight_record",
                  "agent_experiences_record", "task_history_record"):
            setattr(experiment, k, state.get(k, []))
        print(f"resumed from {ckpt}")
    results_path = os.path.join(out, "results.jsonl")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                done[r["gaia_task_id"]] = r
    for a in experiment.agent_graph.agents.values():
        a.router_module.get_llm_response = plain_response
        a.executor_module.get_response = executor_response

    with open(run_meta_path, "w") as f:
        json.dump({
            "argv": sys.argv, "mode": args.mode, "split": args.split, "levels": levels,
            "n_tasks": len(rows), "n_train": n_train,
            "llm_base_url": os.environ.get("LLM_BASE_URL"), "llm_model": os.environ.get("LLM_MODEL"),
            "price_input_per_1m": TRACKER.price_in, "price_output_per_1m": TRACKER.price_out,
            "currency": TRACKER.currency, "embedding_backend": os.environ.get("EMBEDDING_BACKEND", "bge"),
            "tools": os.environ.get("GAIA_TOOLS", "1") != "0", "python_tool": not args.no_python,
            "native_tool_calls": os.environ.get("LLM_SUPPORTS_TOOLS", "1") == "1",
            "max_tool_steps": int(os.environ.get("GAIA_MAX_TOOL_STEPS", 10)),
            "agent_num": exp_cfg["agent_num"], "encoder_load_s": round(encoder_load_s, 2),
        }, f, indent=2, ensure_ascii=False)

    run_t0 = time.time()
    try:
        for i, row in enumerate(rows):
            if row["task_id"] in done:
                continue
            phase = "train" if i < n_train else "test"
            task = to_agentnet_task(i, row)
            box = ToolBox(row["file_path"] or None, enable_python=not args.no_python)
            CURRENT["toolbox"] = box
            err, raw, pred, chain = None, "", None, None
            t0 = time.time()
            with TRACKER.task_scope(i):
                try:
                    chain = experiment.solve_a_single_task(task)
                    raw = chain.get_final_result() or ""
                    pred = extract_answer(raw).strip()
                except Exception as e:  # keep the run going; the task counts as wrong
                    err = f"{type(e).__name__}: {e}"
                    log.exception("task %s failed", row["task_id"])
                    # Agent.execute_task increments current_load and never decrements it when
                    # an exception escapes; reset so a failed task does not distort later routing
                    for a in experiment.agent_graph.agents.values():
                        a.current_load = 0
                if err and ("HTTP 401" in err or "HTTP 403" in err):
                    raise SystemExit(f"\nAPI key / auth error, stopping instead of failing every task:\n{err[:300]}")
                wall = time.time() - t0
                success = bool(pred is not None and question_scorer(pred, row["answer"]))
                acc = experiment.train_accuracy if phase == "train" else experiment.test_accuracy
                acc.update(success)
                t_upd = time.time()
                if phase == "train" and chain is not None:
                    try:
                        experiment.update_agent_graph(chain, {
                            "task_chain_id": chain.task_chain_id, "raw_result": raw,
                            "final_result": pred, "correct_answer": row["answer"], "success": success})
                    except Exception as e:
                        log.exception("update_agent_graph failed on %s", row["task_id"])
                        err = (err or "") + f" | update failed: {e}"
                update_s = time.time() - t_upd
            box.cleanup()
            CURRENT["toolbox"] = None
            usage = TRACKER.summarize(i)
            trace = []
            if chain is not None:
                trace = [{"mode": h.mode, "agent": h.current_agent_id} for h in chain.task_history]
            rec = {
                "idx": i, "gaia_task_id": row["task_id"], "level": row["level"], "phase": phase,
                "task_type": task.task_type, "has_file": bool(row["file_name"]),
                "question": row["question"], "ground_truth": row["answer"],
                "prediction": pred, "correct": success, "error": err,
                "wall_time_s": round(wall, 2), "update_time_s": round(update_s, 2),
                "agentnet_trace": trace, **usage,
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[row["task_id"]] = rec
            try:
                with open(ckpt + ".tmp", "wb") as f:
                    pickle.dump({
                        "agent_graph": experiment.agent_graph,
                        "train_accuracy": experiment.train_accuracy, "test_accuracy": experiment.test_accuracy,
                        "task_result_record": experiment.task_result_record,
                        "agent_info_record": experiment.agent_info_record,
                        "edge_weight_record": experiment.edge_weight_record,
                        "agent_experiences_record": experiment.agent_experiences_record,
                        "task_history_record": experiment.task_history_record,
                    }, f)
                os.replace(ckpt + ".tmp", ckpt)
            except Exception as e:
                log.warning("checkpoint failed: %s", e)
            n_done = len(done)
            # Keep an aggregate on disk at all times. summary.md/json used to be written
            # only in the finally block, so a hard kill (SIGKILL / process-group kill)
            # left no aggregate even though results.jsonl was intact.
            if n_done % 10 == 0:
                try:
                    write_summary(out, list(done.values()), TRACKER.currency, time.time() - run_t0)
                except Exception as e:
                    log.warning("interim summary failed: %s", e)
            n_ok = sum(r["correct"] for r in done.values())
            print(f"[{n_done}/{len(rows)}] L{row['level']} {phase} {'OK ' if success else 'X  '} "
                  f"pred={str(pred)[:40]!r} gt={row['answer'][:40]!r} | {wall:.0f}s "
                  f"{usage['llm_calls']} calls {usage['prompt_tokens'] + usage['completion_tokens']} tok "
                  f"{fmt_money(usage['total_cost'], TRACKER.currency)} | acc {n_ok}/{n_done}"
                  + (f" | ERR {err[:80]}" if err else ""), flush=True)
    except KeyboardInterrupt:
        print("interrupted; writing summary of finished tasks (resume with --resume)")
    finally:
        write_summary(out, list(done.values()), TRACKER.currency, time.time() - run_t0)


def write_summary(out, recs, currency, session_wall):
    if not recs:
        return
    recs = sorted(recs, key=lambda r: r["idx"])

    def agg(rs):
        n = len(rs)
        if n == 0:
            return None
        s = lambda k: sum(r[k] for r in rs)  # noqa: E731
        return {
            "n": n, "correct": s("correct"), "accuracy": s("correct") / n,
            "errors": sum(1 for r in rs if r["error"]),
            "total_cost": s("total_cost"), "llm_cost": s("llm_cost"), "embedding_cost": s("embedding_cost"),
            "avg_cost_per_task": s("total_cost") / n,
            "prompt_tokens": s("prompt_tokens"), "completion_tokens": s("completion_tokens"),
            "avg_tokens_per_task": (s("prompt_tokens") + s("completion_tokens")) / n,
            "llm_calls": s("llm_calls"), "avg_llm_calls_per_task": s("llm_calls") / n,
            "tool_calls": s("tool_calls"),
            "total_wall_time_s": s("wall_time_s") + s("update_time_s"),
            "avg_wall_time_per_task_s": (s("wall_time_s") + s("update_time_s")) / n,
            "median_wall_time_s": sorted(r["wall_time_s"] for r in rs)[n // 2],
            "llm_time_s": s("llm_time_s"), "tool_time_s": s("tool_time_s"),
            "embedding_time_s": s("embedding_time_s"),
            "tokens_estimated_calls": s("tokens_estimated_calls"),
        }

    by_role = {}
    for r in recs:
        for role, v in r.get("by_role", {}).items():
            b = by_role.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0})
            for k in b:
                b[k] += v[k]
    summary = {
        "currency": currency,
        "overall": agg(recs),
        "by_phase": {p: agg([r for r in recs if r["phase"] == p]) for p in ("train", "test") if any(r["phase"] == p for r in recs)},
        "by_level": {f"L{l}": agg([r for r in recs if r["level"] == l]) for l in (1, 2, 3) if any(r["level"] == l for r in recs)},
        "by_role": by_role,
        "this_session_wall_time_s": round(session_wall, 1),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def line(name, a):
        return (f"| {name} | {a['n']} | {a['correct']} ({a['accuracy']:.1%}) | {a['total_cost']:.4f} | "
                f"{a['avg_cost_per_task']:.4f} | {a['prompt_tokens']:,} / {a['completion_tokens']:,} | "
                f"{a['avg_llm_calls_per_task']:.1f} | {a['total_wall_time_s'] / 60:.1f} | "
                f"{a['avg_wall_time_per_task_s']:.1f} | {a['errors']} |")
    md = [f"# AgentNet on GAIA — results ({currency})", "",
          "| subset | tasks | correct (acc) | total cost | cost/task | input / output tokens | LLM calls/task | total time (min) | time/task (s) | errors |",
          "|---|---|---|---|---|---|---|---|---|---|",
          line("all", summary["overall"])]
    for k, v in summary["by_level"].items():
        md.append(line(k, v))
    for k, v in summary["by_phase"].items():
        md.append(line(k, v))
    o = summary["overall"]
    md += ["", f"Time split (summed over tasks): LLM {o['llm_time_s'] / 60:.1f} min, tools {o['tool_time_s'] / 60:.1f} min, "
               f"embedding {o['embedding_time_s'] / 60:.1f} min.", "",
           "| module | calls | input tokens | output tokens | cost |", "|---|---|---|---|---|"]
    for role, b in sorted(by_role.items(), key=lambda x: -x[1]["cost"]):
        md.append(f"| {role} | {b['calls']} | {b['prompt_tokens']:,} | {b['completion_tokens']:,} | {b['cost']:.4f} |")
    if o["tokens_estimated_calls"]:
        md.append(f"\nNote: {o['tokens_estimated_calls']} calls had no usage field; their tokens were estimated (chars/4).")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
