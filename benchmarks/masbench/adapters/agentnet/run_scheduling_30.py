"""Run 30 MASBench Scheduling scenarios with thinking disabled.

Runs simulate_meeting_qwen.py for each scenario with thinking disabled, then collects the evaluation, token usage, and elapsed time from meta.json.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from methods.agentnet.envfile import load_env  # noqa: E402
SCHED = ROOT / "Scheduling"
SCENARIOS = sorted((ROOT / "MASBench_selected_30/Scheduling/scenarios").iterdir())
PY = sys.executable
MODEL = "deepseek-v4.1-flash"
def main():
    load_env([ROOT / ".env", REPO_ROOT / ".env"])
    env = dict(os.environ)
    env["QWEN_OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
    env["QWEN_OPENAI_BASE"] = os.environ["LLM_BASE_URL"]

    out = ROOT / "MASBench_selected_30/scheduling_results.jsonl"
    recs = []
    for i, sc in enumerate(SCENARIOS):
        before = set(glob.glob(str(SCHED / "logs/meeting/*/meta.json")))
        t0 = time.time()
        try:
            r = subprocess.run(
                [PY, "simulate_meeting_qwen.py", "--scenario-dir", str(sc),
                 "--model", MODEL, "--provider", "qwen", "--thinking", "false",
                 "--max-rounds", "5", "--memory", "false", "--protocol", "false",
                 "--t", "20", "--max-tokens", "1024"],
                cwd=str(SCHED), env=env, capture_output=True, text=True, timeout=600,
            )
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -1
        wall = time.time() - t0
        after = set(glob.glob(str(SCHED / "logs/meeting/*/meta.json")))
        new = after - before
        meta = None
        if new:
            p = sorted(new)[-1]
            try:
                meta = json.loads(open(p).read())
            except Exception:
                pass
        status = (meta or {}).get("evaluation", {}).get("status") if meta else "error"
        tok = (meta or {}).get("token_usage", {}).get("total", {}).get("total") if meta else 0
        rec = {"scenario": sc.name, "status": status, "completed": (meta or {}).get("completed") if meta else False,
               "rounds": (meta or {}).get("rounds_used") if meta else None,
               "tokens": tok, "wall_s": round(wall, 1),
               "returncode": rc}
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name} status={status} tok={tok} {wall:.0f}s", flush=True)

    ok = sum(1 for r in recs if r["status"] == "ok")
    tot_tok = sum(r["tokens"] or 0 for r in recs)
    tot_time = sum(r["wall_s"] for r in recs)
    print(f"\n=== Summary ===")
    print(f"Scenarios: {len(recs)}  Succeeded (ok): {ok}  Success rate: {ok/len(recs):.1%}")
    print(f"Total tokens: {tot_tok:,}  Total elapsed: {tot_time/60:.1f} min")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
