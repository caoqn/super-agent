"""批量跑 MAS-Bench Scheduling 30 个场景（AgentNet 的 LLM 层 + flash 关思考）。

每个场景调 simulate_meeting_qwen.py（关思考），收集 meta.json 的 evaluation + token + 耗时。
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gaia.envfile import load_env

ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
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
    print(f"\n=== 汇总 ===")
    print(f"场景数: {len(recs)}  成功(ok): {ok}  成功率: {ok/len(recs):.1%}")
    print(f"总 token: {tot_tok:,}  总耗时: {tot_time/60:.1f} min")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
