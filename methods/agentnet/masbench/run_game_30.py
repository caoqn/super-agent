"""批量跑 MAS-Bench Game 30 地图（Python 3.10 环境 + flash 关思考）。

每个地图 subprocess 跑 eval.py，收集 game_log 的 score + agent_log 的 token + 耗时。
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
GAME = ROOT / "Game"
MAPS = sorted((ROOT / "MASBench_selected_30/Game/maps").glob("*.json"))
PY = sys.executable
MODEL = "deepseek-v4.1-flash"
# 完整实验预算 50 轮（eval.py 默认），不传 --max_round
def collect_result(new_run_dirs):
    """从最新 run 目录读 score + tokens。"""
    if not new_run_dirs:
        return 0, 0
    rd = sorted(new_run_dirs)[-1]
    score = 0
    tokens = 0
    try:
        gl = sorted(glob.glob(str(Path(rd) / "game_log*.json")))
        if gl:
            d = json.loads(open(gl[-1]).read())
            last = d[-1] if isinstance(d, list) else d
            score = last.get("score", 0) or 0
        al = sorted(glob.glob(str(Path(rd) / "agent_log*.json")))
        if al:
            d = json.loads(open(al[-1]).read())
            if isinstance(d, list):
                tokens = sum((r.get("total_llm_tokens") or 0) for r in d)
    except Exception as e:
        print(f"  [warn] 读结果失败: {e}")
    return score, tokens


def main():
    load_env([ROOT / ".env", REPO_ROOT / ".env"])
    env = dict(os.environ)
    env["QWEN_OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
    env["QWEN_OPENAI_BASE"] = os.environ["LLM_BASE_URL"]
    env["MASBENCH_DISABLE_THINKING"] = "1"

    out = ROOT / "MASBench_selected_30/game_results.jsonl"
    recs = []
    for i, m in enumerate(MAPS):
        before = set(glob.glob(str(GAME / ".preset_logs/*/")))
        t0 = time.time()
        r = subprocess.run(
            [PY, "eval.py", "--provider", "QWEN", "--model", MODEL],
            cwd=str(GAME), env={**env, "CUSTOM_MAP_PATH": str(m), "MECHANISM_PRESET": "base"},
            capture_output=True, text=True, timeout=1200,
        )
        wall = time.time() - t0
        after = set(glob.glob(str(GAME / ".preset_logs/*/")))
        score, tokens = collect_result(after - before)
        rec = {"map": m.name, "score": score, "tokens": tokens, "wall_s": round(wall, 1)}
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{len(MAPS)}] {m.name} score={score} tok={tokens} {wall:.0f}s", flush=True)

    tot_score = sum(r["score"] for r in recs)
    tot_tok = sum(r["tokens"] for r in recs)
    tot_time = sum(r["wall_s"] for r in recs)
    print(f"\n=== 汇总 ===")
    print(f"地图数: {len(recs)}  总分: {tot_score}  平均分: {tot_score/len(recs):.2f}")
    print(f"总 token: {tot_tok:,}  总耗时: {tot_time/60:.1f} min")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
