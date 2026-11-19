"""ReAct 适配 —— MASBench Reasoning（30 题）。

单 ReAct agent，工具 lookup[agent_a]/lookup[agent_b] 检索两方证据，Finish[answer] 提交答案。
评分复用官方 src.metrics.cal_f1_score。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
MASBENCH = REPO_ROOT / "MASBench-main"
DATA = MASBENCH / "MASBench_selected_30/Reasoning/reasoning_subset.json"
sys.path.insert(0, str(MASBENCH / "Reasoning"))
from src.metrics import cal_f1_score  # noqa: E402

from react import run_react  # noqa: E402

MAX_STEPS = 10
MAX_TOKENS = 1024

SYSTEM = """Solve a question-answering task by interleaving Thought, Action, Observation steps. You are a single reasoning agent.

Available tools:
- lookup[agent_a] : retrieve the private evidence held by agent A.
- lookup[agent_b] : retrieve the private evidence held by agent B.
- Finish[answer]  : submit your final answer.

Each turn you MUST output exactly two lines, nothing else:
Thought: <your reasoning about what to do next>
Action: <tool[args]>

IMPORTANT: you MUST retrieve evidence from BOTH agents before answering. Always call lookup[agent_a] AND lookup[agent_b] first, then reason across all retrieved evidence, then Finish with a complete answer. Never answer from memory alone.

Example flow:
Thought: I need the private evidence held by both agents.
Action: lookup[agent_a]
(you will receive an Observation)
Thought: I have agent A's evidence; now I need agent B's.
Action: lookup[agent_b]
(you will receive an Observation)
Thought: I now have all evidence; let me compare and answer.
Action: Finish[the complete answer]"""


def _fmt_evidence(blocks):
    """context 条目形如 [title, [sentence, ...]]。"""
    lines = []
    for title, sents in blocks:
        lines.append(f"- {title}: " + " ".join(sents))
    return "\n".join(lines)


def run_one(item, required_calls=None):
    ctx = item.get("context", {}) or {}
    evidence = {
        "agent_a": _fmt_evidence(ctx.get("agent_a", [])),
        "agent_b": _fmt_evidence(ctx.get("agent_b", [])),
    }

    def lookup(args):
        who = (args[0] if args else "").strip().lower()
        if who not in evidence:
            return f"未知主体 '{who}'，只能 lookup[agent_a] 或 lookup[agent_b]。"
        return evidence[who] or "(empty)"

    tools = {"lookup": lookup}
    question = item["question"]
    t0 = time.time()
    final, usage, traj = run_react(
        SYSTEM, tools,
        first_message=f"Question: {question}",
        max_steps=MAX_STEPS, max_tokens=MAX_TOKENS,
        required_calls=required_calls,
    )
    wall = time.time() - t0
    f1 = cal_f1_score(item["answer"], final) if final else None
    rec = {
        "item_id": item.get("id"),
        "level": item.get("level"),
        "f1": f1,
        "final_answer": final,
        "ground_truth": item["answer"],
        "steps": len(traj),
        "wall_s": round(wall, 1),
        **usage.as_dict(),
    }
    return rec


def main(out_name="reasoning_results.jsonl", required_calls=None):
    items = json.loads(DATA.read_text(encoding="utf-8"))
    out = HERE / "results" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                resume.add(json.loads(line)["item_id"])
            except Exception:
                pass

    recs = []
    for i, item in enumerate(items):
        if item.get("id") in resume:
            continue
        rec = run_one(item, required_calls=required_calls)
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(items)}] {item.get('id')} F1={rec['f1']} "
              f"steps={rec['steps']} tok={rec['total_tokens']} {rec['wall_s']}s", flush=True)

    scored = [r for r in recs if r["f1"] is not None]
    if scored:
        print(f"\n=== Reasoning 汇总 ===\n题数 {len(scored)}  mean F1 = "
              f"{sum(r['f1'] for r in scored) / len(scored):.4f}")


if __name__ == "__main__":
    # 用法: python3 run_reasoning.py            -> 原始无约束跑
    #      python3 run_reasoning.py constrained -> 强制先查两方证据再 Finish
    if len(sys.argv) > 1 and sys.argv[1] == "constrained":
        main(out_name="reasoning_results_constrained.jsonl",
             required_calls=[("lookup", "agent_a"), ("lookup", "agent_b")])
    else:
        main()
