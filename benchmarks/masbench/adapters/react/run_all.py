"""Summarize the three ReAct MASBench runs into REPORT.md and summary.json.

Usage: python3 run_all.py (run after all three task runners finish).
Outputs: results/REPORT.md and results/summary.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load(name):
    p = RES / name
    if not p.exists():
        return []
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def stats(recs):
    if not recs:
        return {}
    tok = sum(r.get("total_tokens", 0) or 0 for r in recs)
    pt = sum(r.get("prompt_tokens", 0) or 0 for r in recs)
    ct = sum(r.get("completion_tokens", 0) or 0 for r in recs)
    cost = sum(r.get("cost_cny", 0) or 0 for r in recs)
    wall = sum(r.get("wall_s", 0) or 0 for r in recs)
    calls = sum(r.get("llm_calls", 0) or 0 for r in recs)
    return {"n": len(recs), "total_tokens": tok, "prompt_tokens": pt,
            "completion_tokens": ct, "cost_cny": round(cost, 4),
            "wall_s": round(wall, 1), "llm_calls": calls}


def main():
    reason = load("reasoning_results.jsonl")
    sched = load("scheduling_results.jsonl")
    game = load("game_results.jsonl")

    # Reasoning metrics.
    scored = [r for r in reason if r.get("f1") is not None]
    mean_f1 = sum(r["f1"] for r in scored) / len(scored) if scored else None
    f1_1 = sum(1 for r in scored if r["f1"] == 1.0)
    f1_0 = sum(1 for r in scored if r["f1"] == 0.0)

    # Scheduling metrics.
    sched_status = {}
    for r in sched:
        sched_status[r.get("status", "?")] = sched_status.get(r.get("status", "?"), 0) + 1
    sched_ok = sched_status.get("ok", 0)

    # Game metrics.
    game_total = sum(r.get("score", 0) or 0 for r in game)
    game_mean = game_total / len(game) if game else None

    lines = []
    lines.append("# ReAct × MASBench 90-task experiment report\n")
    lines.append("- Method: single-agent ReAct (Thought/Action/Observation loop)")
    lines.append("- Model: deepseek-v4.1-flash, thinking disabled, temperature 0")
    lines.append("- Data: MASBench_selected_30 (30 tasks per category, seed=42)\n")

    lines.append("## 1. Overview\n")
    lines.append("| Task | Count | Metric | Total tokens | Cost (CNY) | Elapsed |")
    lines.append("|---|---|---|---|---|---|")
    def row(name, st, metric):
        lines.append(f"| {name} | {st['n']} | {metric} | {st['total_tokens']:,} | ¥{st['cost_cny']:.4f} | {st['wall_s']/60:.1f} min |")
    row("Reasoning", stats(reason), f"mean F1 = {mean_f1:.4f}" if mean_f1 is not None else "n/a")
    row("Scheduling", stats(sched), f"ok = {sched_ok}/{len(sched)}")
    row("Game", stats(game), f"Total score {game_total} (mean {game_mean:.1f})" if game_mean is not None else "n/a")

    tot = {
        "total_tokens": stats(reason).get("total_tokens", 0) + stats(sched).get("total_tokens", 0) + stats(game).get("total_tokens", 0),
        "cost_cny": round(stats(reason).get("cost_cny", 0) + stats(sched).get("cost_cny", 0) + stats(game).get("cost_cny", 0), 4),
        "wall_s": stats(reason).get("wall_s", 0) + stats(sched).get("wall_s", 0) + stats(game).get("wall_s", 0),
        "llm_calls": stats(reason).get("llm_calls", 0) + stats(sched).get("llm_calls", 0) + stats(game).get("llm_calls", 0),
    }
    lines.append(f"| **Total** | {len(reason)+len(sched)+len(game)} | — | {tot['total_tokens']:,} | ¥{tot['cost_cny']:.4f} | {tot['wall_s']/60:.1f} min |")
    lines.append("")
    lines.append(f"- Total LLM calls: {tot['llm_calls']:,}")
    lines.append(f"- Token breakdown: total {tot['total_tokens']:,} (prompt {stats(reason).get('prompt_tokens',0)+stats(sched).get('prompt_tokens',0)+stats(game).get('prompt_tokens',0):,} / completion {stats(reason).get('completion_tokens',0)+stats(sched).get('completion_tokens',0)+stats(game).get('completion_tokens',0):,})\n")

    lines.append("## 2. Reasoning (30 tasks, F1)\n")
    lines.append(f"- mean F1 = **{mean_f1:.4f}** ({len(scored)} answered tasks)" if mean_f1 is not None else "- No results")
    lines.append(f"- F1=1.0 (exact): {f1_1} tasks; F1=0 (incorrect): {f1_0} tasks")

    lines.append("\n## 3. Scheduling (30 scenarios)\n")
    lines.append(f"- Status counts: {sched_status}")
    lines.append(f"- Success rate (ok / exact ground-truth slots): **{sched_ok}/{len(sched)}**")

    lines.append("\n## 4. Game (30 maps, delivery score)\n")
    lines.append(f"- Total score: **{game_total}**, mean per map {game_mean:.1f}" if game_mean is not None else "- No results")
    # Delivery totals per item type.
    if game:
        agg = {}
        for r in game:
            for k, v in (r.get("delivered") or {}).items():
                agg[k] = agg.get(k, 0) + v
        tgt = {}
        for r in game:
            for k, v in (r.get("target") or {}).items():
                tgt[k] = v
        lines.append(f"- Delivery totals: {agg}")
        lines.append(f"- Targets: {tgt}")

    lines.append("\n## 5. Notes\n")
    lines.append("- Adapter: `benchmarks/masbench/adapters/react/`; imports the official F1 evaluator.")
    lines.append("- Scoring: official Reasoning F1, Scheduling slot-to-unit exact matching, and Game enhanced-interaction physics and delivery rules.")
    lines.append("- Per-task results: `results/{reasoning,scheduling,game}_results.jsonl`.")

    report = "\n".join(lines) + "\n"
    (RES / "REPORT.md").write_text(report, encoding="utf-8")
    summary = {
        "reasoning": {"n": len(reason), "mean_f1": mean_f1, "f1_1": f1_1, "f1_0": f1_0, **stats(reason)},
        "scheduling": {"n": len(sched), "ok": sched_ok, "status": sched_status, **stats(sched)},
        "game": {"n": len(game), "total_score": game_total, "mean_score": game_mean, **stats(game)},
        "total": tot,
    }
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print("Wrote results/REPORT.md and results/summary.json")


if __name__ == "__main__":
    main()
