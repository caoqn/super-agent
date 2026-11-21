"""Summarize AOrchestra MASBench results into REPORT.md and summary.json.

Costs use the CNY rates in .env. The official AOrchestra ModelPricing does not include
deepseek-v4.1-flash, so get_usage_summary().total_cost is zero; this script applies the
configured rates separately.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

PRICE_IN = 1.005 / 1_000_000   # CNY / token
PRICE_OUT = 4.02 / 1_000_000


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


def cost_cny(r):
    it = r.get("total_input_tokens", 0) or 0
    ot = r.get("total_output_tokens", 0) or 0
    return it * PRICE_IN + ot * PRICE_OUT


def stats(recs):
    if not recs:
        return {}
    it = sum(r.get("total_input_tokens", 0) or 0 for r in recs)
    ot = sum(r.get("total_output_tokens", 0) or 0 for r in recs)
    wall = sum(r.get("wall_s", 0) or 0 for r in recs)
    calls = sum(r.get("call_count", 0) or 0 for r in recs)
    return {
        "n": len(recs),
        "input_tokens": it,
        "output_tokens": ot,
        "total_tokens": it + ot,
        "cost_cny": round(sum(cost_cny(r) for r in recs), 4),
        "wall_s": round(wall, 1),
        "llm_calls": calls,
    }


def main():
    reason = load("reasoning.jsonl")
    sched = load("scheduling.jsonl")
    game = load("game.jsonl")

    # Reasoning
    scored = [r for r in reason if r.get("f1") is not None]
    mean_f1 = sum(r["f1"] for r in scored) / len(scored) if scored else None
    f1_1 = sum(1 for r in scored if r["f1"] == 1.0)

    # Scheduling
    sched_ok = sum(1 for r in sched if r.get("status") == "ok")

    # Game
    game_total = sum(r.get("score", 0) or 0 for r in game)
    game_mean = game_total / len(game) if game else None

    L = []
    L.append("# AOrchestra × MASBench 90-task experiment report\n")
    L.append("- Method: AOrchestra (MainAgent with ReActAgent subagents)")
    L.append("- Model: deepseek-v4.1-flash for manager and subagents, temperature 0")
    L.append("- Data: MASBench_selected_30 (30 tasks per category, seed=42)\n")

    L.append("## 1. Overview\n")
    L.append("| Task | Count | Metric | Total tokens | Cost (CNY) | Elapsed |")
    L.append("|---|---|---|---|---|---|")

    def row(name, st, metric):
        L.append(f"| {name} | {st['n']} | {metric} | {st['total_tokens']:,} | ¥{st['cost_cny']:.4f} | {st['wall_s']/60:.1f} min |")

    row("Reasoning", stats(reason), f"mean F1 = {mean_f1:.4f}" if mean_f1 is not None else "n/a")
    row("Scheduling", stats(sched), f"ok = {sched_ok}/{len(sched)}")
    row("Game", stats(game), f"Total score {game_total} (mean {game_mean:.1f})" if game_mean is not None else "n/a")

    s_reason, s_sched, s_game = stats(reason), stats(sched), stats(game)
    tot = {
        "n": len(reason) + len(sched) + len(game),
        "input_tokens": s_reason.get("input_tokens", 0) + s_sched.get("input_tokens", 0) + s_game.get("input_tokens", 0),
        "output_tokens": s_reason.get("output_tokens", 0) + s_sched.get("output_tokens", 0) + s_game.get("output_tokens", 0),
        "total_tokens": s_reason.get("total_tokens", 0) + s_sched.get("total_tokens", 0) + s_game.get("total_tokens", 0),
        "cost_cny": round(s_reason.get("cost_cny", 0) + s_sched.get("cost_cny", 0) + s_game.get("cost_cny", 0), 4),
        "wall_s": s_reason.get("wall_s", 0) + s_sched.get("wall_s", 0) + s_game.get("wall_s", 0),
        "llm_calls": s_reason.get("llm_calls", 0) + s_sched.get("llm_calls", 0) + s_game.get("llm_calls", 0),
    }
    L.append(f"| **Total** | {tot['n']} | — | {tot['total_tokens']:,} | ¥{tot['cost_cny']:.4f} | {tot['wall_s']/60:.1f} min |")
    L.append("")
    L.append(f"- Total LLM calls: {tot['llm_calls']:,}")
    L.append(f"- Token breakdown: prompt {tot['input_tokens']:,} / completion {tot['output_tokens']:,}")
    L.append(f"- Costs use CNY rates (input ¥{PRICE_IN*1e6:.3f}/M, output ¥{PRICE_OUT*1e6:.2f}/M)\n")

    L.append("## 2. Reasoning (30 tasks, F1)\n")
    L.append(f"- mean F1 = **{mean_f1:.4f}** ({len(scored)} answered tasks)" if mean_f1 is not None else "- No results")
    L.append(f"- F1=1.0 (exact): {f1_1} tasks")

    L.append("\n## 3. Scheduling (30 scenarios)\n")
    L.append(f"- Success rate (ok / exact official solutions): **{sched_ok}/{len(sched)}**")

    L.append("\n## 4. Game (30 maps, delivery score)\n")
    L.append(f"- Total score: **{game_total}**, mean per map {game_mean:.1f}" if game_mean is not None else "- No results")
    if game:
        agg = {}
        for r in game:
            for k, v in (r.get("delivered") or {}).items():
                agg[k] = agg.get(k, 0) + v
        tgt = {}
        for r in game:
            for k, v in (r.get("target") or {}).items():
                tgt[k] = v
        L.append(f"- Delivery totals: {agg}")
        L.append(f"- Targets: {tgt}")

    L.append("\n## 5. Notes\n")
    L.append("- Adapter: `benchmarks/masbench/adapters/aorchestra/`; upstream source files are unchanged.")
    L.append("- Scoring: official Reasoning F1, exact Scheduling slots, and Game enhanced-interaction delivery rules.")
    L.append("- Per-task results: `results/{reasoning,scheduling,game}.jsonl`.")

    report = "\n".join(L) + "\n"
    (RES / "REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "reasoning": {"mean_f1": mean_f1, "f1_1": f1_1, **s_reason},
        "scheduling": {"ok": sched_ok, **s_sched},
        "game": {"total_score": game_total, "mean_score": game_mean, **s_game},
        "total": tot,
    }
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print("Wrote results/REPORT.md and results/summary.json")


if __name__ == "__main__":
    main()
