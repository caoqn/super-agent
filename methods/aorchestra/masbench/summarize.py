"""汇总 MASBench × AOrchestra 90 题结果，生成 REPORT.md 与 summary.json。

成本按 .env 里的 CNY 单价计算（AOrchestra 官方 ModelPricing 无 deepseek-v4.1-flash，
其 get_usage_summary().total_cost 为 0，故在此单独用单价折算）。
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
    L.append("# AOrchestra × MASBench 90 题 实验报告\n")
    L.append("- 方法：AOrchestra（MainAgent 编排 + ReActAgent 子代理），完整多智能体编排")
    L.append("- 模型：deepseek-v4.1-flash（主/子代理同模型），温度 0")
    L.append("- 数据：MASBench_selected_30（Reasoning/Scheduling/Game 各 30，seed=42）\n")

    L.append("## 1. 结果概览\n")
    L.append("| 任务 | 题数 | 主指标 | 总 token | 成本(CNY) | 耗时 |")
    L.append("|---|---|---|---|---|---|")

    def row(name, st, metric):
        L.append(f"| {name} | {st['n']} | {metric} | {st['total_tokens']:,} | ¥{st['cost_cny']:.4f} | {st['wall_s']/60:.1f} min |")

    row("Reasoning", stats(reason), f"mean F1 = {mean_f1:.4f}" if mean_f1 is not None else "n/a")
    row("Scheduling", stats(sched), f"ok = {sched_ok}/{len(sched)}")
    row("Game", stats(game), f"总分 {game_total}（均 {game_mean:.1f}）" if game_mean is not None else "n/a")

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
    L.append(f"| **合计** | {tot['n']} | — | {tot['total_tokens']:,} | ¥{tot['cost_cny']:.4f} | {tot['wall_s']/60:.1f} min |")
    L.append("")
    L.append(f"- 总 LLM 调用次数：{tot['llm_calls']:,}")
    L.append(f"- token 明细：prompt {tot['input_tokens']:,} / completion {tot['output_tokens']:,}")
    L.append(f"- 成本按 CNY 单价（input ¥{PRICE_IN*1e6:.3f}/M，output ¥{PRICE_OUT*1e6:.2f}/M）折算\n")

    L.append("## 2. Reasoning（30 题，F1）\n")
    L.append(f"- mean F1 = **{mean_f1:.4f}**（{len(scored)} 题有答案）" if mean_f1 is not None else "- 无结果")
    L.append(f"- F1=1.0（完全正确）：{f1_1} 题")

    L.append("\n## 3. Scheduling（30 场景）\n")
    L.append(f"- 成功率（ok / 精确匹配官方 solutions）：**{sched_ok}/{len(sched)}**")

    L.append("\n## 4. Game（30 地图，交付计分）\n")
    L.append(f"- 总分：**{game_total}**，平均每地图 {game_mean:.1f}" if game_mean is not None else "- 无结果")
    if game:
        agg = {}
        for r in game:
            for k, v in (r.get("delivered") or {}).items():
                agg[k] = agg.get(k, 0) + v
        tgt = {}
        for r in game:
            for k, v in (r.get("target") or {}).items():
                tgt[k] = v
        L.append(f"- 交付汇总：{agg}")
        L.append(f"- 目标：{tgt}")

    L.append("\n## 5. 说明\n")
    L.append("- 适配层位于 `methods/aorchestra/masbench/`，未改动 AOrchestra 与 MASBench 官方文件。")
    L.append("- 评分口径：Reasoning 官方 `cal_f1_score`；Scheduling 官方槽位精确匹配；Game 官方 enhanced-interaction 交付计分。")
    L.append("- 逐题明细见 `results/{reasoning,scheduling,game}.jsonl`。")

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
    print("已写入 results/REPORT.md 与 results/summary.json")


if __name__ == "__main__":
    main()
