"""汇总 ReAct 在 90 题上的结果并生成报告。

用法: python3 run_all.py   （在三个 runner 全部跑完后执行）
产出: results/REPORT.md 与 results/summary.json
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

    # Reasoning 指标
    scored = [r for r in reason if r.get("f1") is not None]
    mean_f1 = sum(r["f1"] for r in scored) / len(scored) if scored else None
    f1_1 = sum(1 for r in scored if r["f1"] == 1.0)
    f1_0 = sum(1 for r in scored if r["f1"] == 0.0)

    # Scheduling 指标
    sched_status = {}
    for r in sched:
        sched_status[r.get("status", "?")] = sched_status.get(r.get("status", "?"), 0) + 1
    sched_ok = sched_status.get("ok", 0)

    # Game 指标
    game_total = sum(r.get("score", 0) or 0 for r in game)
    game_mean = game_total / len(game) if game else None

    lines = []
    lines.append("# ReAct × MASBench 90 题 实验报告\n")
    lines.append("- 方法：ReAct（Reasoning + Acting，Thought/Action/Observation 循环）单 agent")
    lines.append("- 模型：deepseek-v4.1-flash（关思考），温度 0")
    lines.append("- 数据：MASBench_selected_30（Reasoning/Scheduling/Game 各 30，seed=42）\n")

    lines.append("## 1. 结果概览\n")
    lines.append("| 任务 | 题数 | 主指标 | 总 token | 成本(CNY) | 耗时 |")
    lines.append("|---|---|---|---|---|---|")
    def row(name, st, metric):
        lines.append(f"| {name} | {st['n']} | {metric} | {st['total_tokens']:,} | ¥{st['cost_cny']:.4f} | {st['wall_s']/60:.1f} min |")
    row("Reasoning", stats(reason), f"mean F1 = {mean_f1:.4f}" if mean_f1 is not None else "n/a")
    row("Scheduling", stats(sched), f"ok = {sched_ok}/{len(sched)}")
    row("Game", stats(game), f"总分 {game_total}（均 {game_mean:.1f}）" if game_mean is not None else "n/a")

    tot = {
        "total_tokens": stats(reason).get("total_tokens", 0) + stats(sched).get("total_tokens", 0) + stats(game).get("total_tokens", 0),
        "cost_cny": round(stats(reason).get("cost_cny", 0) + stats(sched).get("cost_cny", 0) + stats(game).get("cost_cny", 0), 4),
        "wall_s": stats(reason).get("wall_s", 0) + stats(sched).get("wall_s", 0) + stats(game).get("wall_s", 0),
        "llm_calls": stats(reason).get("llm_calls", 0) + stats(sched).get("llm_calls", 0) + stats(game).get("llm_calls", 0),
    }
    lines.append(f"| **合计** | {len(reason)+len(sched)+len(game)} | — | {tot['total_tokens']:,} | ¥{tot['cost_cny']:.4f} | {tot['wall_s']/60:.1f} min |")
    lines.append("")
    lines.append(f"- 总 LLM 调用次数：{tot['llm_calls']:,}")
    lines.append(f"- token 明细：total {tot['total_tokens']:,}（其中 prompt {stats(reason).get('prompt_tokens',0)+stats(sched).get('prompt_tokens',0)+stats(game).get('prompt_tokens',0):,} / completion {stats(reason).get('completion_tokens',0)+stats(sched).get('completion_tokens',0)+stats(game).get('completion_tokens',0):,}）\n")

    lines.append("## 2. Reasoning（30 题，F1）\n")
    lines.append(f"- mean F1 = **{mean_f1:.4f}**（{len(scored)} 题有答案）" if mean_f1 is not None else "- 无结果")
    lines.append(f"- F1=1.0（完全正确）：{f1_1} 题；F1=0（全错）：{f1_0} 题")

    lines.append("\n## 3. Scheduling（30 场景）\n")
    lines.append(f"- 状态分布：{sched_status}")
    lines.append(f"- 成功率（ok / 精确匹配 ground-truth 槽位）：**{sched_ok}/{len(sched)}**")

    lines.append("\n## 4. Game（30 地图，交付计分）\n")
    lines.append(f"- 总分：**{game_total}**，平均每地图 {game_mean:.1f}" if game_mean is not None else "- 无结果")
    # 每类型交付
    if game:
        agg = {}
        for r in game:
            for k, v in (r.get("delivered") or {}).items():
                agg[k] = agg.get(k, 0) + v
        tgt = {}
        for r in game:
            for k, v in (r.get("target") or {}).items():
                tgt[k] = v
        lines.append(f"- 交付汇总：{agg}")
        lines.append(f"- 目标：{tgt}")

    lines.append("\n## 5. 说明\n")
    lines.append("- 适配层代码位于 `react_adapter/`，未改动 ReAct-master 与 MASBench 官方文件（仅导入官方 F1 评分函数）。")
    lines.append("- 评分口径：Reasoning 复用官方 `cal_f1_score`；Scheduling 复刻官方槽位→units 精确匹配；Game 复刻官方 enhanced-interaction 物理与交付计分。")
    lines.append("- 逐题明细见 `results/{reasoning,scheduling,game}_results.jsonl`。")

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
    print("已写入 results/REPORT.md 与 results/summary.json")


if __name__ == "__main__":
    main()
