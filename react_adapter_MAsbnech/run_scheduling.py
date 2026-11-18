"""ReAct 适配 —— MASBench Scheduling（30 场景）。

单 ReAct agent 扮演协调者，工具 free_slots[member_id] 查询某成员的可行空闲段（基于
官方模拟同款的 Member*_final_schedule.json + 全局不可用时段），推理出共同 20 分钟会议
时间，Finish[Day D HH:MM-HH:MM] 提交。评分严格复刻官方 _evaluate_against_ground_truth：
解析槽位 -> 转 10 分钟 units -> 精确匹配 task_plans.solutions。
"""
import glob
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASBENCH = (Path(__file__).resolve().parents[1] / "MASBench-main")
SCENARIOS = sorted((MASBENCH / "MASBench_selected_30/Scheduling/scenarios").iterdir())

from react import run_react  # noqa: E402

MAX_STEPS = 15
MAX_TOKENS = 1024
DURATION_MIN = 20

# 官方 _global_rules_text：所有人不可用时段（分钟）。
GLOBAL_BLOCKED = [(0, 420), (420, 480), (720, 840), (1080, 1200)]

SYSTEM = """Solve a meeting-scheduling task by interleaving Thought, Action, Observation steps. You are a single coordinator agent.

Available tools:
- free_slots[member_id] : query one member's available (free) time intervals. member_id is 1..5.
- Finish[slot]          : submit the final meeting slot.

Each turn you MUST output exactly two lines:
Thought: <your reasoning>
Action: <tool[args]>

Goal: find ONE common 20-minute time slot where ALL five members are free. Query each member's free slots, compare them across members, and submit a slot in the exact format "Day D HH:MM-HH:MM" (20 minutes long), e.g. Finish[Day 3 09:50-10:10]."""


def _hhmm_to_min(hhmm: str) -> int:
    m = re.match(r"^([0-2]?\d):(\d{2})$", hhmm.strip())
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def _min_to_hhmm(mm: int) -> str:
    return f"{mm // 60:02d}:{mm % 60:02d}"


def _parse_dur(s: str):
    a, b = s.split("-")
    return _hhmm_to_min(a), _hhmm_to_min(b)


def _free_gaps(schedule: dict) -> dict:
    """返回 {day: [(start_min, end_min), ...]}，仅含 >= DURATION_MIN 的空闲段。"""
    days = {}
    for act in schedule.get("activities", []):
        d = act.get("date")
        try:
            days.setdefault(d, []).append(_parse_dur(act["duration"]))
        except Exception:
            continue
    total_days = int(schedule.get("total_days", 5))
    free = {}
    for d in range(1, total_days + 1):
        busy = list(GLOBAL_BLOCKED) + days.get(d, [])
        busy.sort()
        merged = []
        for a, b in busy:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        gaps = []
        cur = 0
        for a, b in merged:
            if a > cur:
                gaps.append((cur, a))
            cur = max(cur, b)
        if cur < 1440:
            gaps.append((cur, 1440))
        free[d] = [g for g in gaps if g[1] - g[0] >= DURATION_MIN]
    return free


def _parse_slot(text: str):
    """复刻官方 _parse_day_slot 正则。"""
    if not text:
        return None
    m = re.search(
        r"(?:\bDay\s*)?(\d+)\s*(?:,|\s)+([0-2]?\d:\d{2})\s*[-–—~]\s*([0-2]?\d:\d{2})",
        text, flags=re.IGNORECASE,
    )
    if not m:
        return None
    return {"day": int(m.group(1)), "start_time": m.group(2), "end_time": m.group(3)}


def _evaluate(final_text: str, scenario: Path) -> str:
    """复刻官方 _evaluate_against_ground_truth 的精确匹配。"""
    gt_path = scenario / "task_plans"
    if not gt_path.exists():
        return "no_ground_truth"
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    slot = _parse_slot(final_text or "")
    if not slot:
        return "no_slot"
    schedules = [json.loads(open(p).read()) for p in sorted(scenario.glob("Member*_final_schedule.json"))]
    time_unit = schedules[0].get("time_unit", "minutes_10") if schedules else "minutes_10"
    if isinstance(time_unit, str) and time_unit.startswith("minutes_"):
        try:
            minutes_per_unit = int(time_unit.split("_")[1])
        except Exception:
            minutes_per_unit = 15
    else:
        minutes_per_unit = 60
    units_per_day = (60 // minutes_per_unit) * 24
    start_min = _hhmm_to_min(slot["start_time"])
    end_min = _hhmm_to_min(slot["end_time"]) or (start_min + DURATION_MIN)
    if end_min <= start_min:
        end_min = start_min + minutes_per_unit
    duration_units = max(1, (end_min - start_min) // minutes_per_unit)
    start_units = (slot["day"] - 1) * units_per_day + start_min // minutes_per_unit
    end_units = start_units + duration_units
    solutions = (gt_data.get("meeting") or {}).get("solutions") or []
    matched = [s for s in solutions if s.get("start_time") == start_units and s.get("end_time") == end_units]
    return "ok" if matched else "mismatch"


def run_one(scenario: Path):
    members = sorted(scenario.glob("Member*_final_schedule.json"))
    free_cache = {}
    for p in members:
        mid = re.search(r"Member(\d+)_final", p.name).group(1)
        free_cache[mid] = _free_gaps(json.loads(p.read_text(encoding="utf-8")))

    def fmt_gaps(mid):
        parts = []
        for d in sorted(free_cache[mid]):
            for a, b in free_cache[mid][d]:
                parts.append(f"Day {d} {_min_to_hhmm(a)}-{_min_to_hhmm(b)}")
        return "; ".join(parts) if parts else "(无 ≥20 分钟空闲段)"

    def free_slots(args):
        mid = (args[0] if args else "").strip()
        if mid not in free_cache:
            return f"未知成员 '{mid}'，只能查询 1..5。"
        return f"Member{mid} 空闲段（≥20 分钟）: {fmt_gaps(mid)}"

    tools = {"free_slots": free_slots}
    member_names = ", ".join(f"Member{i}" for i in range(1, 6))
    t0 = time.time()
    final, usage, traj = run_react(
        SYSTEM, tools,
        first_message=f"Find one common {DURATION_MIN}-minute meeting slot for: {member_names}.",
        max_steps=MAX_STEPS, max_tokens=MAX_TOKENS,
    )
    wall = time.time() - t0
    status = _evaluate(final, scenario)
    return {
        "scenario": scenario.name,
        "status": status,
        "final_slot": final,
        "steps": len(traj),
        "wall_s": round(wall, 1),
        **usage.as_dict(),
    }


def main():
    out = HERE / "results" / "scheduling_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                resume.add(json.loads(line)["scenario"])
            except Exception:
                pass
    recs = []
    for i, sc in enumerate(SCENARIOS):
        if sc.name in resume:
            continue
        rec = run_one(sc)
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(SCENARIOS)}] {sc.name} status={rec['status']} "
              f"steps={rec['steps']} tok={rec['total_tokens']} {rec['wall_s']}s", flush=True)

    ok = sum(1 for r in recs if r["status"] == "ok")
    print(f"\n=== Scheduling 汇总 ===\n场景 {len(recs)}  成功(ok) {ok}  成功率 {ok / len(recs):.1%}" if recs else "无")


if __name__ == "__main__":
    main()
