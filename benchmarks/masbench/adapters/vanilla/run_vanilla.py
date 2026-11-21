"""Single-call MASBench baseline with all task evidence supplied in one prompt.

Protocol: one LLM call, with no ReAct loop or tool calls. All task information is provided in one prompt, and the model returns its final answer or plan. Scoring matches ReAct exactly:
- Reasoning: reuse the official cal_f1_score.
- Scheduling: reuse run_scheduling's exact slot-to-units matching.
- Game: replay the complete model-generated action plan in run_game's GridWorld and apply the same score() function.

Outputs: results/vanilla_{reasoning,scheduling,game}_results.jsonl
"""
import json
import argparse
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
REACT_ADAPTER = REPO_ROOT / "benchmarks" / "masbench" / "adapters" / "react"  # Reuse ReAct scoring helpers.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REACT_ADAPTER))
sys.path.insert(0, str(HERE))

from methods.vanilla.llm import complete, Usage  # noqa: E402   # this adapter's llm.py
MAX_TOKENS = 2048


# --------------------------------------------------------------------------
# Reasoning: provide both evidence sets and the question in one prompt.
# --------------------------------------------------------------------------
REASON_SYSTEM = """You are a question-answering assistant. Answer the question using ONLY the evidence provided below. Be complete but concise; reproduce the relevant facts (names, materials, functions, locations) exactly as stated in the evidence."""


def run_reasoning_one(item):
    ctx = item.get("context", {}) or {}
    ev_a = _fmt_evidence(ctx.get("agent_a", []))
    ev_b = _fmt_evidence(ctx.get("agent_b", []))
    prompt = (
        f"Evidence held by Agent A:\n{ev_a}\n\n"
        f"Evidence held by Agent B:\n{ev_b}\n\n"
        f"Question: {item['question']}\n\n"
        f"Answer:"
    )
    t0 = time.time()
    final, pt, ct = complete(
        [{"role": "system", "content": REASON_SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    wall = time.time() - t0
    usage = Usage(); usage.add(pt, ct)
    f1 = cal_f1_score(item["answer"], final) if final else None
    return {
        "item_id": item.get("id"), "level": item.get("level"), "f1": f1,
        "final_answer": final, "ground_truth": item["answer"],
        "steps": 1, "wall_s": round(wall, 1), **usage.as_dict(),
    }


# --------------------------------------------------------------------------
# Scheduling: provide all five members free slots in one prompt.
# --------------------------------------------------------------------------
SCHED_SYSTEM = """You are a meeting scheduler. Below are the free (available) time intervals of five members. Find ONE common 20-minute time slot where ALL five members are free, and output it in exactly the format "Day D HH:MM-HH:MM" (e.g. Day 3 09:50-10:10). Output only that slot, nothing else."""


def _fmt_gaps(mid, free):
    parts = []
    for d in sorted(free):
        for a, b in free[d]:
            parts.append(f"Day {d} {_min_to_hhmm(a)}-{_min_to_hhmm(b)}")
    return "; ".join(parts) if parts else "(no >=20min free interval)"


def run_scheduling_one(scenario):
    import glob as _glob
    members = sorted(scenario.glob("Member*_final_schedule.json"))
    lines = []
    for p in members:
        mid = re.search(r"Member(\d+)_final", p.name).group(1)
        free = _free_gaps(json.loads(p.read_text(encoding="utf-8")))
        lines.append(f"Member{mid}: {_fmt_gaps(mid, free)}")
    prompt = (
        "Free intervals of all members (all are free during these times):\n"
        + "\n".join(lines)
        + f"\n\nFind ONE common {DURATION_MIN}-minute slot where all five members are free. Output format: Day D HH:MM-HH:MM"
    )
    t0 = time.time()
    final, pt, ct = complete(
        [{"role": "system", "content": SCHED_SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=256,
    )
    wall = time.time() - t0
    usage = Usage(); usage.add(pt, ct)
    status = _evaluate(final, scenario)
    return {
        "scenario": scenario.name, "status": status, "final_slot": final,
        "steps": 1, "wall_s": round(wall, 1), **usage.as_dict(),
    }


# --------------------------------------------------------------------------
# Game: provide the map and replay the complete generated action plan.
# --------------------------------------------------------------------------
GAME_SYSTEM = """You control an agent in a grid world. You can see the entire map. Output the full sequence of commands to collect resources and deliver them to tables.

Output format — ONLY these commands, one per line, nothing else:
move UP
move DOWN
move LEFT
move RIGHT
pick_up TYPE QTY   (TYPE = wood, sand, or stone)
put_down QTY

Rules:
- UP increases y, RIGHT increases x.
- Carry only ONE resource type at a time, up to capacity.
- Pick up only from a resource tile (never a table). Deliver by put_down on a table tile.
- Tables accept limited per-type amounts.

Legend: W=wall, T=table, Y=you, w=wood, s=sand, S=stone, .=empty.

IMPORTANT: Do NOT write any reasoning, explanation, or commentary. Do NOT start with "Let me...". Output ONLY the command list. The very first line must be a command such as "move ..." or "pick_up ..."."""

ITEM_SYM = {"wood": "w", "sand": "s", "stone": "S"}


def _full_map_desc(world):
    # Full grid, with north (+y) at the top.
    rows = []
    for y in range(world.h, 0, -1):
        row = []
        for x in range(1, world.w + 1):
            if x == world.agent["x"] and y == world.agent["y"]:
                row.append("Y"); continue
            c = world.cells.get((x, y))
            if c is None:
                row.append("*")
            elif c["terrain"] == "wall":
                row.append("W")
            elif c["terrain"] == "table":
                row.append("T")
            elif c["items"]:
                top = sorted(c["items"].items(), key=lambda kv: -kv[1])[0][0]
                row.append(ITEM_SYM.get(top, "?"))
            else:
                row.append(".")
        rows.append("".join(row))

    resources, tables = [], []
    for (x, y), c in world.cells.items():
        if c["terrain"] == "resource" and c["items"]:
            for t, q in c["items"].items():
                resources.append(f"  {t} x{q} at ({x},{y})")
        elif c["terrain"] == "table":
            tc = c.get("type_caps") or {}
            capstr = " ".join(f"{k}:{v}" for k, v in tc.items()) if tc else f"total {c['capacity']}"
            tables.append(f"  table at ({x},{y}) accept [{capstr}]")
    return (
        "Full map (Y=you):\n" + "\n".join(rows)
        + "\n\nResources:\n" + ("\n".join(resources) if resources else "  (none)")
        + "\nTables:\n" + ("\n".join(tables) if tables else "  (none)")
        + f"\nStart: ({world.agent['x']},{world.agent['y']}), carry capacity {world.carry_cap}."
        + f"\nTarget: {world.target}"
    )


def _parse_plan(text):
    # Scan the full response for commands, allowing reasoning, numbering, and code fences.
    actions = []
    pat = re.compile(
        r"\b(move\s+(?:up|down|left|right))"
        r"|\b(pick_up\s+(\w+)\s+(\d+))"
        r"|\b(put_down\s+(\d+))",
        re.I,
    )
    for m in pat.finditer(text or ""):
        if m.group(1):
            actions.append(("move", m.group(1).split()[1].upper()))
        elif m.group(2):
            actions.append(("pick_up", m.group(3).lower(), int(m.group(4))))
        elif m.group(5):
            actions.append(("put_down", int(m.group(6))))
    return actions


def run_game_one(map_path):
    world = GridWorld(json.loads(map_path.read_text(encoding="utf-8")))
    prompt = _full_map_desc(world) + "\n\nCommands (one per line):"
    t0 = time.time()
    final, pt, ct = complete(
        [{"role": "system", "content": GAME_SYSTEM},
         {"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    wall = time.time() - t0
    usage = Usage(); usage.add(pt, ct)

    # Replay the plan, ignoring invalid actions.
    plan = _parse_plan(final)
    for a in plan:
        if a[0] == "move":
            world.move(a[1])
        elif a[0] == "pick_up":
            world.pick_up(a[1], a[2])
        elif a[0] == "put_down":
            world.put_down(a[1])
    score, delivered = world.score()
    return {
        "map": map_path.name, "score": score, "delivered": delivered,
        "target": world.target, "plan_len": len(plan),
        "steps": 1, "wall_s": round(wall, 1), **usage.as_dict(),
    }


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------
def _load_task_helpers(task):
    """Load benchmark modules only for the selected task, after CLI parsing."""
    global REASON_DATA, _fmt_evidence, cal_f1_score
    global _free_gaps, _evaluate, _min_to_hhmm, SCENARIOS, DURATION_MIN
    global GridWorld, MAPS

    if task == "reasoning":
        from run_reasoning import _fmt_evidence, cal_f1_score, DATA
        REASON_DATA = DATA
    elif task == "scheduling":
        from run_scheduling import _free_gaps, _evaluate, _min_to_hhmm, SCENARIOS, DURATION_MIN
    elif task == "game":
        from run_game import GridWorld, MAPS


def _run(task, out_name):
    out = HERE / "results" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                key = json.loads(line).get("item_id") or json.loads(line).get("scenario") or json.loads(line).get("map")
                resume.add(key)
            except Exception:
                pass

    if task == "reasoning":
        items = json.loads(REASON_DATA.read_text(encoding="utf-8"))
        def key_of(it): return it.get("id")
        def run_it(it): return run_reasoning_one(it)
        label = lambda it: it.get("id")
    elif task == "scheduling":
        items = list(SCENARIOS)
        def key_of(sc): return sc.name
        def run_it(sc): return run_scheduling_one(sc)
        label = lambda sc: sc.name
    elif task == "game":
        items = list(MAPS)
        def key_of(m): return m.name
        def run_it(m): return run_game_one(m)
        label = lambda m: m.name
    else:
        raise ValueError(task)

    recs = []
    for i, it in enumerate(items):
        if key_of(it) in resume:
            continue
        rec = run_it(it)
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        extra = (f"score={rec['score']}" if task == "game"
                 else f"status={rec['status']}" if task == "scheduling"
                 else f"F1={rec['f1']}")
        print(f"[{i+1}/{len(items)}] {label(it)} {extra} tok={rec['total_tokens']} {rec['wall_s']}s", flush=True)
    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", choices=["reasoning", "scheduling", "game"], default="reasoning")
    task = parser.parse_args().task
    _load_task_helpers(task)
    names = {
        "reasoning": "vanilla_reasoning_results.jsonl",
        "scheduling": "vanilla_scheduling_results.jsonl",
        "game": "vanilla_game_results.jsonl",
    }
    recs = _run(task, names[task])
    print(f"\n=== vanilla {task} completed: {len(recs)} records ===")
