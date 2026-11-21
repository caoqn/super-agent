"""AOrchestra environments for MASBench Reasoning, Scheduling, and Game.

Uses the AOrchestra benchmark.common.env.Environment interface:
    get_basic_info() -> BasicInfo
    reset() -> Observation
    step(action) -> (observation, reward, done, info)

The SubAgent (ReActAgent) returns actions as JSON:
    {"action": "<tool_name>", "params": {...}, "memory": "..."}
It finishes with the finish action:
    {"action": "finish", "params": {"result": ..., "status": ..., "summary": ...}}

Scoring follows the official MASBench protocol:
    Reasoning  -> official src.metrics.cal_f1_score
    Scheduling -> exact slot matching in the official _evaluate_against_ground_truth
    Game       -> official enhanced-interaction delivery score
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from benchmark.common.env import Action, BasicInfo, Environment, Observation

REPO_ROOT = Path(__file__).resolve().parents[4]
MASBENCH = REPO_ROOT / "MASBench-main"
SEL = MASBENCH / "MASBench_selected_30"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def _fmt_evidence(blocks):
    """Format context entries of the form [title, [sentence, ...]]."""
    lines = []
    for title, sents in blocks:
        lines.append(f"- {title}: " + " ".join(sents))
    return "\n".join(lines)


class ReasoningEnv(Environment):
    """Retrieve and combine the private evidence held by two agents to answer a question."""

    def __init__(self, item: dict):
        self.item_id = item.get("id")
        self.question = item["question"]
        self.ground_truth = item["answer"]
        self.level = item.get("level")
        ctx = item.get("context", {}) or {}
        self.evidence = {
            "agent_a": _fmt_evidence(ctx.get("agent_a", [])),
            "agent_b": _fmt_evidence(ctx.get("agent_b", [])),
        }
        self._steps = 0
        self._done = False
        self.max_steps = 10

    def _build_action_space(self) -> str:
        return (
            "Available actions:\n\n"
            "### lookup\n"
            'Description: Retrieve the private evidence held by one agent.\n'
            'Parameters: {"who": "agent_a" or "agent_b"}\n\n'
            "### finish\n"
            'Description: Report your final answer to the MainAgent.\n'
            'Parameters: {"result": "<final answer>", "status": "done|partial|blocked", "summary": "<brief>"}\n'
        )

    def _build_instruction(self) -> str:
        return (
            f"Question: {self.question}\n\n"
            "You can query the private evidence held by agent_a and agent_b using lookup. "
            "You MUST retrieve evidence from BOTH agents before answering, then reason across "
            "all retrieved evidence and report a complete answer via finish."
        )

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.item_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.max_steps,
            meta_data={"task_id": self.item_id, "level": self.level},
        )

    async def reset(self, seed=None) -> Observation:
        self._steps = 0
        self._done = False
        return {"message": "Environment ready.", "question": self.question}

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._steps += 1
        a = action.get("action", "")
        params = action.get("params", {}) or {}

        if a == "finish":
            self._done = True
            result = params.get("result", "")
            status = params.get("status", "done")
            summary = params.get("summary", "")
            fr = {"result": result, "status": status, "summary": summary}
            return (
                {"message": "Result reported to MainAgent.", "finish_result": fr},
                0.0, True, {"finished": True, "finish_result": fr},
            )

        if a == "lookup":
            who = str(params.get("who", "")).strip().lower()
            if who not in self.evidence:
                obs = {"error": f"Unknown agent '{who}'. Use who=agent_a or who=agent_b."}
            else:
                obs = {"output": self.evidence[who] or "(empty)"}
        else:
            obs = {"error": f"Unknown action '{a}'. Available: lookup, finish."}

        obs["current_step"] = self._steps
        if self._steps >= self.max_steps:
            self._done = True
            fr = {"result": "", "status": "timeout", "summary": "max steps"}
            obs["finish_result"] = fr
            return obs, 0.0, True, {"finished": True, "finish_result": fr}
        return obs, 0.0, False, {}


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

DURATION_MIN = 20
# Global unavailability intervals in minutes, matching official _global_rules_text.
GLOBAL_BLOCKED = [(0, 420), (420, 480), (720, 840), (1080, 1200)]


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
    if not text:
        return None
    m = re.search(
        r"(?:\bDay\s*)?(\d+)\s*(?:,|\s)+([0-2]?\d:\d{2})\s*[-–—~]\s*([0-2]?\d:\d{2})",
        text, flags=re.IGNORECASE,
    )
    if not m:
        return None
    return {"day": int(m.group(1)), "start_time": m.group(2), "end_time": m.group(3)}


class SchedulingEnv(Environment):
    """Find a common 20-minute free slot for five members."""

    def __init__(self, scenario_dir: Path):
        self.scenario = scenario_dir
        self.scenario_id = scenario_dir.name
        self.members = sorted(scenario_dir.glob("Member*_final_schedule.json"))
        self.free_cache = {}
        for p in self.members:
            mid = re.search(r"Member(\d+)_final", p.name).group(1)
            self.free_cache[mid] = _free_gaps(json.loads(p.read_text(encoding="utf-8")))
        self._steps = 0
        self._done = False
        self.max_steps = 15

    def _build_action_space(self) -> str:
        return (
            "Available actions:\n\n"
            "### free_slots\n"
            'Description: Query one member\'s available free intervals (>=20 min).\n'
            'Parameters: {"member_id": "1".."5"}\n\n'
            "### finish\n"
            'Description: Report the final meeting slot to the MainAgent.\n'
            'Parameters: {"result": "Day D HH:MM-HH:MM", "status": "done", "summary": "<brief>"}\n'
        )

    def _build_instruction(self) -> str:
        names = ", ".join(f"Member{i}" for i in range(1, 6))
        return (
            f"Find ONE common {DURATION_MIN}-minute time slot where all five members "
            f"({names}) are free. Query each member's free slots, compare, and report a "
            f"slot in the exact format 'Day D HH:MM-HH:MM' (20 minutes long), e.g. 'Day 3 09:50-10:10'."
        )

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.scenario_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.max_steps,
            meta_data={"task_id": self.scenario_id},
        )

    async def reset(self, seed=None) -> Observation:
        self._steps = 0
        self._done = False
        return {"message": "Environment ready.", "task": self._build_instruction()}

    def _fmt_gaps(self, mid):
        parts = []
        for d in sorted(self.free_cache[mid]):
            for a, b in self.free_cache[mid][d]:
                parts.append(f"Day {d} {_min_to_hhmm(a)}-{_min_to_hhmm(b)}")
        return "; ".join(parts) if parts else "(no >=20min free interval)"

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._steps += 1
        a = action.get("action", "")
        params = action.get("params", {}) or {}

        if a == "finish":
            self._done = True
            result = params.get("result", "")
            status = params.get("status", "done")
            summary = params.get("summary", "")
            fr = {"result": result, "status": status, "summary": summary}
            return (
                {"message": "Slot reported.", "finish_result": fr},
                0.0, True, {"finished": True, "finish_result": fr},
            )

        if a == "free_slots":
            mid = str(params.get("member_id", "")).strip()
            if mid not in self.free_cache:
                obs = {"error": f"Unknown member '{mid}'. Use member_id in 1..5."}
            else:
                obs = {"output": f"Member{mid} free intervals (>=20 min): {self._fmt_gaps(mid)}"}
        else:
            obs = {"error": f"Unknown action '{a}'. Available: free_slots, finish."}

        obs["current_step"] = self._steps
        if self._steps >= self.max_steps:
            self._done = True
            fr = {"result": "", "status": "timeout", "summary": "max steps"}
            obs["finish_result"] = fr
            return obs, 0.0, True, {"finished": True, "finish_result": fr}
        return obs, 0.0, False, {}

    def evaluate(self, final_text: str) -> str:
        """Match the official _evaluate_against_ground_truth scheduling rules."""
        gt_path = self.scenario / "task_plans"
        if not gt_path.exists():
            return "no_ground_truth"
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        slot = _parse_slot(final_text or "")
        if not slot:
            return "no_slot"
        schedules = [json.loads(p.read_text(encoding="utf-8")) for p in self.members]
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


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class GridWorld:
    """Implement MASBench enhanced-interaction physics and delivery scoring."""

    ITEM_SYM = {"wood": "w", "sand": "s", "stone": "S"}

    def __init__(self, map_json):
        m = map_json["map"]
        self.w = int(m["width"])
        self.h = int(m["height"])
        cfg = m.get("agent_config", {}) or {}
        self.view_size = int(cfg.get("view_size", 4))
        self.carry_cap = int(cfg.get("item_number", 5))
        self.target = {str(k): int(v) for k, v in map_json["task"]["scoring"].items()}
        self.cells = {}
        for x in range(1, self.w + 1):
            for y in range(1, self.h + 1):
                if x in (1, self.w) or y in (1, self.h):
                    self.cells[(x, y)] = self._cell("wall", 0, None)
                else:
                    self.cells[(x, y)] = self._cell("empty", 10, None)
        for el in m.get("elements", []):
            x, y = int(el["pos"][0]), int(el["pos"][1])
            t = el["type"]
            c = self.cells[(x, y)]
            if t == "wall":
                c["terrain"] = "wall"; c["capacity"] = 0; c["items"] = {}
            elif t in ("wood", "sand", "stone"):
                c["terrain"] = "resource"
                q = int(el.get("capacity", 1))
                c["capacity"] = max(c["capacity"], q)
                c["items"][t] = c["items"].get(t, 0) + q
            elif t == "table":
                c["terrain"] = "table"
                c["capacity"] = int(el.get("capacity", 10))
                tc = el.get("type_caps") or {}
                c["type_caps"] = {str(k): int(v) for k, v in tc.items()} if tc else None
        a = m.get("agents", [{}])[0]
        self.agent = {"x": int(a["pos"][0]), "y": int(a["pos"][1]), "inv_type": None, "inv_qty": 0}

    @staticmethod
    def _cell(terrain, capacity, type_caps):
        return {"terrain": terrain, "capacity": capacity, "type_caps": type_caps, "items": {}}

    def _cell_at(self, x, y):
        return self.cells.get((x, y))

    def _total_items(self, c):
        return sum(c["items"].values())

    def _remaining_total(self, c):
        return max(0, c["capacity"] - self._total_items(c))

    def _remaining_for_type(self, c, t):
        if c["type_caps"] is None or t not in c["type_caps"]:
            return None
        return max(0, c["type_caps"][t] - c["items"].get(t, 0))

    def _render_view(self):
        R = self.view_size // 2
        ax, ay = self.agent["x"], self.agent["y"]
        rows = []
        for dy in range(R, -R - 1, -1):
            row = []
            for dx in range(-R, R + 1):
                x, y = ax + dx, ay + dy
                if x == ax and y == ay:
                    row.append("Y"); continue
                c = self._cell_at(x, y)
                if c is None:
                    row.append("*")
                elif c["terrain"] == "wall":
                    row.append("W")
                elif c["terrain"] == "table":
                    row.append("T")
                else:
                    if c["items"]:
                        first = sorted(c["items"].items(), key=lambda kv: -kv[1])[0][0]
                        row.append(self.ITEM_SYM.get(first, "?"))
                    else:
                        row.append(".")
            rows.append(" ".join(row))
        return "\n".join(rows)

    def observe(self):
        ax, ay = self.agent["x"], self.agent["y"]
        c = self._cell_at(ax, ay)
        items = ", ".join(f"{k}:{v}" for k, v in c["items"].items()) or "none"
        if c["terrain"] == "table":
            rem = c["type_caps"] and {k: self._remaining_for_type(c, k) for k in c["type_caps"]} or {}
            extra = f"table remaining_by_type={rem}"
        else:
            extra = f"placeable_free={self._remaining_total(c)}"
        inv = (f"carrying {self.agent['inv_type']} x{self.agent['inv_qty']}"
               if self.agent["inv_type"] else "empty")
        return (
            f"Position: ({ax}, {ay})\n"
            f"Local view (Y=you):\n{self._render_view()}\n"
            f"Current tile: terrain={c['terrain']}; items={items}; {extra}\n"
            f"Inventory: {inv} (capacity {self.carry_cap})\n"
            f"Target: {self.target}"
        )

    def move(self, d):
        ax, ay = self.agent["x"], self.agent["y"]
        dx, dy = {"UP": (0, 1), "DOWN": (0, -1), "RIGHT": (1, 0), "LEFT": (-1, 0)}.get(d, (0, 0))
        nx, ny = ax + dx, ay + dy
        c = self._cell_at(nx, ny)
        if c is None or c["terrain"] == "wall":
            return f"move {d} failed (out of bounds or wall)."
        self.agent["x"], self.agent["y"] = nx, ny
        return f"moved {d} to ({nx}, {ny})."

    def pick_up(self, typ, qty):
        ax, ay = self.agent["x"], self.agent["y"]
        c = self._cell_at(ax, ay)
        if c["terrain"] == "table":
            return "pick_up failed (cannot pick up from table)."
        avail = c["items"].get(typ, 0)
        cur = self.agent["inv_type"]
        if cur is not None and cur != typ:
            return f"pick_up failed (already carrying {cur})."
        remaining_cap = self.carry_cap - self.agent["inv_qty"]
        if remaining_cap <= 0:
            return "pick_up failed (inventory full)."
        if avail <= 0:
            return f"pick_up failed (no {typ} on this tile)."
        max_pick = min(avail, remaining_cap)
        if qty > max_pick:
            return f"pick_up failed (qty {qty} > max {max_pick})."
        c["items"][typ] -= qty
        if c["items"][typ] <= 0:
            del c["items"][typ]
        self.agent["inv_type"] = typ
        self.agent["inv_qty"] += qty
        return f"picked up {typ} x{qty}."

    def put_down(self, qty):
        if not self.agent["inv_type"] or self.agent["inv_qty"] <= 0:
            return "put_down failed (not carrying)."
        ax, ay = self.agent["x"], self.agent["y"]
        c = self._cell_at(ax, ay)
        if c["terrain"] == "wall":
            return "put_down failed (wall)."
        typ = self.agent["inv_type"]
        rem_type = self._remaining_for_type(c, typ)
        free_cap = self._remaining_total(c)
        cap_limit = rem_type if rem_type is not None else free_cap
        max_put = min(self.agent["inv_qty"], cap_limit)
        if qty > max_put:
            return f"put_down failed (qty {qty} > max {max_put})."
        c["items"][typ] = c["items"].get(typ, 0) + qty
        self.agent["inv_qty"] -= qty
        if self.agent["inv_qty"] <= 0:
            self.agent["inv_type"] = None
            self.agent["inv_qty"] = 0
        where = "on table" if c["terrain"] == "table" else "on ground"
        return f"put down {typ} x{qty} {where}."

    def score(self):
        delivered = {}
        for (x, y), c in self.cells.items():
            if c["terrain"] == "table":
                for t, q in c["items"].items():
                    delivered[t] = delivered.get(t, 0) + q
        s = 0
        for t, target in self.target.items():
            s += min(delivered.get(t, 0), target)
        return s, delivered


class GameEnv(Environment):
    """Expose grid-world collection and delivery actions."""

    def __init__(self, map_path: Path):
        self.map_path = map_path
        self.map_id = map_path.name
        self.world = GridWorld(json.loads(map_path.read_text(encoding="utf-8")))
        self.target = self.world.target
        self._steps = 0
        self._done = False
        self.max_steps = 50
        # Retain the best delivery so far; delegation can reset the world.
        self.best_score = 0
        self.best_delivered = {}

    def _build_action_space(self) -> str:
        return (
            "Available actions:\n\n"
            "### observe\n"
            "Description: See your position, local view, current tile, inventory, target.\n"
            "Parameters: {}\n\n"
            "### move\n"
            'Description: Move one cell. UP=+y, DOWN=-y, RIGHT=+x, LEFT=-x.\n'
            'Parameters: {"direction": "UP|DOWN|LEFT|RIGHT"}\n\n'
            "### pick_up\n"
            'Description: Pick items from current tile (type: wood/sand/stone).\n'
            'Parameters: {"type": "wood|sand|stone", "qty": <int>}\n\n'
            "### put_down\n"
            'Description: Place carried items on current tile.\n'
            'Parameters: {"qty": <int>}\n\n'
            "### finish\n"
            'Description: Report completion to MainAgent.\n'
            'Parameters: {"result": "<summary>", "status": "done", "summary": "<delivered summary>"}\n'
        )

    def _build_instruction(self) -> str:
        return (
            "Collect resources (wood/sand/stone) in a grid world and deliver them onto TABLE tiles "
            "to satisfy the target amounts. Legend: W=wall, T=table, Y=you, w=wood, s=sand, S=stone, "
            ".=empty. You can carry ONE type at a time (up to capacity). Pick up only from resource tiles "
            "(never from table); score by putting items down on a TABLE. Observe often. "
            f"Target: {self.target}."
        )

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.map_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.max_steps,
            meta_data={"task_id": self.map_id, "target": self.target},
        )

    async def reset(self, seed=None) -> Observation:
        self._steps = 0
        self._done = False
        self.world = GridWorld(json.loads(self.map_path.read_text(encoding="utf-8")))
        return {"message": "Environment ready.", "observation": self.world.observe()}

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._steps += 1
        a = action.get("action", "")
        params = action.get("params", {}) or {}

        if a == "finish":
            self._done = True
            s, delivered = self.world.score()
            self._update_best(s, delivered)
            result = params.get("result", "")
            status = params.get("status", "done")
            summary = params.get("summary", "") or f"score={s} delivered={delivered}"
            fr = {"result": result, "status": status, "summary": summary}
            return (
                {"message": "Result reported.", "score": s, "delivered": delivered, "finish_result": fr},
                0.0, True, {"finished": True, "finish_result": fr, "score": s, "delivered": delivered},
            )

        if a == "observe":
            out = self.world.observe()
        elif a == "move":
            out = self.world.move(str(params.get("direction", "")).strip().upper())
        elif a == "pick_up":
            typ = str(params.get("type", "")).strip().lower()
            try:
                qty = int(params.get("qty", 0))
            except (TypeError, ValueError):
                out = "pick_up qty must be an integer."
            else:
                out = self.world.pick_up(typ, qty)
        elif a == "put_down":
            try:
                qty = int(params.get("qty", 0))
            except (TypeError, ValueError):
                out = "put_down qty must be an integer."
            else:
                out = self.world.put_down(qty)
        else:
            out = f"Unknown action '{a}'. Available: observe, move, pick_up, put_down, finish."

        obs = {"output": out, "current_step": self._steps}
        if self._steps >= self.max_steps:
            self._done = True
            s, delivered = self.world.score()
            self._update_best(s, delivered)
            fr = {"result": "", "status": "timeout", "summary": f"score={s}"}
            obs["finish_result"] = fr
            return obs, 0.0, True, {"finished": True, "finish_result": fr, "score": s, "delivered": delivered}
        return obs, 0.0, False, {}

    def _update_best(self, s, delivered):
        if s > self.best_score:
            self.best_score = s
            self.best_delivered = delivered
