"""ReAct adapter for 30 MASBench Game maps.

A single ReAct agent explores, collects, and delivers items in a grid world using observe / move / pick_up / put_down tools.
Physics and scoring reproduce the official enhanced-interaction semantics:
- Movement is blocked by walls and boundaries; x increases left to right, y increases bottom to top (UP=+y).
- Agents carry one item type at a time. PICK_UP cannot collect from a table; QTY is limited by remaining capacity and the tile's available items.
- PUT_DOWN is limited by the tile's remaining per-type table capacity.
- Score = sum(min(item count on table, target_items)).
"""
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
MASBENCH = REPO_ROOT / "MASBench-main"
MAPS = sorted((MASBENCH / "MASBench_selected_30/Game/maps").glob("*.json"))

sys.path.insert(0, str(REPO_ROOT))
from methods.react.react import run_react  # noqa: E402

MAX_STEPS = 50
MAX_TOKENS = 512

SYSTEM = """You are a single agent in a grid-world sandbox. Solve the resource-collection task by interleaving Thought, Action, Observation steps.

Tools:
- observe[]          : see your position, local view, current tile, inventory.
- move[UP|DOWN|LEFT|RIGHT] : move one cell (UP=+y, DOWN=-y, RIGHT=+x, LEFT=-x).
- pick_up[TYPE,QTY]  : pick items from your current tile (TYPE: wood/sand/stone).
- put_down[QTY]      : place your carried items on your current tile.
- Finish[]           : stop early (optional).

Each turn output exactly two lines:
Thought: <reasoning>
Action: <tool[args]>

Rules:
- Legend: W=wall, T=table, Y=you, w=wood, s=sand, S=stone, .=empty.
- You can carry only ONE item type at a time, up to your capacity.
- Pick up only from your current tile (never from a table). To pick, first move onto the resource tile.
- To score, carry items onto a TABLE tile and put_down there. Tables accept limited per-type amounts.
- Observe often to keep track of where resources and tables are."""


class GridWorld:
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
        for dy in range(R, -R - 1, -1):  # Render rows from +R to -R, with north (+y) at the top.
            row = []
            for dx in range(-R, R + 1):
                x, y = ax + dx, ay + dy
                if x == ax and y == ay:
                    row.append("Y")
                    continue
                c = self._cell_at(x, y)
                if c is None:
                    row.append("*")
                elif c["terrain"] == "wall":
                    row.append("W")
                elif c["terrain"] == "table":
                    row.append("T")
                else:
                    # Display an item symbol for occupied tiles.
                    if c["items"]:
                        first = sorted(c["items"].items(), key=lambda kv: -kv[1])[0][0]
                        row.append(self.ITEM_SYM.get(first, "?"))
                    elif c["terrain"] == "resource":
                        row.append(".")
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


def run_one(map_path: Path):
    world = GridWorld(json.loads(map_path.read_text(encoding="utf-8")))

    def tools_observe(args):
        return world.observe()

    def tools_move(args):
        return world.move((args[0] if args else "").strip().upper())

    def tools_pick(args):
        if len(args) < 2:
            return "pick_up needs TYPE,QTY."
        try:
            qty = int(args[1])
        except ValueError:
            return "pick_up QTY must be an integer."
        return world.pick_up(args[0].strip().lower(), qty)

    def tools_put(args):
        if not args:
            return "put_down needs QTY."
        try:
            qty = int(args[0])
        except ValueError:
            return "put_down QTY must be an integer."
        return world.put_down(qty)

    tools = {"observe": tools_observe, "move": tools_move,
             "pick_up": tools_pick, "put_down": tools_put}

    t0 = time.time()
    final, usage, traj = run_react(
        SYSTEM, tools,
        first_message="Observation: " + world.observe(),
        max_steps=MAX_STEPS, max_tokens=MAX_TOKENS,
    )
    wall = time.time() - t0
    score, delivered = world.score()
    return {
        "map": map_path.name,
        "score": score,
        "delivered": delivered,
        "target": world.target,
        "steps": len(traj),
        "finished": final is not None,
        "wall_s": round(wall, 1),
        **usage.as_dict(),
    }


def main():
    out = HERE / "results" / "game_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    resume = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                resume.add(json.loads(line)["map"])
            except Exception:
                pass
    recs = []
    for i, m in enumerate(MAPS):
        if m.name in resume:
            continue
        rec = run_one(m)
        recs.append(rec)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i + 1}/{len(MAPS)}] {m.name} score={rec['score']} "
              f"steps={rec['steps']} tok={rec['total_tokens']} {rec['wall_s']}s", flush=True)

    if recs:
        print(f"\n=== Game summary ===\nMaps {len(recs)}  Total score {sum(r['score'] for r in recs)}  "
              f"Mean {sum(r['score'] for r in recs) / len(recs):.2f}")


if __name__ == "__main__":
    main()
