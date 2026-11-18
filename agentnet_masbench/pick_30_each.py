"""从 MAS-Bench 三个任务各挑 30 个，数据放到独立文件夹（seed=42 可复现）。

- Reasoning: 30 题 = 20 easy + 10 hard（对齐 2:1 原始比例）
- Scheduling: 30 场景（随机）
- Game: 30 地图 = 15×15 各 15（对齐 1:1 原始比例）
"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MASBench_selected_30"
SEED = 42

rng = random.Random(SEED)


def pick_reasoning():
    data = json.loads((ROOT / "Reasoning/reasoning.json").read_text())
    easy = [x for x in data if x.get("level") == "easy"]
    hard = [x for x in data if x.get("level") == "hard"]
    rng.shuffle(easy)
    rng.shuffle(hard)
    picked = easy[:20] + hard[:10]
    dst = OUT / "Reasoning"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "reasoning_subset.json").write_text(json.dumps(picked, ensure_ascii=False, indent=2))
    return [x.get("id") for x in picked]


def pick_scheduling():
    base = ROOT / "Scheduling/meeting/minutes_10/5days"
    scenarios = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            for s in sorted(d.iterdir()):
                if s.is_dir():
                    scenarios.append((d.name, s.name, s))
    rng.shuffle(scenarios)
    picked = scenarios[:30]
    dst = OUT / "Scheduling/scenarios"
    dst.mkdir(parents=True, exist_ok=True)
    for days, sid, src in picked:
        shutil.copytree(src, dst / f"{days}_{sid}")
    return [f"{d}_{s}" for d, s, _ in picked]


def pick_game():
    src = ROOT / "Game/custom_map"
    maps15 = sorted(src.glob("15x15_*.json"))
    maps18 = sorted(src.glob("18x18_*.json"))
    rng.shuffle(maps15)
    rng.shuffle(maps18)
    picked = maps15[:15] + maps18[:15]
    dst = OUT / "Game/maps"
    dst.mkdir(parents=True, exist_ok=True)
    for m in picked:
        shutil.copyfile(m, dst / m.name)
    return [m.name for m in picked]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    r_ids = pick_reasoning()
    s_ids = pick_scheduling()
    g_ids = pick_game()
    manifest = {
        "seed": SEED,
        "reasoning": {"n": len(r_ids), "ids": r_ids},
        "scheduling": {"n": len(s_ids), "ids": s_ids},
        "game": {"n": len(g_ids), "ids": g_ids},
    }
    (OUT / "selection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"挑选完成，输出目录: {OUT}")
    print(f"  Reasoning: {len(r_ids)} 题")
    print(f"  Scheduling: {len(s_ids)} 场景")
    print(f"  Game: {len(g_ids)} 地图")
    print(f"  合计: {len(r_ids) + len(s_ids) + len(g_ids)} 个任务")


if __name__ == "__main__":
    main()
