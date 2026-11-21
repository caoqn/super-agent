#!/usr/bin/env python3
"""Read-only MAC-Bench run monitor. Usage: python monitor_macbench.py [result_dir]."""
import json
import os
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "save")


def pick_dir(argv):
    if len(argv) > 1:
        return argv[1] if os.path.isabs(argv[1]) else os.path.join(HERE, argv[1])
    cands = []
    for name in os.listdir(SAVE):
        p = os.path.join(SAVE, name)
        if name.startswith("macbench_") and os.path.isdir(p):
            cands.append((os.path.getmtime(p), p))
    return max(cands)[1]


def proc_alive(run_dir):
    try:
        out = subprocess.run(["pgrep", "-fl", "run_macbench.py"], capture_output=True, text=True).stdout
    except Exception:
        return None
    base = os.path.basename(run_dir.rstrip("/"))
    for line in out.splitlines():
        if base in line or run_dir in line:
            return line.split()[0]
    return None


def load_results(path):
    if not os.path.exists(path):
        return []
    recs, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                recs.append(json.loads(l))
            except json.JSONDecodeError:
                bad += 1
    return recs


def main():
    run_dir = pick_dir(sys.argv)
    rel = os.path.relpath(run_dir, HERE)
    recs = load_results(os.path.join(run_dir, "results.jsonl"))
    pid = proc_alive(run_dir)
    n = len(recs)
    ps_vals = [r.get("ps") or 0 for r in recs]
    rr_vals = [r.get("rr") or 0 for r in recs]
    print(f"== MAC-Bench monitor {time.strftime('%H:%M:%S')} ==")
    print(f"Directory: {rel}")
    print(f"Process: {'running PID ' + pid if pid else 'not running'}")
    print(f"Completed: {n} tasks")
    if n:
        print(f"Mean PS: {sum(ps_vals)/n:.3f}   Mean RR: {sum(rr_vals)/n:.3f}")
        print(f"Tasks with PS>0: {sum(1 for p in ps_vals if p > 0)}/{n}")
        # Per-task PS values.
        for r in recs:
            print(f"  {r.get('task_id','?'):30s} PS={r.get('ps')} RR={r.get('rr')} {r.get('stop_reason','')}")
        cost = sum(r.get("cost") or 0 for r in recs)
        print(f"Estimated cost: ¥{cost:.2f}")
    else:
        print("No completed tasks yet (the first sandbox may still be starting)")
    # Report unsuccessful episodes.
    errs = [r for r in recs if r.get("status") not in ("completed", None)]
    if errs:
        print(f"Non-completed tasks: {len(errs)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
