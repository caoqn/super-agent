#!/usr/bin/env python3
"""Read-only GAIA monitor. Defaults to the latest save/gaia_* run.

  python3 monitor_gaia.py                  # Automatically select the latest gaia_* directory under save/
  python3 monitor_gaia.py save/gaia_xxx    # Select a directory

Prints a short report and appends a "!! Warning ..." line when issues are found.
Exit codes: 0 for healthy or completed; 1 when warnings are present.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "save")
EXPECTED = 165                      # Full validation-set task count.
STALL_SECONDS = 900                 # Flag a stall after 15 minutes without new results.


def pick_dir(argv):
    if len(argv) > 1:
        return argv[1] if os.path.isabs(argv[1]) else os.path.join(HERE, argv[1])
    cands = []
    for name in os.listdir(SAVE):
        p = os.path.join(SAVE, name)
        if name.startswith("gaia_") and os.path.isdir(p):
            cands.append((os.path.getmtime(p), p))
    if not cands:
        sys.exit("No save/gaia_* directory found")
    return max(cands)[1]


def proc_alive(run_dir):
    """Find the run_gaia.py process writing this directory, matching its directory name."""
    try:
        out = subprocess.run(["pgrep", "-fl", "run_gaia.py"], capture_output=True, text=True).stdout
    except Exception:
        return None
    base = os.path.basename(run_dir.rstrip("/"))
    for line in out.splitlines():
        if base in line or run_dir in line:
            return line.split()[0]
    return None


def load_results(path):
    recs, bad = [], 0
    if not os.path.exists(path):
        return recs, bad
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return recs, bad


def truncation_stats(path):
    st = defaultdict(Counter)
    if not os.path.exists(path):
        return st
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if x.get("kind") == "llm":
                st[x.get("role")][x.get("finish_reason")] += 1
    return st


def fmt_money(x):
    return f"¥{x:.3f}"


def main():
    run_dir = pick_dir(sys.argv)
    rel = os.path.relpath(run_dir, HERE)
    results_path = os.path.join(run_dir, "results.jsonl")
    log_path = os.path.join(run_dir, "agentnet.log")

    recs, bad = load_results(results_path)
    pid = proc_alive(run_dir)
    n = len(recs)
    ok = sum(1 for r in recs if r.get("correct"))
    cost = sum(r.get("total_cost") or 0 for r in recs)
    work_s = sum((r.get("wall_time_s") or 0) + (r.get("update_time_s") or 0) for r in recs)
    errs = [r for r in recs if r.get("error")]

    now = time.time()
    last_write = os.path.getmtime(results_path) if os.path.exists(results_path) else None
    silent = (now - last_write) if last_write else None

    print(f"== GAIA monitor {time.strftime('%H:%M:%S')} ==")
    print(f"Directory: {rel}")
    print(f"Process: {'running PID ' + pid if pid else 'not running'}")
    print(f"Progress: {n}/{EXPECTED} ({n / EXPECTED:.1%})  correct {ok}/{n}" +
          (f" ({ok / n:.1%})" if n else ""))

    if n:
        avg_s = work_s / n
        avg_c = cost / n
        eta_h = (EXPECTED - n) * avg_s / 3600
        print(f"Per task: {avg_s / 60:.1f} min, {fmt_money(avg_c)}")
        print(f"Total: {work_s / 3600:.1f} h wall time, {fmt_money(cost)}")
        if pid:
            print(f"Estimated remaining: {eta_h:.1f} h   Estimated full-run cost {fmt_money(avg_c * EXPECTED)}")

    if silent is not None:
        print(f"Last write: {silent / 60:.0f} minutes ago")

    st = truncation_stats(os.path.join(run_dir, "calls.jsonl"))
    if st:
        parts = []
        for role in ("router", "router_next_agent", "thought", "executor"):
            c = st.get(role)
            if not c:
                continue
            tot = sum(c.values())
            cut = c.get("length", 0)
            parts.append(f"{role} {cut}/{tot}")
        if parts:
            print("Truncated (finish=length): " + " | ".join(parts))

    log_txt = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log_txt = f.read()
    fallbacks = log_txt.count("unusable router decision")
    converged = log_txt.count("router did not converge")
    if fallbacks or converged:
        print(f"Fallbacks: empty decision {fallbacks} times, round limit {converged} times")

    alerted = []
    if errs:
        kinds = Counter((r.get("error") or "").split(":")[0] for r in errs)
        print(f"Failed tasks: {len(errs)}/{n}  " + ", ".join(f"{k}×{v}" for k, v in kinds.most_common(3)))
        if len(errs) / n > 0.2:
            alerted.append(f"{len(errs)}/{n} tasks failed; failure rate too high")
    if bad:
        alerted.append(f"results.jsonl has {bad} unparsable lines")

    if pid is None:
        if n >= EXPECTED:
            print("Status: completed")
        elif os.path.exists(os.path.join(run_dir, "summary.json")):
            print("Status: process exited with summary (possibly interrupted)")
            alerted.append("Process exited after completing only %d/%d" % (n, EXPECTED))
        else:
            print("Status: process exited without a summary")
            alerted.append("Process exited without writing a summary")
    elif n == 0:
        print("Status: healthy (starting; no task completed yet)")
    elif silent is not None and silent > STALL_SECONDS and n < EXPECTED:
        print("Status: possible stall")
        alerted.append(f"{silent / 60:.0f} minutes without new results")
    else:
        print("Status: healthy")

    for a in alerted:
        print(f"!! Warning {a}")
    return 1 if alerted else 0


if __name__ == "__main__":
    sys.exit(main())
