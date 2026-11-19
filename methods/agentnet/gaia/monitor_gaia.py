#!/usr/bin/env python3
"""GAIA 运行监视器：只读，不干扰正在跑的实验。

  python3 monitor_gaia.py                  # 自动选 save/ 下最新的 gaia_* 目录
  python3 monitor_gaia.py save/gaia_xxx    # 指定目录

输出一段简报；若有异常，末尾追加 "!! 告警 ..." 行。
退出码：0 正常/已完成，1 有告警。
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "save")
EXPECTED = 165                      # validation 全量题数
STALL_SECONDS = 900                 # 超过 15 分钟没有新结果就判定卡住


def pick_dir(argv):
    if len(argv) > 1:
        return argv[1] if os.path.isabs(argv[1]) else os.path.join(HERE, argv[1])
    cands = []
    for name in os.listdir(SAVE):
        p = os.path.join(SAVE, name)
        if name.startswith("gaia_") and os.path.isdir(p):
            cands.append((os.path.getmtime(p), p))
    if not cands:
        sys.exit("找不到 save/gaia_* 目录")
    return max(cands)[1]


def proc_alive(run_dir):
    """找出正在写这个目录的 run_gaia.py 进程（命令行里可能是相对路径，用目录名匹配）。"""
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

    print(f"== GAIA 监视 {time.strftime('%H:%M:%S')} ==")
    print(f"目录: {rel}")
    print(f"进程: {'运行中 PID ' + pid if pid else '未在运行'}")
    print(f"进度: {n}/{EXPECTED} ({n / EXPECTED:.1%})  正确 {ok}/{n}" +
          (f" ({ok / n:.1%})" if n else ""))

    if n:
        avg_s = work_s / n
        avg_c = cost / n
        eta_h = (EXPECTED - n) * avg_s / 3600
        print(f"单题: {avg_s / 60:.1f} min, {fmt_money(avg_c)}")
        print(f"累计: {work_s / 3600:.1f} h 墙钟, {fmt_money(cost)}")
        if pid:
            print(f"预计剩余: {eta_h:.1f} h   全量成本约 {fmt_money(avg_c * EXPECTED)}")

    if silent is not None:
        print(f"最后写入: {silent / 60:.0f} 分钟前")

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
            print("截断(finish=length): " + " | ".join(parts))

    log_txt = ""
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            log_txt = f.read()
    fallbacks = log_txt.count("unusable router decision")
    converged = log_txt.count("router did not converge")
    if fallbacks or converged:
        print(f"兜底触发: 空决策回退 {fallbacks} 次, 轮次超限 {converged} 次")

    alerted = []
    if errs:
        kinds = Counter((r.get("error") or "").split(":")[0] for r in errs)
        print(f"出错题数: {len(errs)}/{n}  " + ", ".join(f"{k}×{v}" for k, v in kinds.most_common(3)))
        if len(errs) / n > 0.2:
            alerted.append(f"{len(errs)}/{n} 题异常，比例过高")
    if bad:
        alerted.append(f"results.jsonl 有 {bad} 行无法解析")

    if pid is None:
        if n >= EXPECTED:
            print("状态: 已跑完")
        elif os.path.exists(os.path.join(run_dir, "summary.json")):
            print("状态: 进程已退出并写了 summary（可能被中断）")
            alerted.append("进程已退出但只完成 %d/%d" % (n, EXPECTED))
        else:
            print("状态: 进程已消失且无 summary")
            alerted.append("进程消失且未写 summary")
    elif n == 0:
        print("状态: 正常（刚启动，还没有完成第一题）")
    elif silent is not None and silent > STALL_SECONDS and n < EXPECTED:
        print("状态: 疑似卡住")
        alerted.append(f"{silent / 60:.0f} 分钟无新结果")
    else:
        print("状态: 正常")

    for a in alerted:
        print(f"!! 告警 {a}")
    return 1 if alerted else 0


if __name__ == "__main__":
    sys.exit(main())
