"""跑一个 Game 地图，正确收集输入/输出 token。

monkeypatch MASLogger.log_agent_action，让每条 agent_log 额外记录：
    usage_prompt_tokens / usage_completion_tokens / usage_total_tokens（该 agent 累计值）。
跑完后取每个 agent 最后一轮的累计值，得到准确的输入/输出 token。

用法: python run_game_one_token.py <map.json>
"""
import os
import sys
import json
import glob
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(ROOT)))
GAME = os.path.join(ROOT, "Game")
sys.path.insert(0, GAME)

# 清代理（httpx 会因 NO_PROXY 里的 [::1] 崩溃）
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
          "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy", "grpc_proxy"]:
    os.environ.pop(k, None)

from gaia.envfile import load_env

load_env([os.path.join(ROOT, ".env"), os.path.join(REPO_ROOT, ".env")])
KEY = os.environ["LLM_API_KEY"]
BASE = os.environ["LLM_BASE_URL"]
os.environ["QWEN_OPENAI_API_KEY"] = KEY
os.environ["QWEN_OPENAI_BASE"] = BASE
os.environ["MASBENCH_DISABLE_THINKING"] = "1"
os.environ["MECHANISM_PRESET"] = "base"

map_arg = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "MASBench_selected_30/Game/maps/15x15_v10.json")
map_path = os.path.abspath(map_arg)

# ---- monkeypatch logger ----
from masbench.logger import MASLogger  # noqa: E402
from masbench import MASFramework  # noqa: E402

_orig = MASLogger.log_agent_action


def _patched(self, env, agent_id, prompt, response, action, message,
             api_call_time, action_success, view, total_llm_tokens, extra=None):
    extra = dict(extra or {})
    try:
        brain = env.agents[agent_id].brain
        extra["usage_prompt_tokens"] = int(getattr(brain, "usage_prompt", 0) or 0)
        extra["usage_completion_tokens"] = int(getattr(brain, "usage_completion", 0) or 0)
        extra["usage_total_tokens"] = int(getattr(brain, "usage_total", 0) or 0)
    except Exception:
        pass
    return _orig(self, env, agent_id, prompt, response, action, message,
                 api_call_time, action_success, view, total_llm_tokens, extra)


MASLogger.log_agent_action = _patched

# ---- 解析 map 参数（复现 eval.py 逻辑）----
with open(map_path) as f:
    custom_map = json.load(f)
map_cfg = custom_map.get("map", {})
agent_cfg = map_cfg.get("agent_config", {}) or {}
num_agents = len(map_cfg.get("agents", []) or []) or 2
agent_number = agent_cfg.get("agent_number")
if isinstance(agent_number, int) and agent_number > 0:
    num_agents = max(num_agents, agent_number)
width = map_cfg.get("width") or 12
height = map_cfg.get("height") or 12
view_size = agent_cfg.get("view_size") or 9
communication_size = agent_cfg.get("communication_size") or max(9, (view_size or 9) + 3)
item_cap = agent_cfg.get("item_number")
if not (isinstance(item_cap, int) and item_cap > 0):
    item_cap = 4
max_round = 50

# ---- 生成 log_dir（和 eval.py 相同格式）----
from datetime import datetime  # noqa: E402
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
map_label = os.path.splitext(os.path.basename(map_path))[0]
log_dir = os.path.join(GAME, ".preset_logs", f"{timestamp}_{map_label}_base_deepseek-v4.1-flash_viewjson")

before = set(glob.glob(os.path.join(GAME, ".preset_logs", "*")))

# ---- 跑 ----
t0 = time.time()
MASFramework.submit(
    "exp_1",
    MASFramework.model_config("deepseek-v4.1-flash", KEY, BASE, provider="QWEN"),
    "Foraging",
    log_dir=log_dir,
    num_agents=num_agents,
    max_round=max_round,
    width=width,
    height=height,
    seed=42,
    view_size=view_size,
    communication_size=communication_size,
    custom_map=custom_map,
    custom_map_path=map_path,
    agent_carrying_capacity=item_cap,
    messages_enabled=True,
    mechanism_preset="base",
)
MASFramework.run_all(max_parallel=1)
wall = time.time() - t0

# ---- 收集结果 ----
al_files = sorted(glob.glob(os.path.join(log_dir, "agent_log*.json")))
if not al_files:
    print("ERROR: 没有 agent_log 在", log_dir)
    sys.exit(1)

d = json.load(open(al_files[-1]))
score = 0
for gl in sorted(glob.glob(os.path.join(log_dir, "game_log*.json"))):
    g = json.load(open(gl))
    last = g[-1] if isinstance(g, list) else g
    score = last.get("score", 0) or 0

by_agent = {}
for r in d:
    by_agent.setdefault(r["agent_id"], []).append((r.get("round", 0), r))

prompt_total = 0
comp_total = 0
total_total = 0
print(f"\n=== 各 agent 最后一轮累计 token ===")
for ag in sorted(by_agent):
    seq = sorted(by_agent[ag], key=lambda x: x[0])
    last = seq[-1][1]
    pt = last.get("usage_prompt_tokens", 0) or 0
    ct = last.get("usage_completion_tokens", 0) or 0
    tt = last.get("usage_total_tokens", 0) or 0
    prompt_total += pt
    comp_total += ct
    total_total += tt
    print(f"  {ag}: 输入(prompt)={pt:,} 输出(completion)={ct:,} 总={tt:,}")

print(f"\n=== 汇总 ===")
print(f"  map: {os.path.basename(map_path)}  (num_agents={num_agents}, view={view_size}, cap={item_cap})")
print(f"  score: {score}")
print(f"  输入 token: {prompt_total:,}")
print(f"  输出 token: {comp_total:,}")
print(f"  总 token: {total_total:,}")
print(f"  耗时: {wall:.1f}s")
