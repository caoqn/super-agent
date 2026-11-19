"""用 AgentNet 跑 MAS-Bench Reasoning 任务的适配脚本。

- 复用 MAS-Bench 的数据加载、prompt、评分（F1）
- 用 AgentNet 的 LLM 调用层（gaia/llm.py，关思考 + 中转）替代 openai 客户端
- 通过 gaia.tracker 记录每次调用的 token / 成本 / 耗时

用法（在 Reasoning/ 目录下）：
  python run_agentnet_reasoning.py --num-samples 1 --max-rounds 6 --seed 42
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# AgentNet 适配层（打包后 gaia/ 位于本包根目录）
AGENTNET = str(Path(__file__).resolve().parents[1])
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, AGENTNET)
from gaia.envfile import load_env  # noqa: E402
load_env([os.path.join(AGENTNET, ".env"), str(REPO_ROOT / ".env")])
from gaia.llm import chat  # noqa: E402
from gaia.tracker import TRACKER  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.data_loader_v4_mas import V4MasDataLoader  # noqa: E402
from src.prompts import QA_INFO_EXCHANGE_PROMPT  # noqa: E402
from src.metrics import cal_f1_score  # noqa: E402
from src.conversation import parse_answer, run_conversation  # noqa: E402
from src.agents import BaseAgent  # noqa: E402


class AgentNetAgent(BaseAgent):
    """MAS-Bench 的 agent 接口，但 LLM 调用走 AgentNet 的 gaia/llm.py（关思考）。"""

    def __init__(self, name: str, max_tokens: int = 1024):
        super().__init__(name=name)
        self.system_prompt = ""
        self.memory = []
        self.max_tokens = max_tokens

    def step(self) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        my_prefix = f"{self.name}:"
        for msg in self.memory:
            role = "assistant" if msg.content.strip().startswith(my_prefix) else "user"
            messages.append({"role": role, "content": msg.content})
        reply = chat(messages, role="executor", max_tokens=self.max_tokens)
        content = (reply.get("content") or "").strip()
        if not content.startswith(f"{self.name}:"):
            content = f"{self.name}: {content}"
        self.add_memory(content)
        return content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=1, help="-1 = 全部 300 题")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--mode", default="context", choices=["context", "support"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="输出 jsonl 路径")
    p.add_argument("--data", default=None, help="数据文件路径（默认 MASBench_selected_30）")
    args = p.parse_args()

    data_path = args.data or (HERE.parent / "MASBench_selected_30/Reasoning/reasoning_subset.json")
    loader = V4MasDataLoader(str(data_path), mode=args.mode, seed=args.seed)
    total = len(loader)
    n = total if args.num_samples < 0 else min(args.num_samples, total)

    out = args.out or f"results_agentnet_{int(time.time())}.jsonl"
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = HERE / out_path

    # 断点续跑：跳过已有结果里的 item_id
    done_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for l in f:
                try:
                    done_ids.add(json.loads(l)["item_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    run_t0 = time.time()
    recs = []
    for i in range(n):
        sample = loader.sample(i)
        if sample.item_id in done_ids:
            continue
        alice = AgentNetAgent("Alice", max_tokens=args.max_tokens)
        bob = AgentNetAgent("Bob", max_tokens=args.max_tokens)
        alice.init_system_prompt(QA_INFO_EXCHANGE_PROMPT, {
            "name": "Alice", "partner": "Bob", "question": sample.question,
            "information": sample.context_first or "(empty for Alice)",
        })
        bob.init_system_prompt(QA_INFO_EXCHANGE_PROMPT, {
            "name": "Bob", "partner": "Alice", "question": sample.question,
            "information": sample.context_second or "(empty for Bob)",
        })

        with TRACKER.task_scope(sample.item_id):
            transcript, final_answer = run_conversation([alice, bob], max_rounds=args.max_rounds)
        usage = TRACKER.summarize(sample.item_id)
        f1 = cal_f1_score(sample.answer, final_answer) if final_answer is not None else None

        rec = {
            "item_id": sample.item_id, "level": sample.level,
            "f1": f1, "final_answer": final_answer, "ground_truth": sample.answer,
            "rounds": len(transcript),
            "wall_s": usage["llm_time_s"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "llm_calls": usage["llm_calls"],
            "llm_cost": usage["llm_cost"],
        }
        recs.append(rec)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i+1}/{n}] {sample.item_id} F1={f1 if f1 is not None else 'n/a'} "
              f"rounds={len(transcript)} tok={usage['prompt_tokens']}/{usage['completion_tokens']} "
              f"llm={usage['llm_time_s']:.0f}s", flush=True)

    scored = [r for r in recs if r["f1"] is not None]
    mean_f1 = sum(r["f1"] for r in scored) / len(scored) if scored else None
    total_tok = sum(r["prompt_tokens"] + r["completion_tokens"] for r in recs)
    total_time = time.time() - run_t0
    summary = {
        "n": n, "mean_f1": mean_f1, "answered": len(scored),
        "total_tokens": total_tok, "total_cost": sum(r["llm_cost"] for r in recs),
        "total_wall_s": round(total_time, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
