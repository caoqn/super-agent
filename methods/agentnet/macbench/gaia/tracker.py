"""Records every model / embedding / tool call with tokens, latency and cost.

A "scope" (thread-local) tags calls with the current task id and the AgentNet
module that issued them (router / thought / executor / tool ...).
"""
import json
import os
import threading
import time
from contextlib import contextmanager

_local = threading.local()
_lock = threading.Lock()


def _f(name, default=0.0):
    try:
        return float(os.environ.get(name, default) or default)
    except ValueError:
        return float(default)


class CostTracker:
    def __init__(self):
        self.records = []
        self.log_path = None
        self.price_in = _f("LLM_PRICE_INPUT_PER_1M")
        self.price_out = _f("LLM_PRICE_OUTPUT_PER_1M")
        self.price_cached_in = _f("LLM_PRICE_CACHED_INPUT_PER_1M", self.price_in)
        self.emb_price = _f("EMB_PRICE_PER_1M")
        self.currency = os.environ.get("LLM_PRICE_CURRENCY", "USD")

    def reload_prices(self):
        self.price_in = _f("LLM_PRICE_INPUT_PER_1M")
        self.price_out = _f("LLM_PRICE_OUTPUT_PER_1M")
        self.price_cached_in = _f("LLM_PRICE_CACHED_INPUT_PER_1M", self.price_in)
        self.emb_price = _f("EMB_PRICE_PER_1M")
        self.currency = os.environ.get("LLM_PRICE_CURRENCY", "USD")

    def set_log_path(self, path):
        self.log_path = path

    # ---- scope -----------------------------------------------------------
    @staticmethod
    def current_task():
        return getattr(_local, "task_id", None)

    @staticmethod
    def current_role():
        return getattr(_local, "role", None)

    @contextmanager
    def task_scope(self, task_id):
        prev = getattr(_local, "task_id", None)
        _local.task_id = task_id
        try:
            yield
        finally:
            _local.task_id = prev

    @contextmanager
    def role_scope(self, role):
        prev = getattr(_local, "role", None)
        _local.role = role
        try:
            yield
        finally:
            _local.role = prev

    # ---- recording -------------------------------------------------------
    def llm_cost(self, prompt_tokens, completion_tokens, cached_tokens=0):
        uncached = max(0, prompt_tokens - cached_tokens)
        return (uncached * self.price_in + cached_tokens * self.price_cached_in
                + completion_tokens * self.price_out) / 1e6

    def record(self, kind, latency, prompt_tokens=0, completion_tokens=0,
               cached_tokens=0, estimated=False, extra=None):
        if kind == "llm":
            cost = self.llm_cost(prompt_tokens, completion_tokens, cached_tokens)
        elif kind == "embedding":
            cost = prompt_tokens * self.emb_price / 1e6
        else:
            cost = 0.0
        rec = {
            "ts": time.time(),
            "task_id": self.current_task(),
            "role": self.current_role(),
            "kind": kind,
            "latency_s": round(latency, 4),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cached_tokens": int(cached_tokens),
            "tokens_estimated": bool(estimated),
            "cost": cost,
        }
        if extra:
            rec.update(extra)
        with _lock:
            self.records.append(rec)
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---- aggregation -----------------------------------------------------
    def summarize(self, task_id=None):
        recs = [r for r in self.records if task_id is None or r["task_id"] == task_id]
        out = {
            "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "cached_tokens": 0, "llm_time_s": 0.0, "llm_cost": 0.0,
            "embedding_calls": 0, "embedding_time_s": 0.0, "embedding_cost": 0.0,
            "tool_calls": 0, "tool_time_s": 0.0, "tokens_estimated_calls": 0,
            "by_role": {},
        }
        for r in recs:
            if r["kind"] == "llm":
                out["llm_calls"] += 1
                out["prompt_tokens"] += r["prompt_tokens"]
                out["completion_tokens"] += r["completion_tokens"]
                out["cached_tokens"] += r["cached_tokens"]
                out["llm_time_s"] += r["latency_s"]
                out["llm_cost"] += r["cost"]
                out["tokens_estimated_calls"] += int(r["tokens_estimated"])
                role = r.get("role") or "other"
                br = out["by_role"].setdefault(role, {"calls": 0, "prompt_tokens": 0,
                                                      "completion_tokens": 0, "cost": 0.0})
                br["calls"] += 1
                br["prompt_tokens"] += r["prompt_tokens"]
                br["completion_tokens"] += r["completion_tokens"]
                br["cost"] += r["cost"]
            elif r["kind"] == "embedding":
                out["embedding_calls"] += 1
                out["embedding_time_s"] += r["latency_s"]
                out["embedding_cost"] += r["cost"]
            elif r["kind"] == "tool":
                out["tool_calls"] += 1
                out["tool_time_s"] += r["latency_s"]
        out["total_cost"] = out["llm_cost"] + out["embedding_cost"]
        for k in ("llm_time_s", "embedding_time_s", "tool_time_s"):
            out[k] = round(out[k], 3)
        return out


TRACKER = CostTracker()
