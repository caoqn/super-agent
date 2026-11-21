"""OpenAI-compatible chat client for the Vanilla baseline (standard library only).

Uses the calling convention validated with AgentNet:
- Reads the base URL from LLM_BASE_URL and calls {base}/chat/completions.
- Disables thinking for deepseek-v4.1-flash by default to avoid timeouts.
- Tracks prompt/completion tokens and cost per call (CNY rates from .env).
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = (Path(__file__).resolve().parents[2] / ".env")


def _load_env() -> dict:
    env = {}
    if not ENV_FILE.exists():
        return env
    for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.split(" #")[0].strip()  # Remove trailing comments.
        env[k] = v
    return env


_ENV = {**_load_env(), **os.environ}
BASE = _ENV.get("LLM_BASE_URL", "").rstrip("/")
KEY = _ENV.get("LLM_API_KEY", "")
MODEL = _ENV.get("LLM_MODEL", "deepseek-v4.1-flash")
PRICE_INPUT_PER_1M = float(_ENV.get("LLM_PRICE_INPUT_PER_1M", "1.005"))
PRICE_OUTPUT_PER_1M = float(_ENV.get("LLM_PRICE_OUTPUT_PER_1M", "4.02"))
TIMEOUT = float(_ENV.get("LLM_TIMEOUT_SECONDS", "300"))
MAX_RETRIES = int(_ENV.get("LLM_MAX_RETRIES", "6"))


class Usage:
    """Accumulate token usage, cost, and elapsed time for one task."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.cost_cny = 0.0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int):
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.calls += 1
        self.cost_cny += (
            int(prompt or 0) / 1_000_000 * PRICE_INPUT_PER_1M
            + int(completion or 0) / 1_000_000 * PRICE_OUTPUT_PER_1M
        )

    def as_dict(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "cost_cny": round(self.cost_cny, 6),
        }


def _post(payload: dict):
    if not BASE or not KEY:
        raise RuntimeError("Set LLM_BASE_URL and LLM_API_KEY before running the benchmark.")
    url = f"{BASE}/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    # Cloudflare-fronted gateways may reject the default Python urllib user agent (error 1010).
    req.add_header("User-Agent", "OpenAI/Python 1.54.0")
    if KEY:
        req.add_header("Authorization", f"Bearer {KEY}")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def complete(messages, max_tokens=1024, temperature=0.0) -> tuple:
    """Call chat/completions and return (content, prompt_tokens, completion_tokens).

    Retries failures and raises RuntimeError when all attempts fail.
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _post(payload)
            if "choices" not in resp or not resp["choices"]:
                raise RuntimeError(f"bad response: {str(resp)[:400]}")
            msg = resp["choices"][0].get("message", {}) or {}
            usage = resp.get("usage") or {}
            pt = usage.get("prompt_tokens") or 0
            ct = usage.get("completion_tokens") or 0
            return (msg.get("content") or "").strip(), int(pt), int(ct)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:500]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {body}"
        except Exception as e:  # Timeout, connection reset, or invalid JSON.
            last_err = str(e)
        time.sleep(min(20, 2 ** attempt))
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} retries: {last_err}")
