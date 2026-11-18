"""OpenAI-compatible chat client (stdlib only) with usage / cost / latency tracking.

Configured through environment variables (usually from a .env file):
  LLM_BASE_URL   e.g. https://api.openai.com/v1  or  https://api.deepseek.com/v1
  LLM_API_KEY
  LLM_MODEL      e.g. gpt-4o-mini
  LLM_MAX_TOKENS (default 2048, same as original AgentNet)
  LLM_TEMPERATURE (default 0.0, same as original AgentNet)
  LLM_TIMEOUT_SECONDS (default 180), LLM_MAX_RETRIES (default 5)
  LLM_THINKING ("disabled"/"enabled") pin the reasoning channel for every call
  LLM_THINKING_FALLBACK (default 1) retry once with thinking disabled when the
                 model truncates and returns nothing
"""
import json
import os
import random
import time
import urllib.error
import urllib.request

from .tracker import TRACKER


class LLMError(RuntimeError):
    pass


def _cfg():
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    return base, key, model


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def _post(url, payload, key, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    # Cloudflare-fronted API relays reject the default "Python-urllib" agent (error 1010)
    req.add_header("User-Agent", os.environ.get("LLM_USER_AGENT", "OpenAI/Python 1.54.0"))
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat(messages, tools=None, max_tokens=None, temperature=None, model=None, role=None):
    """Returns the assistant message dict (content / tool_calls). Records usage."""
    base, key, default_model = _cfg()
    model = model or default_model
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", 180))
    retries = int(os.environ.get("LLM_MAX_RETRIES", 6))
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.0)) if temperature is None else temperature,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", 2048)) if max_tokens is None else max_tokens,
    }
    if os.environ.get("LLM_TOP_P"):
        payload["top_p"] = float(os.environ["LLM_TOP_P"])
    if tools:
        payload["tools"] = tools
    url = f"{base}/chat/completions"

    # Reasoning models can burn the entire max_tokens budget in their hidden
    # thinking channel and then return an empty message with finish_reason
    # "length". One retry with thinking switched off recovers a usable answer.
    pinned_thinking = os.environ.get("LLM_THINKING")
    if pinned_thinking:
        payload["thinking"] = {"type": pinned_thinking}
    thinking_fallback = (
        os.environ.get("LLM_THINKING_FALLBACK", "1") != "0"
        and pinned_thinking != "disabled"
    )

    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            resp = _post(url, payload, key, timeout)
            latency = time.time() - t0
            if "choices" not in resp or not resp["choices"]:
                raise LLMError(f"bad response: {str(resp)[:500]}")
            msg = resp["choices"][0].get("message", {}) or {}
            usage = resp.get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            cached = 0
            details = usage.get("prompt_tokens_details") or {}
            if isinstance(details, dict):
                cached = details.get("cached_tokens") or 0
            cached = cached or usage.get("prompt_cache_hit_tokens") or 0  # DeepSeek style
            estimated = False
            if pt is None or ct is None:
                estimated = True
                pt = _estimate_tokens(json.dumps(messages, ensure_ascii=False))
                ct = _estimate_tokens((msg.get("content") or "") + json.dumps(msg.get("tool_calls") or ""))
            with TRACKER.role_scope(role or TRACKER.current_role()):
                TRACKER.record("llm", latency, pt, ct, cached, estimated,
                               extra={"model": model, "attempt": attempt,
                                      "finish_reason": resp["choices"][0].get("finish_reason")})
            truncated_empty = (
                resp["choices"][0].get("finish_reason") == "length"
                and not (msg.get("content") or "").strip()
                and not msg.get("tool_calls")
            )
            if truncated_empty and thinking_fallback:
                thinking_fallback = False
                payload["thinking"] = {"type": "disabled"}
                print(f"[llm] empty content, truncated at max_tokens={payload['max_tokens']} "
                      f"(reasoning used the whole budget); retrying with thinking disabled")
                continue
            return msg
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:800]
            except Exception:
                pass
            last_err = LLMError(f"HTTP {e.code}: {body}")
            if e.code == 400:
                try:  # keep the rejected request for debugging (contains no API key)
                    dbg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm_400_debug.jsonl")
                    with open(dbg, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"error": body, "payload": payload}, ensure_ascii=False)[:200000] + "\n")
                except Exception:
                    pass
            # 401/404 are configuration errors. A 400 can be a flaky relay channel
            # (same request succeeds on retry), so it is retried like a 5xx.
            if e.code in (401, 404) and attempt >= 1:
                break
        except Exception as e:  # timeouts, connection resets, bad json
            last_err = e
        sleep = min(20, 2 ** attempt) + random.random()
        print(f"[llm] attempt {attempt + 1} failed: {last_err}; retry in {sleep:.1f}s")
        time.sleep(sleep)
    raise LLMError(f"LLM call failed after retries: {last_err}")


def simple_response(system_prompt, query_prompt, role=None):
    msg = chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_prompt},
    ], role=role)
    return (msg.get("content") or "").strip()


def embed(texts):
    """OpenAI-compatible /embeddings (used when EMBEDDING_BACKEND=api)."""
    base = os.environ.get("EMB_BASE_URL", os.environ.get("LLM_BASE_URL", "")).rstrip("/")
    key = os.environ.get("EMB_API_KEY", os.environ.get("LLM_API_KEY", ""))
    model = os.environ.get("EMB_MODEL", "text-embedding-3-small")
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", 180))
    last = None
    for attempt in range(5):
        t0 = time.time()
        try:
            resp = _post(f"{base}/embeddings", {"model": model, "input": texts}, key, timeout)
            usage = resp.get("usage") or {}
            pt = usage.get("prompt_tokens") or usage.get("total_tokens") or sum(_estimate_tokens(t) for t in texts)
            TRACKER.record("embedding", time.time() - t0, pt, 0, extra={"model": model, "n": len(texts)})
            data = sorted(resp["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise LLMError(f"embedding failed: {last}")
