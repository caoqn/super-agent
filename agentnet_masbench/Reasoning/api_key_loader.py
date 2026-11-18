"""
Utility helpers for creating OpenAI-compatible clients and mapping logical model
keys (e.g. 'gemini_pro', 'gemini_flash', 'gpt_4o_mini') to real model ids.

Configuration is taken from:
- environment variables (recommended), and
- a local `.env` file in the project root (fallback).

Expected environment variables:
- OPENAI_API_KEY   : API key for the OpenAI-compatible endpoint.
- OPENAI_BASE_URL  : Base URL for the endpoint, e.g. https://api.openai.com/v1

Optional overrides for model mapping:
- GEMINI_PRO_MODEL      : real model id for logical key 'gemini_pro'
- GEMINI_FLASH_MODEL    : real model id for logical key 'gemini_flash'
- GPT_4O_MINI_MODEL     : real model id for logical key 'gpt_4o_mini'
- DEEPSEEK_V3_2_EXP_MODEL   : real model id for logical key 'deepseek_v3_2_exp'
- QWEN_PLUS_MODEL       : real model id for logical key 'qwen_plus'
- GPT_5_MINI_MODEL    : real model id for logical key 'gpt_5_mini'
- LLAMA_3_1_8B_INSTRUCT_MODEL : real model id for logical key 'llama_3_1_8b_instruct'
- LLAMA_3_3_70B_INSTRUCT_MODEL : real model id for logical key 'llama_3_3_70b_instruct'
- GPT_5_NANO_MODEL     : real model id for logical key 'gpt_5_nano'
- GEMINI_2_5_FLASH_LITE_MODEL : real model id for logical key 'gemini_2_5_flash_lite'
- DEEPSEEK_V3_2_MODEL  : real model id for logical key 'deepseek_v3_2'
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment specific
    OpenAI = None  # type: ignore[assignment]


def _load_env_from_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader: reads KEY=VALUE lines and injects into os.environ
    if the key is not already present.
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # trim quotes if present
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # Do not crash if .env is unreadable; just skip.
        return


# Eagerly load .env once at import time so OPENAI_* variables are available.
_load_env_from_dotenv()


_MODEL_KEY_ENV_VARS: Dict[str, str] = {
    "gemini_flash": "GEMINI_FLASH_MODEL",
    "gpt_4o_mini": "GPT_4O_MINI_MODEL",
    'gemini_pro':"GEMINI_PRO_MODEL",
    "deepseek_v3_2_exp": "DEEPSEEK_V3_2_EXP_MODEL",
    "qwen_plus": "QWEN_PLUS_MODEL",
    "gpt_5_mini": "GPT_5_MINI_MODEL",
    "llama_3_1_8b_instruct": "LLAMA_3_1_8B_INSTRUCT_MODEL",
    "llama_3_3_70b_instruct": "LLAMA_3_3_70B_INSTRUCT_MODEL",
    "gpt_5_nano": "GPT_5_NANO_MODEL",
    "gemini_2_5_flash_lite": "GEMINI_2_5_FLASH_LITE_MODEL",
    "deepseek_v3_2": "DEEPSEEK_V3_2_MODEL"
}

_MODEL_KEY_DEFAULTS: Dict[str, str] = {
    "gemini_flash": "gemini-2.5-flash",
    "gpt_4o_mini": "gpt-4o-mini",
    "gemini_pro":'gemini-2.5-pro',
    "deepseek_v3_2_exp": "deepseek-v3.2-exp",
    "qwen_plus": "qwen-plus",
    "gpt_5_mini": "gpt-5-mini",
    "llama_3_1_8b_instruct": "llama-3-1-8b-instruct",
    "llama_3_3_70b_instruct": "llama-3-3-70b-instruct",
    "gpt_5_nano": "gpt-5-nano",
    "gemini_2_5_flash_lite": "gemini-2.5-flash-lite",
    "deepseek_v3_2": "deepseek-v3.2"
}

def get_model_name(model_key: str) -> str:
    """
    Map a logical model key like 'gemini_pro' or 'gpt_4o_mini' to the real
    model id used by the OpenAI-compatible endpoint.

    If no explicit mapping exists, the key itself is returned so that you can
    pass a raw model id directly.
    """
    env_var = _MODEL_KEY_ENV_VARS.get(model_key)
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override

    default = _MODEL_KEY_DEFAULTS.get(model_key)
    if default:
        return default

    # Fallback: assume caller passed a real model id already.
    return model_key


@lru_cache(maxsize=None)
def _build_client(api_key: str, base_url: str) -> OpenAI:
    """Create and cache an OpenAI client for the given key/base_url."""
    if OpenAI is None:
        raise ImportError(
            "The 'openai' package is required for running experiments. "
            "Install it in the active environment or use the llm-mas-bench environment."
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def get_api(model_key: str | None = None) -> OpenAI:
    """
    Return an OpenAI-compatible client.

    The model_key argument is accepted for convenience (so callers can write
    get_api("gemini_pro")), but currently it is not used to vary credentials;
    all logical models share the same underlying client. Model selection is
    done via get_model_name(model_key) when calling .chat.completions.create().
    """
    # Ensure env is loaded (no-op if already done).
    _load_env_from_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please set it in your environment or .env file."
        )

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return _build_client(api_key, base_url)
