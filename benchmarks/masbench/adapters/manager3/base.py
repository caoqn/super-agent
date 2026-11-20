"""Small benchmark interface used by the portable MASBench adapter."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class NativeTask:
    task_id: str
    prompt: str
    raw: dict[str, Any] = field(default_factory=dict)
    split: str = "test"

@dataclass
class NativeEvalResult:
    success: bool
    score: float
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
