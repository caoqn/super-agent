from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoundContext:
    round_num: int
    agent_name: str
    obs: dict[str, Any]
    received_messages: dict[str, str] = field(default_factory=dict)
    phase: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageCandidate:
    agent_name: str
    position: dict[str, Any] = field(default_factory=dict)
    in_view: bool = False
    raw_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    enabled: bool = False
    prompt_sections: list[str] = field(default_factory=list)
    memory_hints: list[str] = field(default_factory=list)
    memory_sections: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    send: bool = False
    recipients: list[str] = field(default_factory=list)
    mode: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    recipient: str | None = None
    send: bool = False
    prompt: str = ""
    raw_response: str = ""
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    api_call_time: float = 0.0
    tokens: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedIncomingMessage:
    sender: str
    raw_message: str
    parsed_facts: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
