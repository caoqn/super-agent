from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .types import (
    MemoryContext,
    MessageCandidate,
    OutgoingMessage,
    ParsedIncomingMessage,
    RoundContext,
    RoutingDecision,
)


class MemoryMechanism(ABC):
    def __init__(self, owner, config: dict[str, Any] | None = None):
        self.owner = owner
        self.config = dict(config or {})

    def reset(self) -> None:
        return None

    def observe(self, ctx: RoundContext) -> None:
        return None

    async def before_message_routing(self, ctx: RoundContext) -> None:
        del ctx
        return None

    def receive_incoming(
        self,
        ctx: RoundContext,
        protocol: "ProtocolMechanism",
        sender: str,
        raw_message: str,
    ) -> None:
        parsed = protocol.parse_incoming(ctx, sender, raw_message)
        self.ingest_incoming(ctx, [parsed])

    @abstractmethod
    def ingest_incoming(self, ctx: RoundContext, parsed_messages: list[ParsedIncomingMessage]) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_action_context(self, ctx: RoundContext) -> MemoryContext:
        raise NotImplementedError

    @abstractmethod
    def build_message_context(self, ctx: RoundContext) -> MemoryContext:
        raise NotImplementedError

    @abstractmethod
    def record_outgoing(self, ctx: RoundContext, outgoing: OutgoingMessage) -> None:
        raise NotImplementedError

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        raise NotImplementedError


class RoutingMechanism(ABC):
    def __init__(self, owner, config: dict[str, Any] | None = None):
        self.owner = owner
        self.config = dict(config or {})

    def reset(self) -> None:
        return None

    def requires_raw_message_obs(self) -> bool:
        return False

    @abstractmethod
    async def select_targets(
        self,
        ctx: RoundContext,
        candidates: list[MessageCandidate],
        memory_ctx: MemoryContext,
    ) -> RoutingDecision:
        raise NotImplementedError


class ProtocolMechanism(ABC):
    def __init__(self, owner, config: dict[str, Any] | None = None):
        self.owner = owner
        self.config = dict(config or {})

    def reset(self) -> None:
        return None

    def format_history_response(
        self,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> str:
        del recipient, decision_meta
        route_response = str(route_response or "").strip()
        content_response = str(getattr(outgoing, "raw_response", "") or "").strip()
        outgoing_message = str(getattr(outgoing, "message", "") or "").strip()
        if route_response and content_response:
            return f"[ROUTING]\n{route_response}\n[CONTENT]\n{content_response}"
        if route_response:
            return route_response
        return content_response or outgoing_message

    def should_record_history_response(
        self,
        ctx: RoundContext,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> bool:
        del ctx, recipient, route_response, outgoing, decision_meta
        return True

    @abstractmethod
    async def compose(
        self,
        ctx: RoundContext,
        recipient: str,
        memory_ctx: MemoryContext,
        routing_meta: dict[str, Any],
    ) -> OutgoingMessage:
        raise NotImplementedError

    @abstractmethod
    def parse_incoming(
        self,
        ctx: RoundContext,
        sender: str,
        raw_message: str,
    ) -> ParsedIncomingMessage:
        raise NotImplementedError


@dataclass
class MechanismBundle:
    preset_name: str
    memory: MemoryMechanism
    routing: RoutingMechanism
    protocol: ProtocolMechanism
    config: dict[str, Any]
