from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import MechanismBundle


ConfigFactory = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class PresetSpec:
    memory_cls: type
    routing_cls: type
    protocol_cls: type
    config_factory: ConfigFactory


_PRESET_REGISTRY: dict[str, PresetSpec] = {}


def _int_config(raw: dict[str, Any], key: str, default: int) -> int:
    try:
        value = raw.get(key, default)
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def register_preset(
    name: str,
    memory_cls: type,
    routing_cls: type,
    protocol_cls: type,
    config_factory: ConfigFactory,
) -> None:
    _PRESET_REGISTRY[str(name)] = PresetSpec(
        memory_cls=memory_cls,
        routing_cls=routing_cls,
        protocol_cls=protocol_cls,
        config_factory=config_factory,
    )


def get_preset(name: str) -> PresetSpec:
    preset_name = str(name or "base")
    if preset_name not in _PRESET_REGISTRY:
        raise KeyError(f"Unknown mechanism preset: {preset_name}")
    return _PRESET_REGISTRY[preset_name]


def list_presets() -> list[str]:
    return sorted(_PRESET_REGISTRY.keys())


def build_effective_config(preset_name: str = "base", raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    preset = str(preset_name or "base")
    spec = get_preset(preset)
    return spec.config_factory(raw_config or {})


def base_config_factory(raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    del raw_config
    return {
        "memory": {
            "base_no_mechanisms": True,
            "disabled": True,
            "enable_structured_view_memory": False,
            "enable_message_memory": False,
            "response_memory_k": 0,
        },
        "routing": {"base_no_mechanisms": True, "disabled": True, "disable_send": True},
        "protocol": {"base_no_mechanisms": True, "disabled": True},
        "runtime": {
            "raw_message_obs": False,
            "message_phase_order": "before_update",
            "message_phase_use_cached_obs": True,
            "message_phase_log_immediately": False,
            "messages_enabled": False,
        },
        "baseline": {
            "name": "base",
            "mechanisms_enabled": False,
        },
    }


def stable_fair_config_factory(raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(raw_config or {})
    view_max_lines = _int_config(raw, "view_memory_max_lines", 8)
    messages_enabled = not bool(raw.get("disable_send", False))
    return {
        "memory": {
            "enable_structured_view_memory": bool(raw.get("enable_structured_view_memory", True)),
            "view_memory_k": _int_config(raw, "view_memory_k", 8),
            "view_memory_max_lines": view_max_lines,
            "kb_ttl_rounds": _int_config(raw, "kb_ttl_rounds", 8),
            "kb_memory_max_entries": _int_config(raw, "kb_memory_max_entries", view_max_lines),
            "kb_nearest_tables_k": _int_config(raw, "kb_nearest_tables_k", 1),
            "kb_nearest_items_k": _int_config(raw, "kb_nearest_items_k", 2),
            "response_memory_k": _int_config(raw, "response_memory_k", 3),
            "enable_message_memory": bool(raw.get("enable_message_memory", True)),
            "message_memory_k_sent": _int_config(raw, "message_memory_k_sent", 3),
            "message_memory_k_recv": _int_config(raw, "message_memory_k_recv", 3),
        },
        "routing": {
            "single_message_per_round": bool(raw.get("single_message_per_round", True)),
            "single_message_prompt_mode": str(raw.get("single_message_prompt_mode", "gate_llm") or "gate_llm"),
            "ring_mode": str(raw.get("ring_mode", "angle_halves") or "angle_halves"),
            "ask_cooldown_rounds": _int_config(raw, "ask_cooldown_rounds", 2),
            "seed": _int_config(raw, "seed", 0),
            "disable_send": bool(raw.get("disable_send", False)),
        },
        "protocol": {
            "single_message_prompt_mode": str(raw.get("single_message_prompt_mode", "gate_llm") or "gate_llm"),
            "protocol_v2_enabled": bool(raw.get("protocol_v2_enabled", False)),
            "protocol_v2_auto": bool(raw.get("protocol_v2_auto", raw.get("protocol_v2_enabled", False))),
            "ask_cooldown_rounds": _int_config(raw, "ask_cooldown_rounds", 2),
            "max_items_in_message": _int_config(raw, "max_items_in_message", 2),
            "max_tables_in_message": _int_config(raw, "max_tables_in_message", 2),
            "proto_v2_free_enable": bool(raw.get("proto_v2_free_enable", raw.get("protocol_v2_enabled", False))),
            "proto_v2_free_slot": str(raw.get("proto_v2_free_slot", "C") or "C"),
            "proto_v2_free_max": _int_config(raw, "proto_v2_free_max", 150),
            "proto_v2_header_max_len": _int_config(raw, "proto_v2_header_max_len", 70),
            "proto_v2_total_max_len": _int_config(raw, "proto_v2_total_max_len", 220),
        },
        "runtime": {
            "raw_message_obs": True,
            "message_phase_order": str(raw.get("message_phase_order", "before_update") or "before_update"),
            "message_phase_use_cached_obs": bool(raw.get("message_phase_use_cached_obs", True)),
            "message_phase_log_immediately": bool(raw.get("message_phase_log_immediately", True)),
            "message_phase_include_action_intent": bool(raw.get("message_phase_include_action_intent", True)),
            "messages_enabled": messages_enabled,
        },
        "compat": {
            "raw_message_obs": True,
            "profile": "stable_fair",
            "messages_enabled": messages_enabled,
        },
    }


def build_bundle(owner, preset_name: str = "base", raw_config: dict[str, Any] | None = None) -> MechanismBundle:
    preset = str(preset_name or "base")
    spec = get_preset(preset)
    bundle_config = build_effective_config(preset, raw_config)
    return MechanismBundle(
        preset_name=preset,
        memory=spec.memory_cls(owner, bundle_config.get("memory")),
        routing=spec.routing_cls(owner, bundle_config.get("routing")),
        protocol=spec.protocol_cls(owner, bundle_config.get("protocol")),
        config=bundle_config,
    )


def _stable_fair_preset_factory(
    *,
    routing_mode: str,
    protocol_enabled: bool,
    disable_send: bool = False,
) -> ConfigFactory:
    def _factory(raw_config: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = dict(raw_config or {})
        fixed_raw = {
            **raw,
            "enable_structured_view_memory": True,
            "enable_message_memory": True,
            "single_message_per_round": True,
            "single_message_prompt_mode": routing_mode,
            "protocol_v2_enabled": bool(protocol_enabled),
            "protocol_v2_auto": bool(protocol_enabled),
            "proto_v2_free_enable": bool(protocol_enabled),
            "disable_send": bool(disable_send),
        }
        return stable_fair_config_factory(fixed_raw)

    return _factory


def ensure_default_presets_registered() -> None:
    from .stable_fair import (
        StableFairMemoryMechanism,
        StableFairProtocolMechanism,
        StableFairRoutingMechanism,
    )

    if "base" not in _PRESET_REGISTRY:
        register_preset(
            name="base",
            memory_cls=StableFairMemoryMechanism,
            routing_cls=StableFairRoutingMechanism,
            protocol_cls=StableFairProtocolMechanism,
            config_factory=base_config_factory,
        )

    if "stable_fair_memory_only" not in _PRESET_REGISTRY:
        fair_presets = [
            (
                "stable_fair_memory_only",
                _stable_fair_preset_factory(
                    routing_mode="random",
                    protocol_enabled=False,
                    disable_send=True,
                ),
            ),
            (
                "stable_fair_all_random",
                _stable_fair_preset_factory(
                    routing_mode="random",
                    protocol_enabled=False,
                ),
            ),
            (
                "stable_fair_gate_llm",
                _stable_fair_preset_factory(
                    routing_mode="gate_llm",
                    protocol_enabled=False,
                ),
            ),
            (
                "stable_fair_protocol_all_random",
                _stable_fair_preset_factory(
                    routing_mode="random",
                    protocol_enabled=True,
                ),
            ),
            (
                "stable_fair_protocol_gate_llm",
                _stable_fair_preset_factory(
                    routing_mode="gate_llm",
                    protocol_enabled=True,
                ),
            ),
        ]
        for preset_name, config_factory in fair_presets:
            register_preset(
                name=preset_name,
                memory_cls=StableFairMemoryMechanism,
                routing_cls=StableFairRoutingMechanism,
                protocol_cls=StableFairProtocolMechanism,
                config_factory=config_factory,
            )


ensure_default_presets_registered()
