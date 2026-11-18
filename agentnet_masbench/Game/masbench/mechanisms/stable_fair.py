"""Release-facing StableFair mechanisms.

This module keeps the StableFair algorithm behavior from ``stable_fair.py`` but
flattens the historical class hierarchy into three public mechanism classes.
``Reference`` here is only a small, readable baseline/example used by the
StableFair implementation.
"""

from __future__ import annotations

import os
import random
import re
import time
from collections import deque
from typing import Any

from .base import MemoryMechanism, ProtocolMechanism, RoutingMechanism
from .types import (
    MemoryContext,
    MessageCandidate,
    OutgoingMessage,
    ParsedIncomingMessage,
    RoundContext,
    RoutingDecision,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _current_round_action_prompt_block(obs: dict[str, Any] | None) -> str:
    info = dict((obs or {}).get("current_round_action_intent") or {})
    summary = str(info.get("summary") or (obs or {}).get("current_round_action_summary") or "").strip()
    if not summary:
        return ""
    return (
        "\nThis round uses one shared observation snapshot for both action and message phases.\n"
        f"{summary}\n"
        "Keep the message consistent with that chosen action.\n"
        "Do not simply restate the movement.\n"
        "Prioritize one actionable local fact from the current observation or one specific request to the recipient.\n"
        "Prefer this order: current position / visible resource / visible table / local status first; one specific request second; one short plan clause only if useful.\n"
        "Avoid bare movement-only sentences and generic coordination filler.\n"
    )


def _plain_basic_target_prompt_guidance() -> str:
    return (
        "Plain message context:\n"
        "- Write one short natural message to the selected teammate.\n"
        "- Use your current observation naturally if it is useful.\n"
        "- Mention a location or item only when you are confident.\n"
        "- Prefer a stable factual update over asking the teammate to change plans.\n"
        "- Avoid extra headings and generic social filler.\n"
    )


def _plain_current_round_action_context(obs: dict[str, Any] | None) -> str:
    info = dict((obs or {}).get("current_round_action_intent") or {})
    summary = str(info.get("summary") or (obs or {}).get("current_round_action_summary") or "").strip()
    if not summary:
        return ""
    return (
        "\nCurrent action context:\n"
        "- This message uses the same observation snapshot as your current action.\n"
        f"- {summary}\n"
    )


def _format_items_compact(items: dict[str, Any] | None) -> str:
    if not isinstance(items, dict) or not items:
        return "none"
    parts: list[str] = []
    for item_type in ("wood", "sand", "stone"):
        try:
            qty = int(items.get(item_type, 0) or 0)
        except Exception:
            qty = 0
        if qty > 0:
            parts.append(f"{item_type}:{qty}")
    return ", ".join(parts) if parts else "none"


def _parse_resource_quantity(value: Any, fallback_qty: Any = None) -> tuple[str, int | None]:
    text = str(value or "").strip().lower()
    qty = _coerce_int(fallback_qty)
    if not text or text in {"none", "empty", "unknown", "null"}:
        return "none", qty if qty and qty > 0 else None
    match = re.search(r"\b(wood|sand|stone)\b(?:\s*(?:[:=]|x)\s*(\d+))?", text)
    if not match:
        return "none", qty if qty and qty > 0 else None
    resource_type = str(match.group(1) or "").strip().lower()
    parsed_qty = _coerce_int(match.group(2))
    if parsed_qty is not None and parsed_qty > 0:
        qty = int(parsed_qty)
    return resource_type, qty if qty and qty > 0 else None


def _inventory_resource_quantity(obs: dict[str, Any] | None) -> tuple[str, int | None]:
    inv = dict((obs or {}).get("inventory") or {})
    explicit_type = (
        inv.get("type")
        or inv.get("item_type")
        or inv.get("resource_type")
        or inv.get("carrying_resource")
    )
    explicit_qty = (
        inv.get("quantity")
        or inv.get("qty")
        or inv.get("amount")
        or inv.get("total")
        or inv.get("carrying_qty")
    )
    resource_type, qty = _parse_resource_quantity(explicit_type, explicit_qty)
    if resource_type != "none":
        return resource_type, qty
    return _parse_resource_quantity(inv.get("display"), explicit_qty)


def _candidate_resource_quantity(candidate_info: dict[str, Any] | None) -> tuple[str, int | None]:
    info = dict(candidate_info or {})
    explicit_qty = None
    for key in ("carrying_qty", "carrying_quantity", "item_quantity", "quantity", "carrying_amount", "item_qty"):
        explicit_qty = _coerce_int(info.get(key))
        if explicit_qty is not None and explicit_qty > 0:
            break
    return _parse_resource_quantity(
        info.get("carrying_resource") or info.get("carrying_resource_name") or info.get("inventory"),
        explicit_qty,
    )


def _compact_visible_resource_summary(obs: dict[str, Any], max_lines: int = 3) -> str:
    cap = obs.get("capacity_info") or {}
    details = (cap.get("_resource_details") if isinstance(cap, dict) else {}) or {}
    lines: list[str] = []
    try:
        entries = sorted(
            details.values(),
            key=lambda rec: (
                int((rec.get("position") or (0, 0))[1]),
                int((rec.get("position") or (0, 0))[0]),
                str(rec.get("terrain_type") or ""),
            ),
        )
    except Exception:
        entries = list(details.values())
    for rec in entries:
        if str(rec.get("terrain_type") or "") == "table":
            continue
        pos = rec.get("position") or (None, None)
        items = rec.get("items") or []
        for item in items:
            try:
                qty = int(item.get("quantity", 0) or 0)
            except Exception:
                qty = 0
            if qty <= 0:
                continue
            item_type = str(item.get("type") or "unknown")
            lines.append(f"{item_type}@({pos[0]},{pos[1]}):{qty}")
            if len(lines) >= max_lines:
                return "; ".join(lines)
    return "; ".join(lines) if lines else "none"


def _stable_fair_remaining_by_type_text(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    parts: list[str] = []
    for item_type in ("wood", "sand", "stone"):
        qty = _coerce_int(value.get(item_type))
        if qty is not None:
            parts.append(f"{item_type}:{max(0, int(qty))}")
    return ",".join(parts)


def _stable_fair_table_capacity_text(rec: dict[str, Any] | None) -> str:
    data = dict(rec or {})
    parts: list[str] = []
    rem = _coerce_int(data.get("remaining_capacity"))
    if rem is not None:
        parts.append(f"total_rem={max(0, int(rem))}")
    type_text = _stable_fair_remaining_by_type_text(data.get("remaining_by_type"))
    if type_text:
        parts.append(f"type_rem={type_text}")
    return "; ".join(parts) if parts else "capacity=unknown"


def _stable_fair_table_deposit_capacity_text(rec: dict[str, Any] | None) -> str:
    data = dict(rec or {})
    parts: list[str] = []
    rem = _coerce_int(data.get("remaining_capacity"))
    if rem is not None:
        parts.append(f"total_deposit_space={max(0, int(rem))}")
    type_text = _stable_fair_remaining_by_type_text(data.get("remaining_by_type"))
    if type_text:
        parts.append(f"can_deposit={type_text}")
    return "; ".join(parts) if parts else "deposit_capacity=unknown"


def _compact_visible_table_summary(obs: dict[str, Any], max_lines: int = 3, *, deposit_terms: bool = False) -> str:
    tables = list(obs.get("tables_info") or [])
    lines: list[str] = []
    for rec in tables[:max_lines]:
        pos = rec.get("position") or (None, None)
        if isinstance(pos, dict):
            pos = (pos.get("x"), pos.get("y"))
        capacity = _stable_fair_table_deposit_capacity_text(rec) if deposit_terms else _stable_fair_table_capacity_text(rec)
        lines.append(f"table@({pos[0]},{pos[1]}) {capacity}")
    return "; ".join(lines) if lines else "none"


def _compact_memory_hint_summary(obs: dict[str, Any], max_entries: int = 3, *, deposit_terms: bool = False) -> str:
    lines: list[str] = []
    for rec in list(obs.get("kb_nearest_tables") or [])[:2]:
        pos = rec.get("position") or {}
        capacity = _stable_fair_table_deposit_capacity_text(rec) if deposit_terms else _stable_fair_table_capacity_text(rec)
        lines.append(f"mem_table@({pos.get('x')},{pos.get('y')}) {capacity}")
        if len(lines) >= max_entries:
            return "; ".join(lines)
    for rec in list(obs.get("kb_nearest_items") or [])[:2]:
        pos = rec.get("position") or {}
        items = _format_items_compact(rec.get("items"))
        lines.append(f"mem_items@({pos.get('x')},{pos.get('y')}): {items}")
        if len(lines) >= max_entries:
            return "; ".join(lines)
    return "; ".join(lines) if lines else "none"


def _compact_current_facts_block(obs: dict[str, Any] | None, *, usage: str = "message") -> str:
    data = dict(obs or {})
    deposit_terms = str(usage or "").strip().lower() == "protocol"
    inv = data.get("inventory") or {}
    inventory_display = str(inv.get("display") or "empty")
    tile_items = _format_items_compact(data.get("tile_items_counts"))
    tile_type = str(data.get("tile_terrain_type") or "unknown")
    tile_free = data.get("tile_placeable_free_capacity")
    tile_type_remaining = _stable_fair_remaining_by_type_text(data.get("tile_remaining_by_type"))
    visible_resources = _compact_visible_resource_summary(data)
    visible_tables = _compact_visible_table_summary(data, deposit_terms=deposit_terms)
    memory_hints = _compact_memory_hint_summary(data, deposit_terms=deposit_terms)
    tile_capacity = f"total_free_capacity={tile_free}"
    if tile_type_remaining:
        tile_capacity_key = "can_deposit" if deposit_terms and tile_type == "table" else "type_remaining"
        tile_capacity = f"{tile_capacity}; {tile_capacity_key}={tile_type_remaining}"
    lines = [
        "Current local facts:",
        f"- Inventory: {inventory_display}",
        f"- Current tile: terrain={tile_type}; items={tile_items}; {tile_capacity}",
        f"- Visible resources: {visible_resources}",
        f"- Visible tables: {visible_tables}",
        f"- Relevant memory hints: {memory_hints}",
    ]
    if str(usage).lower() == "routing":
        lines.extend(
            [
                "Use these facts to judge whether contacting any recipient is worthwhile right now.",
                "If you decide to send, prefer the recipient whose location and likely role best match your inventory, visible resources/tables, and relevant memory.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Use one concrete fact from this block when it helps the recipient coordinate.",
                "Prefer an actionable fact or a specific request over generic coordination filler.",
                "",
            ]
        )
    return "\n".join(lines)


def _compact_current_facts_plain_context_block(obs: dict[str, Any] | None) -> str:
    block = _compact_current_facts_block(obs, usage="message")
    drop_prefixes = (
        "Use one concrete fact from this block",
        "Prefer an actionable fact or a specific request",
    )
    lines = [
        line for line in str(block or "").splitlines()
        if not any(line.startswith(prefix) for prefix in drop_prefixes)
    ]
    return "\n".join(lines) + "\n"


def _direct_actionable_message_guidance() -> str:
    return (
        "When the current facts support it, prefer one short factual update with a concrete local fact: visible resource, visible table capacity, current inventory, or planned delivery.\n"
        "Table is deposit-only: never ask another agent to pick up from a table, collect from a table, or use a table as a handoff pickup point.\n"
        "If you just deposited at a table, ask others to gather from nearby resource piles or bring their own load to the table, not to collect from the table.\n"
        "Do not propose handoff or pickup-from-teammate plans in this setting; share compatible table facts or resource facts instead.\n"
        "When you mention a table, prefer delivery destination, remaining capacity, or status wording.\n"
        "When useful, add a mild request such as checking an area, reporting nearby resources, or considering a table, but avoid ordering the recipient to abandon its current task.\n"
        "Avoid commands that force a specific pickup or delivery unless the current facts make that action immediately obvious.\n"
        "State concrete resource/table facts plainly; do not append soft qualifiers such as 'if needed' or 'if you need'.\n"
        "Avoid hedging such as 'if you need', 'I'm near', 'let me know', or vague availability statements.\n"
    )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _extract_xy(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict):
        x = _coerce_int(value.get("x"))
        y = _coerce_int(value.get("y"))
    elif isinstance(value, (tuple, list)) and len(value) >= 2:
        x = _coerce_int(value[0])
        y = _coerce_int(value[1])
    else:
        return None
    if x is None or y is None:
        return None
    return int(x), int(y)


def _extract_xy_from_text(value: Any) -> tuple[int, int] | None:
    match = re.search(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", str(value or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _stable_fair_message_coords(value: Any) -> list[tuple[int, int]]:
    text = str(value or "")
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    matches: list[tuple[int, str, str]] = []

    def add(x_value: Any, y_value: Any) -> None:
        x = _coerce_int(x_value)
        y = _coerce_int(y_value)
        if x is None or y is None:
            return
        coord = (int(x), int(y))
        if coord in seen:
            return
        seen.add(coord)
        coords.append(coord)

    for match in re.finditer(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text):
        matches.append((match.start(), match.group(1), match.group(2)))
    for match in re.finditer(
        r"\b(?:wood|sand|stone|resource|table)\s*@\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
        text,
        re.IGNORECASE,
    ):
        matches.append((match.start(), match.group(1), match.group(2)))
    for _, x_value, y_value in sorted(matches, key=lambda item: item[0]):
        add(x_value, y_value)
    return coords


def _manhattan_distance(a: tuple[int, int] | None, b: tuple[int, int] | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _consecutive_round_streak(round_values: list[int], current_round: int) -> int:
    if not round_values:
        return 0
    streak = 0
    seen = set(round_values)
    probe = int(current_round) - 1
    while probe in seen:
        streak += 1
        probe -= 1
    return streak


def _inventory_nonempty(obs: dict[str, Any] | None) -> bool:
    resource_type, qty = _inventory_resource_quantity(obs)
    if resource_type != "none":
        return qty is None or qty > 0
    inv = dict((obs or {}).get("inventory") or {})
    display = str(inv.get("display") or "").strip().lower()
    if display and display != "empty":
        return True
    qty = _coerce_int(inv.get("quantity"))
    return bool(qty and qty > 0)


def _has_visible_resources(obs: dict[str, Any] | None) -> bool:
    return _compact_visible_resource_summary(dict(obs or {}), max_lines=1) != "none"


def _has_table_knowledge(obs: dict[str, Any] | None) -> bool:
    data = dict(obs or {})
    return bool(data.get("tables_info") or data.get("kb_nearest_tables"))


_STABLE_FAIR_INTENT_TO_CODE = {
    "discover": "F",
    "deliver": "D",
    "table_for_carry": "T",
    "carry": "C",
    "request": "Q",
    "reply": "R",
    "handoff": "H",
    "status": "S",
}
_STABLE_FAIR_CODE_TO_INTENT = {v: k for k, v in _STABLE_FAIR_INTENT_TO_CODE.items()}


def _stable_fair_protocol_fact_compact(fact: dict[str, Any] | None) -> str:
    if not isinstance(fact, dict):
        return ""
    kind = str(fact.get("kind") or "status").strip().lower()
    pos = _extract_xy(fact.get("position"))
    if kind == "resource" and pos is not None:
        resource_type = str(fact.get("resource_type") or "unknown").strip().lower()
        qty = _coerce_int(fact.get("quantity"))
        return f"{resource_type}@({pos[0]},{pos[1]}):{int(qty) if qty is not None else 0}"
    if kind == "table" and pos is not None:
        rem = _coerce_int(fact.get("remaining_capacity"))
        if rem is None:
            rb = fact.get("remaining_by_type") or {}
            if isinstance(rb, dict):
                rem = sum(max(0, int(v)) for v in rb.values() if _coerce_int(v) is not None)
        return f"table@({pos[0]},{pos[1]}):{int(rem) if rem is not None else 0}"
    return ""


def _stable_fair_protocol_need_compact(fact: dict[str, Any] | None) -> str:
    if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
        return ""
    rb = fact.get("remaining_by_type") or {}
    if not isinstance(rb, dict) or not rb:
        return ""
    parts: list[str] = []
    for item_type in ("wood", "sand", "stone"):
        value = _coerce_int(rb.get(item_type))
        if value is not None:
            parts.append(f"{item_type}:{max(0, int(value))}")
    return ",".join(parts)


def _parse_stable_fair_compact_fact(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if not value or value.lower() in {"none", "status"}:
        return None
    table_match = re.search(
        r"^table@\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*:\s*(\d+)",
        value,
        re.IGNORECASE,
    )
    if table_match:
        return {
            "kind": "table",
            "position": (int(table_match.group(1)), int(table_match.group(2))),
            "remaining_capacity": int(table_match.group(3)),
        }
    resource_match = re.search(
        r"^(wood|sand|stone)@\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?\s*:\s*(\d+)",
        value,
        re.IGNORECASE,
    )
    if resource_match:
        return {
            "kind": "resource",
            "resource_type": str(resource_match.group(1)).lower(),
            "position": (int(resource_match.group(2)), int(resource_match.group(3))),
            "quantity": int(resource_match.group(4)),
        }
    return None


def _parse_stable_fair_need_compact(text: str) -> dict[str, int]:
    needs: dict[str, int] = {}
    for item_type, value in re.findall(r"\b(wood|sand|stone)\s*:\s*(\d+)", str(text or ""), re.IGNORECASE):
        needs[str(item_type).lower()] = int(value)
    return needs


def _parse_stable_fair_compact_protocol(raw_message: str) -> dict[str, Any] | None:
    text = str(raw_message or "").strip()
    text = text.splitlines()[0].strip() if text else ""
    if " NOTE:" in text:
        text = text.split(" NOTE:", 1)[0].rstrip()
    if not text.startswith("SF1|"):
        return None
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if len(parts) < 2:
        return None
    code = parts[1].strip().upper()
    fields: dict[str, str] = {}
    for part in parts[2:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    proto: dict[str, Any] = {
        "version": "1",
        "intent": _STABLE_FAIR_CODE_TO_INTENT.get(code, "status"),
        "ask": fields.get("a") or "none",
    }
    loc = _extract_xy_from_text(fields.get("l"))
    if loc is not None:
        proto["loc"] = loc
    carry_type, carry_qty = _parse_resource_quantity(fields.get("c") or "none")
    proto["carry_type"] = carry_type
    proto["carry_qty"] = carry_qty
    fact = _parse_stable_fair_compact_fact(fields.get("f") or "")
    need = _parse_stable_fair_need_compact(fields.get("n") or "")
    if fact and str(fact.get("kind") or "") == "table" and need:
        fact["remaining_by_type"] = need
    if fact:
        proto["fact"] = fact
    goal = _extract_xy_from_text(fields.get("g"))
    if goal is not None:
        proto["goal"] = goal
    return proto


def _stable_fair_protocol_note(raw_message: str) -> str:
    text = str(raw_message or "")
    match = re.search(r"(?:^|\n)\s*NOTE\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    note = " ".join(str(match.group(1) or "").strip().split())
    return note[:140]


def _stable_fair_protocol_note_supported(proto: dict[str, Any] | None, note: str) -> bool:
    if not isinstance(proto, dict):
        return True
    text = str(note or "").strip()
    if not text:
        return False

    intent = str(proto.get("intent") or "status").strip().lower()
    carry_type = str(proto.get("carry_type") or "none").strip().lower()
    fact = proto.get("fact")
    fact_kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
    if fact_kind != "table":
        return True

    low = text.lower()
    if re.search(r"\b(?:resources?|materials?|pickup|pick\s+up|collect|gather|available\s+for\s+pickup)\b", low):
        return False
    if re.search(r"\btable\b[^.]*\bhas\b[^.]*\b(?:wood|sand|stone)\b", low) and not re.search(
        r"\b(?:can\s+deposit|can\s+accept|accepts?|deposit|remaining|capacity|space|slot|room)\b",
        low,
    ):
        return False
    if intent in {"deliver", "carry", "handoff"} or carry_type in {"wood", "sand", "stone"}:
        return True
    resource_term = r"(?:wood|sand|stone)"
    unsupported_patterns = (
        rf"\bi(?:'ll| will| am going to)?\s+(?:handle|collect|pick up|pickup|take|gather)\s+{resource_term}\b",
        rf"\byou\s+(?:take|handle|collect|pick up|pickup|gather)\s+{resource_term}\b",
        rf"\b(?:handle|collect|pick up|pickup|take|gather)\s+{resource_term}\s+from\s+table\b",
        rf"\bcollecting\s+{resource_term}\b",
    )
    return not any(re.search(pattern, low) for pattern in unsupported_patterns)


def _stable_fair_one_line_message(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines:
        match = re.match(r"^\s*MSG\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip()
    for line in lines:
        if re.match(r"^\s*(?:TYPE|SEND|TO|FACT|ASK|STATE|LANE|OBS(?:_[A-Z]+)?)\s*:", line, re.IGNORECASE):
            continue
        return re.sub(r"^\s*(?:MSG|NOTE|PLAN|GOAL|HEADER)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    return re.sub(r"^\s*(?:MSG|NOTE|PLAN|GOAL|HEADER)\s*:\s*", "", lines[0], flags=re.IGNORECASE).strip()


def _stable_fair_positive_remaining_by_type(fact: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(fact, dict):
        return {}
    rb = fact.get("remaining_by_type") or {}
    if not isinstance(rb, dict):
        return {}
    parsed: dict[str, int] = {}
    for item_type in ("wood", "sand", "stone"):
        value = _coerce_int(rb.get(item_type))
        if value is not None and value > 0:
            parsed[item_type] = int(value)
    return parsed


def _stable_fair_table_has_type_caps(fact: dict[str, Any] | None) -> bool:
    if not isinstance(fact, dict) or str(fact.get("kind") or "").strip().lower() != "table":
        return False
    rb = fact.get("remaining_by_type")
    return isinstance(rb, dict) and bool(rb)


def _stable_fair_table_remaining_for_type(fact: dict[str, Any] | None, item_type: str) -> int | None:
    if not isinstance(fact, dict) or str(fact.get("kind") or "").strip().lower() != "table":
        return None
    resource_type = str(item_type or "").strip().lower()
    if resource_type not in {"wood", "sand", "stone"}:
        return None
    rb = fact.get("remaining_by_type")
    if isinstance(rb, dict) and rb:
        value = _coerce_int(rb.get(resource_type))
        return max(0, int(value)) if value is not None else 0
    rem = _coerce_int(fact.get("remaining_capacity"))
    return max(0, int(rem)) if rem is not None else None


def _stable_fair_table_supports_carry(
    fact: dict[str, Any] | None,
    carry_type: str,
    carry_qty: int | None = None,
    *,
    require_full_qty: bool = True,
) -> bool:
    resource_type = str(carry_type or "").strip().lower()
    if resource_type not in {"wood", "sand", "stone"}:
        return False
    remaining = _stable_fair_table_remaining_for_type(fact, resource_type)
    if remaining is None or remaining <= 0:
        return False
    qty = _coerce_int(carry_qty)
    if require_full_qty and qty is not None and qty > 0:
        return remaining >= int(qty)
    return True


def _stable_fair_negative_table_capacity_mention(text: str, item_type: str) -> bool:
    low = str(text or "").strip().lower()
    item = str(item_type or "").strip().lower()
    if item not in {"wood", "sand", "stone"}:
        return False
    patterns = (
        rf"\bno\s+(?:remaining\s+)?{item}\b",
        rf"\bno\s+{item}\s+(?:remaining|capacity|space|room)\b",
        rf"\b{item}\s+(?:remaining|capacity|space|room)\s*(?:=|:|is)?\s*0\b",
        rf"\b{item}\s+(?:is\s+)?(?:full|unavailable)\b",
        rf"\b(?:cannot|can't|doesn't|does\s+not)\s+accept\s+{item}\b",
        rf"\baccepts?\s+only\b[^.;]*\b(?:wood|sand|stone)\b[^.;]*\bno\s+{item}\b",
    )
    return any(re.search(pattern, low) for pattern in patterns)


def _stable_fair_current_tile_table_fact(obs: dict[str, Any] | None) -> dict[str, Any] | None:
    data = dict(obs or {})
    if str(data.get("tile_terrain_type") or "").strip().lower() != "table":
        return None
    pos = _extract_xy(data.get("position"))
    fact: dict[str, Any] = {
        "kind": "table",
        "position": pos if pos is not None else None,
        "remaining_capacity": _coerce_int(data.get("tile_placeable_free_capacity")),
        "remaining_by_type": dict(data.get("tile_remaining_by_type") or {}),
        "source": "visible",
    }
    return fact


def _stable_fair_current_tile_resource_fact(obs: dict[str, Any] | None) -> dict[str, Any] | None:
    data = dict(obs or {})
    pos = _extract_xy(data.get("position"))
    if pos is None:
        return None
    items = data.get("tile_items_counts") or {}
    if not isinstance(items, dict):
        return None
    best_type = "none"
    best_qty = 0
    for item_type in ("wood", "sand", "stone"):
        qty = _coerce_int(items.get(item_type))
        if qty is not None and qty > best_qty:
            best_type = item_type
            best_qty = int(qty)
    if best_type == "none":
        return None
    return {
        "kind": "resource",
        "resource_type": best_type,
        "position": pos,
        "quantity": best_qty,
        "source": "current_tile",
    }


def _stable_fair_semantic_delivery_is_safe(proto: dict[str, Any] | None) -> bool:
    if not isinstance(proto, dict):
        return False
    fact = proto.get("fact")
    carry_type = str(proto.get("carry_type") or "none").strip().lower()
    carry_qty = _coerce_int(proto.get("carry_qty"))
    return _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True)


def _stable_fair_table_accepts_text(fact: dict[str, Any] | None) -> str:
    rb = _stable_fair_positive_remaining_by_type(fact)
    if rb:
        types = [item_type for item_type in ("wood", "sand", "stone") if item_type in rb]
        if len(types) == 1:
            accepts = f"{types[0]} only"
        elif len(types) == 2:
            accepts = f"{types[0]} and {types[1]}"
        else:
            accepts = ", ".join(types[:-1]) + f", and {types[-1]}"
        remaining = ", ".join(f"{item_type}={rb[item_type]}" for item_type in types)
        return f"accepts {accepts}; remaining {remaining}"
    rem = _coerce_int((fact or {}).get("remaining_capacity") if isinstance(fact, dict) else None)
    if rem is not None and rem > 0:
        return f"has remaining capacity {int(rem)}"
    return "has unknown remaining capacity"


def _stable_fair_table_for_type_text(fact: dict[str, Any] | None, item_type: str) -> str:
    resource_type = str(item_type or "").strip().lower()
    if resource_type not in {"wood", "sand", "stone"}:
        return _stable_fair_table_accepts_text(fact)
    remaining = _stable_fair_table_remaining_for_type(fact, resource_type)
    if remaining is not None:
        return f"can accept {resource_type}; type_rem {resource_type}={max(0, int(remaining))}"
    return f"can accept {resource_type}; type_rem unknown"


def _stable_fair_message_table_for_type(text: str) -> str:
    low = str(text or "").lower()
    for item_type in ("wood", "sand", "stone"):
        patterns = (
            rf"\b(?:your|my|carried|carrying)\s+{item_type}\b",
            rf"\b{item_type}\b[^.]*\b(?:deposit|delivery|slot|space|type_rem)\b",
            rf"\b(?:can\s+accept|space\s+for)\s+{item_type}\b",
        )
        if any(re.search(pattern, low) for pattern in patterns):
            return item_type
    return "none"


def _stable_fair_parse_remaining_by_type(text: str) -> dict[str, int]:
    raw = str(text or "")
    remaining_match = re.search(
        r"\b(?:remaining|type_rem|type_remaining|can_deposit|deposit_capacity|can_accept)\b(.+)$",
        raw,
        re.IGNORECASE,
    )
    source = str(remaining_match.group(1) or "") if remaining_match else raw
    rb: dict[str, int] = {}
    for item_type, value in re.findall(
        r"\b(wood|sand|stone)\s*(?:=|:|is\s+|remaining\s+|space\s*=?\s*)?\s*(\d+)",
        source,
        re.IGNORECASE,
    ):
        qty = _coerce_int(value)
        if qty is not None and qty > 0:
            rb[str(item_type).lower()] = int(qty)
    return rb


def _stable_fair_parse_state_obs_prefix(raw_message: str) -> dict[str, Any]:
    text = str(raw_message or "")
    if not text.strip():
        return {}
    state: dict[str, Any] = {}
    obs_facts: list[dict[str, Any]] = []
    obs_lane = ""
    found = False

    def parse_fields(payload: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for part in str(payload or "").split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key_norm = str(key or "").strip().lower()
            if key_norm:
                fields[key_norm] = str(value or "").strip()
        return fields

    def parse_obs_fact(payload: str) -> list[dict[str, Any]]:
        obs_text = str(payload or "").strip()
        if not obs_text or obs_text.lower() in {"none", "no_fact"}:
            return []
        if "|" in obs_text:
            facts: list[dict[str, Any]] = []
            for segment in obs_text.split("|"):
                facts.extend(parse_obs_fact(segment))
            return facts
        facts: list[dict[str, Any]] = []
        table_match = re.search(
            r"\btable\d*\s*(?:=|@|at)?\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?",
            obs_text,
            re.IGNORECASE,
        )
        if table_match:
            rb = _stable_fair_parse_remaining_by_type(obs_text)
            total_match = re.search(
                r"\b(?:total_rem|total_deposit_space|total_deposit|deposit_space|remaining_capacity|rem|remaining)\s*(?:=|:|is)?\s*(\d+)",
                obs_text,
                re.IGNORECASE,
            )
            rem = _coerce_int(total_match.group(1) if total_match else None)
            if rem is None and rb:
                rem = sum(rb.values())
            table: dict[str, Any] = {
                "kind": "table",
                "position": (int(table_match.group(1)), int(table_match.group(2))),
                "source": "state_obs",
            }
            if rem is not None:
                table["remaining_capacity"] = int(rem)
            if rb:
                table["remaining_by_type"] = rb
            facts.append(table)
        resource_match = re.search(
            r"\bresource\d*\s*=\s*(wood|sand|stone)@\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*:\s*(\d+)",
            obs_text,
            re.IGNORECASE,
        )
        if not resource_match:
            resource_match = re.search(
                r"\b(?:resource\s+)?(wood|sand|stone)(?:\s+resource)?\s*(?:at|@)\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?.*?(?:qty\s*=?|has|contains|available|with|:)\s*(\d+)",
                obs_text,
                re.IGNORECASE,
            )
        if resource_match:
            facts.append(
                {
                    "kind": "resource",
                    "resource_type": str(resource_match.group(1)).lower(),
                    "position": (int(resource_match.group(2)), int(resource_match.group(3))),
                    "quantity": int(resource_match.group(4)),
                    "source": "state_obs",
                }
            )
        return facts

    for raw_line in text.splitlines():
        state_match = re.match(r"^\s*STATE\s*:\s*(.+?)\s*$", raw_line, re.IGNORECASE)
        if state_match:
            found = True
            fields = parse_fields(state_match.group(1))
            if fields.get("id"):
                state["id"] = fields["id"]
            loc = _extract_xy_from_text(fields.get("loc"))
            if loc is not None:
                state["loc"] = loc
            carry_type, carry_qty = _parse_resource_quantity(fields.get("carry") or "none")
            state["carry_type"] = carry_type
            state["carry_qty"] = carry_qty
            goal = _extract_xy_from_text(fields.get("goal"))
            if goal is not None:
                state["goal"] = goal
            if fields.get("action"):
                state["action"] = fields["action"]
            continue
        obs_match = re.match(r"^\s*OBS(?:_([A-Z]+))?\s*:\s*(.+?)\s*$", raw_line, re.IGNORECASE)
        if obs_match:
            found = True
            if obs_match.group(1) and not obs_lane:
                obs_lane = str(obs_match.group(1) or "").strip().upper()
            obs_facts.extend(parse_obs_fact(obs_match.group(2)))

    if not found:
        return {}
    return {"state_obs_prefix": True, "state": state, "obs_facts": obs_facts, "obs_lane": obs_lane}


def _parse_stable_fair_semantic_message(raw_message: str) -> dict[str, Any] | None:
    prefix = _stable_fair_parse_state_obs_prefix(raw_message)
    line = _stable_fair_one_line_message(raw_message)
    if (not line and not prefix) or line.upper().startswith("SF1|") or "SFPROTO:" in line.upper():
        return None
    proto: dict[str, Any] = {
        "version": "semantic_state_obs" if prefix else "semantic",
        "intent": "status",
        "ask": "none",
        "carry_type": "none",
        "carry_qty": None,
        "note": line,
    }
    if prefix:
        state = dict(prefix.get("state") or {})
        if state.get("id"):
            proto["id"] = str(state.get("id") or "")
        if isinstance(state.get("loc"), tuple):
            proto["loc"] = state.get("loc")
        if str(state.get("carry_type") or "none") in {"wood", "sand", "stone"}:
            proto["carry_type"] = str(state.get("carry_type") or "none")
            proto["carry_qty"] = state.get("carry_qty")
        if isinstance(state.get("goal"), tuple):
            proto["goal"] = state.get("goal")
        if state.get("action"):
            proto["action"] = str(state.get("action") or "")
        obs_facts = [fact for fact in list(prefix.get("obs_facts") or []) if isinstance(fact, dict)]
        if obs_facts:
            proto["obs_facts"] = obs_facts
            obs_fact = dict(obs_facts[0])
            proto["fact"] = obs_fact
            kind = str(obs_fact.get("kind") or "").strip().lower()
            if kind == "table":
                table_for_type = _stable_fair_message_table_for_type(line)
                carry_type = str(proto.get("carry_type") or "none").strip().lower()
                if not (
                    table_for_type in {"wood", "sand", "stone"}
                    and _stable_fair_table_supports_carry(
                        obs_fact,
                        table_for_type,
                        None,
                        require_full_qty=False,
                    )
                ):
                    table_for_type = "none"
                if (
                    table_for_type == "none"
                    and carry_type in {"wood", "sand", "stone"}
                    and _stable_fair_table_supports_carry(
                        obs_fact,
                        carry_type,
                        proto.get("carry_qty"),
                        require_full_qty=False,
                    )
                ):
                    table_for_type = carry_type
                if table_for_type in {"wood", "sand", "stone"}:
                    proto["table_for_type"] = table_for_type
                    proto["intent"] = "table_for_carry"
                    proto["goal"] = obs_fact.get("position")
                else:
                    proto["intent"] = "discover"
            elif kind == "resource":
                proto["intent"] = "discover"
        return proto

    loc_match = re.search(
        r"\b(?:loc(?:ation)?(?:\s+is)?|my\s+(?:position|location)\s+is|i\s+am\s+at)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        line,
        re.IGNORECASE,
    )
    if loc_match:
        proto["loc"] = (int(loc_match.group(1)), int(loc_match.group(2)))

    carry_match = re.search(
        r"\bcarrying\s+(wood|sand|stone)\s*(?:[:=x]|\s+x\s+|\s+)?\s*(\d+)?",
        line,
        re.IGNORECASE,
    )
    if carry_match:
        proto["carry_type"] = str(carry_match.group(1)).lower()
        qty = _coerce_int(carry_match.group(2))
        proto["carry_qty"] = int(qty) if qty is not None and qty > 0 else None
        proto["intent"] = "carry"

    deliver_match = re.search(
        r"\b(?:delivering|bringing|carrying|deliver|bring)\s+(wood|sand|stone)\s*(?:[:=x]|\s+x\s+|\s+)?\s*(\d+)?.*?\btable(?:\s+at)?\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        line,
        re.IGNORECASE,
    )
    if deliver_match:
        proto["intent"] = "deliver"
        proto["carry_type"] = str(deliver_match.group(1)).lower()
        qty = _coerce_int(deliver_match.group(2))
        proto["carry_qty"] = int(qty) if qty is not None and qty > 0 else proto.get("carry_qty")
        proto["goal"] = (int(deliver_match.group(3)), int(deliver_match.group(4)))

    pickup_match = re.search(
        r"\b(?:picking\s+up|pick\s+up|pickup)\s+(?:(\d+)\s+)?(wood|sand|stone)\b.*?\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        line,
        re.IGNORECASE,
    )
    if pickup_match:
        proto["intent"] = "carry"
        proto["carry_type"] = str(pickup_match.group(2)).lower()
        qty = _coerce_int(pickup_match.group(1))
        proto["carry_qty"] = int(qty) if qty is not None and qty > 0 else proto.get("carry_qty")
        proto["fact"] = {
            "kind": "resource",
            "resource_type": str(pickup_match.group(2)).lower(),
            "position": (int(pickup_match.group(3)), int(pickup_match.group(4))),
            "quantity": int(qty) if qty is not None and qty > 0 else None,
        }

    resource_match = re.search(
        r"\b(?:resource\s+)?(wood|sand|stone)(?:\s+resource)?\s*(?:at|@)(?:\s+position)?\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?.*?(?:has|contains|qty\s*=?|available|with|:)\s*(\d+)",
        line,
        re.IGNORECASE,
    )
    if resource_match:
        proto["intent"] = "discover"
        proto["fact"] = {
            "kind": "resource",
            "resource_type": str(resource_match.group(1)).lower(),
            "position": (int(resource_match.group(2)), int(resource_match.group(3))),
            "quantity": int(resource_match.group(4)),
        }

    table_match = re.search(r"\btable(?:\s+at|@)?\s*\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?", line, re.IGNORECASE)
    if table_match:
        table_pos = (int(table_match.group(1)), int(table_match.group(2)))
        obs_facts_for_table = [fact for fact in list(proto.get("obs_facts") or []) if isinstance(fact, dict)]
        fact = next(
            (
                dict(obs_fact)
                for obs_fact in obs_facts_for_table
                if str(obs_fact.get("kind") or "") == "table" and _extract_xy(obs_fact.get("position")) == table_pos
            ),
            None,
        )
        if fact is None:
            rb = _stable_fair_parse_remaining_by_type(line)
            rem = sum(rb.values()) if rb else None
            if rem is None:
                cap_match = re.search(r"\bremaining(?:\s+capacity)?\s*(?:=|:|is)?\s*(\d+)", line, re.IGNORECASE)
                cap = _coerce_int(cap_match.group(1) if cap_match else None)
                if cap is not None and cap > 0:
                    rem = int(cap)
            fact = {
                "kind": "table",
                "position": table_pos,
            }
            if rem is not None:
                fact["remaining_capacity"] = int(rem)
            if rb:
                fact["remaining_by_type"] = rb
        proto["fact"] = fact
        table_for_type = _stable_fair_message_table_for_type(line)
        if table_for_type != "none":
            proto["table_for_type"] = table_for_type
        proto["goal"] = fact["position"] if proto.get("carry_type") != "none" else proto.get("goal")
        if proto.get("carry_type") != "none" and re.search(r"\b(?:delivering|deliver|bringing|bring)\b", line, re.IGNORECASE):
            proto["intent"] = "deliver"
        elif table_for_type != "none":
            proto["intent"] = "table_for_carry"
        else:
            proto["intent"] = "discover"

    request_match = re.search(r"\b(?:report|find|check|tell me)\b.*\btable\b.*\baccepts?\s+(wood|sand|stone)\b", line, re.IGNORECASE)
    if request_match:
        if str(proto.get("intent") or "status") not in {"carry", "deliver"}:
            proto["intent"] = "request"
        proto["ask"] = f"table_for={str(request_match.group(1)).lower()}"
    elif re.search(r"\bneed\b.*\bcompatible\s+table\b", line, re.IGNORECASE) and str(proto.get("carry_type") or "none") in {"wood", "sand", "stone"}:
        proto["ask"] = f"table_for={proto.get('carry_type')}"
    elif re.search(r"\b(?:report|find|check|tell me)\b", line, re.IGNORECASE):
        if str(proto.get("intent") or "status") in {"carry", "deliver"}:
            if str(proto.get("carry_type") or "none") in {"wood", "sand", "stone"}:
                proto["ask"] = f"table_for={proto.get('carry_type')}"
            return proto
        proto["intent"] = "request"
        proto["ask"] = "info"

    obs_facts = [fact for fact in list(proto.get("obs_facts") or []) if isinstance(fact, dict)]
    if not isinstance(proto.get("fact"), dict) and obs_facts and str(proto.get("intent") or "status") != "request":
        obs_fact = dict(obs_facts[0])
        proto["fact"] = obs_fact
        kind = str(obs_fact.get("kind") or "").strip().lower()
        if kind == "table":
            table_for_type = _stable_fair_message_table_for_type(line)
            carry_type = str(proto.get("carry_type") or "none").strip().lower()
            if (
                table_for_type == "none"
                and carry_type in {"wood", "sand", "stone"}
                and _stable_fair_table_supports_carry(obs_fact, carry_type, proto.get("carry_qty"), require_full_qty=False)
            ):
                table_for_type = carry_type
            if table_for_type in {"wood", "sand", "stone"}:
                proto["table_for_type"] = table_for_type
                if str(proto.get("intent") or "status") == "status":
                    proto["intent"] = "table_for_carry"
                if not isinstance(proto.get("goal"), tuple):
                    proto["goal"] = obs_fact.get("position")
            elif str(proto.get("intent") or "status") == "status":
                proto["intent"] = "discover"
        elif kind == "resource" and str(proto.get("intent") or "status") == "status":
            proto["intent"] = "discover"

    if proto.get("intent") == "status" and not proto.get("fact") and proto.get("carry_type") == "none":
        low = line.lower()
        if not any(term in low for term in ("resource", "table", "carrying", "report", "check", "find")):
            if not prefix:
                return None
    return proto


def _stable_fair_protocol_memory_summary(raw_message: str, parsed_facts: dict[str, Any] | None = None) -> str:
    facts = parsed_facts if isinstance(parsed_facts, dict) else {}
    proto = facts.get("stable_fair_protocol") if isinstance(facts, dict) else None
    if not isinstance(proto, dict):
        proto = facts.get("stable_fair_semantic") if isinstance(facts, dict) else None
    if not isinstance(proto, dict):
        proto = _parse_stable_fair_compact_protocol(raw_message)
    if not isinstance(proto, dict):
        proto = _parse_stable_fair_semantic_message(raw_message)
    if not isinstance(proto, dict):
        return " ".join(str(raw_message or "").split())

    parts = [f"proto {str(proto.get('intent') or 'status')}"]
    loc = proto.get("loc")
    if isinstance(loc, tuple) and len(loc) == 2:
        parts.append(f"loc={loc[0]},{loc[1]}")
    carry_type = str(proto.get("carry_type") or "none")
    carry_qty = proto.get("carry_qty")
    if carry_type != "none":
        parts.append(f"carry={carry_type}:{carry_qty if carry_qty is not None else '?'}")
    table_for_type = str(proto.get("table_for_type") or "none").strip().lower()
    if table_for_type in {"wood", "sand", "stone"}:
        parts.append(f"table_for={table_for_type}")
    fact = proto.get("fact")
    if isinstance(fact, dict):
        kind = str(fact.get("kind") or "")
        pos = _extract_xy(fact.get("position"))
        if kind == "table" and pos is not None:
            fact_text = f"table@({pos[0]},{pos[1]})"
            rem = _coerce_int(fact.get("remaining_capacity"))
            if rem is not None:
                fact_text = f"{fact_text} total_rem={int(rem)}"
            rb = fact.get("remaining_by_type") or {}
            if isinstance(rb, dict):
                needs = [
                    f"{item_type}:{int(rb[item_type])}"
                    for item_type in ("wood", "sand", "stone")
                    if _coerce_int(rb.get(item_type)) is not None and int(rb[item_type]) > 0
                ]
                if needs:
                    fact_text = f"{fact_text} can_deposit={','.join(needs)}"
            parts.append(f"fact={fact_text}")
        elif kind == "resource" and pos is not None:
            parts.append(
                f"fact={fact.get('resource_type')}@({pos[0]},{pos[1]}) qty={fact.get('quantity', 'unknown')}"
            )
    goal = proto.get("goal")
    if isinstance(goal, tuple) and len(goal) == 2:
        parts.append(f"goal=({goal[0]},{goal[1]})")
    ask = str(proto.get("ask") or "none")
    if ask and ask.lower() != "none":
        parts.append(f"ask={ask}")
    note = str(proto.get("note") or _stable_fair_protocol_note(raw_message) or "").strip()
    if str(proto.get("version") or "").lower() == "semantic" and isinstance(fact, dict):
        note = ""
    if note and _stable_fair_protocol_note_supported(proto, note):
        parts.append(f"note={note}")
    return " ".join(parts)


def _stable_fair_semantic_is_active(proto: dict[str, Any] | None) -> bool:
    if not isinstance(proto, dict):
        return False
    intent = str(proto.get("intent") or "status").strip().lower()
    ask = str(proto.get("ask") or "none").strip().lower()
    carry_type = str(proto.get("carry_type") or "none").strip().lower()
    table_for_type = str(proto.get("table_for_type") or "none").strip().lower()
    note_low = str(proto.get("note") or "").strip().lower()
    if intent == "carry" and (
        "moving " in note_low
        or " move " in f" {note_low} "
        or "picking up" in note_low
        or "pick up" in note_low
    ):
        return False
    if ask and ask != "none":
        return True
    if intent in {"reply", "request"}:
        return True
    if intent == "table_for_carry" or table_for_type in {"wood", "sand", "stone"}:
        fact = proto.get("fact")
        return _stable_fair_table_supports_carry(fact, table_for_type, None, require_full_qty=False)
    if intent == "deliver":
        return _stable_fair_semantic_delivery_is_safe(proto)
    fact = proto.get("fact")
    kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
    if intent == "carry":
        if kind == "resource":
            return False
        if kind == "table":
            return _stable_fair_table_supports_carry(fact, carry_type, proto.get("carry_qty"), require_full_qty=False)
    return False


def _stable_fair_semantic_payload_kind(proto: dict[str, Any] | None) -> str:
    if not isinstance(proto, dict):
        return "STATUS"
    intent = str(proto.get("intent") or "status").strip().lower()
    ask = str(proto.get("ask") or "none").strip().lower()
    carry_type = str(proto.get("carry_type") or "none").strip().lower()
    table_for_type = str(proto.get("table_for_type") or "none").strip().lower()
    fact = proto.get("fact")
    kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
    if intent == "reply":
        return "REPLY"
    if intent == "table_for_carry" or table_for_type in {"wood", "sand", "stone"}:
        return "TABLE_FOR_CARRY"
    if intent == "deliver":
        if _stable_fair_semantic_delivery_is_safe(proto):
            return "DELIVERY"
        return "CARRY" if carry_type in {"wood", "sand", "stone"} else "STATUS"
    if intent == "request" or (ask and ask != "none"):
        return "REQUEST"
    if carry_type in {"wood", "sand", "stone"} or intent == "carry":
        return "CARRY"
    if kind == "resource":
        return "RESOURCE"
    if kind == "table":
        return "TABLE"
    return "STATUS"


_STABLE_FAIR_ROUTING_OBJECTIVES = {
    "answer_request",
    "ask_missing_table",
    "share_table_for_carrier",
    "delivery_ready",
    "share_resource",
    "share_table",
    "coverage",
}


def _stable_fair_recipient_signal_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "coverage_edge"
    lifecycle = str(feature.get("payload_lifecycle") or "").strip().lower()
    if lifecycle in {"stale_static_table", "stale_static_resource", "exhausted_table", "request_cooldown"}:
        return "coverage_edge"
    if bool(feature.get("reply_recent")) and bool(feature.get("reply_info_request")):
        return "recent_request_edge"
    if bool(feature.get("reply_recent")):
        return "recent_reply_edge"
    if bool(feature.get("safe_table_delivery")) and str(feature.get("carrying") or "none") in {"wood", "sand", "stone"}:
        return "recipient_can_use_table_edge"
    if bool(feature.get("need_table_for_type")):
        return "sender_needs_recipient_knowledge_edge"
    if str(feature.get("best_fact_kind") or "none") in {"resource", "table"}:
        return "fresh_fact_edge"
    return "coverage_edge"


def _stable_fair_routing_edge_reason_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "low_value_coverage"
    lifecycle = str(feature.get("payload_lifecycle") or "").strip().lower()
    if lifecycle in {"stale_static_table", "stale_static_resource", "exhausted_table", "request_cooldown"}:
        return "low_value_coverage"
    if bool(feature.get("reply_recent")) and bool(feature.get("reply_info_request")):
        return "answerable_recent_request"
    carrying = str(feature.get("carrying") or "none").strip().lower()
    if bool(feature.get("safe_table_delivery")) and carrying in {"wood", "sand", "stone"}:
        return "recipient_needs_known_table"
    if bool(feature.get("own_table_relevant_to_candidate")):
        return "recipient_needs_known_table"
    if bool(feature.get("need_table_for_type")):
        return "sender_needs_table_knowledge"
    if bool(feature.get("reply_recent")):
        return "dynamic_status_refresh"
    kind = str(feature.get("best_fact_kind") or "none").strip().lower()
    if kind == "resource":
        return "fresh_resource_coverage"
    if kind == "table":
        return "fresh_table_coverage"
    if bool(feature.get("dynamic_status_refresh")):
        return "dynamic_status_refresh"
    return "low_value_coverage"


def _stable_fair_routing_recipient_value_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "coverage_only"
    edge_reason = _stable_fair_routing_edge_reason_for_feature(feature)
    if edge_reason == "answerable_recent_request":
        return "can_receive_answer"
    if edge_reason == "recipient_needs_known_table":
        return "can_act_now"
    if edge_reason == "sender_needs_table_knowledge":
        return "can_answer_sender"
    if edge_reason in {"fresh_resource_coverage", "fresh_table_coverage"}:
        return "can_update_memory"
    if edge_reason == "dynamic_status_refresh":
        return "can_update_status"
    return "coverage_only"


def _stable_fair_routing_send_risk_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "none"
    risks: list[str] = []
    lifecycle = str(feature.get("payload_lifecycle") or "").strip().lower()
    if lifecycle in {"stale_static_table", "stale_static_resource"}:
        risks.append(lifecycle)
    elif lifecycle == "exhausted_table":
        risks.append("exhausted_table")
    elif lifecycle == "request_cooldown":
        risks.append("request_cooldown")
    if bool(feature.get("duplicate_fact_recent")) and not bool(feature.get("dynamic_status_refresh")):
        risks.append("duplicate_fact_recent")
    if bool(feature.get("concentrated_pair")):
        risks.append("recent_pair_loop")
    elif bool(feature.get("stale_pair")):
        risks.append("stale_pair")
    kind = str(feature.get("best_fact_kind") or "none").strip().lower()
    try:
        target_need = float(feature.get("target_need", 1.0) or 1.0)
    except Exception:
        target_need = 1.0
    if kind in {"resource", "table"} and target_need < 1.05 and not bool(feature.get("reply_info_request")):
        risks.append("static_low_value")
    return ",".join(risks) if risks else "none"


def _stable_fair_routing_recipient_answerability_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "low_answerability"
    if bool(feature.get("reply_recent")) and bool(feature.get("reply_info_request")):
        return "direct_requester"
    if bool(feature.get("latest_incoming_table_for_current_carry")):
        return "known_table_provider"
    if bool(feature.get("latest_incoming_has_table")):
        return "recent_table_reporter"
    if bool(feature.get("in_view")) and int(feature.get("rolling_recent_contact_count", 0) or 0) == 0:
        return "uncontacted_explorer"
    if not bool(feature.get("contacted_before")) or int(feature.get("rolling_recent_contact_count", 0) or 0) == 0:
        return "uncontacted_explorer"
    return "low_answerability"


def _stable_fair_recipient_can_answer_table_request(feature: dict[str, Any] | None) -> bool:
    answerability = str(
        (feature or {}).get("recipient_answerability")
        or _stable_fair_routing_recipient_answerability_for_feature(feature)
        or ""
    ).strip().lower()
    return answerability in {"direct_requester", "known_table_provider", "recent_table_reporter"}


def _stable_fair_routing_edge_class_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "weak"
    edge_reason = str(feature.get("edge_reason") or _stable_fair_routing_edge_reason_for_feature(feature)).strip()
    if edge_reason == "answerable_recent_request":
        return "reply_request"
    if edge_reason == "recipient_needs_known_table":
        return "table_for_carrier"
    if edge_reason == "sender_needs_table_knowledge":
        return "ask_table_provider"
    if edge_reason in {"fresh_resource_coverage", "fresh_table_coverage"}:
        if bool(feature.get("duplicate_fact_recent")):
            return "weak"
        return "fresh_discovery"
    if edge_reason == "dynamic_status_refresh":
        return "status_refresh"
    return "weak"


def _stable_fair_routing_payload_state_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "blocked"
    lifecycle = str(feature.get("payload_lifecycle") or "").strip().lower()
    if lifecycle in {"exhausted_table", "request_cooldown"}:
        return "blocked"
    if lifecycle in {"stale_static_table", "stale_static_resource"}:
        return "stale"
    if lifecycle.startswith("active_") or lifecycle in {"recent_table_refresh", "usable_table", "partial_table"}:
        return "active"
    if bool(feature.get("duplicate_fact_recent")):
        return "duplicate"
    if str(feature.get("best_fact_kind") or "none").strip().lower() in {"resource", "table"}:
        return "fresh"
    return "blocked"


def _stable_fair_routing_audience_fit_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "weak"
    edge_class = _stable_fair_routing_edge_class_for_feature(feature)
    if edge_class == "reply_request":
        return "can_receive"
    if edge_class == "table_for_carrier":
        return "can_act"
    if edge_class == "ask_table_provider":
        return "can_answer" if _stable_fair_recipient_can_answer_table_request(feature) else "weak"
    if edge_class == "fresh_discovery":
        return "memory_only"
    if edge_class == "status_refresh":
        return "memory_only"
    return "weak"


def _stable_fair_routing_objective_from_intent(routing_intent: str, fact_kind: str = "") -> str:
    intent = str(routing_intent or "").strip().lower()
    kind = str(fact_kind or "").strip().lower()
    if intent == "reply":
        return "answer_request"
    if intent == "need_table_for_type":
        return "ask_missing_table"
    if intent == "table_for_carry":
        return "share_table_for_carrier"
    if intent == "delivery":
        return "delivery_ready"
    if intent == "fact_share":
        if kind == "resource":
            return "share_resource"
        if kind == "table":
            return "share_table"
    if not intent:
        if kind == "resource":
            return "share_resource"
        if kind == "table":
            return "share_table"
    return "coverage"


def _stable_fair_routing_objective_for_feature(feature: dict[str, Any] | None) -> str:
    if not isinstance(feature, dict):
        return "coverage"
    objective = str(feature.get("routing_objective") or "").strip().lower()
    if objective in _STABLE_FAIR_ROUTING_OBJECTIVES:
        return objective
    return _stable_fair_routing_objective_from_intent(
        str(feature.get("routing_intent") or ""),
        str(feature.get("best_fact_kind") or ""),
    )


def _stable_fair_routing_brief_for_feature(
    feature: dict[str, Any] | None,
    *,
    recipient: str = "",
    pressure: str = "",
) -> dict[str, Any]:
    if not isinstance(feature, dict):
        return {"recipient": str(recipient or ""), "objective": "coverage"}
    objective = _stable_fair_routing_objective_for_feature(feature)
    item_type = "none"
    if objective == "ask_missing_table":
        item_type = str(feature.get("own_carrying") or feature.get("table_for_type") or "none").strip().lower()
    elif objective == "share_table_for_carrier":
        item_type = str(
            feature.get("table_for_type")
            or feature.get("carrying")
            or feature.get("own_carrying")
            or "none"
        ).strip().lower()
    elif objective == "delivery_ready":
        item_type = str(feature.get("own_carrying") or feature.get("table_for_type") or "none").strip().lower()
    elif objective == "share_resource":
        item_type = str(feature.get("best_fact_type") or "none").strip().lower()
    if item_type not in {"wood", "sand", "stone"}:
        item_type = "none"

    constraints: list[str] = []
    if bool(feature.get("duplicate_fact_recent")):
        constraints.append("duplicate_fact_recent")
    lifecycle = str(feature.get("payload_lifecycle") or "").strip().lower()
    if lifecycle:
        constraints.append(f"lifecycle={lifecycle}")
    if objective == "ask_missing_table":
        constraints.append("request_only_no_carry_override")
    elif objective == "share_table_for_carrier":
        constraints.append("table_must_match_carried_type")
    elif objective == "delivery_ready":
        constraints.append("delivery_requires_current_carry_and_compatible_table")
    elif objective in {"share_resource", "share_table"}:
        constraints.append("keep_static_fact_kind")
    if bool(feature.get("concentrated_pair")) or bool(feature.get("stale_pair")):
        constraints.append("avoid_pair_loop_without_reply")

    ask = f"table_for={item_type}" if objective == "ask_missing_table" and item_type != "none" else "none"
    return {
        "recipient": str(recipient or feature.get("name") or ""),
        "objective": objective,
        "pressure": str(pressure or ""),
        "item_type": item_type,
        "evidence": str(feature.get("best_fact_label") or "none"),
        "evidence_kind": str(feature.get("best_fact_kind") or "none"),
        "evidence_source": str(feature.get("best_fact_source") or "none"),
        "coverage_novelty": str(feature.get("coverage_novelty") or "none"),
        "fact_area_key": str(feature.get("fact_area_key") or "unknown"),
        "payload_lifecycle": lifecycle or "none",
        "ask": ask,
        "recipient_carrying": str(feature.get("carrying") or "none"),
        "sender_carrying": str(feature.get("own_carrying") or "none"),
        "constraints": constraints,
    }


def _stable_fair_routing_message_contract_for_feature(
    feature: dict[str, Any] | None,
    *,
    recipient: str = "",
    pressure: str = "",
) -> dict[str, Any]:
    """Compact handoff from routing to protocol.

    Routing owns SEND/TO and the communication purpose. Protocol may rephrase,
    but should not silently replace a strong routing purpose with a generic fact.
    """
    brief = _stable_fair_routing_brief_for_feature(feature, recipient=recipient, pressure=pressure)
    if not isinstance(feature, dict):
        return {
            "recipient": str(recipient or ""),
            "objective": str(brief.get("objective") or "coverage"),
            "item_type": str(brief.get("item_type") or "none"),
            "ask": str(brief.get("ask") or "none"),
            "primary_fact": None,
            "fact_kind": "none",
            "fact_label": "none",
            "sender_carrying": "none",
            "sender_carrying_qty": None,
            "recipient_carrying": "none",
            "recipient_carrying_qty": None,
            "pressure": str(pressure or ""),
            "source": "routing",
        }

    primary_fact = feature.get("best_fact")
    if isinstance(primary_fact, dict):
        primary_fact = dict(primary_fact)
    else:
        primary_fact = None
    backup_fact = feature.get("backup_fact")
    if isinstance(backup_fact, dict):
        backup_fact = dict(backup_fact)
    else:
        backup_fact = None

    item_type = str(brief.get("item_type") or "none").strip().lower()
    if item_type not in {"wood", "sand", "stone"}:
        item_type = str(
            feature.get("table_for_type")
            or feature.get("own_carrying")
            or feature.get("carrying")
            or feature.get("best_fact_type")
            or "none"
        ).strip().lower()
    if item_type not in {"wood", "sand", "stone"}:
        item_type = "none"

    return {
        "recipient": str(recipient or feature.get("name") or ""),
        "objective": str(brief.get("objective") or _stable_fair_routing_objective_for_feature(feature)),
        "routing_intent": str(feature.get("routing_intent") or feature.get("routing_hint") or "coverage"),
        "routing_edge": str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)),
        "payload_state": str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)),
        "audience_fit": str(feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature)),
        "item_type": item_type,
        "ask": str(brief.get("ask") or "none"),
        "primary_fact": primary_fact,
        "backup_fact": backup_fact,
        "fact_kind": str(feature.get("best_fact_kind") or (primary_fact or {}).get("kind") or "none"),
        "fact_label": str(feature.get("best_fact_label") or _stable_fair_protocol_fact_compact(primary_fact)),
        "fact_source": str(feature.get("best_fact_source") or (primary_fact or {}).get("source") or "none"),
        "coverage_novelty": str(feature.get("coverage_novelty") or "none"),
        "fact_area_key": str(feature.get("fact_area_key") or "unknown"),
        "backup_fact_kind": str(feature.get("backup_fact_kind") or (backup_fact or {}).get("kind") or "none"),
        "backup_fact_label": str(feature.get("backup_fact_label") or _stable_fair_protocol_fact_compact(backup_fact)),
        "backup_fact_source": str(feature.get("backup_fact_source") or (backup_fact or {}).get("source") or "none"),
        "sender_carrying": str(feature.get("own_carrying") or "none"),
        "sender_carrying_qty": feature.get("own_carrying_qty"),
        "recipient_carrying": str(feature.get("carrying") or "none"),
        "recipient_carrying_qty": feature.get("carrying_qty"),
        "latest_incoming_ask_type": str(feature.get("latest_incoming_ask_type") or "none"),
        "duplicate_fact_recent": bool(feature.get("duplicate_fact_recent")),
        "payload_lifecycle": str(feature.get("payload_lifecycle") or "none"),
        "pressure": str(pressure or ""),
        "reason": str(brief.get("constraints") or ""),
        "source": "routing",
    }


def _stable_fair_contract_objective(contract: dict[str, Any] | None) -> str:
    if not isinstance(contract, dict):
        return ""
    return str(contract.get("objective") or "").strip().lower()

class StableFairMemoryMechanism(MemoryMechanism):
    """StableFair memory mechanism with the required Reference helpers inlined."""

    def __init__(self, owner, config: dict[str, Any] | None = None):
        MemoryMechanism.__init__(self, owner, config)
        self.enable_structured_view_memory = bool(self.config.get("enable_structured_view_memory", False))
        self.view_memory_k = int(self.config.get("view_memory_k", 5) or 5)
        self.view_memory_max_lines = int(self.config.get("view_memory_max_lines", 12) or 12)
        self.kb_ttl_rounds = max(0, _safe_int(self.config.get("kb_ttl_rounds", 0), 0))
        self.kb_memory_max_entries = max(
            0,
            _safe_int(self.config.get("kb_memory_max_entries", self.view_memory_max_lines), self.view_memory_max_lines),
        )
        self.kb_nearest_tables_k = max(0, _safe_int(self.config.get("kb_nearest_tables_k", 2), 2))
        self.kb_nearest_items_k = max(0, _safe_int(self.config.get("kb_nearest_items_k", 3), 3))
        self.response_memory_k = max(0, _safe_int(self.config.get("response_memory_k", 5), 5))
        self.enable_message_memory = bool(self.config.get("enable_message_memory", False))
        self.message_memory_k_sent = int(self.config.get("message_memory_k_sent", 3) or 3)
        self.message_memory_k_recv = int(self.config.get("message_memory_k_recv", 3) or 3)
        self._sent = deque(maxlen=self.message_memory_k_sent)
        self._recv = deque(maxlen=self.message_memory_k_recv)
        self._bind_owner_state()

    def _bind_owner_state(self) -> None:
        self.owner.enable_structured_view_memory = self.enable_structured_view_memory
        self.owner.view_memory_k = self.view_memory_k
        self.owner.view_memory_max_lines = self.view_memory_max_lines
        self.owner.kb_ttl_rounds = self.kb_ttl_rounds
        self.owner.kb_memory_max_entries = self.kb_memory_max_entries
        self.owner.kb_nearest_tables_k = self.kb_nearest_tables_k
        self.owner.kb_nearest_items_k = self.kb_nearest_items_k
        self.owner.response_memory_k = self.response_memory_k
        self.owner.enable_response_memory = self.response_memory_k > 0
        self.owner.enable_message_memory = self.enable_message_memory
        if not hasattr(self.owner, "view_struct_mem"):
            self.owner.view_struct_mem = deque(maxlen=self.view_memory_k)
        else:
            try:
                self.owner.view_struct_mem = deque(list(self.owner.view_struct_mem), maxlen=self.view_memory_k)
            except Exception:
                self.owner.view_struct_mem = deque(maxlen=self.view_memory_k)
        if not hasattr(self.owner, "known_map") or not isinstance(getattr(self.owner, "known_map"), dict):
            self.owner.known_map = {}
        try:
            current_response_memory = list(getattr(self.owner, "response_memory", []) or [])
            self.owner.response_memory = deque(current_response_memory[-self.response_memory_k :], maxlen=self.response_memory_k)
        except Exception:
            self.owner.response_memory = deque(maxlen=self.response_memory_k)

    def reset(self) -> None:
        self._sent = deque(maxlen=self.message_memory_k_sent)
        self._recv = deque(maxlen=self.message_memory_k_recv)
        try:
            self.owner.view_struct_mem = deque(maxlen=self.view_memory_k)
        except Exception:
            pass
        try:
            self.owner.known_map = {}
        except Exception:
            pass
        try:
            self.owner.response_memory = deque(maxlen=self.response_memory_k)
        except Exception:
            pass

    def observe(self, ctx: RoundContext) -> None:
        del ctx
        return None

    async def before_message_routing(self, ctx: RoundContext) -> None:
        updater = getattr(self.owner, "kb_update_from_messages_via_llm", None)
        if callable(updater):
            await updater(ctx.obs)

    def receive_incoming(
        self,
        ctx: RoundContext,
        protocol: ProtocolMechanism,
        sender: str,
        raw_message: str,
    ) -> None:
        del protocol
        if not self.enable_message_memory:
            return
        self._recv.append(
            {
                "from": str(sender or ""),
                "message": str(raw_message or ""),
                "round": int(ctx.round_num),
            }
        )

    def ingest_incoming(self, ctx: RoundContext, parsed_messages: list[ParsedIncomingMessage]) -> None:
        if not self.enable_message_memory:
            return
        for item in parsed_messages:
            self._recv.append(
                {
                    "from": str(item.sender or ""),
                    "message": str(item.raw_message or ""),
                    "round": int(ctx.round_num),
                }
            )
            self._merge_parsed_facts_into_known_map(ctx, item)

    def _merge_parsed_facts_into_known_map(self, ctx: RoundContext, item: ParsedIncomingMessage) -> None:
        facts = item.parsed_facts if isinstance(item.parsed_facts, dict) else {}
        merge = getattr(self.owner, "_kb_merge_entry", None)
        if not callable(merge):
            return

        sender = str(item.sender or "")
        round_num = int(ctx.round_num)

        def _merge_one(pos_xy: tuple[int, int], patch: dict[str, Any], evidence: str) -> None:
            try:
                merge(pos_xy, patch, "msg", round_num, sender=sender, evidence=evidence)
            except Exception:
                pass

        for table in facts.get("tables") or []:
            try:
                x = int(table.get("x"))
                y = int(table.get("y"))
                rem = table.get("remaining_capacity")
                patch: dict[str, Any] = {"terrain_type": "table"}
                if rem is not None:
                    patch["remaining_capacity"] = int(rem)
                _merge_one((x, y), patch, "PARSED_INCOMING_TABLE")
            except Exception:
                continue

        for resource in facts.get("resources") or []:
            try:
                x = int(resource.get("x"))
                y = int(resource.get("y"))
                item_type = str(resource.get("type") or "").strip().lower()
                qty = int(resource.get("quantity"))
                if not item_type or qty <= 0:
                    continue
                _merge_one((x, y), {"terrain_type": "resource", "items": {item_type: qty}}, f"PARSED_INCOMING_RESOURCE:{item_type}")
            except Exception:
                continue

        goal = facts.get("goal")
        if isinstance(goal, dict):
            try:
                x = int(goal.get("x"))
                y = int(goal.get("y"))
                existing = {}
                try:
                    existing = dict((getattr(self.owner, "known_map", {}) or {}).get((x, y)) or {})
                except Exception:
                    existing = {}
                if str(existing.get("terrain_type") or "").lower() != "table" and not (existing.get("items") or {}):
                    _merge_one((x, y), {"terrain_type": "goal"}, "PARSED_INCOMING_GOAL")
            except Exception:
                pass

        reconcile = getattr(self.owner, "_kb_reconcile_and_prune", None)
        if callable(reconcile):
            try:
                reconcile(round_num)
            except Exception:
                pass

    def build_action_context(self, ctx: RoundContext) -> MemoryContext:
        builder = getattr(self.owner, "_build_action_memory_context", None)
        if callable(builder):
            return builder(ctx.obs, ctx.received_messages, self)
        return MemoryContext()

    def build_message_context(self, ctx: RoundContext) -> MemoryContext:
        facts = {
            "known_map_size": len(getattr(self.owner, "known_map", {}) or {}),
            "recent_messages_received": list(self._recv),
            "recent_messages_sent": list(self._sent),
        }
        builder = getattr(self.owner, "_build_message_memory_context", None)
        if callable(builder):
            return builder(ctx.obs, self, facts)
        return MemoryContext(enabled=bool(self.enable_structured_view_memory or self.enable_message_memory), facts=facts)

    def record_outgoing(self, ctx: RoundContext, outgoing: OutgoingMessage) -> None:
        if not self.enable_message_memory:
            return
        if outgoing.send and outgoing.recipient and outgoing.message:
            self._sent.append(
                {
                    "to": str(outgoing.recipient),
                    "message": str(outgoing.message),
                    "round": int(ctx.round_num),
                }
            )

    def export_state(self) -> dict[str, Any]:
        tables = []
        items = []
        known_map_state = {}
        known_map = getattr(self.owner, "known_map", {}) or {}
        for (x, y), rec in known_map.items():
            entry = {"x": int(x), "y": int(y)}
            terrain = rec.get("terrain_type")
            if terrain:
                entry["terrain"] = str(terrain)
            rem = rec.get("remaining_capacity")
            if rem is not None:
                entry["remaining_capacity"] = int(rem)
            item_map = rec.get("items") or {}
            if item_map:
                try:
                    entry["items"] = {str(k): int(v) for k, v in item_map.items()}
                except Exception:
                    entry["items"] = item_map
            if str(terrain) == "table":
                tables.append(entry)
            elif item_map:
                items.append(entry)
            known_map_state[f"{int(x)},{int(y)}"] = dict(entry)
        return {
            "tables": tables,
            "items": items,
            "messages": {"sent": list(self._sent), "recv": list(self._recv)},
            "known_map": known_map_state,
            "known_map_size": len(known_map_state),
        }

    _reference_bind_owner_state = _bind_owner_state
    _reference_reset = reset
    _reference_before_message_routing = before_message_routing
    _reference_merge_parsed_facts_into_known_map = _merge_parsed_facts_into_known_map
    _reference_build_action_context = build_action_context
    _reference_build_message_context = build_message_context
    _reference_export_state = export_state

    TEAM_STATE_MAX_AGE_ROUNDS = 12
    TEAM_STATE_MAX_AGENTS = 4
    TEAM_STATE_MAX_REQUESTS = 3
    TEAM_STATE_MAX_COORDINATION = 4
    TEAM_STATE_MAX_COORDINATION_AGE_ROUNDS = 6
    TEAM_COMM_MAX_LINES = 4

    def _disabled_by_preset(self) -> bool:
        return bool(self.config.get("base_no_mechanisms") or self.config.get("disabled"))

    def _bind_disabled_owner_state(self) -> None:
        self.enable_structured_view_memory = False
        self.enable_message_memory = False
        self.response_memory_k = 0
        self.owner.enable_structured_view_memory = False
        self.owner.enable_message_memory = False
        self.owner.enable_response_memory = False
        self.owner.response_memory_k = 0
        self.owner.stable_fair_team_state = {"agents": {}, "requests": [], "coordination": []}
        try:
            self.owner.response_memory = deque(maxlen=0)
        except Exception:
            self.owner.response_memory = []
        try:
            self.owner.view_struct_mem = deque(maxlen=0)
        except Exception:
            self.owner.view_struct_mem = []
        try:
            self.owner.known_map = {}
        except Exception:
            pass

    def _bind_owner_state(self) -> None:
        if self._disabled_by_preset():
            self._bind_disabled_owner_state()
            return
        self._reference_bind_owner_state()
        if not isinstance(getattr(self.owner, "stable_fair_team_state", None), dict):
            self.owner.stable_fair_team_state = {"agents": {}, "requests": [], "coordination": []}

    def reset(self) -> None:
        if self._disabled_by_preset():
            self._bind_disabled_owner_state()
            return
        self._reference_reset()
        try:
            self.owner.stable_fair_team_state = {"agents": {}, "requests": [], "coordination": []}
        except Exception:
            pass

    async def before_message_routing(self, ctx: RoundContext) -> None:
        if self._disabled_by_preset():
            del ctx
            return None
        return await self._reference_before_message_routing(ctx)

    def receive_incoming(
        self,
        ctx: RoundContext,
        protocol,
        sender: str,
        raw_message: str,
    ) -> None:
        if self._disabled_by_preset() or not self.enable_message_memory:
            return
        try:
            parsed = protocol.parse_incoming(ctx, sender, raw_message)
        except Exception:
            parsed = ParsedIncomingMessage(
                sender=str(sender or ""),
                raw_message=str(raw_message or ""),
                parsed_facts={},
                meta={"protocol_profile": "stable_fair_semantic_v1", "parse_error": True},
            )
        self.ingest_incoming(ctx, [parsed])

    def ingest_incoming(self, ctx: RoundContext, parsed_messages: list[ParsedIncomingMessage]) -> None:
        if self._disabled_by_preset() or not self.enable_message_memory:
            return
        for item in parsed_messages:
            self._recv.append(
                {
                    "from": str(item.sender or ""),
                    "message": _stable_fair_protocol_memory_summary(item.raw_message, item.parsed_facts),
                    "round": int(ctx.round_num),
                }
            )
            self._merge_parsed_facts_into_known_map(ctx, item)
            self._merge_protocol_team_state(ctx, item)

    def record_outgoing(self, ctx: RoundContext, outgoing: OutgoingMessage) -> None:
        if self._disabled_by_preset() or not self.enable_message_memory:
            return
        if outgoing.send and outgoing.recipient and outgoing.message:
            self._sent.append(
                {
                    "to": str(outgoing.recipient),
                    "message": _stable_fair_protocol_memory_summary(outgoing.message),
                    "round": int(ctx.round_num),
                }
            )

    def format_action_received_message(self, sender: str, raw_message: str, obs: dict[str, Any] | None = None) -> str:
        del sender, obs
        summary = _stable_fair_protocol_memory_summary(raw_message)
        proto = _parse_stable_fair_compact_protocol(raw_message)
        if not isinstance(proto, dict):
            proto = _parse_stable_fair_semantic_message(raw_message)
        note = str((proto or {}).get("note") or "").strip() if isinstance(proto, dict) else ""
        if note and _stable_fair_protocol_note_supported(proto, note) and self._team_communication_summary_actionable(note):
            return note[:160]
        if self._team_communication_summary_actionable(summary):
            return summary
        text = " ".join(str(raw_message or "").split())
        if not text:
            return ""
        low = text.lower()
        if any(term in low for term in ("wood", "sand", "stone", "table", "carrying", "deliver", "pickup", "pick up", "report")):
            return text[:140]
        return ""

    def _merge_parsed_facts_into_known_map(self, ctx: RoundContext, item: ParsedIncomingMessage) -> None:
        self._reference_merge_parsed_facts_into_known_map(ctx, item)
        facts = item.parsed_facts if isinstance(item.parsed_facts, dict) else {}
        merge = getattr(self.owner, "_kb_merge_entry", None)
        if not callable(merge):
            return
        sender = str(item.sender or "")
        round_num = int(ctx.round_num)
        for table in facts.get("tables") or []:
            if not isinstance(table, dict):
                continue
            rb = table.get("remaining_by_type")
            if not isinstance(rb, dict) or not rb:
                continue
            try:
                x = int(table.get("x"))
                y = int(table.get("y"))
                patch = {
                    "terrain_type": "table",
                    "remaining_capacity": _coerce_int(table.get("remaining_capacity")),
                    "remaining_by_type": {
                        str(k).lower(): int(v)
                        for k, v in rb.items()
                        if str(k).lower() in {"wood", "sand", "stone"} and _coerce_int(v) is not None
                    },
                }
                merge((x, y), patch, "msg", round_num, sender=sender, evidence="PARSED_INCOMING_TABLE_NEED")
            except Exception:
                continue

    def _team_state(self) -> dict[str, Any]:
        state = getattr(self.owner, "stable_fair_team_state", None)
        if not isinstance(state, dict):
            state = {"agents": {}, "requests": [], "coordination": []}
            try:
                self.owner.stable_fair_team_state = state
            except Exception:
                pass
        state.setdefault("agents", {})
        state.setdefault("requests", [])
        state.setdefault("coordination", [])
        return state

    def _merge_protocol_team_state(self, ctx: RoundContext, item: ParsedIncomingMessage) -> None:
        facts = item.parsed_facts if isinstance(item.parsed_facts, dict) else {}
        proto = facts.get("stable_fair_protocol")
        if not isinstance(proto, dict):
            proto = facts.get("stable_fair_semantic")
        if not isinstance(proto, dict):
            return
        sender = str(item.sender or proto.get("id") or "").strip()
        if not sender:
            return
        round_num = int(ctx.round_num)
        state = self._team_state()
        agents = state.setdefault("agents", {})
        payload_kind = _stable_fair_semantic_payload_kind(proto)
        sender_state = {
            "loc": proto.get("loc"),
            "carry_type": str(proto.get("carry_type") or "none"),
            "carry_qty": proto.get("carry_qty"),
            "goal": proto.get("goal"),
            "action": proto.get("action"),
        }
        payload_about = {
            "intent": str(proto.get("intent") or "status"),
            "payload_kind": payload_kind,
            "table_for_type": str(proto.get("table_for_type") or "none"),
            "fact": proto.get("fact"),
            "ask": str(proto.get("ask") or "none"),
            "note": str(proto.get("note") or "").strip(),
        }
        agent_state = {
            "round": round_num,
            "intent": str(proto.get("intent") or "status"),
            "payload_kind": payload_kind,
            "loc": proto.get("loc"),
            "carry_type": str(proto.get("carry_type") or "none"),
            "carry_qty": proto.get("carry_qty"),
            "table_for_type": str(proto.get("table_for_type") or "none"),
            "goal": proto.get("goal"),
            "fact": proto.get("fact"),
            "ask": str(proto.get("ask") or "none"),
            "note": str(proto.get("note") or "").strip(),
            "sender_state": sender_state,
            "payload_about": payload_about,
        }
        agents[sender] = agent_state
        ask = str(agent_state.get("ask") or "none").strip()
        if ask and ask.lower() != "none":
            requests = list(state.setdefault("requests", []))
            requests.append({"from": sender, "round": round_num, "ask": ask})
            state["requests"] = requests[-int(self.TEAM_STATE_MAX_REQUESTS) :]
        if _stable_fair_semantic_is_active(proto):
            coordination = list(state.setdefault("coordination", []))
            entry = {
                "from": sender,
                "round": round_num,
                "payload_kind": _stable_fair_semantic_payload_kind(proto),
                "intent": str(proto.get("intent") or "status"),
                "carry_type": str(proto.get("carry_type") or "none"),
                "carry_qty": proto.get("carry_qty"),
                "table_for_type": str(proto.get("table_for_type") or "none"),
                "goal": proto.get("goal"),
                "ask": ask,
                "fact": proto.get("fact"),
                "note": str(proto.get("note") or "").strip(),
            }

            def coordination_key(rec: dict[str, Any]) -> tuple[Any, ...]:
                fact = rec.get("fact") if isinstance(rec, dict) else None
                fact_kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
                fact_pos = _extract_xy((fact or {}).get("position") if isinstance(fact, dict) else None)
                fact_label = ""
                if isinstance(fact, dict):
                    if fact_kind == "resource":
                        fact_label = str(fact.get("resource_type") or "")
                    elif fact_kind == "table":
                        fact_label = _stable_fair_protocol_need_compact(fact)
                goal = rec.get("goal") if isinstance(rec, dict) else None
                return (
                    str((rec or {}).get("from") or ""),
                    str((rec or {}).get("payload_kind") or ""),
                    str((rec or {}).get("carry_type") or "none"),
                    str((rec or {}).get("table_for_type") or "none"),
                    str((rec or {}).get("ask") or "none"),
                    goal if isinstance(goal, tuple) else None,
                    fact_kind,
                    fact_pos,
                    fact_label,
                )

            entry_key = coordination_key(entry)
            coordination = [rec for rec in coordination if not isinstance(rec, dict) or coordination_key(rec) != entry_key]
            coordination.append(entry)
            state["coordination"] = coordination[-int(self.TEAM_STATE_MAX_COORDINATION) :]
        self._prune_protocol_team_state(round_num)

    def _prune_protocol_team_state(self, round_num: int) -> None:
        state = self._team_state()
        max_age = int(self.TEAM_STATE_MAX_AGE_ROUNDS)
        agents = {}
        for name, rec in dict(state.get("agents") or {}).items():
            try:
                rec_round = int((rec or {}).get("round", 0) or 0)
            except Exception:
                rec_round = 0
            if int(round_num) - rec_round <= max_age:
                agents[str(name)] = rec
        state["agents"] = agents
        requests = []
        for rec in list(state.get("requests") or []):
            try:
                rec_round = int((rec or {}).get("round", 0) or 0)
            except Exception:
                rec_round = 0
            if int(round_num) - rec_round <= max_age:
                requests.append(rec)
        state["requests"] = requests[-int(self.TEAM_STATE_MAX_REQUESTS) :]
        coordination = []
        max_coord_age = int(self.TEAM_STATE_MAX_COORDINATION_AGE_ROUNDS)
        for rec in list(state.get("coordination") or []):
            try:
                rec_round = int((rec or {}).get("round", 0) or 0)
            except Exception:
                rec_round = 0
            if int(round_num) - rec_round <= max_coord_age:
                coordination.append(rec)
        state["coordination"] = coordination[-int(self.TEAM_STATE_MAX_COORDINATION) :]

    def _active_coordination_section(self, round_num: int) -> str:
        self._prune_protocol_team_state(int(round_num))
        state = self._team_state()
        records = list(state.get("coordination") or [])
        if not records:
            return ""
        lines = ["Memory - active coordination:"]
        ordered = sorted(
            [rec for rec in records if isinstance(rec, dict)],
            key=lambda rec: int(rec.get("round", 0) or 0),
            reverse=True,
        )[: int(self.TEAM_STATE_MAX_COORDINATION)]
        for rec in ordered:
            sender = str(rec.get("from") or "?")
            payload_kind = str(rec.get("payload_kind") or rec.get("intent") or "STATUS")
            carry_type = str(rec.get("carry_type") or "none")
            carry_qty = rec.get("carry_qty")
            carry_text = ""
            if carry_type in {"wood", "sand", "stone"}:
                carry_text = f"; sender_carry={carry_type}:{carry_qty if carry_qty is not None else '?'}"
            ask = str(rec.get("ask") or "none")
            ask_text = f"; ask={ask}" if ask and ask.lower() != "none" else ""
            table_for_type = str(rec.get("table_for_type") or "none")
            table_for_text = f"; payload_table_for={table_for_type}" if table_for_type in {"wood", "sand", "stone"} else ""
            goal = rec.get("goal")
            goal_text = f"; goal=({goal[0]},{goal[1]})" if isinstance(goal, tuple) and len(goal) == 2 else ""
            fact = rec.get("fact")
            fact_text = ""
            if isinstance(fact, dict):
                pos = _extract_xy(fact.get("position"))
                kind = str(fact.get("kind") or "")
                if kind == "resource" and pos is not None:
                    fact_text = f"; fact={fact.get('resource_type')}@({pos[0]},{pos[1]})"
                elif kind == "table" and pos is not None:
                    rb = _stable_fair_positive_remaining_by_type(fact)
                    if rb:
                        caps = ",".join(f"{t}:{rb[t]}" for t in ("wood", "sand", "stone") if t in rb)
                        total_rem = _coerce_int(fact.get("remaining_capacity"))
                        total_text = f" total_rem={int(total_rem)}" if total_rem is not None else ""
                        fact_text = f"; fact=table@({pos[0]},{pos[1]}){total_text} can_deposit={caps}"
                    else:
                        total_rem = _coerce_int(fact.get("remaining_capacity"))
                        total_text = f" total_rem={int(total_rem)}" if total_rem is not None else ""
                        fact_text = f"; fact=table@({pos[0]},{pos[1]}){total_text}"
            note = str(rec.get("note") or "").strip()
            note_text = f"; note={note[:90]}" if note else ""
            lines.append(
                f"- from {sender} r{rec.get('round')}: {payload_kind}{carry_text}{table_for_text}{ask_text}{goal_text}{fact_text}{note_text}"
            )
        return "\n".join(lines) if len(lines) > 1 else ""

    def _protocol_team_state_section(self, round_num: int) -> str:
        self._prune_protocol_team_state(int(round_num))
        state = self._team_state()
        agents = dict(state.get("agents") or {})
        if not agents and not state.get("requests"):
            return ""
        lines = ["Memory - protocol team state:"]
        ordered_agents = sorted(
            agents.items(),
            key=lambda pair: int((pair[1] or {}).get("round", 0) or 0),
            reverse=True,
        )[: int(self.TEAM_STATE_MAX_AGENTS)]
        for name, rec in ordered_agents:
            loc = rec.get("loc") if isinstance(rec, dict) else None
            loc_text = f"@({loc[0]},{loc[1]})" if isinstance(loc, tuple) and len(loc) == 2 else "@unknown"
            carry_type = str(rec.get("carry_type") or "none")
            carry_qty = rec.get("carry_qty")
            carry_text = "none" if carry_type == "none" else f"{carry_type}:{carry_qty if carry_qty is not None else '?'}"
            table_for_type = str(rec.get("table_for_type") or "none")
            table_for_text = f"; payload_table_for={table_for_type}" if table_for_type in {"wood", "sand", "stone"} else ""
            goal = rec.get("goal")
            goal_text = f"; goal=({goal[0]},{goal[1]})" if isinstance(goal, tuple) and len(goal) == 2 else ""
            fact = rec.get("fact") if isinstance(rec, dict) else None
            fact_text = ""
            if isinstance(fact, dict):
                kind = str(fact.get("kind") or "status")
                if kind == "table":
                    pos = fact.get("position")
                    if isinstance(pos, tuple) and len(pos) == 2:
                        fact_text = f"; fact=table@({pos[0]},{pos[1]}) total_rem={fact.get('remaining_capacity', 'unknown')}"
                        rb = _stable_fair_positive_remaining_by_type(fact)
                        if rb:
                            needs = ",".join(f"{item_type}:{rb[item_type]}" for item_type in ("wood", "sand", "stone") if item_type in rb)
                            fact_text = f"{fact_text} can_deposit={needs}"
                elif kind == "resource":
                    pos = fact.get("position")
                    if isinstance(pos, tuple) and len(pos) == 2:
                        fact_text = f"; fact={fact.get('resource_type')}@({pos[0]},{pos[1]}) qty={fact.get('quantity', 'unknown')}"
            note = str(rec.get("note") or "").strip()
            note_proto = {
                "intent": rec.get("intent"),
                "carry_type": rec.get("carry_type"),
                "table_for_type": rec.get("table_for_type"),
                "fact": fact,
            }
            note_text = f"; note={note[:100]}" if note and _stable_fair_protocol_note_supported(note_proto, note) else ""
            payload_kind = str(rec.get("payload_kind") or rec.get("intent") or "STATUS")
            payload_prefix = f"; payload={payload_kind}"
            lines.append(
                f"- {name} r{rec.get('round')}: {rec.get('intent')} {loc_text}; "
                f"carry={carry_text}{goal_text}{payload_prefix}{fact_text}{note_text}"
            )
            if table_for_text:
                lines[-1] = f"{lines[-1]}{table_for_text}"
        for req in list(state.get("requests") or [])[-int(self.TEAM_STATE_MAX_REQUESTS) :]:
            lines.append(f"- request from {req.get('from')} r{req.get('round')}: {req.get('ask')}")
        return "\n".join(lines)

    def _team_communication_summary_actionable(self, summary: str) -> bool:
        text = str(summary or "").strip().lower()
        if not text:
            return False
        if text.startswith("proto "):
            return True
        return any(
            term in text
            for term in (
                "fact=",
                "ask=",
                "resource",
                "table",
                "carrying",
                "carry=",
                "deliver",
                "pickup",
                "pick up",
                "wood",
                "sand",
                "stone",
                "wood@",
                "sand@",
                "stone@",
            )
        )

    def _team_communication_section(self) -> str:
        records = list(self._recv or [])
        if not records:
            return ""
        lines = ["Memory - team communication facts:"]
        seen: set[tuple[str, str]] = set()
        for rec in reversed(records):
            sender = str((rec or {}).get("from") or "?")
            summary = str((rec or {}).get("message") or "").strip()
            if not self._team_communication_summary_actionable(summary):
                continue
            key = (sender, summary)
            if key in seen:
                continue
            seen.add(key)
            round_text = (rec or {}).get("round")
            round_part = f" r{round_text}" if round_text is not None else ""
            lines.append(f"- from {sender}{round_part}: {summary}")
            if len(lines) > int(self.TEAM_COMM_MAX_LINES):
                break
        return "\n".join(lines) if len(lines) > 1 else ""

    def _remove_raw_message_memory_sections(self, memory_ctx: MemoryContext) -> None:
        filtered: list[str] = []
        for section in list(memory_ctx.memory_sections or []):
            text = str(section or "")
            if text.startswith("Memory \u2014 messages you received") or text.startswith("Recent received messages:"):
                continue
            filtered.append(text)
        memory_ctx.memory_sections = filtered

    def _attach_protocol_team_state(self, ctx: RoundContext, memory_ctx: MemoryContext) -> MemoryContext:
        active = self._active_coordination_section(int(ctx.round_num))
        if active:
            memory_ctx.enabled = True
            memory_ctx.memory_sections.append(active)
            memory_ctx.memory_hints.append("- Use active coordination only when it is recent and type-compatible; do not override visible table compatibility.")
            memory_ctx.memory_hints.append("- Treat table facts as movement targets; choose PUT_DOWN for score only when your current tile is a compatible table.")
        section = self._protocol_team_state_section(int(ctx.round_num))
        if not section:
            return memory_ctx
        memory_ctx.enabled = True
        memory_ctx.memory_sections.append(section)
        memory_ctx.memory_hints.append("- Use protocol team state to avoid duplicate deliveries and match carrying resources to compatible tables.")
        memory_ctx.memory_hints.append("- Table facts are map evidence, not permission to PUT_DOWN on empty/resource tiles.")
        facts = dict(memory_ctx.facts or {})
        facts["stable_fair_team_state"] = self._team_state()
        memory_ctx.facts = facts
        return memory_ctx

    def build_action_context(self, ctx: RoundContext) -> MemoryContext:
        if self._disabled_by_preset():
            del ctx
            return MemoryContext(enabled=False)
        memory_ctx = self._reference_build_action_context(ctx)
        self._remove_raw_message_memory_sections(memory_ctx)
        comm_section = self._team_communication_section()
        if comm_section:
            memory_ctx.enabled = True
            memory_ctx.memory_sections.append(comm_section)
            memory_ctx.memory_hints.append("- Use team communication facts as compact evidence; ignore generic status that lacks resources, tables, carrying, delivery, or a request.")
        return self._attach_protocol_team_state(ctx, memory_ctx)

    def build_message_context(self, ctx: RoundContext) -> MemoryContext:
        if self._disabled_by_preset():
            del ctx
            return MemoryContext(enabled=False)
        return self._attach_protocol_team_state(ctx, self._reference_build_message_context(ctx))

    def export_state(self) -> dict[str, Any]:
        if self._disabled_by_preset():
            return {
                "tables": [],
                "items": [],
                "messages": {"sent": [], "recv": []},
                "known_map": {},
                "known_map_size": 0,
                "stable_fair_team_state": {"agents": {}, "requests": [], "coordination": []},
            }
        state = self._reference_export_state()
        state["stable_fair_team_state"] = self._team_state()
        return state


class StableFairRoutingMechanism(RoutingMechanism):
    """StableFair routing mechanism with routing baselines flattened in place."""

    def __init__(self, owner, config: dict[str, Any] | None = None):
        RoutingMechanism.__init__(self, owner, config)
        self.single_message_per_round = bool(self.config.get("single_message_per_round", False))
        self.mode = str(self.config.get("single_message_prompt_mode", "basic") or "basic").lower()
        self.seed = int(self.config.get("seed", 0) or 0)
        self.disable_send = bool(self.config.get("disable_send", False))

    def requires_raw_message_obs(self) -> bool:
        return True

    async def select_targets(
        self,
        ctx: RoundContext,
        candidates: list[MessageCandidate],
        memory_ctx: MemoryContext,
    ) -> RoutingDecision:
        del memory_ctx
        if self.disable_send:
            return RoutingDecision(send=False, recipients=[], mode=self.mode, meta={"mode": self.mode, "disabled_by_preset": True})
        if not candidates:
            return RoutingDecision(send=False, recipients=[], mode=self.mode, meta={"mode": self.mode, "error_reason": "no_candidates"})

        names = [candidate.agent_name for candidate in candidates]
        if not self.single_message_per_round:
            return RoutingDecision(send=True, recipients=names, mode="per_target", meta={"mode": "per_target"})

        if self.mode == "random":
            recipient = self._select_random_target(ctx, names)
            if recipient:
                return RoutingDecision(send=True, recipients=[recipient], mode="random", meta={"mode": "random", "route_strategy": "faithful_random"})
            return RoutingDecision(send=False, recipients=[], mode="random", meta={"mode": "random", "error_reason": "no_candidates"})

        if self.mode in {"basic", "enhanced"}:
            return await self._select_llm_single_target(ctx, names)

        recipient = names[0] if names else None
        if recipient:
            return RoutingDecision(send=True, recipients=[recipient], mode=self.mode, meta={"mode": self.mode})
        return RoutingDecision(send=False, recipients=[], mode=self.mode, meta={"mode": self.mode, "error_reason": "no_candidates"})

    async def _select_llm_single_target(self, ctx: RoundContext, candidate_names: list[str]) -> RoutingDecision:
        prompt = self._build_original_single_message_prompt(ctx.obs, list(candidate_names), mode=self.mode or "basic")
        response = ""
        api_call_time = 0.0
        try:
            p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            t0 = time.time()
            sampling_override = None
            if str(self.mode or "").lower() == "basic":
                try:
                    sampler = getattr(self.owner.brain, "sampling_override_for", None)
                    if callable(sampler):
                        sampling_override = sampler("basic_routing")
                except Exception:
                    sampling_override = None
            if sampling_override:
                response = await self.owner.brain.generate(prompt, sampling_override=sampling_override)
            else:
                response = await self.owner.brain.generate(prompt)
            api_call_time = time.time() - t0
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            self._jump_back()
        except Exception as exc:
            response = f"SEND: no\nTO: none\nMSG: routing_error {exc}"

        send, recipient, draft_message, error_reason = self._parse_route_message_decision(response)
        normalized_recipient = self._normalize_candidate_name(recipient, candidate_names)
        if send and normalized_recipient:
            recipient = normalized_recipient

        meta = {
            "mode": self.mode or "basic",
            "route_strategy": "faithful_llm_single",
            "routing_prompt": prompt,
            "routing_response": response,
            "routing_api_call_time": api_call_time,
            "routing_tokens": self._snapshot_brain_usage(),
            "candidate_order": list(candidate_names),
        }

        legal_names = set(candidate_names)
        if not send:
            meta["error_reason"] = error_reason or "model_send_no"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if not recipient:
            meta["error_reason"] = error_reason or "missing_recipient"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if recipient not in legal_names:
            meta["error_reason"] = "invalid_recipient"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if not draft_message:
            meta["error_reason"] = "missing_message"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)

        meta["draft_message"] = draft_message
        return RoutingDecision(send=True, recipients=[recipient], mode=self.mode or "basic", meta=meta)

    def _build_original_single_message_prompt(self, obs: dict, candidates: list[str], mode: str = "basic") -> str:
        round_num = obs.get("round", 0)
        pos = obs.get("position", {}) or {}
        x, y = pos.get("x"), pos.get("y")
        agents_in_comm = obs.get("agents_in_communication_range", {}) or {}
        cand_info = {c: agents_in_comm.get(c, {}) for c in candidates}

        lines: list[str] = []
        if str(mode).lower() == "enhanced":
            last_msgs: dict[str, str] = {}
            try:
                for m in (obs.get("messages") or []):
                    sender = m.get("agent_name")
                    text = str(m.get("message") or "")
                    if sender and sender not in last_msgs:
                        last_msgs[sender] = text
            except Exception:
                last_msgs = {}

            try:
                k_tab = int(os.getenv("PROTO_V2_CAND_TABLES_K", "5"))
                k_res = int(os.getenv("PROTO_V2_CAND_RES_K", "5"))
            except Exception:
                k_tab, k_res = 5, 5
            try:
                carried = int((obs.get("inventory") or {}).get("quantity", 0) or 0)
            except Exception:
                carried = 0

            for c in candidates:
                info = cand_info.get(c, {}) or {}
                p = info.get("position", {}) or {}
                cx, cy = p.get("x"), p.get("y")
                in_view = bool(info.get("in_view_range", False))
                sent = c in last_msgs
                msg_text = last_msgs.get(c, "")
                lines.append(
                    f"- {c}: pos (x={cx}, y={cy}); in_view: {'yes' if in_view else 'no'}; "
                    f"last_round_sent_to_me: {'yes' if sent else 'no'}; last_round_msg: \"{msg_text}\""
                )

            cand_tables = list(self.owner._collect_candidate_tables(obs, k=k_tab, prefer_fit=True) or [])
            cand_res = list(self.owner._collect_candidate_resources(obs, k=k_res) or [])
            if cand_tables:
                lines.append("Candidate tables (x,y; rem; dist; src):")
                for e in cand_tables:
                    xy = e.get("pos")
                    rem = e.get("remaining")
                    dist = e.get("dist")
                    src = e.get("src")
                    fit = "fit" if (rem is not None and rem >= carried) else "partial"
                    lines.append(f"  - table ({xy[0]},{xy[1]}), rem={rem}, dist={dist}, {fit}, src={src}")
            if cand_res:
                lines.append("Candidate resources (type @ (x,y); qty; dist; src):")
                for e in cand_res:
                    xy = e.get("pos")
                    item_type = e.get("type")
                    qty = e.get("qty")
                    dist = e.get("dist")
                    src = e.get("src")
                    lines.append(f"  - {item_type} @ ({xy[0]},{xy[1]}), qty={qty}, dist={dist}, src={src}")
        else:
            for c in candidates:
                info = cand_info.get(c, {}) or {}
                p = info.get("position", {}) or {}
                cx, cy = p.get("x"), p.get("y")
                lines.append(f"- {c}: pos (x={cx}, y={cy})")

        ctx_lines = [
            f"You are {self.owner.name}. This is MESSAGE PHASE (single-message-per-round) for round {round_num}.",
            f"Your position: ({x},{y}).",
            "Candidates in communication range (choose at most ONE):",
            *lines,
            "Choose at most ONE recipient within the candidates and send ONE message, or send none.",
        ]

        if str(mode).lower() == "enhanced":
            ctx_lines.extend(
                [
                    "Guidance:",
                    "- If a candidate sent you a question last round about resources/tables, prioritize replying with the requested info.",
                    "- If a candidate is not in your view, consider proactively sharing known tables/resources (coordinates and quantities/capacities) from your memory.",
                ]
            )

        instr = [
            "Output EXACTLY three lines:",
            "SEND: yes|no",
            "TO: <one of the candidates>",
            "MSG: <your message text>",
        ]
        try:
            if str(os.getenv("DISALLOW_COMPACT_FORMAT", "0")).strip().lower() in ("1", "true", "yes", "on"):
                instr.extend(
                    [
                        "Do NOT use compact tags (e.g., 'R:', 'T:', '@(x,y)', 'qty=', 'Q=', 'C=').",
                        "Write plain sentences for MSG instead.",
                    ]
                )
        except Exception:
            pass
        return "\n".join(ctx_lines + ["", *instr])

    def _parse_route_message_decision(self, response: str) -> tuple[bool, str | None, str, str | None]:
        send = False
        saw_send = False
        recipient = None
        message = ""
        try:
            for line in str(response or "").splitlines():
                raw = line.strip()
                low = raw.lower()
                if low.startswith("send:"):
                    saw_send = True
                    send = low.split(":", 1)[1].strip() in ("yes", "y", "true", "1")
                elif low.startswith("to:"):
                    recipient = raw.split(":", 1)[1].strip()
                elif low.startswith("msg:"):
                    message = raw.split(":", 1)[1].strip()
        except Exception:
            return False, None, "", "parse_error"
        if not saw_send:
            return False, None, message, "missing_send"
        if not send:
            return False, None, message, "model_send_no"
        if not recipient or str(recipient).strip().lower() == "none":
            return True, None, message, "missing_recipient"
        if not str(message or "").strip():
            return True, recipient, "", "missing_message"
        return True, recipient, message, None

    def _normalize_candidate_name(self, recipient: str | None, candidate_names: list[str]) -> str | None:
        if not recipient:
            return None
        norm = self._normalize_candidate_token(recipient)
        if not norm:
            return None
        for candidate in candidate_names:
            if self._normalize_candidate_token(candidate) == norm:
                return candidate
        return None

    def _normalize_candidate_token(self, text: str) -> str:
        token = str(text or "").strip().lower()
        return re.sub(r"[^a-z0-9]+", "", token)

    def _select_random_target(self, ctx: RoundContext, names: list[str]) -> str | None:
        choices = sorted(str(name) for name in names if str(name))
        if not choices:
            return None
        try:
            round_num = int(getattr(ctx, "round_num", 0) or 0)
        except Exception:
            round_num = 0
        owner_name = str(getattr(self.owner, "name", "") or "")
        salt = (int(self.seed) * 1000003) + (round_num * 9176) + self._name_salt(owner_name)
        return random.Random(salt).choice(choices)

    def _name_salt(self, name: str) -> int:
        return sum(ord(ch) for ch in str(name))

    def _accumulate_message_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        try:
            self.owner.brain.message_prompt_tokens = int(getattr(self.owner.brain, "message_prompt_tokens", 0) or 0) + int(prompt_tokens)
            self.owner.brain.message_completion_tokens = int(getattr(self.owner.brain, "message_completion_tokens", 0) or 0) + int(completion_tokens)
        except Exception:
            pass

    def _snapshot_brain_usage(self) -> dict[str, Any]:
        usage = getattr(self.owner.brain, "usage", {})
        if isinstance(usage, dict):
            return dict(usage)
        if isinstance(usage, (int, float)):
            return {"total": int(usage)}
        return {}

    def _jump_back(self) -> None:
        try:
            if hasattr(self.owner.brain, "jump_back"):
                self.owner.brain.jump_back()
        except Exception:
            pass

    _reference_select_targets = select_targets
    _reference_select_llm_single_target = _select_llm_single_target
    _reference_build_original_single_message_prompt = _build_original_single_message_prompt

    _BASIC_GUIDANCE_LINES = [
        "Guidance:",
        "- A useful message should include one concrete coordination point: your position, your current objective, your immediate target, or one specific question.",
        "- Avoid generic messages with no actionable content.",
    ]

    def _build_original_single_message_prompt(self, obs: dict, candidates: list[str], mode: str = "basic") -> str:
        prompt = self._reference_build_original_single_message_prompt(obs, candidates, mode=mode)
        if str(mode).lower() != "basic":
            return prompt

        marker = "\n\nOutput EXACTLY three lines:\n"
        guidance_block = "\n".join(self._BASIC_GUIDANCE_LINES)
        if marker in prompt:
            return prompt.replace(marker, f"\n{guidance_block}{marker}", 1)
        return f"{prompt}\n{guidance_block}"

    async def _select_llm_single_target(self, ctx: RoundContext, candidate_names: list[str]) -> RoutingDecision:
        if str(self.mode or "basic").lower() != "basic":
            return await self._reference_select_llm_single_target(ctx, candidate_names)

        prompt = self._build_original_target_only_prompt(ctx.obs, list(candidate_names))
        response = ""
        api_call_time = 0.0
        try:
            p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            t0 = time.time()
            sampling_override = None
            try:
                sampler = getattr(self.owner.brain, "sampling_override_for", None)
                if callable(sampler):
                    sampling_override = sampler("basic_routing")
            except Exception:
                sampling_override = None
            if sampling_override:
                response = await self.owner.brain.generate(prompt, sampling_override=sampling_override)
            else:
                response = await self.owner.brain.generate(prompt)
            api_call_time = time.time() - t0
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            self._jump_back()
        except Exception as exc:
            response = f"SEND: no\nTO: none\nERROR: routing_error {exc}"

        send, recipient, error_reason = self._parse_route_target_only_decision(response)
        normalized_recipient = self._normalize_candidate_name(recipient, candidate_names)
        if send and normalized_recipient:
            recipient = normalized_recipient

        meta = {
            "mode": self.mode or "basic",
            "route_strategy": "stable_llm_target_only",
            "routing_prompt": prompt,
            "routing_response": response,
            "routing_api_call_time": api_call_time,
            "routing_tokens": self._snapshot_brain_usage(),
            "candidate_order": list(candidate_names),
        }

        legal_names = set(candidate_names)
        if not send:
            meta["error_reason"] = error_reason or "model_send_no"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if not recipient:
            meta["error_reason"] = error_reason or "missing_recipient"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if recipient not in legal_names:
            meta["error_reason"] = "invalid_recipient"
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)

        return RoutingDecision(send=True, recipients=[recipient], mode=self.mode or "basic", meta=meta)

    def _build_original_target_only_prompt(self, obs: dict, candidates: list[str]) -> str:
        prompt = self._build_original_single_message_prompt(obs, candidates, mode="basic")
        previous_block = "\n\nOutput EXACTLY three lines:\nSEND: yes|no\nTO: <one of the candidates>\nMSG: <your message text>"
        new_block = "\n\nOutput EXACTLY two lines:\nSEND: yes|no\nTO: <one of the candidates>"
        if previous_block in prompt:
            prompt = prompt.replace(previous_block, new_block, 1)
        action_block = _current_round_action_prompt_block(obs)
        marker = "\n\nOutput EXACTLY two lines:\n"
        if action_block and marker in prompt:
            prompt = prompt.replace(marker, f"{action_block}{marker}", 1)
        elif action_block:
            prompt = f"{prompt}{action_block}"
        return prompt

    def _parse_route_target_only_decision(self, response: str) -> tuple[bool, str | None, str | None]:
        send = False
        saw_send = False
        recipient = None
        try:
            for line in str(response or "").splitlines():
                raw = line.strip()
                low = raw.lower()
                if low.startswith("send:"):
                    saw_send = True
                    send = low.split(":", 1)[1].strip() in ("yes", "y", "true", "1")
                elif low.startswith("to:"):
                    recipient = raw.split(":", 1)[1].strip()
        except Exception:
            return False, None, "parse_error"

        if not saw_send:
            return False, None, "missing_send"
        if not send:
            return False, None, "model_send_no"
        if not recipient or str(recipient).strip().lower() == "none":
            return True, None, "missing_recipient"
        return True, recipient, None

    _reference_target_prompt = _build_original_target_only_prompt

    EARLY_COVERAGE_ROUNDS = 16
    ROLLING_CONTACT_WINDOW_ROUNDS = 8
    CONCENTRATION_WINDOW_ROUNDS = 6
    CONCENTRATION_MIN_COUNT = 2
    CONCENTRATION_SHARE_THRESHOLD = 0.4
    DUPLICATE_FACT_LOOKBACK_ROUNDS = 4
    AREA_COVERAGE_LOOKBACK_ROUNDS = 10
    SHORTLIST_DIVISOR = 2
    FACT_RELEVANCE_DISTANCE_SCALE = 0.18

    def _disabled_by_preset(self) -> bool:
        return bool(self.config.get("base_no_mechanisms") or self.config.get("disabled"))

    async def select_targets(
        self,
        ctx: RoundContext,
        candidates: list[MessageCandidate],
        memory_ctx: MemoryContext,
    ) -> RoutingDecision:
        if self._disabled_by_preset():
            del ctx, candidates, memory_ctx
            return RoutingDecision(
                send=False,
                recipients=[],
                mode="base",
                meta={"mode": "base", "disabled_by_preset": True, "error_reason": "base_no_mechanisms"},
            )

        mode = str(self.mode or "").lower()
        if mode != "gate_llm":
            return await self._reference_select_targets(ctx, candidates, memory_ctx)

        del memory_ctx
        if self.disable_send:
            return RoutingDecision(
                send=False,
                recipients=[],
                mode=self.mode,
                meta={"mode": self.mode, "disabled_by_preset": True},
            )
        if not candidates:
            return RoutingDecision(
                send=False,
                recipients=[],
                mode=self.mode,
                meta={"mode": self.mode, "error_reason": "no_candidates"},
            )

        names = [candidate.agent_name for candidate in candidates]
        if not self.single_message_per_round:
            return RoutingDecision(send=True, recipients=names, mode="per_target", meta={"mode": "per_target"})
        return await self._select_llm_single_target(ctx, names)

    def _recent_contact_records(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        records = list((obs or {}).get("recent_messages_sent") or [])
        if records:
            return [rec for rec in records if isinstance(rec, dict)]
        try:
            memory = getattr(getattr(self.owner, "mechanisms", None), "memory", None)
            return [rec for rec in list(getattr(memory, "_sent", []) or []) if isinstance(rec, dict)]
        except Exception:
            return []

    def _recent_received_records(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        records = list((obs or {}).get("recent_messages_received") or [])
        if records:
            return [rec for rec in records if isinstance(rec, dict)]
        try:
            memory = getattr(getattr(self.owner, "mechanisms", None), "memory", None)
            return [rec for rec in list(getattr(memory, "_recv", []) or []) if isinstance(rec, dict)]
        except Exception:
            return []

    def _contacted_candidate_names(self, obs: dict[str, Any] | None) -> set[str]:
        return {
            str(rec.get("to") or "").strip()
            for rec in self._recent_contact_records(obs)
            if str(rec.get("to") or "").strip()
        }

    def _recent_contact_count(self, obs: dict[str, Any] | None, candidate: str) -> int:
        return sum(
            1
            for rec in self._recent_contact_records(obs)
            if str(rec.get("to") or "").strip() == candidate
        )

    def _last_contact_gap(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> int | None:
        last_round = None
        for rec in self._recent_contact_records(obs):
            if str(rec.get("to") or "").strip() != candidate:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None or rec_round >= int(round_num):
                continue
            if last_round is None or int(rec_round) > int(last_round):
                last_round = int(rec_round)
        if last_round is None:
            return None
        return max(0, int(round_num) - int(last_round))

    def _last_incoming_gap(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> int | None:
        last_round = None
        for rec in self._recent_received_records(obs):
            if str(rec.get("from") or "").strip() != candidate:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None or rec_round >= int(round_num):
                continue
            if last_round is None or int(rec_round) > int(last_round):
                last_round = int(rec_round)
        if last_round is None:
            for rec in list((obs or {}).get("messages") or []):
                if not isinstance(rec, dict):
                    continue
                if str(rec.get("agent_name") or "").strip() != candidate:
                    continue
                rec_round = _coerce_int(rec.get("round"))
                if rec_round is None or rec_round >= int(round_num):
                    continue
                if last_round is None or int(rec_round) > int(last_round):
                    last_round = int(rec_round)
        if last_round is None:
            return None
        return max(0, int(round_num) - int(last_round))

    def _latest_incoming_text(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> str:
        latest_round = None
        latest_text = ""

        def consider(sender: str, text: str, rec_round: int | None) -> None:
            nonlocal latest_round, latest_text
            if str(sender or "").strip() != str(candidate or "").strip():
                return
            if rec_round is None or int(rec_round) >= int(round_num):
                return
            if latest_round is None or int(rec_round) > int(latest_round):
                latest_round = int(rec_round)
                latest_text = str(text or "")

        for rec in self._recent_received_records(obs):
            consider(
                str(rec.get("from") or rec.get("agent_name") or ""),
                str(rec.get("message") or rec.get("text") or ""),
                _coerce_int(rec.get("round")),
            )
        for rec in list((obs or {}).get("messages") or []):
            if not isinstance(rec, dict):
                continue
            consider(
                str(rec.get("agent_name") or rec.get("from") or ""),
                str(rec.get("message") or rec.get("text") or ""),
                _coerce_int(rec.get("round")),
            )
        return latest_text

    def _incoming_message_requests_info(self, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        if "?" in low:
            return True
        request_markers = (
            "|q|",
            "a=table_for=",
            "ask:table_for=",
            "ask=table_for=",
            "table_for=",
            "where are",
            "where is",
            "what do you see",
            "what can you see",
            "can you see",
            "do you see",
            "confirm",
            "check",
            "report",
            "share",
            "tell me",
            "status",
            "any table",
            "any resource",
        )
        return any(marker in low for marker in request_markers)

    def _incoming_message_table_request_type(self, text: str) -> str:
        low = str(text or "").strip().lower()
        if not low:
            return "none"
        proto = _parse_stable_fair_semantic_message(text) or _parse_stable_fair_compact_protocol(text)
        if isinstance(proto, dict):
            ask = str(proto.get("ask") or "none").strip().lower()
            match = re.search(r"table_for=(wood|sand|stone)", ask)
            if match:
                return str(match.group(1)).lower()
        match = re.search(r"\btable_for\s*=\s*(wood|sand|stone)\b", low)
        if match:
            return str(match.group(1)).lower()
        match = re.search(r"\btable\b.*\baccepts?\s+(wood|sand|stone)\b", low)
        if match:
            return str(match.group(1)).lower()
        if "table" in low and any(word in low for word in ("report", "find", "check", "tell", "where", "any")):
            return "table"
        if "resource" in low and any(word in low for word in ("report", "find", "check", "tell", "where", "any")):
            return "resource"
        return "info" if self._incoming_message_requests_info(low) else "none"

    def _incoming_message_has_table_fact_for_type(self, text: str, item_type: str) -> tuple[bool, bool]:
        low = str(text or "").strip().lower()
        if not low:
            return False, False
        proto = _parse_stable_fair_semantic_message(text) or _parse_stable_fair_compact_protocol(text)
        fact = proto.get("fact") if isinstance(proto, dict) else None
        has_table = isinstance(fact, dict) and str(fact.get("kind") or "") == "table"
        supports_type = False
        carry_type = str(item_type or "none").strip().lower()
        if has_table and carry_type in {"wood", "sand", "stone"}:
            supports_type = _stable_fair_table_supports_carry(fact, carry_type, None, require_full_qty=False)
        if not has_table:
            has_table = "table" in low and re.search(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", low) is not None
        if has_table and not supports_type and carry_type in {"wood", "sand", "stone"}:
            supports_type = (
                f"table_for={carry_type}" in low
                or re.search(rf"\b(?:accepts?|remaining|space for)\b[^.]*\b{carry_type}\b", low) is not None
            )
        return has_table, supports_type

    def _recent_incoming_has_table_fact_for_type(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
        item_type: str,
        *,
        window: int = 8,
    ) -> tuple[bool, bool]:
        has_any_table = False
        has_type_table = False
        seen: set[tuple[int | None, str]] = set()

        def consider(sender: str, text: str, rec_round: int | None) -> None:
            nonlocal has_any_table, has_type_table
            if str(sender or "").strip() != str(candidate or "").strip():
                return
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    return
                if int(round_num) - int(rec_round) > int(window):
                    return
            key = (rec_round, str(text or ""))
            if key in seen:
                return
            seen.add(key)
            has_table, supports_type = self._incoming_message_has_table_fact_for_type(text, item_type)
            has_any_table = has_any_table or has_table
            has_type_table = has_type_table or supports_type

        for rec in self._recent_received_records(obs):
            consider(
                str(rec.get("from") or rec.get("agent_name") or ""),
                str(rec.get("message") or rec.get("text") or ""),
                _coerce_int(rec.get("round")),
            )
        for rec in list((obs or {}).get("messages") or []):
            if not isinstance(rec, dict):
                continue
            consider(
                str(rec.get("agent_name") or rec.get("from") or ""),
                str(rec.get("message") or rec.get("text") or ""),
                _coerce_int(rec.get("round")),
            )
        return has_any_table, has_type_table

    def _sender_rotation_index(self) -> int:
        raw = str(getattr(self.owner, "name", "") or "")
        digits: list[str] = []
        for ch in reversed(raw):
            if ch.isdigit():
                digits.append(ch)
            elif digits:
                break
        if not digits:
            return 0
        try:
            return int("".join(reversed(digits)))
        except Exception:
            return 0

    def _candidate_display_order(
        self,
        candidate_names: list[str],
        round_num: int,
    ) -> tuple[list[str], dict[str, Any]]:
        names = [str(name) for name in candidate_names if str(name)]
        if len(names) <= 1:
            return names, {"rotation_offset": 0, "base_order": list(names)}
        offset = (int(round_num) + int(self._sender_rotation_index())) % len(names)
        display_order = names[offset:] + names[:offset]
        return display_order, {
            "rotation_offset": offset,
            "base_order": list(names),
        }

    def _rolling_recent_contact_count(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> int:
        window = int(self.ROLLING_CONTACT_WINDOW_ROUNDS)
        count = 0
        for rec in self._recent_contact_records(obs):
            if str(rec.get("to") or "").strip() != candidate:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None:
                continue
            if rec_round >= int(round_num):
                continue
            if int(round_num) - rec_round > window:
                continue
            count += 1
        return count

    def _recent_concentration_stats(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> tuple[int, int, float]:
        sent_records = self._recent_contact_records(obs)
        window = int(self.CONCENTRATION_WINDOW_ROUNDS)
        recent_window_records: list[dict[str, Any]] = []
        for rec in sent_records:
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None:
                continue
            if int(round_num) - rec_round > window:
                continue
            if rec_round >= int(round_num):
                continue
            recent_window_records.append(rec)
        recent_total_sent = len(recent_window_records)
        recent_pair_count = sum(
            1
            for rec in recent_window_records
            if str(rec.get("to") or "").strip() == candidate
        )
        recent_pair_share = (
            float(recent_pair_count) / float(recent_total_sent)
            if recent_total_sent > 0
            else 0.0
        )
        return recent_pair_count, recent_total_sent, recent_pair_share

    def _same_recipient_streak(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        round_num: int,
    ) -> int:
        sent_rounds = [
            rec_round
            for rec_round in (
                _coerce_int(rec.get("round"))
                for rec in self._recent_contact_records(obs)
                if str(rec.get("to") or "").strip() == candidate
            )
            if rec_round is not None and int(rec_round) < int(round_num)
        ]
        return _consecutive_round_streak(sent_rounds, int(round_num))

    def _task_target_items(self, obs: dict[str, Any] | None) -> dict[str, int]:
        data = dict(obs or {})
        raw_targets = data.get("target_items")
        if not isinstance(raw_targets, dict):
            task_obj = data.get("task")
            raw_targets = getattr(task_obj, "target_items", None)
        if not isinstance(raw_targets, dict):
            return {}
        targets: dict[str, int] = {}
        for item_type in ("wood", "sand", "stone"):
            value = _coerce_int(raw_targets.get(item_type))
            if value is not None and value > 0:
                targets[item_type] = int(value)
        return targets

    def _task_target_weights(self, obs: dict[str, Any] | None) -> dict[str, float]:
        targets = self._task_target_items(obs)
        if targets:
            values = [int(v) for v in targets.values() if int(v) > 0]
            average = float(sum(values)) / float(len(values)) if values else 1.0
            static_weights: dict[str, float] = {}
            for item_type in ("wood", "sand", "stone"):
                raw_value = int(targets.get(item_type, int(average)) or int(average))
                ratio = float(raw_value) / average if average > 0 else 1.0
                static_weights[item_type] = max(0.75, min(1.5, ratio))
        else:
            static_weights = {"wood": 1.0, "sand": 1.0, "stone": 1.0}

        observed_remaining = self._observed_remaining_by_type(obs)
        if not observed_remaining:
            return static_weights
        observed_values = [int(v) for v in observed_remaining.values() if int(v) > 0]
        if not observed_values:
            return static_weights
        observed_average = float(sum(observed_values)) / float(len(observed_values))
        dynamic_weights: dict[str, float] = {}
        for item_type in ("wood", "sand", "stone"):
            observed_value = observed_remaining.get(item_type)
            if observed_value is None:
                dynamic_weights[item_type] = static_weights[item_type]
                continue
            ratio = float(observed_value) / observed_average if observed_average > 0 else 1.0
            dynamic_weights[item_type] = max(0.6, min(1.8, ratio))

        blended: dict[str, float] = {}
        for item_type in ("wood", "sand", "stone"):
            blended[item_type] = max(
                0.65,
                min(1.8, 0.35 * float(static_weights[item_type]) + 0.65 * float(dynamic_weights[item_type])),
            )
        return blended

    def _observed_remaining_by_type(self, obs: dict[str, Any] | None) -> dict[str, int]:
        observed: dict[tuple[int, int], dict[str, int]] = {}
        for source_name in ("tables_info", "kb_nearest_tables"):
            for rec in list((obs or {}).get(source_name) or []):
                if not isinstance(rec, dict):
                    continue
                pos = _extract_xy(rec.get("position"))
                if pos is None:
                    continue
                rb = rec.get("remaining_by_type") or {}
                if not isinstance(rb, dict) or not rb:
                    continue
                cur = observed.setdefault(pos, {})
                for item_type in ("wood", "sand", "stone"):
                    value = _coerce_int(rb.get(item_type))
                    if value is None:
                        continue
                    if source_name == "tables_info" or item_type not in cur:
                        cur[item_type] = max(0, int(value))

        totals = {"wood": 0, "sand": 0, "stone": 0}
        for rb in observed.values():
            for item_type in ("wood", "sand", "stone"):
                totals[item_type] += int(rb.get(item_type, 0) or 0)
        return totals

    def _iter_resource_fact_entries(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        data = dict(obs or {})
        current_fact = _stable_fair_current_tile_resource_fact(data)
        if current_fact:
            facts.append(current_fact)
        cap = data.get("capacity_info") or {}
        details = (cap.get("_resource_details") if isinstance(cap, dict) else {}) or {}
        for rec in details.values():
            pos = _extract_xy(rec.get("position"))
            if pos is None:
                continue
            terrain = str(rec.get("terrain_type") or "").strip().lower()
            if terrain == "table":
                continue
            for item in list(rec.get("items") or []):
                if not isinstance(item, dict):
                    continue
                qty = _coerce_int(item.get("quantity"))
                if qty is None or qty <= 0:
                    continue
                resource_type = str(item.get("type") or "").strip().lower()
                if resource_type not in {"wood", "sand", "stone"}:
                    continue
                facts.append(
                    {
                        "kind": "resource",
                        "resource_type": resource_type,
                        "position": pos,
                        "quantity": int(qty),
                        "source": "visible",
                    }
                )
        for rec in list(data.get("kb_nearest_items") or []):
            if not isinstance(rec, dict):
                continue
            pos = _extract_xy(rec.get("position"))
            if pos is None:
                continue
            raw_items = rec.get("items") or {}
            if isinstance(raw_items, dict):
                iterator = raw_items.items()
            elif isinstance(raw_items, list):
                iterator = [
                    (str(item.get("type") or ""), item.get("quantity"))
                    for item in raw_items
                    if isinstance(item, dict)
                ]
            else:
                iterator = []
            for item_type, qty_raw in iterator:
                qty = _coerce_int(qty_raw)
                resource_type = str(item_type or "").strip().lower()
                if qty is None or qty <= 0 or resource_type not in {"wood", "sand", "stone"}:
                    continue
                facts.append(
                    {
                        "kind": "resource",
                        "resource_type": resource_type,
                        "position": pos,
                        "quantity": int(qty),
                        "source": "memory",
                    }
                )
        def source_rank(fact: dict[str, Any]) -> int:
            source = str(fact.get("source") or "").strip().lower()
            if source == "current_tile":
                return 3
            if source == "visible":
                return 2
            return 1

        dedup: dict[tuple[str, str, tuple[int, int]], dict[str, Any]] = {}
        for fact in facts:
            key = (str(fact.get("kind") or ""), str(fact.get("resource_type") or ""), tuple(fact.get("position") or ()))
            existing = dedup.get(key)
            if existing is None or (
                source_rank(fact),
                int(_coerce_int(fact.get("quantity")) or 0),
            ) > (
                source_rank(existing),
                int(_coerce_int(existing.get("quantity")) or 0),
            ):
                dedup[key] = fact
        return list(dedup.values())

    def _iter_table_fact_entries(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        data = dict(obs or {})
        for rec in list(data.get("tables_info") or []):
            pos = _extract_xy(rec.get("position"))
            if pos is None:
                continue
            facts.append(
                {
                    "kind": "table",
                    "position": pos,
                    "remaining_capacity": _coerce_int(rec.get("remaining_capacity")),
                    "remaining_by_type": dict(rec.get("remaining_by_type") or {}),
                    "source": "visible",
                }
            )
        for rec in list(data.get("kb_nearest_tables") or []):
            if not isinstance(rec, dict):
                continue
            pos = _extract_xy(rec.get("position"))
            if pos is None:
                continue
            facts.append(
                {
                    "kind": "table",
                    "position": pos,
                    "remaining_capacity": _coerce_int(rec.get("remaining_capacity")),
                    "remaining_by_type": dict(rec.get("remaining_by_type") or {}),
                    "source": "memory",
                }
            )
        dedup: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}
        for fact in facts:
            key = (str(fact.get("kind") or ""), tuple(fact.get("position") or ()))
            if key not in dedup or str(fact.get("source") or "") == "visible":
                dedup[key] = fact
        return list(dedup.values())

    def _format_fact_label(self, fact: dict[str, Any] | None) -> str:
        if not isinstance(fact, dict):
            return "none"
        pos = tuple(fact.get("position") or ())
        if len(pos) != 2:
            return "none"
        if str(fact.get("kind") or "") == "table":
            return f"table@({pos[0]},{pos[1]})"
        resource_type = str(fact.get("resource_type") or "unknown")
        return f"{resource_type}@({pos[0]},{pos[1]})"

    def _message_mentions_fact(self, text: str, fact: dict[str, Any] | None) -> bool:
        if not isinstance(fact, dict):
            return False
        raw_text = str(text or "")
        if not raw_text:
            return False
        pos = tuple(fact.get("position") or ())
        if len(pos) != 2:
            return False
        x, y = int(pos[0]), int(pos[1])
        coord_pattern = re.compile(rf"(?:\(\s*{x}\s*,\s*{y}\s*\)|@\s*{x}\s*,\s*{y}\b)")
        if coord_pattern.search(raw_text) is None:
            return False
        lower = raw_text.lower()
        if str(fact.get("kind") or "") == "table":
            return "table" in lower
        resource_type = str(fact.get("resource_type") or "").strip().lower()
        return bool(resource_type) and resource_type in lower

    def _duplicate_fact_recent(
        self,
        obs: dict[str, Any] | None,
        candidate: str,
        fact: dict[str, Any] | None,
        round_num: int,
    ) -> bool:
        if not isinstance(fact, dict):
            return False
        lookback = int(self.DUPLICATE_FACT_LOOKBACK_ROUNDS)
        for rec in self._recent_contact_records(obs):
            if str(rec.get("to") or "").strip() != candidate:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None or rec_round >= int(round_num):
                continue
            if int(round_num) - int(rec_round) > lookback:
                continue
            if self._message_mentions_fact(str(rec.get("message") or ""), fact):
                return True
        return False

    def _fact_recent_message_count(
        self,
        obs: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        round_num: int,
        *,
        window: int | None = None,
        candidate: str | None = None,
        include_received: bool = True,
    ) -> int:
        if not isinstance(fact, dict):
            return 0
        lookback = int(window if window is not None else self.DUPLICATE_FACT_LOOKBACK_ROUNDS)
        target = str(candidate or "").strip()
        records: list[dict[str, Any]] = list(self._recent_contact_records(obs))
        if include_received:
            records.extend(self._recent_received_records(obs))
            for rec in list((obs or {}).get("messages") or []):
                if isinstance(rec, dict):
                    records.append(rec)
        count = 0
        seen: set[tuple[int | None, str, str, str]] = set()
        for rec in records:
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    continue
                if int(round_num) - int(rec_round) > lookback:
                    continue
            if target and str(rec.get("to") or rec.get("from") or rec.get("agent_name") or "").strip() != target:
                continue
            text = str(rec.get("message") or rec.get("text") or "")
            key = (
                rec_round,
                str(rec.get("to") or ""),
                str(rec.get("from") or rec.get("agent_name") or ""),
                text,
            )
            if key in seen:
                continue
            seen.add(key)
            if self._message_mentions_fact(text, fact):
                count += 1
        return count

    def _coord_area_key(self, obs: dict[str, Any] | None, pos: tuple[int, int] | None) -> str:
        if pos is None or len(pos) != 2:
            return "unknown"
        width = _coerce_int((obs or {}).get("map_width") or (obs or {}).get("width"))
        height = _coerce_int((obs or {}).get("map_height") or (obs or {}).get("height"))
        x = int(pos[0])
        y = int(pos[1])
        if width is None or width <= 0:
            width = max(1, x + 1)
        if height is None or height <= 0:
            height = max(1, y + 1)
        x_bin = min(2, max(0, int(x * 3 / max(1, int(width)))))
        y_bin = min(2, max(0, int(y * 3 / max(1, int(height)))))
        return f"{x_bin},{y_bin}"

    def _fact_area_key(self, obs: dict[str, Any] | None, fact: dict[str, Any] | None) -> str:
        return self._coord_area_key(obs, _extract_xy((fact or {}).get("position")) if isinstance(fact, dict) else None)

    def _message_mentions_area_kind(
        self,
        obs: dict[str, Any] | None,
        text: str,
        area_key: str,
        kind: str,
        resource_type: str = "",
    ) -> bool:
        if not area_key or area_key == "unknown":
            return False
        raw_text = str(text or "")
        if not raw_text:
            return False
        kind_norm = str(kind or "").strip().lower()
        resource_norm = str(resource_type or "").strip().lower()
        lower = raw_text.lower()
        if kind_norm == "table" and "table" not in lower:
            return False
        if kind_norm == "resource" and not any(token in lower for token in ("resource", "wood", "sand", "stone")):
            return False
        for match in re.finditer(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", raw_text):
            pos = (int(match.group(1)), int(match.group(2)))
            if self._coord_area_key(obs, pos) != area_key:
                continue
            window = raw_text[max(0, match.start() - 36) : min(len(raw_text), match.end() + 36)].lower()
            if kind_norm == "table":
                if "table" in window:
                    return True
                continue
            if kind_norm == "resource":
                resource_context = any(
                    marker in window
                    for marker in ("resource", "picking up", "pickup", " at ", " has ", "@")
                )
                if "table" in window and "resource" not in window and "picking" not in window and "pickup" not in window:
                    continue
                if resource_context and (
                    "resource" in window
                    or (resource_norm and resource_norm in window)
                    or any(token in window for token in ("wood", "sand", "stone"))
                ):
                    return True
        return False

    def _area_recent_message_count(
        self,
        obs: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        round_num: int,
        *,
        window: int | None = None,
        candidate: str | None = None,
        include_received: bool = True,
    ) -> int:
        if not isinstance(fact, dict):
            return 0
        area_key = self._fact_area_key(obs, fact)
        if area_key == "unknown":
            return 0
        lookback = int(window if window is not None else self.AREA_COVERAGE_LOOKBACK_ROUNDS)
        target = str(candidate or "").strip()
        kind = str(fact.get("kind") or "").strip().lower()
        resource_type = str(fact.get("resource_type") or "").strip().lower()
        records: list[dict[str, Any]] = list(self._recent_contact_records(obs))
        if include_received:
            records.extend(self._recent_received_records(obs))
            for rec in list((obs or {}).get("messages") or []):
                if isinstance(rec, dict):
                    records.append(rec)
        count = 0
        seen: set[tuple[int | None, str, str, str]] = set()
        for rec in records:
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    continue
                if int(round_num) - rec_round > lookback:
                    continue
            if target and str(rec.get("to") or rec.get("from") or rec.get("agent_name") or "").strip() != target:
                continue
            text = str(rec.get("message") or rec.get("text") or "")
            key = (
                rec_round,
                str(rec.get("to") or ""),
                str(rec.get("from") or rec.get("agent_name") or ""),
                text,
            )
            if key in seen:
                continue
            seen.add(key)
            if self._message_mentions_area_kind(obs, text, area_key, kind, resource_type):
                count += 1
        return count

    def _coverage_novelty_for_fact(
        self,
        obs: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        round_num: int,
        fact_recent_message_count: int,
    ) -> tuple[str, str, int]:
        if not isinstance(fact, dict):
            return "none", "unknown", 0
        kind = str(fact.get("kind") or "").strip().lower()
        source = str(fact.get("source") or "").strip().lower()
        if kind not in {"resource", "table"} or source not in {"visible", "current_tile"}:
            return "repeat", self._fact_area_key(obs, fact), 0
        area_key = self._fact_area_key(obs, fact)
        area_recent = self._area_recent_message_count(obs, fact, int(round_num), include_received=True)
        if area_recent <= 0:
            return "new_area", area_key, int(area_recent)
        if int(fact_recent_message_count or 0) <= 0:
            return "new_fact", area_key, int(area_recent)
        return "repeat", area_key, int(area_recent)

    def _recent_table_request_count(
        self,
        obs: dict[str, Any] | None,
        item_type: str,
        round_num: int,
        *,
        candidate: str | None = None,
        window: int | None = None,
    ) -> int:
        resource_type = str(item_type or "").strip().lower()
        if resource_type not in {"wood", "sand", "stone"}:
            return 0
        lookback = int(window if window is not None else max(4, self.DUPLICATE_FACT_LOOKBACK_ROUNDS + 2))
        target = str(candidate or "").strip()
        count = 0
        for rec in self._recent_contact_records(obs):
            if target and str(rec.get("to") or "").strip() != target:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is None or int(rec_round) >= int(round_num):
                continue
            if int(round_num) - int(rec_round) > lookback:
                continue
            requested_type = self._incoming_message_table_request_type(str(rec.get("message") or rec.get("text") or ""))
            if requested_type in {resource_type, "table"}:
                count += 1
        return count

    def _payload_lifecycle_for_feature(self, feature: dict[str, Any] | None) -> str:
        if not isinstance(feature, dict):
            return "none"
        if bool(feature.get("reply_recent")) and bool(feature.get("reply_info_request")):
            return "active_reply"
        if bool(feature.get("need_table_for_type")):
            if bool(feature.get("request_cooldown")):
                return "request_cooldown"
            return "active_request"
        if bool(feature.get("safe_table_delivery")) and str(feature.get("table_for_type") or "none") in {"wood", "sand", "stone"}:
            return "active_table_for_carry"
        if bool(feature.get("own_table_relevant_to_candidate")):
            return "active_shared_table"
        kind = str(feature.get("best_fact_kind") or "none").strip().lower()
        duplicate = bool(feature.get("duplicate_fact_recent"))
        recent_count = int(feature.get("fact_recent_message_count", 0) or 0)
        recipient_carrying = str(feature.get("carrying") or "none").strip().lower()
        sender_carrying = str(feature.get("own_carrying") or "none").strip().lower()
        if kind == "table":
            fit = str(feature.get("table_delivery_fit") or "none").strip().lower()
            if fit == "saturated":
                return "exhausted_table"
            if fit == "partial":
                return "partial_table"
            if duplicate or recent_count >= 2:
                if recipient_carrying == "none" and sender_carrying == "none":
                    return "stale_static_table"
                return "recent_table_refresh"
            if recipient_carrying == "none" and sender_carrying == "none":
                return "fresh_static_table" if str(feature.get("best_fact_source") or "") == "visible" else "generic_table"
            return "usable_table"
        if kind == "resource":
            if duplicate or recent_count >= 2:
                return "stale_static_resource"
            return "fresh_resource"
        if bool(feature.get("carrying_reason")):
            return "carry_status"
        return "low_value"

    def _fact_target_need(
        self,
        fact: dict[str, Any] | None,
        target_weights: dict[str, float],
        candidate_carrying: str,
        candidate_load_qty: int | None = None,
    ) -> float:
        if not isinstance(fact, dict):
            return 1.0
        if str(fact.get("kind") or "") == "table":
            fit = self._table_delivery_fit(fact, candidate_carrying, candidate_load_qty)
            if fit["status"] == "not_carrying":
                return max(0.65, self._table_broadcast_value(fact))
            if fit["status"] == "saturated":
                return 0.0
            if fit["status"] == "partial":
                return 0.45
            rb = fact.get("remaining_by_type") or {}
            if isinstance(rb, dict) and rb and candidate_carrying in rb:
                values = [int(v) for v in rb.values() if _coerce_int(v) is not None]
                local_average = float(sum(values)) / float(len(values)) if values else 0.0
                local_value = float(_coerce_int(rb.get(candidate_carrying)) or 0)
                if local_average > 0:
                    local_ratio = max(0.6, min(1.8, local_value / local_average))
                    return max(
                        0.65,
                        min(
                            1.9,
                            0.45 * float(target_weights.get(candidate_carrying, 1.0)) + 0.55 * float(local_ratio),
                        ),
                    )
            if candidate_carrying in target_weights:
                return float(target_weights.get(candidate_carrying, 1.0))
            return float(sum(target_weights.values())) / float(len(target_weights) or 1)
        resource_type = str(fact.get("resource_type") or "")
        return float(target_weights.get(resource_type, 1.0))

    def _table_broadcast_value(self, fact: dict[str, Any] | None) -> float:
        if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
            return 0.0
        rb = fact.get("remaining_by_type") or {}
        positive_values: list[int] = []
        if isinstance(rb, dict):
            for item_type in ("wood", "sand", "stone"):
                value = _coerce_int(rb.get(item_type))
                if value is not None and int(value) > 0:
                    positive_values.append(int(value))
        remaining = _coerce_int(fact.get("remaining_capacity"))
        total_remaining = int(remaining) if remaining is not None else sum(positive_values)
        if total_remaining <= 0 and not positive_values:
            return 0.0
        carrying_capacity = _coerce_int(getattr(self.owner, "carrying_capacity", None))
        unit = int(carrying_capacity) if carrying_capacity is not None and carrying_capacity > 0 else 6
        value = 0.0
        if len(positive_values) >= 2:
            value += 0.45
        if len(positive_values) >= 3:
            value += 0.25
        if total_remaining >= unit * 4:
            value += 0.45
        elif total_remaining >= unit * 2:
            value += 0.25
        if str(fact.get("source") or "") == "visible":
            value += 0.20
        if positive_values and max(positive_values) >= unit * 2:
            value += 0.15
        return min(1.35, max(0.0, value))

    def _is_high_value_table_fact(self, fact: dict[str, Any] | None) -> bool:
        return self._table_broadcast_value(fact) >= 0.85

    def _fact_actionability(
        self,
        fact: dict[str, Any] | None,
        candidate_carrying: str,
        candidate_load_qty: int | None = None,
    ) -> float:
        if not isinstance(fact, dict):
            return 0.0
        if str(fact.get("kind") or "") == "table":
            fit = self._table_delivery_fit(fact, candidate_carrying, candidate_load_qty)
            status = str(fit.get("status") or "")
            if status == "safe":
                return 1.25
            if status == "partial":
                return 0.25
            if status == "unknown":
                return 0.35
            if status == "saturated":
                return 0.0
            return max(0.15, self._table_broadcast_value(fact))
        if candidate_carrying == "none":
            return 1.0
        return 0.35 if candidate_carrying == str(fact.get("resource_type") or "") else 0.4

    def _candidate_load_qty(
        self,
        candidate_info: dict[str, Any] | None,
        candidate_carrying: str,
    ) -> int | None:
        carrying = str(candidate_carrying or "").strip().lower()
        if carrying not in {"wood", "sand", "stone"}:
            return None
        parsed_carrying, parsed_qty = _candidate_resource_quantity(candidate_info)
        if parsed_carrying == carrying and parsed_qty is not None and parsed_qty > 0:
            return int(parsed_qty)
        for key in ("carrying_qty", "carrying_quantity", "item_quantity", "quantity"):
            value = _coerce_int((candidate_info or {}).get(key))
            if value is not None and value > 0:
                return int(value)
        for key in ("carrying_amount", "item_qty"):
            value = _coerce_int((candidate_info or {}).get(key))
            if value is not None and value > 0:
                return int(value)
        owner_capacity = _coerce_int(getattr(self.owner, "carrying_capacity", None))
        if owner_capacity is not None and owner_capacity > 0:
            return int(owner_capacity)
        return None

    def _table_delivery_fit(
        self,
        fact: dict[str, Any] | None,
        candidate_carrying: str,
        candidate_load_qty: int | None = None,
    ) -> dict[str, Any]:
        carrying = str(candidate_carrying or "").strip().lower()
        if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
            return {"status": "not_table", "safe": False}
        if carrying not in {"wood", "sand", "stone"}:
            return {"status": "not_carrying", "safe": False}

        required_qty = _coerce_int(candidate_load_qty)
        if required_qty is None or required_qty <= 0:
            owner_capacity = _coerce_int(getattr(self.owner, "carrying_capacity", None))
            required_qty = int(owner_capacity) if owner_capacity is not None and owner_capacity > 0 else None

        rb = fact.get("remaining_by_type") or {}
        if isinstance(rb, dict) and carrying in rb:
            remaining_for_type = _coerce_int(rb.get(carrying))
            if remaining_for_type is None:
                return {
                    "status": "unknown",
                    "safe": False,
                    "remaining_for_type": None,
                    "required_qty": required_qty,
                }
            remaining_for_type = max(0, int(remaining_for_type))
            if remaining_for_type <= 0:
                status = "saturated"
                safe = False
            elif required_qty is not None and remaining_for_type < int(required_qty):
                status = "partial"
                safe = False
            else:
                status = "safe"
                safe = True
            return {
                "status": status,
                "safe": safe,
                "remaining_for_type": remaining_for_type,
                "required_qty": required_qty,
            }

        remaining_capacity = _coerce_int(fact.get("remaining_capacity"))
        if remaining_capacity is None:
            return {
                "status": "unknown",
                "safe": False,
                "remaining_for_type": None,
                "required_qty": required_qty,
            }
        remaining_capacity = max(0, int(remaining_capacity))
        if remaining_capacity <= 0:
            status = "saturated"
            safe = False
        elif required_qty is not None and remaining_capacity < int(required_qty):
            status = "partial"
            safe = False
        else:
            status = "safe"
            safe = True
        return {
            "status": status,
            "safe": safe,
            "remaining_for_type": None,
            "required_qty": required_qty,
            "remaining_capacity": remaining_capacity,
        }

    def _fact_key_for_routing(self, fact: dict[str, Any] | None) -> tuple[Any, ...]:
        if not isinstance(fact, dict):
            return ("none",)
        kind = str(fact.get("kind") or "").strip().lower()
        pos = _extract_xy(fact.get("position"))
        if kind == "resource":
            return (kind, str(fact.get("resource_type") or "none"), pos)
        if kind == "table":
            return (kind, pos)
        return (kind or "none", pos)

    def _fact_relevance_for_candidate(
        self,
        fact: dict[str, Any],
        candidate_pos: tuple[int, int],
        candidate_carrying: str,
        candidate_load_qty: int | None,
        target_weights: dict[str, float],
    ) -> tuple[float, int | None, float]:
        distance = _manhattan_distance(candidate_pos, tuple(fact.get("position") or ()))
        if distance is None:
            return 0.0, None, 1.0
        target_need = self._fact_target_need(fact, target_weights, candidate_carrying, candidate_load_qty)
        actionability = self._fact_actionability(fact, candidate_carrying, candidate_load_qty)
        source_bonus = 0.2 if str(fact.get("source") or "") == "visible" else 0.0
        if str(fact.get("kind") or "") == "table":
            base = 1.1 * target_need * actionability + source_bonus
        else:
            base = 1.05 * target_need * actionability + source_bonus
        relevance = float(base) / (1.0 + float(self.FACT_RELEVANCE_DISTANCE_SCALE) * float(distance))
        return relevance, distance, target_need

    def _strategic_table_fact_for_candidate(
        self,
        obs: dict[str, Any] | None,
        candidate_pos: tuple[int, int],
        candidate_carrying: str,
        candidate_load_qty: int | None,
        target_weights: dict[str, float],
    ) -> tuple[dict[str, Any] | None, int | None, float, float]:
        best_fact: dict[str, Any] | None = None
        best_distance: int | None = None
        best_relevance = 0.0
        best_target_need = 1.0
        best_key: tuple[int, int, int] | None = None
        for fact in self._iter_table_fact_entries(obs):
            table_value = int(round(100 * self._table_broadcast_value(fact)))
            if table_value < 85:
                continue
            relevance, distance, target_need = self._fact_relevance_for_candidate(
                fact,
                candidate_pos,
                candidate_carrying,
                candidate_load_qty,
                target_weights,
            )
            if distance is None:
                continue
            source_bonus = 1 if str(fact.get("source") or "") in {"visible", "current_tile"} else 0
            key = (source_bonus, table_value, -int(distance))
            if best_key is None or key > best_key:
                best_key = key
                best_fact = dict(fact)
                best_distance = distance
                best_relevance = relevance
                best_target_need = target_need
        return best_fact, best_distance, best_relevance, best_target_need

    def _should_promote_strategic_table_over_resource(
        self,
        resource_fact: dict[str, Any] | None,
        resource_relevance: float,
        resource_target_need: float,
        table_fact: dict[str, Any] | None,
        table_relevance: float,
    ) -> bool:
        if not isinstance(resource_fact, dict) or str(resource_fact.get("kind") or "") != "resource":
            return False
        if not isinstance(table_fact, dict) or str(table_fact.get("kind") or "") != "table":
            return False
        if str(table_fact.get("source") or "") not in {"visible", "current_tile"}:
            return False
        if not self._is_high_value_table_fact(table_fact):
            return False
        if float(resource_target_need or 0.0) >= 1.35:
            return False
        return float(table_relevance or 0.0) >= 0.9 * max(0.001, float(resource_relevance or 0.0))

    def _backup_fact_for_candidate(
        self,
        obs: dict[str, Any] | None,
        candidate_info: dict[str, Any] | None,
        target_weights: dict[str, float],
        primary_fact: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, int | None, float]:
        candidate_pos = _extract_xy((candidate_info or {}).get("position"))
        candidate_carrying, _candidate_qty = _candidate_resource_quantity(candidate_info)
        candidate_load_qty = self._candidate_load_qty(candidate_info, candidate_carrying)
        if candidate_pos is None:
            return None, None, 0.0
        primary_key = self._fact_key_for_routing(primary_fact)
        primary_kind = str((primary_fact or {}).get("kind") or "").strip().lower() if isinstance(primary_fact, dict) else ""
        best_fact: dict[str, Any] | None = None
        best_distance: int | None = None
        best_relevance = 0.0
        for fact in self._iter_resource_fact_entries(obs) + self._iter_table_fact_entries(obs):
            if self._fact_key_for_routing(fact) == primary_key:
                continue
            kind = str(fact.get("kind") or "").strip().lower()
            if primary_kind == "resource" and kind != "table":
                continue
            if primary_kind == "table" and kind != "resource":
                continue
            if kind == "table" and not self._is_high_value_table_fact(fact):
                continue
            relevance, distance, _target_need = self._fact_relevance_for_candidate(
                fact,
                candidate_pos,
                candidate_carrying,
                candidate_load_qty,
                target_weights,
            )
            if relevance > best_relevance:
                best_fact = dict(fact)
                best_distance = distance
                best_relevance = relevance
        return best_fact, best_distance, best_relevance

    def _best_table_fact_for_carry(
        self,
        obs: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> tuple[dict[str, Any] | None, int | None, dict[str, Any]]:
        resource_type = str(carry_type or "").strip().lower()
        if resource_type not in {"wood", "sand", "stone"}:
            return None, None, {"status": "not_carrying", "safe": False}
        pos = _extract_xy((obs or {}).get("position")) or (0, 0)
        best_fact: dict[str, Any] | None = None
        best_distance: int | None = None
        best_fit: dict[str, Any] = {"status": "not_table", "safe": False}
        best_key: tuple[int, int, int] | None = None
        for fact in self._iter_table_fact_entries(obs):
            fit = self._table_delivery_fit(fact, resource_type, carry_qty)
            if not bool(fit.get("safe")):
                continue
            distance = _manhattan_distance(pos, _extract_xy(fact.get("position")))
            remaining = _coerce_int(fit.get("remaining_for_type") or fit.get("remaining_capacity"))
            key = (
                1 if str(fact.get("source") or "") == "visible" else 0,
                int(remaining) if remaining is not None else 0,
                -int(distance or 0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_fact = dict(fact)
                best_distance = distance
                best_fit = dict(fit)
        return best_fact, best_distance, best_fit

    def _shortlist_size_for_pool(self, pool_size: int) -> int:
        total = max(0, int(pool_size))
        if total <= 2:
            return total
        divisor = max(1, int(self.SHORTLIST_DIVISOR))
        return max(2, (total + divisor - 1) // divisor)

    def _best_fact_for_candidate(
        self,
        obs: dict[str, Any] | None,
        candidate_info: dict[str, Any] | None,
        target_weights: dict[str, float],
    ) -> tuple[dict[str, Any] | None, int | None, float, float]:
        candidate_pos = _extract_xy((candidate_info or {}).get("position"))
        candidate_carrying, _candidate_qty = _candidate_resource_quantity(candidate_info)
        candidate_load_qty = self._candidate_load_qty(candidate_info, candidate_carrying)
        if candidate_pos is None:
            return None, None, 0.0, 1.0
        best_fact = None
        best_distance = None
        best_relevance = 0.0
        best_target_need = 1.0
        for fact in self._iter_resource_fact_entries(obs) + self._iter_table_fact_entries(obs):
            relevance, distance, target_need = self._fact_relevance_for_candidate(
                fact,
                candidate_pos,
                candidate_carrying,
                candidate_load_qty,
                target_weights,
            )
            if distance is None:
                continue
            if best_fact is None or relevance > best_relevance:
                best_fact = fact
                best_distance = distance
                best_relevance = relevance
                best_target_need = target_need
        table_fact, table_distance, table_relevance, table_target_need = self._strategic_table_fact_for_candidate(
            obs,
            candidate_pos,
            candidate_carrying,
            candidate_load_qty,
            target_weights,
        )
        if self._should_promote_strategic_table_over_resource(
            best_fact,
            best_relevance,
            best_target_need,
            table_fact,
            table_relevance,
        ):
            best_fact = table_fact
            best_distance = table_distance
            best_relevance = table_relevance
            best_target_need = table_target_need
        return best_fact, best_distance, best_relevance, best_target_need

    def _candidate_utility_features(
        self,
        obs: dict[str, Any] | None,
        candidate_names: list[str],
        round_num: int,
    ) -> list[dict[str, Any]]:
        agents_in_comm = (obs or {}).get("agents_in_communication_range", {}) or {}
        sent_records = self._recent_contact_records(obs)
        recv_records = self._recent_received_records(obs)
        own_carrying = _inventory_nonempty(obs)
        own_carry_type, own_carry_qty = _inventory_resource_quantity(obs)
        own_table_fact, own_table_distance, own_table_fit = self._best_table_fact_for_carry(
            obs,
            own_carry_type,
            own_carry_qty,
        )
        own_has_compatible_table = bool(own_table_fact)
        has_table_knowledge = _has_table_knowledge(obs)
        contacted_names = self._contacted_candidate_names(obs)
        target_weights = self._task_target_weights(obs)

        features: list[dict[str, Any]] = []
        for original_index, candidate in enumerate(candidate_names):
            info = agents_in_comm.get(candidate, {}) or {}
            carrying, carrying_qty = _candidate_resource_quantity(info)
            carrying_reason = carrying != "none" and (has_table_knowledge or own_carrying)
            in_view = bool(info.get("in_view_range", False))
            sent_rounds = [
                round_value
                for round_value in (_coerce_int(rec.get("round")) for rec in sent_records if str(rec.get("to") or "").strip() == candidate)
                if round_value is not None
            ]
            pair_streak = _consecutive_round_streak(sent_rounds, int(round_num))
            last_contact_gap = self._last_contact_gap(obs, candidate, int(round_num))
            last_incoming_gap = self._last_incoming_gap(obs, candidate, int(round_num))
            reply_recent = last_incoming_gap == 1
            latest_incoming_text = self._latest_incoming_text(obs, candidate, int(round_num))
            reply_info_request = bool(reply_recent and self._incoming_message_requests_info(latest_incoming_text))
            latest_incoming_ask_type = self._incoming_message_table_request_type(latest_incoming_text)
            latest_has_table, latest_table_for_current_carry = self._incoming_message_has_table_fact_for_type(
                latest_incoming_text,
                own_carry_type,
            )
            recent_has_table, recent_table_for_current_carry = self._recent_incoming_has_table_fact_for_type(
                obs,
                candidate,
                int(round_num),
                own_carry_type,
            )
            latest_has_table = bool(latest_has_table or recent_has_table)
            latest_table_for_current_carry = bool(
                latest_table_for_current_carry or recent_table_for_current_carry
            )
            rolling_recent_contact_count = self._rolling_recent_contact_count(obs, candidate, int(round_num))
            recent_pair_count, recent_total_sent, recent_pair_share = self._recent_concentration_stats(
                obs,
                candidate,
                int(round_num),
            )
            best_fact, fact_distance, fact_relevance_score, target_need = self._best_fact_for_candidate(
                obs,
                info,
                target_weights,
            )
            backup_fact, backup_fact_distance, backup_fact_relevance_score = self._backup_fact_for_candidate(
                obs,
                info,
                target_weights,
                best_fact,
            )
            own_table_for_carry = bool(own_has_compatible_table and own_carry_type in {"wood", "sand", "stone"})
            own_table_relevant_to_candidate = bool(
                own_table_for_carry
                and carrying == own_carry_type
                and carrying in {"wood", "sand", "stone"}
            )
            if own_table_relevant_to_candidate and not (
                isinstance(best_fact, dict)
                and str(best_fact.get("kind") or "") == "table"
                and bool(self._table_delivery_fit(best_fact, carrying, self._candidate_load_qty(info, carrying)).get("safe"))
            ):
                best_fact = dict(own_table_fact or {})
                fact_distance = own_table_distance
                fact_relevance_score = max(float(fact_relevance_score), 1.05)
                target_need = max(float(target_need), 1.0)
            duplicate_fact_recent = self._duplicate_fact_recent(obs, candidate, best_fact, int(round_num))
            fact_recent_message_count = self._fact_recent_message_count(
                obs,
                best_fact,
                int(round_num),
                include_received=True,
            )
            coverage_novelty, fact_area_key, fact_area_recent_message_count = self._coverage_novelty_for_fact(
                obs,
                best_fact,
                int(round_num),
                int(fact_recent_message_count),
            )
            table_fit = self._table_delivery_fit(
                best_fact,
                carrying,
                self._candidate_load_qty(info, carrying),
            )
            safe_table_delivery = bool(table_fit.get("safe"))
            candidate_table_for_carry = bool(carrying != "none" and str((best_fact or {}).get("kind") or "") == "table" and safe_table_delivery)
            table_for_type = carrying if candidate_table_for_carry else (own_carry_type if own_table_relevant_to_candidate else "none")
            sender_needs_table_for_type = bool(
                own_carry_type in {"wood", "sand", "stone"}
                and not own_has_compatible_table
            )
            request_recent_count = self._recent_table_request_count(
                obs,
                own_carry_type,
                int(round_num),
                candidate=candidate,
            )
            request_recent_global_count = self._recent_table_request_count(
                obs,
                own_carry_type,
                int(round_num),
            )
            request_cooldown = bool(
                sender_needs_table_for_type
                and (
                    request_recent_count > 0
                    or (
                        request_recent_global_count >= 2
                        and not reply_info_request
                        and not latest_table_for_current_carry
                        and not latest_has_table
                    )
                )
            )
            need_table_for_type = bool(
                sender_needs_table_for_type
                and not request_cooldown
                and last_contact_gap != 1
                and (
                    reply_info_request
                    or int(rolling_recent_contact_count or 0) <= 1
                )
            )
            routing_intent = self._classify_candidate_intent(
                reply_recent=reply_recent,
                reply_info_request=reply_info_request,
                carrying_reason=carrying_reason,
                best_fact=best_fact,
                safe_table_delivery=safe_table_delivery,
                candidate_table_for_carry=candidate_table_for_carry,
                own_table_for_carry=own_table_relevant_to_candidate,
                need_table_for_type=need_table_for_type,
            )
            best_fact_kind = str((best_fact or {}).get("kind") or "none")
            routing_objective = _stable_fair_routing_objective_from_intent(routing_intent, best_fact_kind)
            dynamic_status_refresh = bool(
                routing_intent in {"table_for_carry", "need_table_for_type", "reply"}
                or carrying_reason
            )
            stale_pair = bool(
                len(candidate_names) > 1
                and pair_streak >= 2
            )
            concentrated_pair = bool(
                len(candidate_names) > 1
                and recent_pair_count >= int(self.CONCENTRATION_MIN_COUNT)
                and recent_pair_share >= float(self.CONCENTRATION_SHARE_THRESHOLD)
            )
            positive_signal = bool(
                best_fact is not None
                or reply_recent
                or carrying_reason
                or own_table_relevant_to_candidate
                or need_table_for_type
            )
            feature = {
                "name": candidate,
                "original_index": original_index,
                "contact_last_round": last_contact_gap == 1,
                "pair_streak": pair_streak,
                "same_recipient_streak": pair_streak,
                "reply_recent": reply_recent,
                "reply_info_request": reply_info_request,
                "latest_incoming_ask_type": latest_incoming_ask_type,
                "latest_incoming_has_table": latest_has_table,
                "latest_incoming_table_for_current_carry": latest_table_for_current_carry,
                "carrying_reason": carrying_reason,
                "stale_pair": stale_pair,
                "recent_pair_count": recent_pair_count,
                "recent_total_sent": recent_total_sent,
                "recent_pair_share": recent_pair_share,
                "concentrated_pair": concentrated_pair,
                "contacted_before": candidate in contacted_names,
                "recent_contact_count": self._recent_contact_count(obs, candidate),
                "rolling_recent_contact_count": rolling_recent_contact_count,
                "carrying": carrying,
                "carrying_qty": carrying_qty,
                "in_view": in_view,
                "fact_distance": fact_distance,
                "best_fact": best_fact,
                "best_fact_label": self._format_fact_label(best_fact),
                "best_fact_kind": best_fact_kind,
                "best_fact_type": str((best_fact or {}).get("resource_type") or "none"),
                "best_fact_source": str((best_fact or {}).get("source") or "none"),
                "best_fact_table_value": round(self._table_broadcast_value(best_fact), 3),
                "backup_fact": backup_fact,
                "backup_fact_label": self._format_fact_label(backup_fact),
                "backup_fact_kind": str((backup_fact or {}).get("kind") or "none"),
                "backup_fact_type": str((backup_fact or {}).get("resource_type") or "none"),
                "backup_fact_source": str((backup_fact or {}).get("source") or "none"),
                "backup_fact_distance": backup_fact_distance,
                "backup_fact_relevance_score": round(float(backup_fact_relevance_score), 4),
                "target_need": float(target_need),
                "duplicate_fact_recent": duplicate_fact_recent,
                "fact_recent_message_count": int(fact_recent_message_count),
                "fact_area_key": fact_area_key,
                "fact_area_recent_message_count": int(fact_area_recent_message_count),
                "coverage_novelty": coverage_novelty,
                "dynamic_status_refresh": dynamic_status_refresh,
                "table_delivery_fit": str(table_fit.get("status") or "none"),
                "safe_table_delivery": safe_table_delivery,
                "table_remaining_for_type": table_fit.get("remaining_for_type"),
                "table_required_qty": table_fit.get("required_qty"),
                "table_for_type": table_for_type,
                "own_carrying": own_carry_type,
                "own_carrying_qty": own_carry_qty,
                "own_table_for_carry": own_table_for_carry,
                "own_table_relevant_to_candidate": own_table_relevant_to_candidate,
                "sender_needs_table_for_type": sender_needs_table_for_type,
                "need_table_for_type": need_table_for_type,
                "request_recent_count": int(request_recent_count),
                "request_recent_global_count": int(request_recent_global_count),
                "request_cooldown": request_cooldown,
                "routing_intent": routing_intent,
                "routing_hint": routing_intent,
                "routing_objective": routing_objective,
                "intent_signal": routing_intent,
                "positive_signal": positive_signal,
                "fact_relevance_score": round(float(fact_relevance_score), 4),
            }
            feature["payload_lifecycle"] = self._payload_lifecycle_for_feature(feature)
            feature["edge_reason"] = _stable_fair_routing_edge_reason_for_feature(feature)
            feature["recipient_value"] = _stable_fair_routing_recipient_value_for_feature(feature)
            feature["recipient_answerability"] = _stable_fair_routing_recipient_answerability_for_feature(feature)
            feature["send_risk"] = _stable_fair_routing_send_risk_for_feature(feature)
            feature["routing_edge"] = _stable_fair_routing_edge_class_for_feature(feature)
            feature["payload_state"] = _stable_fair_routing_payload_state_for_feature(feature)
            feature["audience_fit"] = _stable_fair_routing_audience_fit_for_feature(feature)
            feature["routing_brief"] = _stable_fair_routing_brief_for_feature(feature)
            features.append(feature)
        return features

    def _classify_candidate_intent(
        self,
        *,
        reply_recent: bool,
        reply_info_request: bool,
        carrying_reason: bool,
        best_fact: dict[str, Any] | None,
        safe_table_delivery: bool,
        candidate_table_for_carry: bool = False,
        own_table_for_carry: bool = False,
        need_table_for_type: bool = False,
    ) -> str:
        if reply_recent and reply_info_request:
            return "reply"
        if candidate_table_for_carry or (
            carrying_reason
            and str((best_fact or {}).get("kind") or "") == "table"
            and safe_table_delivery
        ):
            return "table_for_carry"
        if own_table_for_carry:
            return "table_for_carry"
        if need_table_for_type:
            return "need_table_for_type"
        if str((best_fact or {}).get("kind") or "") in {"resource", "table"}:
            return "fact_share"
        return "coverage"

    def _reorder_candidates_for_basic_routing(
        self,
        obs: dict[str, Any] | None,
        candidate_names: list[str],
        round_num: int,
    ) -> tuple[list[str], dict[str, Any]]:
        display_names, display_meta = self._candidate_display_order(list(candidate_names), int(round_num))
        features = self._candidate_utility_features(obs, display_names, round_num)
        if not features:
            return list(candidate_names), {
                "history_available": False,
                "early_coverage_mode": False,
                "features": [],
            }
        history_available = any(
            bool(
                feature.get("positive_signal")
                or feature.get("pair_streak", 0)
                or feature.get("reply_recent")
                or feature.get("carrying_reason")
                or feature.get("contacted_before")
            )
            for feature in features
        )
        early_coverage_round_limit = int(self.EARLY_COVERAGE_ROUNDS)
        early_coverage_mode = int(round_num) < int(early_coverage_round_limit)
        ordered_features_full = list(features)
        ordered_features = list(features)
        ordered_names = [str(feature.get("name") or "") for feature in ordered_features if str(feature.get("name") or "")]
        return ordered_names or list(candidate_names), {
            "history_available": history_available,
            "early_coverage_mode": early_coverage_mode,
            "early_coverage_round_limit": early_coverage_round_limit,
            "features": ordered_features,
            "features_full": ordered_features_full,
            "shortlist_applied": False,
            "selection_regime": "plain_llm_candidate_order",
            "candidate_pool_size": len(ordered_features_full),
            "shortlist_size": len(ordered_features_full),
            "rotation_offset": int(display_meta.get("rotation_offset", 0) or 0),
            "base_order": list(display_meta.get("base_order") or list(candidate_names)),
            "shortlist_gap_next": None,
            "shortlist_gap_third": None,
        }

    def _should_auto_send(
        self,
        reorder_meta: dict[str, Any] | None,
        round_num: int,
    ) -> tuple[bool, str | None]:
        del reorder_meta, round_num
        return False, None

    def _feature_is_blocked_for_sparse_gate(self, feature: dict[str, Any] | None) -> bool:
        if not isinstance(feature, dict):
            return True
        state = str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)).lower()
        if state in {"blocked", "stale", "duplicate"}:
            return True
        lifecycle = str(feature.get("payload_lifecycle") or "").lower()
        if lifecycle in {"stale_static_table", "stale_static_resource", "exhausted_table", "request_cooldown"}:
            return True
        risk = str(feature.get("send_risk") or "").lower()
        return any(token in risk for token in ("duplicate_fact_recent", "stale_", "exhausted_table", "request_cooldown"))

    def _feature_has_sparse_coverage_debt(
        self,
        feature: dict[str, Any] | None,
        *,
        allow_memory_resource: bool = False,
    ) -> bool:
        if not isinstance(feature, dict):
            return False
        coverage_novelty = str(feature.get("coverage_novelty") or "repeat").strip().lower()
        if bool(feature.get("duplicate_fact_recent")) or bool(feature.get("contact_last_round")):
            return False
        contact_limit = 2 if coverage_novelty == "new_area" else 1
        if int(feature.get("rolling_recent_contact_count", 0) or 0) > contact_limit:
            return False
        fact = feature.get("best_fact")
        if not isinstance(fact, dict):
            return False
        kind = str(feature.get("best_fact_kind") or fact.get("kind") or "").strip().lower()
        recent_count = int(feature.get("fact_recent_message_count", 0) or 0)
        source = str(feature.get("best_fact_source") or fact.get("source") or "").strip().lower()
        if recent_count > 1 and source not in {"visible", "current_tile"}:
            return False
        if coverage_novelty == "new_area" and source in {"visible", "current_tile"}:
            if kind == "table":
                remaining = _coerce_int(fact.get("remaining_capacity"))
                return self._is_high_value_table_fact(fact) or bool(remaining is not None and int(remaining) >= 8)
            if kind == "resource":
                qty = _coerce_int(fact.get("quantity"))
                if qty is not None and int(qty) <= 0:
                    return False
                return float(feature.get("target_need", 0.0) or 0.0) >= 0.9
        if source not in {"visible", "current_tile"}:
            return False
        if kind == "table":
            if self._is_high_value_table_fact(fact):
                return recent_count < 2
            remaining = _coerce_int(fact.get("remaining_capacity"))
            return bool(source == "visible" and remaining is not None and int(remaining) >= 12)
        if kind == "resource":
            if source not in {"visible", "current_tile"} and not allow_memory_resource:
                return False
            qty = _coerce_int(fact.get("quantity"))
            if qty is not None and int(qty) <= 0:
                return False
            return float(feature.get("target_need", 0.0) or 0.0) >= 1.1
        return False

    def _sparse_send_reason_for_feature(
        self,
        feature: dict[str, Any] | None,
        round_num: int,
        pressure: dict[str, Any] | None,
    ) -> tuple[bool, str, int]:
        if self._feature_is_blocked_for_sparse_gate(feature):
            return False, "blocked_or_repeated", 0
        assert isinstance(feature, dict)
        edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)).lower()
        state = str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)).lower()
        audience = str(feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature)).lower()
        kind = str(feature.get("best_fact_kind") or "none").lower()
        fact = feature.get("best_fact")
        source = str(feature.get("best_fact_source") or (fact or {}).get("source") or "").strip().lower()
        coverage_novelty = str(feature.get("coverage_novelty") or "repeat").strip().lower()
        early_stage = int(round_num) < int(self.EARLY_COVERAGE_ROUNDS)
        pressure_level = str((pressure or {}).get("level") or "low").strip().lower()
        if edge in {"reply_request", "table_for_carrier", "ask_table_provider"} and state == "active":
            if audience in {"can_receive", "can_act", "can_answer"}:
                if edge == "ask_table_provider" and not (
                    str(feature.get("own_carrying") or "none") in {"wood", "sand", "stone"}
                    and bool(feature.get("sender_needs_table_for_type"))
                    and not bool(feature.get("request_cooldown"))
                ):
                    return False, "low_value_static_or_status", 0
                if edge == "table_for_carrier" and not (
                    str(feature.get("carrying") or "none") in {"wood", "sand", "stone"}
                    and bool(feature.get("safe_table_delivery"))
                ):
                    return False, "low_value_static_or_status", 0
                return True, f"{edge}_active", 100
        if edge == "fresh_discovery" and state in {"fresh", "active"} and isinstance(fact, dict):
            if state == "active" and source not in {"visible", "current_tile"}:
                return False, "low_value_static_or_status", 0
            resource_allowed, resource_reason, resource_priority = self._resource_coverage_priority_for_feature(
                feature,
                round_num,
            )
            if resource_allowed:
                return True, resource_reason, resource_priority
            if coverage_novelty == "new_area" and self._feature_has_sparse_coverage_debt(feature):
                return True, "new_area_coverage", 84
            if kind == "table" and self._is_high_value_table_fact(fact) and self._feature_has_sparse_coverage_debt(feature):
                return True, "high_value_table_coverage", 82
            if self._feature_has_sparse_coverage_debt(feature):
                return True, "coverage_debt", 72
            if early_stage and source == "visible":
                return True, "early_visible_discovery", 62
            if (
                pressure_level == "high"
                and kind == "resource"
                and source in {"visible", "current_tile"}
                and float(feature.get("target_need", 0.0) or 0.0) >= 1.1
            ):
                return True, "high_pressure_resource", 58
        return False, "low_value_static_or_status", 0

    def _best_sparse_send_feature(
        self,
        reorder_meta: dict[str, Any] | None,
        round_num: int,
        pressure: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None, int]:
        best_feature: dict[str, Any] | None = None
        best_reason: str | None = None
        best_key: tuple[int, int, int, int, int] | None = None
        features = list((reorder_meta or {}).get("features") or []) + list((reorder_meta or {}).get("features_full") or [])
        for feature in features:
            if not isinstance(feature, dict):
                continue
            allowed, reason, priority = self._sparse_send_reason_for_feature(feature, round_num, pressure)
            if not allowed:
                continue
            contact_count = int(feature.get("rolling_recent_contact_count", 0) or 0)
            table_value = int(round(100 * self._table_broadcast_value(feature.get("best_fact"))))
            fact_relevance = int(round(100 * float(feature.get("fact_relevance_score", 0.0) or 0.0)))
            novelty_bonus = 1 if str(feature.get("coverage_novelty") or "") == "new_area" else 0
            key = (int(priority), int(novelty_bonus), -contact_count, table_value, fact_relevance)
            if best_key is None or key > best_key:
                best_key = key
                best_feature = feature
                best_reason = reason
        return best_feature, best_reason, int(best_key[0]) if best_key else 0

    def _sparse_gate_veto_after_send_yes(
        self,
        reorder_meta: dict[str, Any] | None,
        round_num: int,
        pressure: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        best_feature, reason, _priority = self._best_sparse_send_feature(reorder_meta, round_num, pressure)
        if best_feature is not None:
            return None, None
        features = [feature for feature in list((reorder_meta or {}).get("features") or []) if isinstance(feature, dict)]
        return (features[0] if features else None), f"sparse_gate_veto:{reason or 'no_useful_payload'}"

    def _feature_has_resource_coverage_floor(
        self,
        feature: dict[str, Any] | None,
        round_num: int,
        obs: dict[str, Any] | None,
    ) -> bool:
        allowed, _reason, _priority = self._resource_coverage_priority_for_feature(
            feature,
            round_num,
            obs=obs,
            require_recipient_fresh=True,
        )
        return bool(allowed)

    def _resource_coverage_priority_for_feature(
        self,
        feature: dict[str, Any] | None,
        round_num: int,
        *,
        obs: dict[str, Any] | None = None,
        require_recipient_fresh: bool = False,
    ) -> tuple[bool, str, int]:
        if self._feature_is_blocked_for_sparse_gate(feature):
            return False, "blocked_or_repeated", 0
        assert isinstance(feature, dict)
        edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)).lower()
        state = str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)).lower()
        objective = _stable_fair_routing_objective_for_feature(feature)
        if edge != "fresh_discovery" or state not in {"fresh", "active"} or objective not in {"coverage", "share_resource"}:
            return False, "not_resource_coverage", 0
        fact = feature.get("best_fact")
        if not isinstance(fact, dict):
            return False, "missing_fact", 0
        if str(feature.get("best_fact_kind") or fact.get("kind") or "").strip().lower() != "resource":
            return False, "not_resource_fact", 0
        source = str(feature.get("best_fact_source") or fact.get("source") or "").strip().lower()
        if source not in {"visible", "current_tile"}:
            return False, "nonlocal_resource", 0
        if bool(feature.get("contact_last_round")) or int(feature.get("rolling_recent_contact_count", 0) or 0) > 1:
            return False, "recent_contact", 0
        qty = _coerce_int(fact.get("quantity"))
        if qty is not None and int(qty) <= 0:
            return False, "empty_resource", 0
        target_need = float(feature.get("target_need", 1.0) or 1.0)
        if target_need < 0.9:
            return False, "low_target_need", 0
        recipient = str(feature.get("name") or "").strip()
        if require_recipient_fresh and recipient and self._fact_recent_message_count(
            obs,
            fact,
            int(round_num),
            window=6,
            candidate=recipient,
            include_received=False,
        ) > 0:
            return False, "recipient_recent_resource_duplicate", 0

        novelty = str(feature.get("coverage_novelty") or "repeat").strip().lower()
        has_debt = self._feature_has_sparse_coverage_debt(feature)
        if source == "current_tile":
            priority = 88 if novelty in {"new_area", "new_fact"} else 86
        elif novelty in {"new_area", "new_fact"}:
            priority = 86
        elif has_debt:
            priority = 80
        elif target_need >= 1.1:
            priority = 78
        else:
            return False, "low_value_resource_coverage", 0

        if novelty == "new_area" and has_debt:
            reason = "new_area_coverage_floor"
        else:
            reason = "resource_coverage_floor"
        return True, reason, int(priority)

    def _best_resource_coverage_floor_feature(
        self,
        features: list[dict[str, Any]],
        round_num: int,
        obs: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        best_feature: dict[str, Any] | None = None
        best_reason: str | None = None
        best_key: tuple[int, int, int, int, int] | None = None
        for feature in features:
            allowed, reason, priority = self._resource_coverage_priority_for_feature(
                feature,
                round_num,
                obs=obs,
                require_recipient_fresh=True,
            )
            if not allowed:
                continue
            fact = feature.get("best_fact")
            source = str(feature.get("best_fact_source") or (fact or {}).get("source") or "").strip().lower()
            source_rank = 2 if source == "current_tile" else 1
            contact_count = int(feature.get("rolling_recent_contact_count", 0) or 0)
            qty = int(_coerce_int((fact or {}).get("quantity")) or 0)
            relevance = int(round(100 * float(feature.get("fact_relevance_score", 0.0) or 0.0)))
            key = (int(priority), int(source_rank), -contact_count, int(qty), int(relevance))
            if best_key is None or key > best_key:
                best_key = key
                best_feature = feature
                best_reason = reason
        return best_feature, best_reason

    def _maybe_rebalance_recipient_for_coverage(
        self,
        recipient: str,
        reorder_meta: dict[str, Any] | None,
        round_num: int,
        pressure: dict[str, Any] | None,
    ) -> tuple[str, str | None]:
        features = [
            feature
            for feature in list((reorder_meta or {}).get("features") or []) + list((reorder_meta or {}).get("features_full") or [])
            if isinstance(feature, dict)
        ]
        feature_by_name: dict[str, dict[str, Any]] = {}
        for feature in features:
            name = str(feature.get("name") or "").strip()
            if name and name not in feature_by_name:
                feature_by_name[name] = feature
        selected = feature_by_name.get(str(recipient or "").strip())
        if not selected:
            return recipient, None
        selected_allowed, _selected_reason, selected_priority = self._sparse_send_reason_for_feature(
            selected,
            round_num,
            pressure,
        )
        selected_edge = str(selected.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(selected)).lower()
        if selected_edge in {"reply_request", "table_for_carrier", "ask_table_provider"}:
            return recipient, None
        selected_count = int(selected.get("rolling_recent_contact_count", 0) or 0)
        selected_concentrated = bool(selected.get("concentrated_pair")) or bool(selected.get("contact_last_round"))
        selected_novelty = str(selected.get("coverage_novelty") or "repeat").strip().lower()
        other_new_area = any(
            name != recipient
            and str(feature.get("coverage_novelty") or "").strip().lower() == "new_area"
            for name, feature in feature_by_name.items()
        )
        if selected_count <= 1 and not selected_concentrated and not (other_new_area and selected_novelty != "new_area"):
            return recipient, None
        best_name = recipient
        best_key: tuple[int, int, int, int, int] | None = None
        for name, feature in feature_by_name.items():
            if name == recipient:
                continue
            allowed, _reason, priority = self._sparse_send_reason_for_feature(feature, round_num, pressure)
            if not allowed:
                continue
            if selected_allowed and priority < selected_priority:
                continue
            contact_count = int(feature.get("rolling_recent_contact_count", 0) or 0)
            feature_novelty = str(feature.get("coverage_novelty") or "repeat").strip().lower()
            if (
                contact_count >= selected_count
                and not selected_concentrated
                and not (feature_novelty == "new_area" and selected_novelty != "new_area")
            ):
                continue
            table_value = int(round(100 * self._table_broadcast_value(feature.get("best_fact"))))
            fact_relevance = int(round(100 * float(feature.get("fact_relevance_score", 0.0) or 0.0)))
            novelty_bonus = 1 if feature_novelty == "new_area" else 0
            key = (int(priority), int(novelty_bonus), -contact_count, table_value, fact_relevance)
            if best_key is None or key > best_key:
                best_key = key
                best_name = name
        if best_name != recipient:
            return best_name, "recipient_coverage_debt"
        return recipient, None

    def _coverage_floor_feature_after_send_no(
        self,
        reorder_meta: dict[str, Any] | None,
        round_num: int,
        pressure: dict[str, Any] | None,
        obs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Keep sparse gate from dropping simple map-coverage evidence."""
        features = [
            feature
            for feature in list((reorder_meta or {}).get("features") or [])
            if isinstance(feature, dict)
        ]
        if not features:
            return None, None
        early_stage = int(round_num) < int(self.EARLY_COVERAGE_ROUNDS)
        pressure_level = str((pressure or {}).get("level") or "low").strip().lower()

        def active_floor_reason(feature: dict[str, Any]) -> str | None:
            if self._feature_is_blocked_for_sparse_gate(feature):
                return None
            edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)).lower()
            state = str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)).lower()
            audience = str(feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature)).lower()
            if edge not in {"reply_request", "table_for_carrier", "ask_table_provider"} or state != "active":
                return None
            if audience not in {"can_receive", "can_act", "can_answer"}:
                return None
            if edge == "ask_table_provider" and not (
                str(feature.get("own_carrying") or "none") in {"wood", "sand", "stone"}
                and bool(feature.get("sender_needs_table_for_type"))
                and not bool(feature.get("request_cooldown"))
            ):
                return None
            if edge == "table_for_carrier" and not (
                str(feature.get("carrying") or "none") in {"wood", "sand", "stone"}
                and bool(feature.get("safe_table_delivery"))
            ):
                return None
            return f"{edge}_active_floor"

        for feature in features:
            reason = active_floor_reason(feature)
            if reason:
                return feature, reason

        resource_feature, resource_reason = self._best_resource_coverage_floor_feature(
            features,
            int(round_num),
            obs,
        )
        if resource_feature is not None and resource_reason:
            return resource_feature, resource_reason

        for feature in features:
            if self._feature_is_blocked_for_sparse_gate(feature):
                continue
            edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)).lower()
            state = str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)).lower()
            kind = str(feature.get("best_fact_kind") or "none").lower()
            if edge == "fresh_discovery" and state in {"fresh", "active"} and kind in {"resource", "table"}:
                if bool(feature.get("contact_last_round")):
                    continue
                source = str(feature.get("best_fact_source") or "").strip().lower()
                if source not in {"visible", "current_tile"}:
                    continue
                coverage_novelty = str(feature.get("coverage_novelty") or "repeat").strip().lower()
                has_debt = self._feature_has_sparse_coverage_debt(feature)
                high_value_table = bool(
                    kind == "table"
                    and self._is_high_value_table_fact(feature.get("best_fact"))
                    and has_debt
                )
                early_visible = bool(early_stage and source == "visible")
                high_pressure_resource = bool(
                    pressure_level == "high"
                    and kind == "resource"
                    and source in {"visible", "current_tile"}
                    and float(feature.get("target_need", 0.0) or 0.0) >= 1.1
                )
                if coverage_novelty == "new_area" and has_debt:
                    return feature, "new_area_coverage_floor"
                if not early_stage and coverage_novelty not in {"new_area", "new_fact"} and not high_value_table and not high_pressure_resource:
                    continue
                if early_visible or high_value_table or high_pressure_resource or (
                    has_debt and (early_stage or pressure_level in {"medium", "high"})
                ):
                    if has_debt and not early_stage and pressure_level in {"medium", "high"}:
                        return feature, "fresh_discovery_coverage_debt_floor"
                    return feature, "fresh_discovery_coverage_floor"
        return None, None

    def _should_apply_minimal_floor_override(
        self,
        feature: dict[str, Any] | None,
        reason: str | None,
        round_num: int,
    ) -> bool:
        """Turn only high-confidence sparse-gate misses into real sends."""
        if not isinstance(feature, dict):
            return False
        reason_norm = str(reason or "").strip().lower()
        if reason_norm in {
            "reply_request_active_floor",
            "table_for_carrier_active_floor",
        }:
            return True
        if reason_norm == "ask_table_provider_active_floor":
            return _stable_fair_recipient_can_answer_table_request(feature)
        if reason_norm == "new_area_coverage_floor":
            return True
        if reason_norm == "resource_coverage_floor":
            fact = feature.get("best_fact")
            if not isinstance(fact, dict):
                return False
            if str(feature.get("best_fact_kind") or fact.get("kind") or "").strip().lower() != "resource":
                return False
            if str(feature.get("best_fact_source") or fact.get("source") or "").strip().lower() not in {"visible", "current_tile"}:
                return False
            if bool(feature.get("duplicate_fact_recent")) or bool(feature.get("contact_last_round")):
                return False
            return float(feature.get("target_need", 1.0) or 1.0) >= 0.9
        if reason_norm in {"fresh_discovery_coverage_floor", "fresh_discovery_coverage_debt_floor"}:
            if str(feature.get("best_fact_source") or "").strip().lower() not in {"visible", "current_tile"}:
                return False
            kind = str(feature.get("best_fact_kind") or "").strip().lower()
            if kind == "resource":
                return int(round_num) < int(self.EARLY_COVERAGE_ROUNDS) or self._feature_has_sparse_coverage_debt(feature)
            if kind == "table":
                return self._feature_has_sparse_coverage_debt(feature)
            return False
        return False

    @staticmethod
    def _coverage_floor_feature_summary(feature: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(feature, dict):
            return None
        return {
            "name": str(feature.get("name") or ""),
            "routing_edge": str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)),
            "payload_state": str(feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)),
            "audience_fit": str(feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature)),
            "best_fact_kind": str(feature.get("best_fact_kind") or "none"),
            "best_fact_label": str(feature.get("best_fact_label") or "none"),
            "fact_area_key": str(feature.get("fact_area_key") or "unknown"),
            "coverage_novelty": str(feature.get("coverage_novelty") or "none"),
            "duplicate_fact_recent": bool(feature.get("duplicate_fact_recent")),
            "contact_last_round": bool(feature.get("contact_last_round")),
        }

    def _parse_send_only_decision(self, response: str) -> tuple[bool, str | None]:
        saw_send = False
        send = False
        try:
            for line in str(response or "").splitlines():
                raw = line.strip()
                low = raw.lower()
                if low.startswith("send:"):
                    saw_send = True
                    send = low.split(":", 1)[1].strip() in ("yes", "y", "true", "1")
        except Exception:
            return False, "parse_error"
        if not saw_send:
            return False, "missing_send"
        if not send:
            return False, "model_send_no"
        return True, None

    def _parse_recipient_only_decision(self, response: str) -> tuple[str | None, str | None]:
        recipient = None
        try:
            for line in str(response or "").splitlines():
                raw = line.strip()
                low = raw.lower()
                if low.startswith("to:"):
                    recipient = raw.split(":", 1)[1].strip()
                    token = recipient.split()[0].strip(",:;") if recipient else ""
                    if token:
                        recipient = token
        except Exception:
            return None, "parse_error"
        if not recipient or str(recipient).strip().lower() == "none":
            return None, "missing_recipient"
        return recipient, None

    async def _run_basic_routing_prompt(self, prompt: str, sampling_override=None) -> tuple[str, float]:
        response = ""
        api_call_time = 0.0
        try:
            t0 = time.time()
            if sampling_override:
                response = await self.owner.brain.generate(prompt, sampling_override=sampling_override)
            else:
                response = await self.owner.brain.generate(prompt)
            api_call_time = time.time() - t0
            self._jump_back()
        except Exception as exc:
            response = f"ERROR: routing_error {exc}"
        return response, api_call_time

    def _build_candidate_status_block(self, obs: dict, candidates: list[str]) -> str:
        facts_block = _compact_current_facts_block(obs, usage="routing")
        agents_in_comm = (obs or {}).get("agents_in_communication_range", {}) or {}
        feature_map = {
            str(feature.get("name") or ""): feature
            for feature in self._candidate_utility_features(obs, candidates, int((obs or {}).get("round", 0) or 0))
        }
        pressure = self._communication_pressure_profile(obs, candidates, list(feature_map.values()))
        reasons = ", ".join(list(pressure.get("reasons") or [])[:4]) or "ordinary local state"
        candidate_status_lines: list[str] = [
            "Candidate local status:",
            f"Communication pressure: {pressure.get('level')} (score={pressure.get('score')}; {reasons}).",
        ]
        for candidate in candidates:
            info = agents_in_comm.get(candidate, {}) or {}
            feature = feature_map.get(candidate, {})
            carrying = str(feature.get("carrying") or info.get("carrying_resource") or "none")
            carrying_qty = feature.get("carrying_qty")
            rolling_recent_contact_count = int(feature.get("rolling_recent_contact_count", 0) or 0)
            best_fact_label = str(feature.get("best_fact_label") or "none")
            best_fact_source = str(feature.get("best_fact_source") or "none")
            routing_edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature))
            audience_fit = str(feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature))
            coverage_novelty = str(feature.get("coverage_novelty") or "none")
            area_key = str(feature.get("fact_area_key") or "unknown")
            table_for_type = str(feature.get("table_for_type") or "none")
            latest_incoming_ask_type = str(feature.get("latest_incoming_ask_type") or "none")
            contact_status = "recent" if bool(feature.get("contact_last_round")) or rolling_recent_contact_count > 1 else "ok"
            if routing_edge in {"reply_request", "table_for_carrier", "ask_table_provider"}:
                value = "action"
            elif routing_edge == "fresh_discovery" and coverage_novelty in {"new_area", "new_fact"}:
                value = "coverage"
            else:
                value = "low"
            fact_bits = [f"{best_fact_label}", f"source={best_fact_source}", f"novelty={coverage_novelty}", f"area={area_key}"]
            if table_for_type != "none":
                fact_bits.append(f"table_for={table_for_type}")
            if latest_incoming_ask_type != "none":
                fact_bits.append(f"latest_request={latest_incoming_ask_type}")
            candidate_status_lines.append(
                f"- {candidate}: edge={routing_edge}; value={value}; fact={', '.join(fact_bits)}; "
                f"audience={audience_fit}; contact={contact_status}; "
                f"carry={carrying}:{carrying_qty if carrying_qty is not None else 'unknown'}"
            )
        candidate_status_lines.append("")
        candidate_status_lines.append(
            "Use only these core cues: edge, value, fact novelty, audience, and contact."
        )
        candidate_status_lines.append(
            "Routing chooses SEND/TO only; protocol will decide the final wording."
        )
        candidate_status_lines.append(
            "Send only when one concrete payload could change a teammate's next few actions or add a fresh map fact they are unlikely to know."
        )
        candidate_status_lines.append(
            "Action value means reply/request, compatible table for a carried item, or asking for a compatible table."
        )
        candidate_status_lines.append(
            "Coverage value means one fresh visible/current resource or table from a new area or new fact; otherwise prefer SEND:no."
        )
        candidate_status_lines.append("")
        candidate_status_block = "\n".join(candidate_status_lines)
        return f"\n{facts_block}{candidate_status_block}\n"

    def _build_candidate_status_block_lowcost(self, obs: dict, candidates: list[str]) -> str:
        facts_block = _compact_current_facts_block(obs, usage="routing")
        agents_in_comm = (obs or {}).get("agents_in_communication_range", {}) or {}
        feature_map = {
            str(feature.get("name") or ""): feature
            for feature in self._candidate_utility_features(obs, candidates, int((obs or {}).get("round", 0) or 0))
        }
        candidate_status_lines: list[str] = [
            "Compact routing evidence:",
            "Use these fields only as evidence; routing chooses SEND/TO, and protocol writes the final message.",
        ]
        for candidate in candidates:
            info = agents_in_comm.get(candidate, {}) or {}
            feature = feature_map.get(candidate, {})
            carrying = str(feature.get("carrying") or info.get("carrying_resource") or "none")
            carrying_qty = feature.get("carrying_qty")
            rolling_recent_contact_count = int(feature.get("rolling_recent_contact_count", 0) or 0)
            best_fact_label = str(feature.get("best_fact_label") or "none")
            routing_edge = str(feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature))
            coverage_novelty = str(feature.get("coverage_novelty") or "none")
            table_for_type = str(feature.get("table_for_type") or "none")
            latest_incoming_ask_type = str(feature.get("latest_incoming_ask_type") or "none")
            contact_status = "recent" if bool(feature.get("contact_last_round")) or rolling_recent_contact_count > 1 else "not_recent"
            if routing_edge in {"reply_request", "table_for_carrier", "ask_table_provider"}:
                purpose = "action"
            elif routing_edge == "fresh_discovery" and coverage_novelty in {"new_area", "new_fact"}:
                purpose = "coverage"
            else:
                purpose = "low"
            need_bits: list[str] = []
            if table_for_type != "none":
                need_bits.append(f"table_for={table_for_type}")
            if latest_incoming_ask_type != "none":
                need_bits.append(f"latest_request={latest_incoming_ask_type}")
            need_text = ",".join(need_bits) if need_bits else "none"
            candidate_status_lines.append(
                f"- {candidate}: purpose={purpose}; payload={best_fact_label}; freshness={coverage_novelty}; "
                f"contact={contact_status}; need={need_text}; "
                f"carry={carrying}:{carrying_qty if carrying_qty is not None else 'unknown'}"
            )
        candidate_status_lines.append("")
        candidate_status_lines.append(
            "Core cues: payload, recipient carry/need, freshness, and recent contact."
        )
        candidate_status_lines.append(
            "Send only for one useful payload: help action soon, answer/ask a concrete need, or add a fresh visible/current map fact."
        )
        candidate_status_lines.append("")
        candidate_status_block = "\n".join(candidate_status_lines)
        return f"\n{facts_block}{candidate_status_block}\n"

    def _communication_pressure_profile(
        self,
        obs: dict[str, Any] | None,
        candidates: list[str],
        features: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        data = dict(obs or {})
        agent_count = _coerce_int(data.get("num_agents"))
        carry_type, carry_qty = _inventory_resource_quantity(data)
        if features is None:
            feature_list = self._candidate_utility_features(data, candidates, int(data.get("round", 0) or 0))
        else:
            feature_list = list(features or [])

        visible_resources = [
            fact for fact in self._iter_resource_fact_entries(data)
            if str(fact.get("source") or "") == "visible"
        ]
        visible_tables = [
            fact for fact in self._iter_table_fact_entries(data)
            if str(fact.get("source") or "") == "visible"
        ]
        direct_reply_count = sum(
            1
            for feature in feature_list
            if bool(feature.get("reply_recent"))
            and (
                bool(feature.get("reply_info_request"))
                or _stable_fair_routing_objective_for_feature(feature) == "answer_request"
            )
        )
        delivery_ready_count = sum(
            1
            for feature in feature_list
            if _stable_fair_routing_objective_for_feature(feature) == "delivery_ready"
            and bool(feature.get("safe_table_delivery"))
            and not bool(feature.get("duplicate_fact_recent"))
            and (
                bool(feature.get("reply_recent"))
                or int(feature.get("rolling_recent_contact_count", 0) or 0) == 0
            )
            and (
                str(feature.get("best_fact_source") or "") == "visible"
                or (
                    feature.get("fact_distance") is not None
                    and int(feature.get("fact_distance") or 999) <= 4
                )
            )
        )
        table_for_carry_count = sum(
            1
            for feature in feature_list
            if _stable_fair_routing_objective_for_feature(feature) == "share_table_for_carrier"
            and not bool(feature.get("duplicate_fact_recent"))
            and str(feature.get("payload_lifecycle") or "") not in {"stale_static_table", "exhausted_table"}
            and int(feature.get("rolling_recent_contact_count", 0) or 0) == 0
        )
        need_table_request_count = sum(
            1
            for feature in feature_list
            if _stable_fair_routing_objective_for_feature(feature) == "ask_missing_table"
            and str(feature.get("payload_lifecycle") or "") != "request_cooldown"
            and int(feature.get("rolling_recent_contact_count", 0) or 0) == 0
            and _stable_fair_recipient_can_answer_table_request(feature)
        )
        actionable_resource_count = sum(
            1
            for feature in feature_list
            if str(feature.get("best_fact_kind") or "") == "resource"
            and isinstance(feature.get("best_fact"), dict)
            and not bool(feature.get("duplicate_fact_recent"))
            and int(feature.get("rolling_recent_contact_count", 0) or 0) == 0
            and str(feature.get("best_fact_source") or "") == "visible"
            and (
                feature.get("fact_distance") is None
                or int(feature.get("fact_distance") or 999) <= 6
            )
            and float(feature.get("target_need", 1.0) or 1.0) >= 1.0
        )
        fresh_fact_count = sum(
            1
            for feature in feature_list
            if isinstance(feature.get("best_fact"), dict) and not bool(feature.get("duplicate_fact_recent"))
            and str(feature.get("payload_lifecycle") or "") not in {
                "stale_static_table",
                "stale_static_resource",
                "exhausted_table",
                "request_cooldown",
            }
        )
        clear_fact_count = sum(
            1
            for feature in feature_list
            if isinstance(feature.get("best_fact"), dict)
            and not bool(feature.get("duplicate_fact_recent"))
            and str(feature.get("payload_lifecycle") or "") not in {
                "stale_static_table",
                "stale_static_resource",
                "exhausted_table",
                "request_cooldown",
            }
            and (
                str(feature.get("best_fact_source") or "") == "visible"
                or int(feature.get("rolling_recent_contact_count", 0) or 0) == 0
                or (
                    feature.get("fact_distance") is not None
                    and int(feature.get("fact_distance") or 999) <= 6
                )
            )
        )
        duplicate_static_count = sum(
            1
            for feature in feature_list
            if bool(feature.get("duplicate_fact_recent")) and not bool(feature.get("dynamic_status_refresh"))
        )
        stale_lifecycle_count = sum(
            1
            for feature in feature_list
            if str(feature.get("payload_lifecycle") or "") in {
                "stale_static_table",
                "stale_static_resource",
                "exhausted_table",
                "request_cooldown",
            }
        )
        recent_contact_total = sum(
            int(feature.get("rolling_recent_contact_count", 0) or 0)
            for feature in feature_list
        )
        self_delivery_ready = False
        if carry_type != "none":
            self_delivery_ready = any(
                bool(self._table_delivery_fit(fact, carry_type, carry_qty).get("safe"))
                for fact in self._iter_table_fact_entries(data)
            )

        score = 0
        reasons: list[str] = []
        strong_signal_count = (
            direct_reply_count
            + delivery_ready_count
            + table_for_carry_count
            + need_table_request_count
            + actionable_resource_count
        )

        if delivery_ready_count:
            score += 3
            reasons.append("delivery-ready table/carry signal")
        if direct_reply_count:
            score += 3
            reasons.append("direct reply/request")
        if table_for_carry_count:
            score += 2
            reasons.append("table-for-carry signal")
        if need_table_request_count:
            score += 1
            reasons.append("carrying agent may need one compatible-table request")
        if actionable_resource_count:
            score += 2
            reasons.append("non-duplicate resource fact")
        if self_delivery_ready:
            score += 1
            reasons.append(f"self carrying {carry_type} with usable table")
        elif carry_type != "none":
            score += 1
            reasons.append(f"self carrying {carry_type} may need one table request")

        static_score = 0
        if len(visible_resources) + len(visible_tables) >= 2:
            static_score += 1
            reasons.append("multiple visible facts")
        if fresh_fact_count >= 2:
            static_score += 1
            reasons.append("fresh facts for recipients")
        if clear_fact_count:
            static_score += 1
            reasons.append("clear fresh recipient fact")

        if strong_signal_count:
            score += min(static_score, 1)
            if agent_count is not None and int(agent_count) >= 6:
                score += 1
                reasons.append(f"team size {int(agent_count)} weak amplifier")
        else:
            score += min(static_score, 2)
            if agent_count is not None and int(agent_count) >= 6:
                reasons.append(f"team size {int(agent_count)} weak signal")
        if duplicate_static_count >= max(1, fresh_fact_count) and strong_signal_count == 0:
            score -= 1
            reasons.append("mostly repeated static facts")
        if stale_lifecycle_count and strong_signal_count == 0:
            score -= 1
            reasons.append("stale/exhausted payload lifecycle")
        if recent_contact_total >= 3 and direct_reply_count == 0:
            score -= 1
            reasons.append("recent contact already dense")
        if not strong_signal_count:
            score = min(score, 2)
        elif recent_contact_total >= 3 and direct_reply_count == 0 and delivery_ready_count == 0:
            score = min(score, 2)

        score = max(0, int(score))
        if strong_signal_count and score >= 3:
            level = "high"
        elif score >= 2:
            level = "medium"
        else:
            level = "low"
        return {
            "level": level,
            "score": score,
            "reasons": reasons,
            "agent_count": agent_count,
            "fresh_fact_count": fresh_fact_count,
            "dynamic_count": direct_reply_count + delivery_ready_count + table_for_carry_count + need_table_request_count,
            "strong_signal_count": strong_signal_count,
            "direct_reply_count": direct_reply_count,
            "delivery_ready_count": delivery_ready_count,
            "table_for_carry_count": table_for_carry_count,
            "need_table_request_count": need_table_request_count,
            "actionable_resource_count": actionable_resource_count,
            "stale_lifecycle_count": stale_lifecycle_count,
            "self_delivery_ready": self_delivery_ready,
            "clear_fact_count": clear_fact_count,
        }

    def _communication_pressure_guidance(self, pressure: dict[str, Any]) -> str:
        level = str((pressure or {}).get("level") or "low")
        if level == "high":
            return (
                "Communication pressure guidance: high.\n"
                "Default to SEND: yes for one action-value message or one coverage-value discovery.\n"
                "Keep it to one grounded fact; do not turn high pressure into repeated table/resource chatter or generic status.\n"
            )
        if level == "medium":
            return (
                "Communication pressure guidance: medium.\n"
                "Lean SEND: yes for a compatible-table/request message or a fresh new-area fact; choose SEND: no for repeats or dense recent contact.\n"
            )
        return (
            "Communication pressure guidance: low.\n"
            "Low pressure is not automatic silence; use SEND: yes only for a clear action value or first concise new-area discovery.\n"
        )

    def _payload_first_gate_block(self, obs: dict[str, Any] | None, candidates: list[str]) -> str:
        data = dict(obs or {})
        carry_type, carry_qty = _inventory_resource_quantity(data)
        action_block = _current_round_action_prompt_block(data).strip()
        visible_resources = _compact_visible_resource_summary(data, max_lines=2)
        visible_tables = _compact_visible_table_summary(data, max_lines=2)
        lines = [
            "Payload check:",
            "- First identify the one concrete payload, then decide SEND.",
            "- Useful payloads: answer/ask a concrete need, share a compatible table for carried resources, or share one fresh visible/current resource/table.",
            "- Avoid repeated facts, generic status, and movement-only updates.",
            f"- Current carrying: {carry_type}:{carry_qty if carry_qty is not None else '?'}",
            f"- Visible resources now: {visible_resources}",
            f"- Visible tables now: {visible_tables}",
        ]
        if action_block:
            lines.append(f"- Current action context: {action_block.replace(chr(10), ' ')[:220]}")
        return "\n".join(lines) + "\n"

    async def _select_llm_single_target(self, ctx, candidate_names: list[str]):
        ordered_names, reorder_meta = self._reorder_candidates_for_basic_routing(ctx.obs, list(candidate_names), int(ctx.round_num))
        lowcost_gate = str(self.mode or "").lower() == "gate_llm"
        sampling_override = None
        try:
            sampler = getattr(self.owner.brain, "sampling_override_for", None)
            if callable(sampler):
                sampling_override = sampler("basic_routing")
        except Exception:
            sampling_override = None

        p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
        c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
        total_api_call_time = 0.0
        auto_send, auto_send_reason = self._should_auto_send(reorder_meta, int(ctx.round_num))
        if auto_send:
            send_prompt = None
            send_response = "AUTO_SEND: yes"
            send_api_call_time = 0.0
            send = True
            error_reason = None
        else:
            if lowcost_gate:
                send_prompt = self._build_original_gate_target_only_prompt(ctx.obs, ordered_names)
            else:
                send_prompt = self._build_original_target_only_prompt(ctx.obs, ordered_names)
            send_response, send_api_call_time = await self._run_basic_routing_prompt(send_prompt, sampling_override=sampling_override)
            total_api_call_time += send_api_call_time
            send, error_reason = self._parse_send_only_decision(send_response)

        routing_pressure = self._communication_pressure_profile(
            ctx.obs,
            ordered_names,
            list((reorder_meta or {}).get("features") or []),
        )
        send_veto_feature: dict[str, Any] | None = None
        send_veto_reason: str | None = None
        if lowcost_gate and send:
            send_veto_feature, send_veto_reason = self._sparse_gate_veto_after_send_yes(
                reorder_meta,
                int(ctx.round_num),
                routing_pressure,
            )
            if send_veto_reason:
                send = False
                error_reason = send_veto_reason
        send_override_feature: dict[str, Any] | None = None
        send_override_reason: str | None = None
        send_override_recipient: str | None = None
        send_soft_suggestion_feature: dict[str, Any] | None = None
        send_soft_suggestion_reason: str | None = None
        if lowcost_gate and not send:
            floor_feature, floor_reason = self._coverage_floor_feature_after_send_no(
                reorder_meta,
                int(ctx.round_num),
                routing_pressure,
                ctx.obs,
            )
            if floor_feature is not None:
                if self._should_apply_minimal_floor_override(floor_feature, floor_reason, int(ctx.round_num)):
                    name = str(floor_feature.get("name") or "").strip()
                    if name:
                        send = True
                        error_reason = None
                        send_override_feature = floor_feature
                        send_override_reason = str(floor_reason or "minimal_floor")
                        send_override_recipient = name
                else:
                    send_soft_suggestion_feature = floor_feature
                    send_soft_suggestion_reason = floor_reason

        meta = {
            "mode": self.mode or "basic",
            "route_strategy": "stable_llm_target_only",
            "route_strategy_variant": "two_step_send_then_target_lowcost" if lowcost_gate else "two_step_send_then_target",
            "routing_prompt": send_prompt,
            "routing_response": send_response,
            "routing_send_prompt": send_prompt,
            "routing_send_response": send_response,
            "routing_send_policy": (
                auto_send_reason
                or ("llm_send_gate_lowcost" if lowcost_gate else "llm_send_gate")
            ),
            "routing_send_override": send_override_reason,
            "routing_send_override_feature": self._coverage_floor_feature_summary(send_override_feature),
            "routing_send_soft_suggestion": send_soft_suggestion_reason,
            "routing_send_soft_suggestion_feature": self._coverage_floor_feature_summary(send_soft_suggestion_feature),
            "routing_send_veto": send_veto_reason,
            "routing_send_veto_feature": self._coverage_floor_feature_summary(send_veto_feature),
            "routing_api_call_time": total_api_call_time,
            "routing_tokens": {},
            "communication_pressure": routing_pressure,
            "communication_pressure_level": str(routing_pressure.get("level") or "low"),
            "communication_pressure_score": int(routing_pressure.get("score") or 0),
            "candidate_order": list(ordered_names),
            "candidate_order_original": list(candidate_names),
            "candidate_order_full": [
                str(feature.get("name") or "")
                for feature in list((reorder_meta or {}).get("features_full") or [])
                if str(feature.get("name") or "")
            ],
            "candidate_reorder_meta": reorder_meta,
        }

        if not send:
            meta["error_reason"] = error_reason or "model_send_no"
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            meta["routing_tokens"] = self._snapshot_brain_usage()
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)

        if send_override_recipient:
            select_prompt = None
            select_response = f"AUTO_TO: {send_override_recipient}"
            recipient = send_override_recipient
            error_reason = None
            meta["routing_select_prompt"] = select_prompt
            meta["routing_select_response"] = select_response
            meta["routing_select_policy"] = "minimal_floor_recipient"
            meta["routing_response"] = f"{send_response}\nSEND_OVERRIDE: {send_override_reason}\nTO: {recipient}".strip()
        elif lowcost_gate:
            select_prompt = self._build_original_gate_recipient_only_prompt(ctx.obs, ordered_names)
        else:
            select_prompt = self._build_original_recipient_only_prompt(ctx.obs, ordered_names)
        if not send_override_recipient:
            select_response, select_api_call_time = await self._run_basic_routing_prompt(select_prompt, sampling_override=sampling_override)
            total_api_call_time += select_api_call_time
            recipient, error_reason = self._parse_recipient_only_decision(select_response)
            normalized_recipient = self._normalize_candidate_name(recipient, ordered_names)
            if normalized_recipient:
                recipient = normalized_recipient
            meta["routing_api_call_time"] = total_api_call_time
            meta["routing_select_prompt"] = select_prompt
            meta["routing_select_response"] = select_response
            meta["routing_select_policy"] = "llm_select_gate_lowcost" if lowcost_gate else "llm_select_gate"
            meta["routing_response"] = f"{send_response}\n{select_response}".strip()

        legal_names = set(ordered_names)
        if not recipient:
            meta["error_reason"] = error_reason or "missing_recipient"
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            meta["routing_tokens"] = self._snapshot_brain_usage()
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)
        if recipient not in legal_names:
            meta["error_reason"] = "invalid_recipient"
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            meta["routing_tokens"] = self._snapshot_brain_usage()
            return RoutingDecision(send=False, recipients=[], mode=self.mode or "basic", meta=meta)

        if lowcost_gate:
            rebalanced_recipient, rebalance_reason = self._maybe_rebalance_recipient_for_coverage(
                str(recipient),
                reorder_meta,
                int(ctx.round_num),
                routing_pressure,
            )
            if rebalance_reason and rebalanced_recipient in legal_names:
                meta["routing_select_original_recipient"] = str(recipient)
                meta["routing_select_soft_rebalance_reason"] = rebalance_reason
                meta["routing_select_soft_rebalance_recipient"] = rebalanced_recipient
                recipient = rebalanced_recipient
                meta["routing_response"] = f"{meta.get('routing_response', '').strip()}\nREBALANCED_TO: {recipient}".strip()

        p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
        c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
        self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
        meta["routing_tokens"] = self._snapshot_brain_usage()
        for feature in list((reorder_meta or {}).get("features") or []) + list((reorder_meta or {}).get("features_full") or []):
            if not isinstance(feature, dict):
                continue
            if str(feature.get("name") or "").strip() != str(recipient or "").strip():
                continue
            routing_objective = _stable_fair_routing_objective_for_feature(feature)
            meta["routing_selected_recipient"] = str(recipient)
            meta["routing_selected_intent"] = str(feature.get("routing_intent") or "coverage")
            meta["routing_selected_hint"] = str(feature.get("routing_hint") or feature.get("routing_intent") or "coverage")
            meta["routing_selected_objective"] = routing_objective
            meta["routing_selected_brief"] = _stable_fair_routing_brief_for_feature(
                feature,
                recipient=str(recipient),
                pressure=str((routing_pressure or {}).get("level") or ""),
            )
            meta["routing_message_contract"] = _stable_fair_routing_message_contract_for_feature(
                feature,
                recipient=str(recipient),
                pressure=str((routing_pressure or {}).get("level") or ""),
            )
            meta["routing_selected_edge_reason"] = str(
                feature.get("edge_reason") or _stable_fair_routing_edge_reason_for_feature(feature)
            )
            meta["routing_selected_recipient_value"] = str(
                feature.get("recipient_value") or _stable_fair_routing_recipient_value_for_feature(feature)
            )
            meta["routing_selected_recipient_answerability"] = str(
                feature.get("recipient_answerability")
                or _stable_fair_routing_recipient_answerability_for_feature(feature)
            )
            meta["routing_selected_send_risk"] = str(
                feature.get("send_risk") or _stable_fair_routing_send_risk_for_feature(feature)
            )
            meta["routing_selected_edge"] = str(
                feature.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature)
            )
            meta["routing_selected_payload_state"] = str(
                feature.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature)
            )
            meta["routing_selected_audience_fit"] = str(
                feature.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature)
            )
            meta["routing_selected_latest_incoming_ask_type"] = str(feature.get("latest_incoming_ask_type") or "none")
            meta["routing_selected_focus_label"] = str(feature.get("best_fact_label") or "none")
            meta["routing_selected_focus_kind"] = str(feature.get("best_fact_kind") or "none")
            meta["routing_selected_backup_label"] = str(feature.get("backup_fact_label") or "none")
            meta["routing_selected_backup_kind"] = str(feature.get("backup_fact_kind") or "none")
            meta["routing_selected_carrying"] = str(feature.get("carrying") or "none")
            if feature.get("carrying_qty") is not None:
                meta["routing_selected_carrying_qty"] = feature.get("carrying_qty")
            break
        return RoutingDecision(send=True, recipients=[recipient], mode=self.mode or "basic", meta=meta)

    def _build_original_gate_target_only_prompt(self, obs: dict, candidates: list[str]) -> str:
        prompt = self._reference_target_prompt(obs, candidates)
        previous_block = "\n\nOutput EXACTLY two lines:\nSEND: yes|no\nTO: <one of the candidates>"
        new_block = "\n\nOutput EXACTLY one line:\nSEND: yes|no"
        if previous_block in prompt:
            prompt = prompt.replace(previous_block, new_block, 1)
        marker = "\n\nOutput EXACTLY one line:\nSEND: yes|no"
        round_num = int((obs or {}).get("round", 0) or 0)
        early_limit = int(self.EARLY_COVERAGE_ROUNDS)
        early_stage = round_num < early_limit
        stage_guidance = (
            f"Early exploration stage: yes (round < {early_limit}).\n"
            "Early visible discoveries can be useful if they add fresh map coverage.\n"
        ) if early_stage else (
            f"Early exploration stage: no (round >= {early_limit}).\n"
            "After early exploration, send only for concrete action value or fresh map coverage.\n"
        )
        routing_guidance = (
            "Decide only whether one useful payload is worth sending this round; do not choose a recipient yet.\n"
            "A useful payload helps a teammate act soon, answers/asks one concrete need, or adds one fresh visible/current map fact.\n"
            f"{stage_guidance}"
            "Answer SEND: no for repeated facts, generic status, unsupported updates, or movement-only messages.\n"
        )
        extra = f"{self._build_candidate_status_block_lowcost(obs, candidates)}{self._payload_first_gate_block(obs, candidates)}{routing_guidance}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"

    def _build_original_gate_recipient_only_prompt(self, obs: dict, candidates: list[str]) -> str:
        prompt = self._reference_target_prompt(obs, candidates)
        previous_block = "\n\nOutput EXACTLY two lines:\nSEND: yes|no\nTO: <one of the candidates>"
        new_block = "\n\nOutput EXACTLY one line:\nTO: <one of the candidates>"
        if previous_block in prompt:
            prompt = prompt.replace(previous_block, new_block, 1)
        marker = "\n\nOutput EXACTLY one line:\n"
        routing_guidance = (
            "A message will be sent this round.\n"
            "The candidate list is a neutral display order, not a ranked shortlist.\n"
            "Choose the teammate most likely to use or answer the payload.\n"
            "For fresh map coverage, prefer someone less recently contacted or likely missing that fact.\n"
        )
        extra = f"{self._build_candidate_status_block_lowcost(obs, candidates)}{self._payload_first_gate_block(obs, candidates)}{routing_guidance}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"

    def _build_original_target_only_prompt(self, obs: dict, candidates: list[str]) -> str:
        prompt = self._reference_target_prompt(obs, candidates)
        previous_block = "\n\nOutput EXACTLY two lines:\nSEND: yes|no\nTO: <one of the candidates>"
        new_block = "\n\nOutput EXACTLY one line:\nSEND: yes|no"
        if previous_block in prompt:
            prompt = prompt.replace(previous_block, new_block, 1)
        marker = "\n\nOutput EXACTLY one line:\nSEND: yes|no"
        round_num = int((obs or {}).get("round", 0) or 0)
        early_limit = int(self.EARLY_COVERAGE_ROUNDS)
        early_stage = round_num < early_limit
        pressure = self._communication_pressure_profile(obs, candidates)
        pressure_guidance = self._communication_pressure_guidance(pressure)
        stage_guidance = (
            f"Early exploration stage: yes (round < {early_limit}).\n"
            "In early exploration, allow one first-useful visible discovery when it adds coverage value.\n"
        ) if early_stage else (
            f"Early exploration stage: no (round >= {early_limit}).\n"
            "After early exploration, require action value or a fresh new-area/new-fact coverage value.\n"
        )
        routing_guidance = (
            "First decide whether sending any message is worthwhile this round.\n"
            "Do not choose or imply a recipient yet; this step only decides whether to send any message.\n"
            "Use only the compact cues above: edge, value, fact novelty, audience, and contact.\n"
            f"{pressure_guidance}"
            f"{stage_guidance}"
            "Answer SEND: yes only for action value or a fresh coverage fact that could matter to the recipient's next few actions.\n"
            "Answer SEND: no for generic status, repeated coordinates, unsupported updates, or local facts that are already shared.\n"
        )
        extra = f"{self._build_candidate_status_block(obs, candidates)}{routing_guidance}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"

    def _build_original_recipient_only_prompt(self, obs: dict, candidates: list[str]) -> str:
        prompt = self._reference_target_prompt(obs, candidates)
        previous_block = "\n\nOutput EXACTLY two lines:\nSEND: yes|no\nTO: <one of the candidates>"
        new_block = "\n\nOutput EXACTLY one line:\nTO: <one of the candidates>"
        if previous_block in prompt:
            prompt = prompt.replace(previous_block, new_block, 1)
        marker = "\n\nOutput EXACTLY one line:\n"
        pressure = self._communication_pressure_profile(obs, candidates)
        pressure_guidance = self._communication_pressure_guidance(pressure)
        routing_guidance = (
            "A message will be sent this round.\n"
            "The candidate list is a neutral display order, not a ranked shortlist.\n"
            "Choose the single recipient most likely to benefit from this payload.\n"
            "Use only the compact cues above: edge, value, fact novelty, audience, and contact.\n"
            f"{pressure_guidance}"
            "For action value, choose the agent who can use or answer it.\n"
            "For coverage value, prefer someone less recently contacted or likely missing that area/fact.\n"
        )
        extra = f"{self._build_candidate_status_block(obs, candidates)}{routing_guidance}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"


class StableFairProtocolMechanism(ProtocolMechanism):
    """StableFair protocol mechanism with only the needed Reference example kept."""

    def __init__(self, owner, config: dict[str, Any] | None = None):
        ProtocolMechanism.__init__(self, owner, config)
        self.mode = str(self.config.get("single_message_prompt_mode", "basic") or "basic").lower()
        self.protocol_v2_enabled = bool(self.config.get("protocol_v2_enabled", False))
        self.protocol_v2_auto = bool(self.config.get("protocol_v2_auto", True))
        self.ask_cooldown_rounds = int(self.config.get("ask_cooldown_rounds", 2) or 2)
        self.max_items_in_message = int(self.config.get("max_items_in_message", 5) or 5)
        self.max_tables_in_message = int(self.config.get("max_tables_in_message", 3) or 3)
        self.proto_v2_free_enable = bool(self.config.get("proto_v2_free_enable", False))
        self.proto_v2_free_slot = str(self.config.get("proto_v2_free_slot", "C") or "C")
        self.proto_v2_free_max = int(self.config.get("proto_v2_free_max", 150) or 150)
        self.proto_v2_header_max_len = int(self.config.get("proto_v2_header_max_len", 0) or 0)
        self.proto_v2_total_max_len = int(self.config.get("proto_v2_total_max_len", 220) or 220)

    def should_record_history_response(
        self,
        ctx: RoundContext,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> bool:
        del ctx, recipient, route_response, outgoing
        meta = dict(decision_meta or {})
        mode = str(meta.get("mode") or self.mode or "basic").strip().lower()
        return mode != "random"

    async def compose(
        self,
        ctx: RoundContext,
        recipient: str,
        memory_ctx: MemoryContext,
        routing_meta: dict[str, Any],
    ) -> OutgoingMessage:
        if self.protocol_v2_enabled and self.protocol_v2_auto:
            return await self._compose_original_protocol_v2(ctx, recipient)
        routing_meta = dict(routing_meta or {})
        draft_message = str((routing_meta or {}).get("draft_message") or "").strip()
        if draft_message:
            return self._compose_routing_passthrough(recipient, draft_message)
        return await self._compose_original_random_llm(ctx, recipient, memory_ctx)

    def parse_incoming(
        self,
        ctx: RoundContext,
        sender: str,
        raw_message: str,
    ) -> ParsedIncomingMessage:
        facts: dict[str, Any] = {"tables": [], "resources": [], "goal": None}
        text = str(raw_message or "")
        try:
            parser = getattr(self.owner, "_kb_parse_messages", None)
            if callable(parser):
                entries = parser(
                    obs_messages=[{"agent_name": sender, "message": text, "round": max(0, int(ctx.round_num) - 1)}],
                    msgs_dict={sender: text},
                    round_num=int(ctx.round_num),
                )
                for pos, patch, _msg_sender, _evidence in entries or []:
                    if not (isinstance(pos, (tuple, list)) and len(pos) == 2 and isinstance(patch, dict)):
                        continue
                    x, y = int(pos[0]), int(pos[1])
                    terrain = str(patch.get("terrain_type") or "").lower()
                    if terrain == "table":
                        facts["tables"].append({"x": x, "y": y, "remaining_capacity": patch.get("remaining_capacity")})
                    items = patch.get("items") or {}
                    if isinstance(items, dict):
                        for item_type, qty in items.items():
                            try:
                                facts["resources"].append({"x": x, "y": y, "type": str(item_type), "quantity": int(qty)})
                            except Exception:
                                continue
                    if terrain == "goal" and facts.get("goal") is None:
                        facts["goal"] = {"x": x, "y": y}
            if facts.get("goal") is None:
                match = re.search(r"My\s+goal\s+is\s*\(?(\d+)\s*,\s*(\d+)\)?", text, re.IGNORECASE)
                if match:
                    facts["goal"] = {"x": int(match.group(1)), "y": int(match.group(2))}
        except Exception:
            pass
        return ParsedIncomingMessage(
            sender=sender,
            raw_message=text,
            parsed_facts=facts,
            meta={
                "tables_count": len(facts.get("tables") or []),
                "resources_count": len(facts.get("resources") or []),
                "has_goal": bool(facts.get("goal")),
                "protocol_profile": "stable_fair_reference",
            },
        )

    def format_history_response(
        self,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> str:
        route_response = str(route_response or "").strip()
        content_response = str(getattr(outgoing, "raw_response", "") or "").strip()
        outgoing_message = str(getattr(outgoing, "message", "") or "").strip()
        meta = dict(decision_meta or {})
        protocol_action = str(meta.get("protocol_action") or "")
        if protocol_action == "original_protocol_v2_structured":
            reason = str(meta.get("reason") or "proto_v2_structured")
            return f"SEND: yes\nTO: {recipient}\nMSG:\n{outgoing_message}\n[PROTO_V2_APPLIED] reason={reason}"
        if route_response:
            return route_response
        return content_response or outgoing_message

    async def _compose_original_protocol_v2(self, ctx: RoundContext, recipient: str) -> OutgoingMessage:
        header_message = self._build_original_protocol_v2_message(ctx.obs)
        final_message = header_message
        llm_generated_text = ""
        api_call_time = 0.0
        raw_response = "[ORIGINAL_FAITHFUL_PROTO_V2]"
        prompt = ""
        if self.proto_v2_free_enable and str(header_message or "").strip():
            free_prompt = self._build_proto_v2_free_prompt(self.owner.name, header_message, ctx.obs, self.proto_v2_free_max)
            prompt = free_prompt
            free_resp = ""
            try:
                t0 = time.time()
                p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                free_resp = await self.owner.brain.generate(free_prompt)
                api_call_time = time.time() - t0
                p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
                self._jump_back()
            except Exception as exc:
                free_resp = f"[FREE_SLOT_ERROR] {exc}"
            free_line = self._sanitize_free_line(free_resp, self.proto_v2_free_max)
            if free_line:
                llm_generated_text = free_line
                final_message = self._append_free_slot_text(header_message, free_line)
            raw_response = f"{raw_response}\nFREE:{free_resp}"
        return OutgoingMessage(
            recipient=recipient,
            send=bool(str(final_message or "").strip()),
            prompt=prompt,
            raw_response=raw_response,
            message=str(final_message or "").strip(),
            meta={
                "mode": self.mode or "basic",
                "protocol_profile": "stable_fair_reference",
                "protocol_action": "original_protocol_v2_structured",
                "reason": "proto_v2_structured",
                "llm_generated_text": llm_generated_text,
                "deterministic_text": str(header_message or "").strip(),
                "protocol_free_enabled": bool(self.proto_v2_free_enable),
                "protocol_free_used": bool(llm_generated_text),
            },
            api_call_time=api_call_time,
            tokens=self._snapshot_brain_usage(),
        )

    def _build_original_protocol_v2_message(self, obs: dict) -> str:
        max_tables = max(1, self.max_tables_in_message)
        pos = obs.get("position", {}) or {}
        x, y = int(pos.get("x", 0) or 0), int(pos.get("y", 0) or 0)
        carrying_qty = self.owner.get_total_carrying() if hasattr(self.owner, "get_total_carrying") else 0
        carry_type = self.owner.get_carrying_item_type() if hasattr(self.owner, "get_carrying_item_type") else None

        contrib = obs.get("my_contribution") or {}
        c_stone = _safe_int(contrib.get("stone"), 0)
        c_sand = _safe_int(contrib.get("sand"), 0)
        c_wood = _safe_int(contrib.get("wood"), 0)

        table_items: list[tuple[int, int, int]] = []
        try:
            cand_tabs = list(self.owner._collect_candidate_tables(obs, k=max_tables * 3, prefer_fit=True) or [])
        except Exception:
            cand_tabs = []
        for entry in cand_tabs:
            pos_xy = entry.get("pos") or (None, None)
            rem = int(entry.get("remaining") or 0)
            if pos_xy[0] is None or pos_xy[1] is None or rem <= 0:
                continue
            table_items.append((int(pos_xy[0]), int(pos_xy[1]), rem))
            if len(table_items) >= max_tables:
                break

        goal_xy = (table_items[0][0], table_items[0][1]) if table_items else None
        dir_token = "STAY"
        if goal_xy is not None:
            dir_token = self._dir_from_to(x, y, goal_xy[0], goal_xy[1])

        lines = [
            f"I am {self.owner.name}.",
            f"My location is ({x},{y}).",
            f"I have contributed stone {c_stone}, sand {c_sand}, wood {c_wood}.",
        ]
        if carry_type and carrying_qty > 0:
            lines.append(f"I am carrying {carry_type}: {carrying_qty}.")
        else:
            lines.append("I am not carrying anything.")
        for tx, ty, rem in table_items:
            lines.append(f"Table at ({tx},{ty}) has remaining capacity {rem}.")
        if goal_xy is not None:
            gx, gy = goal_xy
            lines.append(f"My plan is to move {dir_token} toward table at ({gx},{gy}).")
            lines.append(f"My goal is ({gx},{gy}).")
        return "\n".join(lines)

    def _build_proto_v2_free_prompt(self, agent_name: str, header_msg: str, obs: dict, max_chars: int) -> str:
        try:
            pos = obs.get("position", {}) or {}
            x, y = int(pos.get("x", 0) or 0), int(pos.get("y", 0) or 0)
        except Exception:
            x, y = 0, 0
        lines = [
            f"You are {agent_name}. Write ONE short sentence (single line) in natural language.",
            "This sentence will be appended after the protocol header.",
            "Express your plan, goal, or a brief request/comment in plain words.",
            "Do not add prefixes like SEND/TO/MSG/PLAN/GOAL/REQ/C.",
            f"Keep it under {int(max_chars or 150)} characters.",
            "Do NOT repeat the full header or coordinates verbatim unless truly necessary.",
            f"My position: {x},{y}",
            f"Header: {header_msg}",
        ]
        return "\n".join(lines)

    def _sanitize_free_line(self, text: str, max_chars: int) -> str:
        if not isinstance(text, str):
            return ""
        line = text.splitlines()[0] if "\n" in text else text
        for prefix in ("SEND:", "send:", "TO:", "to:", "MSG:", "msg:"):
            if line.strip().startswith(prefix):
                try:
                    line = line.split(":", 1)[1].strip()
                except Exception:
                    pass
        line = " ".join(line.strip().split())
        try:
            limit = int(max_chars or 0)
        except Exception:
            limit = 0
        if limit > 0 and len(line) > limit:
            line = line[:limit].rstrip()
        return line

    def _append_free_slot_text(self, header_msg: str, free_line: str) -> str:
        if self.proto_v2_free_slot.upper() == "REQ" and "REQ:none" in header_msg:
            return header_msg.replace("REQ:none", free_line, 1)
        return (str(header_msg or "").rstrip() + " " + str(free_line or "").strip()).strip()

    def _dir_from_to(self, sx: int, sy: int, gx: int, gy: int) -> str:
        dx = gx - sx
        dy = gy - sy
        horiz = "E" if dx > 0 else ("W" if dx < 0 else "")
        vert = "N" if dy > 0 else ("S" if dy < 0 else "")
        if vert and horiz:
            return vert + horiz
        return vert or horiz or "STAY"

    def _compose_routing_passthrough(self, recipient: str, draft_message: str) -> OutgoingMessage:
        return OutgoingMessage(
            recipient=recipient,
            send=bool(draft_message.strip()),
            prompt="",
            raw_response="[ROUTING_MSG_USED]",
            message=draft_message.strip(),
            meta={
                "mode": self.mode or "basic",
                "protocol_profile": "stable_fair_reference",
                "protocol_action": "passthrough_routing_message",
                "llm_generated_text": draft_message.strip(),
                "deterministic_text": "",
            },
            api_call_time=0.0,
            tokens=self._snapshot_brain_usage(),
        )

    async def _compose_original_random_llm(
        self,
        ctx: RoundContext,
        recipient: str,
        memory_ctx: MemoryContext,
    ) -> OutgoingMessage:
        del memory_ctx
        target_info = (ctx.obs.get("agents_in_communication_range") or {}).get(recipient, {}) or {}
        prompt = self._build_original_random_target_prompt(ctx.obs, recipient, target_info)
        response = ""
        api_call_time = 0.0
        try:
            t0 = time.time()
            p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            response = await self.owner.brain.generate(prompt)
            api_call_time = time.time() - t0
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            self._jump_back()
        except Exception as exc:
            response = f"MSG: generation_error {exc}"
        message_only = self._parse_msg_only(response)
        final_msg = message_only.strip() if message_only.strip() else str(response).strip()
        if not final_msg:
            final_msg = "Hello."
        return OutgoingMessage(
            recipient=recipient,
            send=bool(final_msg.strip()),
            prompt=prompt,
            raw_response=response,
            message=final_msg.strip(),
            meta={
                "mode": self.mode or "random",
                "protocol_profile": "stable_fair_reference",
                "llm_generated_text": final_msg.strip(),
                "deterministic_text": "",
            },
            api_call_time=api_call_time,
            tokens=self._snapshot_brain_usage(),
        )

    def _build_original_random_target_prompt(self, obs: dict, target_name: str, target_info: dict | None = None) -> str:
        round_num = obs.get("round", 0)
        pos = obs.get("position", {}) or {}
        x, y = pos.get("x"), pos.get("y")
        target_info = target_info or {}
        in_view = bool(target_info.get("in_view_range", False))
        tpos = target_info.get("position", {}) or {}
        tx, ty = tpos.get("x"), tpos.get("y")
        carrying = target_info.get("carrying_resource")

        lines = [
            f"You are {self.owner.name}. This is MESSAGE PHASE for round {round_num}.",
            f"Your position: ({x},{y}). Target: {target_name}.",
            f"Target position: ({tx},{ty}) | In view: {in_view} | Carrying: {carrying if carrying else 'unknown'}",
            "Task: Collaborate with others to achieve the goal.",
            "",
            "Output EXACTLY one line:",
            "MSG: <your message text>",
        ]
        return "\n".join(lines)

    def _parse_msg_only(self, response: str) -> str:
        try:
            for line in str(response or "").splitlines():
                raw = line.strip()
                if raw.lower().startswith("msg:"):
                    return raw.split(":", 1)[1].strip()
        except Exception:
            return ""
        return ""

    def _accumulate_message_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        try:
            self.owner.brain.message_prompt_tokens = int(getattr(self.owner.brain, "message_prompt_tokens", 0) or 0) + int(prompt_tokens)
            self.owner.brain.message_completion_tokens = int(getattr(self.owner.brain, "message_completion_tokens", 0) or 0) + int(completion_tokens)
        except Exception:
            pass

    def _snapshot_brain_usage(self) -> dict[str, Any]:
        usage = getattr(self.owner.brain, "usage", {})
        if isinstance(usage, dict):
            return dict(usage)
        if isinstance(usage, (int, float)):
            return {"total": int(usage)}
        return {}

    def _jump_back(self) -> None:
        try:
            if hasattr(self.owner.brain, "jump_back"):
                self.owner.brain.jump_back()
        except Exception:
            pass

    _reference_should_record_history_response = should_record_history_response
    _reference_compose = compose
    _reference_parse_incoming = parse_incoming
    _reference_format_history_response = format_history_response
    _reference_build_original_random_target_prompt = _build_original_random_target_prompt
    _reference_build_proto_v2_free_prompt = _build_proto_v2_free_prompt

    def _build_original_random_target_prompt(self, obs: dict, target_name: str, target_info: dict | None = None) -> str:
        prompt = self._reference_build_original_random_target_prompt(obs, target_name, target_info)
        action_block = _plain_current_round_action_context(obs)
        guidance = _plain_basic_target_prompt_guidance()
        marker = "\nTask: Collaborate with others to achieve the goal.\n"
        if action_block and marker in prompt:
            return prompt.replace(marker, f"{marker}{action_block}{guidance}", 1)
        if marker in prompt:
            return prompt.replace(marker, f"{marker}{guidance}", 1)
        if action_block:
            return f"{prompt}{action_block}{guidance}"
        return f"{prompt}\n{guidance}"

    def _build_proto_v2_free_prompt(self, agent_name: str, header_msg: str, obs: dict, max_chars: int) -> str:
        prompt = self._reference_build_proto_v2_free_prompt(agent_name, header_msg, obs, max_chars)
        action_block = _current_round_action_prompt_block(obs).strip()
        if not action_block:
            return prompt
        marker = f"Header: {header_msg}"
        extra = (
            f"{action_block}\n"
            "Use the appended sentence to complement the header with one useful local fact or one specific request.\n"
            "Do not merely restate the move or repeat information already present in the header.\n"
        )
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}\n{extra}"

    def format_history_response(
        self,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> str:
        meta = dict(decision_meta or {})
        if str(meta.get("route_strategy") or "") == "stable_llm_target_only" and not meta.get("protocol_action"):
            outgoing_message = str(getattr(outgoing, "message", "") or "").strip()
            if outgoing_message:
                return f"SEND: yes\nTO: {recipient}\nMSG: {outgoing_message}"
        return self._reference_format_history_response(recipient, route_response, outgoing, decision_meta)

    _reference_plain_history_response = format_history_response
    _reference_plain_target_prompt = _build_original_random_target_prompt

    PROTOCOL_PROFILE = "stable_fair_compact_v2"
    SEMANTIC_PROTOCOL_PROFILE = "stable_fair_semantic_v1"
    LLM_SELECT_PROTOCOL_PROFILE = "protocol_llm_select_v1"
    PROMPT_PROTOCOL_PROFILE = "semantic_state_obs_prompt_v1"
    LLM_SELECT_PROTOCOL_MODES = {"gate_llm"}
    DUPLICATE_FACT_LOOKBACK_ROUNDS = 4

    def _disabled_by_preset(self) -> bool:
        return bool(self.config.get("base_no_mechanisms") or self.config.get("disabled"))

    def _stable_fair_recent_contact_records(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        records = list((obs or {}).get("recent_messages_sent") or [])
        if records:
            return [rec for rec in records if isinstance(rec, dict)]
        try:
            memory = getattr(getattr(self.owner, "mechanisms", None), "memory", None)
            return [rec for rec in list(getattr(memory, "_sent", []) or []) if isinstance(rec, dict)]
        except Exception:
            return []

    def _stable_fair_recent_received_records(self, obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        records = list((obs or {}).get("recent_messages_received") or [])
        if records:
            return [rec for rec in records if isinstance(rec, dict)]
        try:
            memory = getattr(getattr(self.owner, "mechanisms", None), "memory", None)
            return [rec for rec in list(getattr(memory, "_recv", []) or []) if isinstance(rec, dict)]
        except Exception:
            return []

    def _stable_fair_message_mentions_fact(self, text: str, fact: dict[str, Any] | None) -> bool:
        if not isinstance(fact, dict):
            return False
        raw_text = str(text or "").lower()
        pos = tuple(fact.get("position") or ())
        if len(pos) != 2:
            return False
        x, y = int(pos[0]), int(pos[1])
        coord_pattern = re.compile(rf"(?:\(\s*{x}\s*,\s*{y}\s*\)|@\s*{x}\s*,\s*{y}\b)")
        if coord_pattern.search(raw_text) is None:
            return False
        if str(fact.get("kind") or "") == "table":
            return "table" in raw_text
        resource_type = str(fact.get("resource_type") or "").strip().lower()
        return bool(resource_type) and resource_type in raw_text

    def _stable_fair_fact_recent_message_count(
        self,
        obs: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        round_num: int,
        *,
        window: int | None = None,
        include_received: bool = True,
    ) -> int:
        if not isinstance(fact, dict):
            return 0
        lookback = int(window if window is not None else self.DUPLICATE_FACT_LOOKBACK_ROUNDS)
        records: list[dict[str, Any]] = list(self._stable_fair_recent_contact_records(obs))
        if include_received:
            records.extend(self._stable_fair_recent_received_records(obs))
            for rec in list((obs or {}).get("messages") or []):
                if isinstance(rec, dict):
                    records.append(rec)
        count = 0
        seen: set[tuple[int | None, str, str, str]] = set()
        for rec in records:
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    continue
                if int(round_num) - int(rec_round) > lookback:
                    continue
            text = str(rec.get("message") or rec.get("text") or "")
            key = (
                rec_round,
                str(rec.get("to") or ""),
                str(rec.get("from") or rec.get("agent_name") or ""),
                text,
            )
            if key in seen:
                continue
            seen.add(key)
            if self._stable_fair_message_mentions_fact(text, fact):
                count += 1
        return count

    def _stable_fair_fact_recently_sent_to_recipient(
        self,
        obs: dict[str, Any] | None,
        recipient: str,
        fact: dict[str, Any] | None,
        round_num: int,
        *,
        window: int | None = None,
    ) -> bool:
        if not isinstance(fact, dict):
            return False
        lookback = int(window if window is not None else self.DUPLICATE_FACT_LOOKBACK_ROUNDS)
        recipient_norm = str(recipient or "").strip()
        seen: set[tuple[int | None, str]] = set()
        for rec in self._stable_fair_recent_contact_records(obs):
            if str(rec.get("to") or "").strip() != recipient_norm:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    continue
                if int(round_num) - int(rec_round) > lookback:
                    continue
            text = str(rec.get("message") or rec.get("text") or "")
            key = (rec_round, text)
            if key in seen:
                continue
            seen.add(key)
            if self._stable_fair_message_mentions_fact(text, fact):
                return True
        return False

    def _stable_fair_fact_recently_received_from_recipient(
        self,
        obs: dict[str, Any] | None,
        recipient: str,
        fact: dict[str, Any] | None,
        round_num: int,
        *,
        window: int | None = None,
    ) -> bool:
        if not isinstance(fact, dict):
            return False
        lookback = int(window if window is not None else self.DUPLICATE_FACT_LOOKBACK_ROUNDS)
        recipient_norm = str(recipient or "").strip()
        records = list(self._stable_fair_recent_received_records(obs))
        records.extend(rec for rec in list((obs or {}).get("messages") or []) if isinstance(rec, dict))
        seen: set[tuple[int | None, str]] = set()
        for rec in records:
            sender = str(rec.get("from") or rec.get("agent_name") or "").strip()
            if sender != recipient_norm:
                continue
            rec_round = _coerce_int(rec.get("round"))
            if rec_round is not None:
                if int(rec_round) >= int(round_num):
                    continue
                if int(round_num) - int(rec_round) > lookback:
                    continue
            text = str(rec.get("message") or rec.get("text") or "")
            key = (rec_round, text)
            if key in seen:
                continue
            seen.add(key)
            if self._stable_fair_message_mentions_fact(text, fact):
                return True
        return False

    def should_record_history_response(
        self,
        ctx: RoundContext,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> bool:
        meta = dict(decision_meta or {})
        mode = str(meta.get("mode") or self.mode or "").strip().lower()
        if mode == "gate_llm" or str(meta.get("route_strategy") or "") == "stable_llm_target_only":
            return False
        return self._reference_should_record_history_response(ctx, recipient, route_response, outgoing, meta)

    async def compose(
        self,
        ctx: RoundContext,
        recipient: str,
        memory_ctx: MemoryContext,
        routing_meta: dict[str, Any],
    ) -> OutgoingMessage:
        if self._disabled_by_preset():
            del ctx, memory_ctx, routing_meta
            return OutgoingMessage(
                recipient=recipient,
                send=False,
                message="",
                meta={"mode": "base", "disabled_by_preset": True, "error_reason": "base_no_mechanisms"},
            )

        previous_routing_meta = getattr(self, "_active_routing_meta", None)
        previous_routing_recipient = getattr(self, "_active_routing_recipient", "")
        self._active_routing_meta = dict(routing_meta or {})
        self._active_routing_recipient = str(recipient or "")
        try:
            outgoing = await self._reference_compose(ctx, recipient, memory_ctx, routing_meta)
        finally:
            self._active_routing_meta = previous_routing_meta
            self._active_routing_recipient = previous_routing_recipient

        protocol_meta = dict(getattr(self, "_last_stable_fair_protocol_meta", {}) or {})
        if protocol_meta:
            meta = dict(outgoing.meta or {})
            meta.update(protocol_meta)
            outgoing.meta = meta

        focus_feature = self._routing_focus_feature(routing_meta, recipient)
        if not focus_feature:
            return outgoing
        meta = dict(outgoing.meta or {})
        meta["routing_focus_label"] = self._routing_fact_label(focus_feature.get("best_fact"))
        meta["routing_focus_kind"] = str(focus_feature.get("best_fact_kind") or "")
        outgoing.meta = meta
        draft_message = str((routing_meta or {}).get("draft_message") or "").strip()
        if not self.protocol_v2_enabled and not draft_message:
            return outgoing
        return outgoing

    async def _compose_original_protocol_v2(self, ctx: RoundContext, recipient: str) -> OutgoingMessage:
        return await self._compose_stable_fair_semantic_protocol(ctx, recipient)

    def _use_llm_selected_protocol(self) -> bool:
        routing_meta = getattr(self, "_active_routing_meta", None)
        mode = str((routing_meta or {}).get("mode") or self.mode or "").strip().lower()
        return bool(self.protocol_v2_enabled) and mode in self.LLM_SELECT_PROTOCOL_MODES

    def _stable_fair_routing_message_contract(
        self,
        routing_meta: dict[str, Any] | None,
        feature: dict[str, Any] | None,
        recipient: str,
    ) -> dict[str, Any]:
        contract = (routing_meta or {}).get("routing_message_contract") if isinstance(routing_meta, dict) else None
        if isinstance(contract, dict):
            recipient_text = str(contract.get("recipient") or recipient or "").strip()
            if not recipient_text or recipient_text == str(recipient or "").strip():
                return dict(contract)
        pressure = str((routing_meta or {}).get("communication_pressure_level") or "") if isinstance(routing_meta, dict) else ""
        return _stable_fair_routing_message_contract_for_feature(
            feature,
            recipient=str(recipient or ""),
            pressure=pressure,
        )

    async def _compose_stable_fair_semantic_protocol(
        self,
        ctx: RoundContext,
        recipient: str,
    ) -> OutgoingMessage:
        return await self._compose_stable_fair_state_obs_protocol(ctx, recipient)

    async def _compose_stable_fair_state_obs_protocol(
        self,
        ctx: RoundContext,
        recipient: str,
    ) -> OutgoingMessage:
        return await self._compose_stable_fair_state_obs_protocol_simple(ctx, recipient)

    async def _compose_stable_fair_state_obs_protocol_simple(
        self,
        ctx: RoundContext,
        recipient: str,
    ) -> OutgoingMessage:
        data = dict(ctx.obs or {})
        routing_meta = getattr(self, "_active_routing_meta", None)
        feature = self._routing_selected_feature(routing_meta, recipient)
        routing_contract = (
            self._stable_fair_routing_message_contract(routing_meta, feature, recipient)
            if isinstance(feature, dict)
            else {}
        )
        recipient_evidence = self._stable_fair_protocol_has_recipient_evidence(feature, routing_contract)
        carry_type, carry_qty = _inventory_resource_quantity(data)
        state_line = self._stable_fair_auto_state_line(
            data,
            carry_type=carry_type,
            carry_qty=carry_qty,
            goal=data.get("goal_position"),
        )
        candidate_facts = self._stable_fair_protocol_lane_candidate_facts(
            data,
            recipient=recipient,
            feature=feature,
            routing_contract=routing_contract,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        content_candidate_lines = self._stable_fair_protocol_content_candidate_lines(
            data,
            recipient=recipient,
            feature=feature,
            routing_contract=routing_contract,
            candidate_facts=candidate_facts,
            carry_type=carry_type,
            carry_qty=carry_qty,
            recipient_evidence=recipient_evidence,
        )
        candidate_fact_lines = self._stable_fair_protocol_lane_fact_lines(candidate_facts[:4])
        delivery_context_lines = self._stable_fair_delivery_context_lines(
            data,
            recipient=recipient,
            feature=feature,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        prompt = self._build_stable_fair_state_obs_message_prompt(
            data,
            recipient,
            state_line=state_line,
            content_candidate_lines=content_candidate_lines,
            candidate_fact_lines=candidate_fact_lines,
            delivery_context_lines=delivery_context_lines,
            feature=feature,
            routing_contract=routing_contract,
            recipient_evidence=recipient_evidence,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        free_resp = ""
        api_call_time = 0.0
        protocol_generate_error = ""
        if self.proto_v2_free_enable and prompt:
            try:
                t0 = time.time()
                p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                free_resp = await self.owner.brain.generate(prompt)
                api_call_time += time.time() - t0
                p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
                self._jump_back()
            except Exception as exc:
                free_resp = f"[STATE_OBS_PROTOCOL_ERROR] {exc}"
                protocol_generate_error = "state_obs_protocol_error"
        if protocol_generate_error:
            response_lane, final_message = "STATUS", ""
        else:
            response_lane, final_message = self._parse_stable_fair_protocol_lane_response(
                free_resp,
                max_chars=min(120, int(self.proto_v2_free_max or 120)),
            )
        llm_generated_text = final_message
        fallback_used = False
        protocol_reject_reason = ""
        protocol_guard_warning = ""
        selected_lane = response_lane or self._stable_fair_protocol_lane_for_message(final_message)
        obs_facts = self._stable_fair_protocol_obs_facts_for_plain_message(
            selected_lane,
            final_message,
            candidate_facts,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        primary_fact: dict[str, Any] = dict(obs_facts[0]) if obs_facts else {"kind": "status"}
        obs_line = self._stable_fair_protocol_obs_block_for_facts(selected_lane, obs_facts)
        deterministic_text = ""
        raw_response = f"[STABLE_FAIR_STATE_OBS_PROMPT]\nRESPONSE:{free_resp}"
        if not final_message:
            protocol_reject_reason = protocol_generate_error or "empty_message"
        elif not obs_facts and self._stable_fair_message_has_observable_protocol_claim(selected_lane, final_message):
            protocol_guard_warning = "obs_not_grounded_for_message"
        elif not self._stable_fair_state_obs_message_safe(
            final_message,
            state_line=state_line,
            obs_line=obs_line,
            obs=data,
            recipient=recipient,
            candidate_facts=candidate_facts,
        ):
            protocol_guard_warning = "obs_not_grounded_for_message"
            obs_facts = []
            primary_fact = {"kind": "status"}
            obs_line = "OBS_STATUS: none"
        protocol_downgraded_to_plain = False

        semantic_type = self._infer_stable_fair_semantic_type(final_message)
        if (
            semantic_type == "STATUS"
            and final_message
            and not fallback_used
            and selected_lane in {"RESOURCE", "TABLE", "CARRY", "REQUEST"}
        ):
            semantic_type = selected_lane
        if not obs_facts and str(primary_fact.get("kind") or "") in {"resource", "table"}:
            obs_facts = [dict(primary_fact)]
        ask = "none"
        if semantic_type == "REQUEST":
            match = re.search(r"\b(?:wood|sand|stone)\b", final_message.lower())
            if match:
                ask = f"table_for={match.group(0)}"
        intent = self._stable_fair_protocol_intent_for_semantic(
            semantic_type,
            ask=ask,
            fact=primary_fact,
            carry_type=carry_type,
        )
        if not final_message:
            wire_message = ""
        else:
            wire_message = "\n".join([state_line, obs_line, f"MSG: {final_message}"]).strip()
        routing_context_meta = self._stable_fair_delivery_context_meta(feature)
        recipient_context_used = bool(
            str(recipient or "") in dict(data.get("agents_in_communication_range") or {})
        )
        selected_payload = {
            "type": semantic_type,
            "selected_type": semantic_type,
            "intent": intent,
            "fact": primary_fact,
            "obs_facts": list(obs_facts),
            "lane": selected_lane,
            "carry_type": carry_type,
            "carry_qty": carry_qty,
            "ask": ask,
            "recipient": recipient,
            "selected_message": final_message,
            "llm_generated": bool(llm_generated_text),
            "fallback_used": bool(fallback_used),
            "routing_contract": dict(routing_contract),
        }
        meta = {
            "mode": str((routing_meta or {}).get("mode") or self.mode or "basic"),
            "protocol_profile": self.SEMANTIC_PROTOCOL_PROFILE,
            "protocol_prompt_profile": self.PROMPT_PROTOCOL_PROFILE,
            "protocol_action": "stable_fair_state_obs_prompt_grounded",
            "protocol_intent": intent,
            "protocol_table_for_type": "none",
            "protocol_table_for_qty": None,
            "protocol_goal": None,
            "protocol_ask": ask,
            "protocol_fact": primary_fact,
            "protocol_obs_facts": list(obs_facts),
            "protocol_recipient": recipient,
            "protocol_used_routing_intent": False,
            "protocol_recipient_context_used": bool(recipient_context_used),
            "protocol_routing_context_used": bool(routing_context_meta),
            "protocol_routing_context": routing_context_meta,
            "protocol_routing_contract": dict(routing_contract),
            "protocol_recipient_evidence_used": bool(recipient_evidence),
            "protocol_semantic_payload": selected_payload,
            "protocol_lane": selected_lane,
            "protocol_response_lane": selected_lane,
            "protocol_content_anchor_candidates": [],
            "protocol_content_candidate_lines": list(content_candidate_lines),
            "protocol_selected_anchor_id": "",
            "protocol_selected_anchor_kind": "",
            "protocol_selected_anchor_context_source": "",
            "protocol_selected_anchor_context_reason": "",
            "reason": "state_obs_protocol_message_prompt",
            "llm_generated_text": llm_generated_text,
            "deterministic_text": deterministic_text,
            "protocol_free_enabled": bool(self.proto_v2_free_enable),
            "protocol_free_used": bool(llm_generated_text),
            "protocol_fallback_used": bool(fallback_used),
            "protocol_repair_used": False,
            "protocol_repair_reason": "",
            "protocol_message_prompt": prompt,
            "protocol_message_response": str(free_resp or "").strip(),
            "protocol_message_prompt_used": bool(prompt),
            "protocol_semantic_type": semantic_type,
            "protocol_response_type": semantic_type,
            "protocol_response_focus": "",
            "protocol_response_ask": ask,
            "protocol_reject_reason": protocol_reject_reason,
            "protocol_guard_warning": protocol_guard_warning,
            "protocol_downgraded_to_plain": False,
            "protocol_state_obs_enabled": True,
            "protocol_state_obs_sent": bool(final_message),
            "protocol_contract_guard_rejected": bool(protocol_reject_reason),
            "protocol_state_obs_soft_guarded": bool(protocol_guard_warning),
            "protocol_state_line": state_line,
            "protocol_obs_line": obs_line,
            "protocol_wire_message": wire_message,
            "protocol_selected_focus_id": "",
        }
        self._last_stable_fair_protocol_meta = dict(meta)
        return OutgoingMessage(
            recipient=recipient,
            send=bool(str(wire_message or "").strip()),
            prompt=prompt,
            raw_response=raw_response,
            message=wire_message,
            meta=meta,
            api_call_time=api_call_time,
            tokens=self._snapshot_brain_usage(),
        )

        data = dict(ctx.obs or {})
        routing_meta = getattr(self, "_active_routing_meta", None)
        feature = self._routing_selected_feature(routing_meta, recipient)
        routing_contract = self._stable_fair_routing_message_contract(routing_meta, feature, recipient)
        carry_type, carry_qty = _inventory_resource_quantity(data)
        content_anchors = self._stable_fair_protocol_brief_anchors(
            data,
            recipient=recipient,
            feature=feature,
            routing_contract=routing_contract,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        content_anchors = self._stable_fair_protocol_numbered_anchors(content_anchors)

        def materialize_anchor(anchor: dict[str, Any] | None) -> tuple[
            dict[str, Any],
            str,
            str,
            str,
            int | None,
            str,
        ]:
            anchor_dict = dict(anchor or {})
            anchor_fact = dict(anchor_dict.get("fact") or {"kind": "status"})
            anchor_ask = str(anchor_dict.get("ask") or "none")
            anchor_type = str(anchor_dict.get("semantic_type") or "STATUS").strip().upper()
            anchor_table_for_type = str(anchor_dict.get("table_for_type") or "none").strip().lower()
            anchor_table_for_qty = _coerce_int(anchor_dict.get("table_for_qty"))
            anchor_carry_type = (
                anchor_table_for_type
                if anchor_type == "TABLE_FOR_CARRY" and anchor_table_for_type != "none"
                else carry_type
            )
            anchor_carry_qty = (
                anchor_table_for_qty
                if anchor_type == "TABLE_FOR_CARRY" and anchor_table_for_type != "none"
                else carry_qty
            )
            anchor_fallback = str(anchor_dict.get("message") or "").strip()
            if not anchor_fallback:
                anchor_fallback = self._stable_fair_semantic_fallback_message(
                    semantic_type=anchor_type,
                    fact=anchor_fact,
                    carry_type=anchor_carry_type,
                    carry_qty=anchor_carry_qty,
                    ask=anchor_ask,
                )
            anchor_fallback = self._stable_fair_protocol_compound_fallback_message(
                anchor_fallback,
                anchor_dict,
                content_anchors,
                carry_type=carry_type,
                carry_qty=carry_qty,
            )
            return (
                anchor_fact,
                anchor_ask,
                anchor_type,
                anchor_table_for_type,
                anchor_table_for_qty,
                anchor_fallback,
            )

        selected_anchor = dict(content_anchors[0]) if content_anchors else {}
        fact, ask, semantic_type, table_for_type, table_for_qty, fallback_message = materialize_anchor(selected_anchor)
        intent = self._stable_fair_protocol_intent_for_semantic(
            semantic_type,
            ask=ask,
            fact=fact,
            carry_type=carry_type,
        )
        goal = self._stable_fair_protocol_goal(data, fact, table_for_type if table_for_type != "none" else carry_type)
        action_type = str(selected_anchor.get("action_type") or "")
        action_message = str(selected_anchor.get("message") or "")
        action_is_strong = bool(selected_anchor.get("action_is_strong"))
        state_line = self._stable_fair_auto_state_line(
            data,
            carry_type=carry_type,
            carry_qty=carry_qty,
            goal=goal,
        )
        obs_line = self._stable_fair_auto_obs_line(data, fact)
        routing_context_meta = self._stable_fair_protocol_routing_context_meta(feature)
        anchor_meta = [self._stable_fair_protocol_anchor_meta(anchor) for anchor in content_anchors]
        selected_anchor_meta = self._stable_fair_protocol_anchor_meta(selected_anchor) if selected_anchor else {}
        semantic_payload = {
            "type": semantic_type,
            "intent": intent,
            "fact": fact,
            "carry_type": carry_type,
            "carry_qty": carry_qty,
            "table_for_type": table_for_type if semantic_type == "TABLE_FOR_CARRY" else "none",
            "table_for_qty": table_for_qty if semantic_type == "TABLE_FOR_CARRY" else None,
            "goal": goal,
            "ask": ask,
            "recipient": recipient,
            "recipient_context_used": bool(routing_context_meta),
            "recipient_context_influence": str(selected_anchor_meta.get("context_source") or "protocol"),
            "recipient_context_reason": str(selected_anchor_meta.get("context_reason") or ""),
            "deterministic_message": fallback_message,
            "content_anchor": dict(selected_anchor_meta),
            "routing_contract": dict(routing_contract),
        }
        protocol_meta = {
            "protocol_profile": self.SEMANTIC_PROTOCOL_PROFILE,
            "protocol_intent": intent,
            "protocol_semantic_type": semantic_type,
            "protocol_carry_type": carry_type,
            "protocol_carry_qty": carry_qty,
            "protocol_table_for_type": table_for_type if semantic_type == "TABLE_FOR_CARRY" else "none",
            "protocol_table_for_qty": table_for_qty if semantic_type == "TABLE_FOR_CARRY" else None,
            "protocol_goal": goal,
            "protocol_ask": ask,
            "protocol_fact": fact,
            "protocol_recipient": recipient,
            "protocol_used_routing_intent": False,
            "protocol_recipient_context_used": bool(routing_context_meta),
            "protocol_routing_context_used": bool(routing_context_meta),
            "protocol_routing_context": routing_context_meta,
            "protocol_routing_contract": dict(routing_contract),
            "protocol_semantic_payload": dict(semantic_payload),
            "protocol_content_anchor_candidates": anchor_meta,
            "protocol_selected_anchor_id": str(selected_anchor_meta.get("id") or ""),
            "protocol_selected_anchor_kind": str(selected_anchor_meta.get("kind") or ""),
            "protocol_selected_anchor_context_source": str(selected_anchor_meta.get("context_source") or "protocol"),
            "protocol_selected_anchor_context_reason": str(selected_anchor_meta.get("context_reason") or ""),
        }
        self._last_stable_fair_protocol_meta = dict(protocol_meta)

        prompt = self._build_stable_fair_semantic_protocol_prompt(
            data,
            recipient,
            semantic_type=semantic_type,
            fallback_message=fallback_message,
            fact=fact,
            carry_type=carry_type,
            carry_qty=carry_qty,
            ask=ask,
            feature=feature,
            action_type=action_type,
            action_message=action_message,
            action_is_strong=action_is_strong,
            content_anchors=content_anchors,
            state_line=state_line,
            obs_line=obs_line,
            routing_contract=routing_contract,
        )
        final_message = fallback_message
        llm_generated_text = ""
        selected_type = semantic_type
        response_type = ""
        response_focus = ""
        response_ask = ""
        free_resp = ""
        api_call_time = 0.0
        raw_response = "[STABLE_FAIR_SEMANTIC_PROTOCOL]"
        fallback_used = True
        contract_guard_rejected = False
        protocol_reject_reason = ""
        if self.proto_v2_free_enable and prompt:
            try:
                t0 = time.time()
                p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                free_resp = await self.owner.brain.generate(prompt)
                api_call_time += time.time() - t0
                p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
                self._jump_back()
            except Exception as exc:
                free_resp = f"[SEMANTIC_PROTOCOL_ERROR] {exc}"
            response_focus, response_type, response_ask, candidate_message = self._parse_stable_fair_semantic_decision_response(
                free_resp,
                max_chars=min(120, int(self.proto_v2_free_max or 120)),
            )
            focus_anchor = self._stable_fair_protocol_anchor_for_focus(response_focus, content_anchors)
            if response_focus and focus_anchor is None:
                response_anchor = None
                protocol_reject_reason = "invalid_focus"
            elif focus_anchor is not None:
                response_anchor = focus_anchor
            else:
                response_anchor = self._stable_fair_protocol_anchor_for_message(
                    candidate_message,
                    content_anchors,
                    default=selected_anchor,
                )
            response_base_anchor = response_anchor or selected_anchor
            response_allowed = self._stable_fair_semantic_response_allowed(
                routing_intent="",
                base_semantic_type=str((response_base_anchor or {}).get("semantic_type") or semantic_type),
                response_type=response_type or str((response_base_anchor or {}).get("semantic_type") or semantic_type),
                table_for_type=str((response_base_anchor or {}).get("table_for_type") or "none"),
                routing_objective=_stable_fair_contract_objective(routing_contract),
            )
            if (
                candidate_message
                and response_anchor is not None
                and response_allowed
                and self._stable_fair_semantic_message_safe_with_anchors(
                    candidate_message,
                    content_anchors,
                    data,
                    recipient,
                    semantic_type=response_type or str((response_anchor or {}).get("semantic_type") or semantic_type),
                    ask=response_ask or str((response_anchor or {}).get("ask") or ask),
                    selected_anchor=response_anchor,
                )
            ):
                final_message = candidate_message
                selected_type = response_type or semantic_type
                llm_generated_text = candidate_message
                if response_anchor:
                    selected_anchor = dict(response_anchor)
                    (
                        fact,
                        anchor_ask,
                        semantic_type,
                        table_for_type,
                        table_for_qty,
                        fallback_message,
                    ) = materialize_anchor(selected_anchor)
                    ask = str(response_ask if response_ask and response_ask != "none" else anchor_ask or ask or "none")
                    goal = self._stable_fair_protocol_goal(
                        data,
                        fact,
                        table_for_type if table_for_type != "none" else carry_type,
                    )
                    obs_line = self._stable_fair_auto_obs_line(data, fact)
                    selected_anchor_meta = self._stable_fair_protocol_anchor_meta(selected_anchor)
                fallback_used = False
            elif candidate_message:
                contract_guard_rejected = True
                if not protocol_reject_reason:
                    if response_anchor is None:
                        protocol_reject_reason = "no_matching_focus"
                    elif not response_allowed:
                        protocol_reject_reason = "type_not_allowed_for_focus"
                    else:
                        protocol_reject_reason = "message_not_safe_with_grounding"
                if response_anchor is not None and focus_anchor is not None:
                    selected_anchor = dict(response_anchor)
                    (
                        fact,
                        anchor_ask,
                        anchor_semantic_type,
                        table_for_type,
                        table_for_qty,
                        fallback_message,
                    ) = materialize_anchor(selected_anchor)
                    ask = str(response_ask if response_ask and response_ask != "none" else anchor_ask or ask or "none")
                    semantic_type = anchor_semantic_type
                    selected_type = response_type if response_allowed else anchor_semantic_type
                    goal = self._stable_fair_protocol_goal(
                        data,
                        fact,
                        table_for_type if table_for_type != "none" else carry_type,
                    )
                    obs_line = self._stable_fair_auto_obs_line(data, fact)
                    selected_anchor_meta = self._stable_fair_protocol_anchor_meta(selected_anchor)
                    final_message = fallback_message
            raw_response = f"{raw_response}\nRESPONSE:{free_resp}"

        intent = self._stable_fair_protocol_intent_for_semantic(
            selected_type,
            ask=ask,
            fact=fact,
            carry_type=carry_type,
        )
        meta = dict(protocol_meta)
        selected_payload = dict(semantic_payload)
        selected_payload.update(
            {
                "selected_type": selected_type,
                "selected_message": final_message,
                "llm_generated": bool(llm_generated_text),
                "fallback_used": bool(fallback_used),
                "content_anchor": dict(selected_anchor_meta),
                "routing_contract": dict(routing_contract),
                "fact": fact,
                "ask": ask,
            }
        )
        selected_protocol_intent = {
            "RESOURCE": "discover",
            "TABLE": "discover",
            "TABLE_FOR_CARRY": "table_for_carry",
            "CARRY": "carry",
            "DELIVERY": "deliver",
            "DEPLETION": "status",
            "REQUEST": "request",
            "REPLY": "reply",
            "STATUS": "status",
        }.get(selected_type, intent)
        selected_table_for_type = table_for_type if selected_type == "TABLE_FOR_CARRY" else "none"
        selected_payload["type"] = selected_type
        selected_payload["intent"] = selected_protocol_intent
        selected_payload["table_for_type"] = selected_table_for_type
        selected_payload["table_for_qty"] = table_for_qty if selected_type == "TABLE_FOR_CARRY" else None
        state_line = self._stable_fair_auto_state_line(
            data,
            carry_type=carry_type,
            carry_qty=carry_qty,
            goal=goal,
        )
        wire_message = "\n".join([state_line, obs_line, f"MSG: {final_message}"]).strip()
        meta.update(
            {
                "mode": str((routing_meta or {}).get("mode") or self.mode or "basic"),
                "protocol_profile": self.SEMANTIC_PROTOCOL_PROFILE,
                "protocol_prompt_profile": self.PROMPT_PROTOCOL_PROFILE,
                "protocol_action": "stable_fair_semantic_protocol",
                "protocol_intent": selected_protocol_intent,
                "protocol_table_for_type": selected_table_for_type,
                "protocol_table_for_qty": table_for_qty if selected_type == "TABLE_FOR_CARRY" else None,
                "protocol_goal": goal,
                "protocol_ask": ask,
                "protocol_fact": fact,
                "reason": "semantic_protocol_message_prompt",
                "llm_generated_text": llm_generated_text,
                "deterministic_text": fallback_message,
                "protocol_free_enabled": bool(self.proto_v2_free_enable),
                "protocol_free_used": bool(llm_generated_text),
                "protocol_fallback_used": bool(fallback_used),
                "protocol_message_prompt": prompt,
                "protocol_message_response": str(free_resp or "").strip(),
                "protocol_message_prompt_used": bool(prompt),
                "protocol_semantic_type": selected_type,
                "protocol_semantic_payload": selected_payload,
                "protocol_routing_contract": dict(routing_contract),
                "protocol_response_type": response_type,
                "protocol_response_focus": response_focus,
                "protocol_response_ask": response_ask,
                "protocol_reject_reason": protocol_reject_reason,
                "protocol_state_obs_enabled": True,
                "protocol_contract_guard_rejected": bool(contract_guard_rejected),
                "protocol_state_line": state_line,
                "protocol_obs_line": obs_line,
                "protocol_wire_message": wire_message,
                "protocol_selected_focus_id": str(selected_anchor_meta.get("focus_id") or ""),
                "protocol_selected_anchor_id": str(selected_anchor_meta.get("id") or ""),
                "protocol_selected_anchor_kind": str(selected_anchor_meta.get("kind") or ""),
                "protocol_selected_anchor_context_source": str(selected_anchor_meta.get("context_source") or "protocol"),
                "protocol_selected_anchor_context_reason": str(selected_anchor_meta.get("context_reason") or ""),
            }
        )
        self._last_stable_fair_protocol_meta = dict(meta)
        return OutgoingMessage(
            recipient=recipient,
            send=bool(str(wire_message or "").strip()),
            prompt=prompt,
            raw_response=raw_response,
            message=wire_message,
            meta=meta,
            api_call_time=api_call_time,
            tokens=self._snapshot_brain_usage(),
        )

    def _stable_fair_protocol_intent_for_semantic(
        self,
        semantic_type: str,
        *,
        ask: str,
        fact: dict[str, Any] | None,
        carry_type: str,
    ) -> str:
        semantic_norm = str(semantic_type or "STATUS").strip().upper()
        ask_norm = str(ask or "none").strip().lower()
        if semantic_norm == "REPLY":
            return "reply"
        if semantic_norm == "REQUEST" or (ask_norm and ask_norm != "none"):
            return "request"
        if semantic_norm == "TABLE_FOR_CARRY":
            return "table_for_carry"
        if semantic_norm == "DELIVERY":
            return "deliver"
        if semantic_norm == "CARRY":
            return "carry"
        if semantic_norm in {"RESOURCE", "TABLE"}:
            return "discover"
        if carry_type in {"wood", "sand", "stone"}:
            return "carry"
        if isinstance(fact, dict) and str(fact.get("kind") or "") in {"resource", "table"}:
            return "discover"
        return "status"

    def _stable_fair_protocol_brief_anchors(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        routing_contract: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        data = dict(obs or {})
        anchors: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        target_info = dict((data.get("agents_in_communication_range") or {}).get(recipient, {}) or {})
        recipient_carry_type, recipient_carry_qty = _candidate_resource_quantity(target_info)
        contract = dict(routing_contract or {})
        contract_objective = _stable_fair_contract_objective(contract)
        contract_fact = contract.get("primary_fact")
        if isinstance(contract_fact, dict):
            contract_fact = self._normalize_stable_fair_protocol_candidate_fact(contract_fact)
        else:
            contract_fact = None
        contract_backup_fact = contract.get("backup_fact")
        if isinstance(contract_backup_fact, dict):
            contract_backup_fact = self._normalize_stable_fair_protocol_candidate_fact(contract_backup_fact)
        else:
            contract_backup_fact = None
        contract_item_type = str(contract.get("item_type") or "none").strip().lower()
        if contract_item_type not in {"wood", "sand", "stone"}:
            contract_item_type = "none"
        payload_lifecycle = str((feature or {}).get("payload_lifecycle") or "none").strip().lower()
        suppress_table_facts = payload_lifecycle in {"stale_static_table", "exhausted_table", "request_cooldown"}
        suppress_resource_facts = payload_lifecycle in {"stale_static_resource", "request_cooldown"}

        def anchor_key(anchor_id: str, fact: dict[str, Any], ask: str) -> tuple[Any, ...]:
            fact_key = self._stable_fair_protocol_candidate_fact_key(fact)
            if fact_key == ("status",):
                return (str(anchor_id or "").upper(), str(ask or "none").lower())
            return fact_key

        def add_anchor(
            anchor_id: str,
            anchor_kind: str,
            semantic_type: str,
            fact: dict[str, Any] | None,
            ask: str,
            reason: str,
            *,
            message: str = "",
            table_for_type: str = "none",
            table_for_qty: int | None = None,
            context_source: str = "protocol",
            context_reason: str = "",
            action_is_strong: bool = False,
        ) -> None:
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            ask_norm = self._normalize_stable_fair_protocol_ask(ask, carry_type)
            key = anchor_key(anchor_id, normalized, ask_norm)
            if key in seen:
                return
            seen.add(key)
            semantic_norm = str(semantic_type or "STATUS").strip().upper()
            table_type_norm = str(table_for_type or "none").strip().lower()
            if table_type_norm not in {"wood", "sand", "stone"}:
                table_type_norm = "none"
                table_for_qty_norm = None
            else:
                table_for_qty_norm = table_for_qty
            message_text = str(message or "").strip()
            if not message_text:
                message_carry_type = table_type_norm if semantic_norm == "TABLE_FOR_CARRY" and table_type_norm != "none" else carry_type
                message_carry_qty = table_for_qty_norm if semantic_norm == "TABLE_FOR_CARRY" and table_type_norm != "none" else carry_qty
                message_text = self._stable_fair_semantic_fallback_message(
                    semantic_type=semantic_norm,
                    fact=normalized,
                    carry_type=message_carry_type,
                    carry_qty=message_carry_qty,
                    ask=ask_norm,
                )
            anchors.append(
                {
                    "id": str(anchor_id or "").strip().upper(),
                    "kind": str(anchor_kind or "").strip(),
                    "semantic_type": semantic_norm,
                    "fact": normalized,
                    "ask": ask_norm,
                    "label": self._stable_fair_protocol_candidate_label(
                        normalized,
                        table_type_norm if table_type_norm in {"wood", "sand", "stone"} else carry_type,
                        table_for_qty_norm if table_type_norm in {"wood", "sand", "stone"} else carry_qty,
                    ),
                    "message": message_text,
                    "reason": str(reason or "").strip(),
                    "table_for_type": table_type_norm,
                    "table_for_qty": table_for_qty_norm,
                    "recipient_fit": str(context_source or "protocol"),
                    "freshness": self._stable_fair_protocol_fact_freshness(normalized),
                    "risk": "none",
                    "lifecycle": "brief",
                    "priority": max(0, 100 - len(anchors)),
                    "context_source": str(context_source or "protocol"),
                    "context_reason": str(context_reason or ""),
                    "action_type": semantic_norm,
                    "action_is_strong": bool(action_is_strong),
                }
            )

        def contract_reason() -> str:
            return self._stable_fair_protocol_simple_routing_context(recipient, feature)

        def contract_qty_for(item_type: str) -> int | None:
            if item_type == recipient_carry_type:
                return recipient_carry_qty
            if item_type == carry_type:
                return carry_qty
            qty = _coerce_int(contract.get("recipient_carrying_qty"))
            if str(contract.get("recipient_carrying") or "") == item_type:
                return qty
            qty = _coerce_int(contract.get("sender_carrying_qty"))
            if str(contract.get("sender_carrying") or "") == item_type:
                return qty
            return None

        def active_table_for_owner(item_type: str) -> str:
            item_norm = str(item_type or "none").strip().lower()
            if item_norm == recipient_carry_type and item_norm in {"wood", "sand", "stone"}:
                return "recipient"
            if item_norm == carry_type and item_norm in {"wood", "sand", "stone"}:
                return "sender"
            return "none"

        def add_routing_contract_anchor() -> None:
            item_type = contract_item_type
            if contract_objective == "ask_missing_table" and carry_type in {"wood", "sand", "stone"}:
                item_type = carry_type
            if item_type == "none":
                if carry_type in {"wood", "sand", "stone"}:
                    item_type = carry_type
                elif recipient_carry_type in {"wood", "sand", "stone"}:
                    item_type = recipient_carry_type
            qty = contract_qty_for(item_type)
            if contract_objective == "ask_missing_table" and item_type in {"wood", "sand", "stone"}:
                if carry_type == item_type:
                    known_table = self._stable_fair_best_table_fact(data, item_type, qty)
                    if known_table and _stable_fair_table_supports_carry(
                        known_table,
                        item_type,
                        qty,
                        require_full_qty=False,
                    ):
                        add_anchor(
                            "SENDER_CARRY_TABLE",
                            "sender_carry_table_anchor",
                            "TABLE_FOR_CARRY",
                            known_table,
                            "none",
                            "sender already has a compatible table; routing recipient is only an audience cue",
                            table_for_type=item_type,
                            table_for_qty=qty,
                            context_source="protocol",
                            context_reason=contract_reason(),
                        )
                        return
                qty_text = f":{int(qty)}" if qty is not None else ""
                if carry_type == item_type:
                    message = f"I carry {item_type}{qty_text} and need a table that accepts {item_type}."
                else:
                    message = f"Can you report a table that accepts {item_type}?"
                add_anchor(
                    "SENDER_TABLE_REQUEST",
                    "sender_table_request_anchor",
                    "REQUEST",
                    {"kind": "status"},
                    f"table_for={item_type}",
                    "routing selected a missing-table request",
                    message=message,
                    context_source="routing_message_contract",
                    context_reason=contract_reason(),
                )
                return

            if contract_objective in {"share_table_for_carrier", "delivery_ready"} and item_type in {"wood", "sand", "stone"}:
                table_fact = contract_fact if isinstance(contract_fact, dict) and str(contract_fact.get("kind") or "") == "table" else None
                if not table_fact or not _stable_fair_table_supports_carry(table_fact, item_type, qty, require_full_qty=False):
                    table_fact = self._stable_fair_best_table_fact(data, item_type, qty)
                if table_fact and _stable_fair_table_supports_carry(table_fact, item_type, qty, require_full_qty=False):
                    owner = active_table_for_owner(item_type)
                    if owner == "recipient":
                        anchor_id = "RECIPIENT_CARRY_TABLE"
                        anchor_kind = "recipient_carry_table_anchor"
                        semantic = "TABLE_FOR_CARRY"
                        table_for_type = item_type
                        table_for_qty = recipient_carry_qty
                    elif owner == "sender":
                        anchor_id = "SENDER_CARRY_TABLE"
                        anchor_kind = "sender_carry_table_anchor"
                        semantic = "TABLE_FOR_CARRY"
                        table_for_type = item_type
                        table_for_qty = carry_qty
                    else:
                        anchor_id = "ROUTING_FACT"
                        anchor_kind = "routing_contract_table_anchor"
                        semantic = "TABLE"
                        table_for_type = "none"
                        table_for_qty = None
                    add_anchor(
                        anchor_id,
                        anchor_kind,
                        semantic,
                        table_fact,
                        "none",
                        "routing selected a compatible table-for-carry message",
                        table_for_type=table_for_type,
                        table_for_qty=table_for_qty,
                        context_source="routing_message_contract",
                        context_reason=contract_reason(),
                    )
                return

            if contract_objective == "answer_request" and isinstance(contract_fact, dict):
                fact_kind = str(contract_fact.get("kind") or "").strip().lower()
                ask_item = str(contract.get("latest_incoming_ask_type") or item_type or "none").strip().lower()
                if fact_kind == "table":
                    semantic = "REPLY"
                    message = ""
                    table_for_type = "none"
                    table_for_qty = None
                    if ask_item in {"wood", "sand", "stone"} and _stable_fair_table_supports_carry(
                        contract_fact,
                        ask_item,
                        qty,
                        require_full_qty=False,
                    ):
                        pos = _extract_xy(contract_fact.get("position"))
                        if recipient_carry_type == ask_item:
                            semantic = "TABLE_FOR_CARRY"
                            table_for_type = ask_item
                            table_for_qty = recipient_carry_qty
                            if pos is not None:
                                message = (
                                    f"Table ({pos[0]},{pos[1]}) "
                                    f"{_stable_fair_table_for_type_text(contract_fact, ask_item)} for your carried {ask_item}."
                                )
                        elif carry_type == ask_item:
                            semantic = "TABLE_FOR_CARRY"
                            table_for_type = ask_item
                            table_for_qty = carry_qty
                            if pos is not None:
                                message = (
                                    f"Table ({pos[0]},{pos[1]}) "
                                    f"{_stable_fair_table_for_type_text(contract_fact, ask_item)} for my carried {ask_item}."
                                )
                        elif pos is not None:
                            message = (
                                f"Table ({pos[0]},{pos[1]}) "
                                f"{_stable_fair_table_for_type_text(contract_fact, ask_item)} for requested {ask_item}."
                            )
                    add_anchor(
                        "ROUTING_REPLY",
                        "routing_contract_reply_anchor",
                        semantic,
                        contract_fact,
                        "none",
                        "routing selected a reply to a recent request",
                        message=message,
                        table_for_type=table_for_type,
                        table_for_qty=table_for_qty,
                        context_source="routing_message_contract",
                        context_reason=contract_reason(),
                    )
                elif fact_kind == "resource":
                    add_anchor(
                        "ROUTING_REPLY",
                        "routing_contract_reply_anchor",
                        "REPLY",
                        contract_fact,
                        "none",
                        "routing selected a reply to a recent request",
                        context_source="routing_message_contract",
                        context_reason=contract_reason(),
                    )

        add_routing_contract_anchor()

        current_fact = _stable_fair_current_tile_table_fact(data) or _stable_fair_current_tile_resource_fact(data)
        action_type, action_message, action_is_strong = self._stable_fair_action_semantic_candidate(
            data,
            fact=current_fact,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        if action_message and action_is_strong:
            action_fact = current_fact if isinstance(current_fact, dict) else {"kind": "status"}
            if action_type == "CARRY":
                action_fact = _stable_fair_current_tile_resource_fact(data) or action_fact
            elif action_type == "DELIVERY":
                action_fact = _stable_fair_current_tile_table_fact(data) or action_fact
            add_anchor(
                "CURRENT_ACTION",
                "current_action_anchor",
                action_type or "STATUS",
                action_fact,
                "none",
                "current action is directly useful",
                message=action_message,
                table_for_type=carry_type if action_type in {"DELIVERY", "TABLE_FOR_CARRY"} else "none",
                table_for_qty=carry_qty if action_type in {"DELIVERY", "TABLE_FOR_CARRY"} else None,
                action_is_strong=action_is_strong,
            )

        context_fact = (feature or {}).get("best_fact") if isinstance(feature, dict) else None
        if isinstance(context_fact, dict):
            context_fact = self._normalize_stable_fair_protocol_candidate_fact(context_fact)
            context_kind = str(context_fact.get("kind") or "").strip().lower()
            if (
                context_kind == "resource"
                and not suppress_resource_facts
                and not (carry_type in {"wood", "sand", "stone"} and bool((feature or {}).get("need_table_for_type")))
            ):
                add_anchor(
                    "ROUTING_FACT",
                    "routing_fact_anchor",
                    "RESOURCE",
                    context_fact,
                    "none",
                    "fact attached to the selected recipient",
                    context_source="routing_recipient_context",
                    context_reason=self._stable_fair_protocol_simple_routing_context(recipient, feature),
                )
            elif context_kind == "table" and not suppress_table_facts:
                table_for_type = "none"
                table_for_qty: int | None = None
                semantic_type = "TABLE"
                sender_supported = bool(carry_type in {"wood", "sand", "stone"} and _stable_fair_table_supports_carry(
                    context_fact,
                    carry_type,
                    carry_qty,
                    require_full_qty=False,
                ))
                recipient_supported = bool(
                    recipient_carry_type in {"wood", "sand", "stone"} and _stable_fair_table_supports_carry(
                        context_fact,
                        recipient_carry_type,
                        recipient_carry_qty,
                        require_full_qty=False,
                    )
                )
                if carry_type in {"wood", "sand", "stone"} and not sender_supported and not recipient_supported:
                    context_kind = ""
                elif sender_supported:
                    semantic_type = "TABLE_FOR_CARRY"
                    table_for_type = carry_type
                    table_for_qty = carry_qty
                elif recipient_supported:
                    semantic_type = "TABLE_FOR_CARRY"
                    table_for_type = recipient_carry_type
                    table_for_qty = recipient_carry_qty
                if context_kind == "table":
                    add_anchor(
                        "ROUTING_FACT",
                        "routing_fact_anchor",
                        semantic_type,
                        context_fact,
                        "none",
                        "fact attached to the selected recipient",
                        table_for_type=table_for_type,
                        table_for_qty=table_for_qty,
                        context_source="routing_recipient_context",
                        context_reason=self._stable_fair_protocol_simple_routing_context(recipient, feature),
                    )

        if isinstance(contract_backup_fact, dict):
            backup_kind = str(contract_backup_fact.get("kind") or "").strip().lower()
            backup_key = self._stable_fair_protocol_candidate_fact_key(contract_backup_fact)
            primary_key = self._stable_fair_protocol_candidate_fact_key(contract_fact) if isinstance(contract_fact, dict) else None
            if backup_key != primary_key:
                if backup_kind == "table":
                    item_type = contract_item_type
                    qty = contract_qty_for(item_type)
                    semantic = "TABLE"
                    table_for_type = "none"
                    table_for_qty = None
                    for candidate_type, candidate_qty in (
                        (recipient_carry_type, recipient_carry_qty),
                        (carry_type, carry_qty),
                    ):
                        if candidate_type in {"wood", "sand", "stone"} and _stable_fair_table_supports_carry(
                            contract_backup_fact,
                            candidate_type,
                            candidate_qty,
                            require_full_qty=False,
                        ):
                            semantic = "TABLE_FOR_CARRY"
                            table_for_type = candidate_type
                            table_for_qty = candidate_qty
                            break
                    add_anchor(
                        "ROUTING_BACKUP_FACT",
                        "routing_backup_fact_anchor",
                        semantic,
                        contract_backup_fact,
                        "none",
                        "secondary payload from routing",
                        table_for_type=table_for_type,
                        table_for_qty=table_for_qty,
                        context_source="routing_message_contract",
                        context_reason=contract_reason(),
                    )
                elif backup_kind == "resource":
                    add_anchor(
                        "ROUTING_BACKUP_FACT",
                        "routing_backup_fact_anchor",
                        "RESOURCE",
                        contract_backup_fact,
                        "none",
                        "secondary payload from routing",
                        context_source="routing_message_contract",
                        context_reason=contract_reason(),
                    )

        if recipient_carry_type in {"wood", "sand", "stone"}:
            recipient_table_fact = self._stable_fair_best_table_fact(data, recipient_carry_type, recipient_carry_qty)
            if recipient_table_fact and _stable_fair_table_supports_carry(
                recipient_table_fact,
                recipient_carry_type,
                recipient_carry_qty,
                require_full_qty=False,
            ):
                add_anchor(
                    "RECIPIENT_CARRY_TABLE",
                    "recipient_carry_table_anchor",
                    "TABLE_FOR_CARRY",
                    recipient_table_fact,
                    "none",
                    f"{recipient} carries {recipient_carry_type}",
                    table_for_type=recipient_carry_type,
                    table_for_qty=recipient_carry_qty,
                )

        sender_table_fact = None
        if carry_type in {"wood", "sand", "stone"}:
            sender_table_fact = self._stable_fair_best_table_fact(data, carry_type, carry_qty)
            if sender_table_fact and not _stable_fair_table_supports_carry(
                sender_table_fact,
                carry_type,
                carry_qty,
                require_full_qty=False,
            ):
                sender_table_fact = None
            if sender_table_fact:
                add_anchor(
                    "SENDER_CARRY_TABLE",
                    "sender_carry_table_anchor",
                    "TABLE_FOR_CARRY",
                    sender_table_fact,
                    "none",
                    f"sender carries {carry_type}",
                    table_for_type=carry_type,
                    table_for_qty=carry_qty,
                )
            elif not bool((feature or {}).get("request_cooldown")):
                add_anchor(
                    "SENDER_TABLE_REQUEST",
                    "sender_table_request_anchor",
                    "REQUEST",
                    {"kind": "status"},
                    f"table_for={carry_type}",
                    f"sender carries {carry_type} but lacks a compatible table",
                    context_source="routing_recipient_context" if bool((feature or {}).get("need_table_for_type")) else "protocol",
                    context_reason=self._stable_fair_protocol_simple_routing_context(recipient, feature),
                )

        resource_fact = self._stable_fair_best_resource_fact(data)
        if resource_fact and not suppress_resource_facts:
            add_anchor(
                "FRESH_RESOURCE",
                "fresh_resource_anchor",
                "RESOURCE",
                resource_fact,
                "none",
                "fresh visible or remembered resource fact",
            )
        general_table_fact = self._stable_fair_best_table_fact(data, "none", None)
        if general_table_fact and not suppress_table_facts:
            add_anchor(
                "FRESH_TABLE",
                "fresh_table_anchor",
                "TABLE",
                general_table_fact,
                "none",
                "fresh visible or remembered table fact",
            )
        add_anchor(
            "STATUS",
            "status_anchor",
            "STATUS",
            {"kind": "status"},
            "none",
            "fallback when no concrete fact is useful",
        )
        return anchors[:5]

    def _stable_fair_protocol_fact_freshness(self, fact: dict[str, Any] | None) -> str:
        if not isinstance(fact, dict):
            return "local"
        source = str(fact.get("source") or "").strip().lower()
        if source in {"visible", "memory", "current_tile"}:
            return source
        return "local"

    def _stable_fair_protocol_simple_routing_context(self, recipient: str, feature: dict[str, Any] | None) -> str:
        if not isinstance(feature, dict):
            return f"selected_recipient={recipient}; chosen_for=coverage"
        edge = _stable_fair_routing_edge_class_for_feature(feature)
        objective = _stable_fair_routing_objective_for_feature(feature)
        latest_request = str(feature.get("latest_incoming_ask_type") or "none")
        table_for = str(feature.get("table_for_type") or "none")
        fact_kind = str(feature.get("best_fact_kind") or "none")
        if edge == "reply_request":
            chosen_for = f"reply_request:{latest_request}"
        elif edge == "table_for_carrier":
            chosen_for = f"table_for_recipient:{table_for}"
        elif edge == "ask_table_provider":
            chosen_for = f"ask_table_provider:{table_for}"
        elif edge == "fresh_discovery":
            chosen_for = f"fresh_fact:{fact_kind}"
        else:
            chosen_for = objective or edge or "coverage"
        return f"selected_recipient={recipient}; chosen_for={chosen_for}"

    def _stable_fair_protocol_numbered_anchors(self, anchors: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        numbered: list[dict[str, Any]] = []
        for idx, anchor in enumerate(list(anchors or [])[:5], start=1):
            item = dict(anchor or {})
            item["focus_id"] = f"F{idx}"
            numbered.append(item)
        return numbered

    def _stable_fair_protocol_anchor_for_focus(
        self,
        focus: str,
        anchors: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        focus_norm = re.sub(r"[^A-Za-z0-9_:-]+", "", str(focus or "").strip()).upper()
        if not focus_norm:
            return None
        for anchor in list(anchors or []):
            focus_id = str((anchor or {}).get("focus_id") or "").strip().upper()
            anchor_id = str((anchor or {}).get("id") or "").strip().upper()
            if focus_norm in {focus_id, anchor_id}:
                return dict(anchor)
        return None

    def _stable_fair_protocol_brief_anchor_lines(self, anchors: list[dict[str, Any]] | None) -> list[str]:
        if not anchors:
            return ["- F1 id=STATUS type=STATUS ask=none fact=status cue=no concrete grounded fact"]
        lines: list[str] = []
        for anchor in list(anchors or [])[:5]:
            meta = self._stable_fair_protocol_anchor_meta(anchor)
            lines.append(
                f"- {meta.get('focus_id') or meta.get('id')}: id={meta.get('id')}; "
                f"type={meta.get('semantic_type')}; ask={meta.get('ask')}; "
                f"fact={meta.get('label')}; cue={meta.get('reason')}"
            )
        return lines

    def _stable_fair_protocol_anchor_for_message(
        self,
        message: str,
        anchors: list[dict[str, Any]] | None,
        *,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        line = str(message or "")
        low = line.lower()
        mentioned_items = [item for item in ("wood", "sand", "stone") if item in low]
        coords = _stable_fair_message_coords(line)
        coord_matches: list[dict[str, Any]] = []
        for anchor in list(anchors or []):
            fact = anchor.get("fact") if isinstance(anchor, dict) else None
            pos = _extract_xy((fact or {}).get("position") if isinstance(fact, dict) else None)
            if pos is not None and pos in coords:
                coord_matches.append(anchor)
        if coord_matches:
            if "table" in low:
                for item_type in mentioned_items:
                    for anchor in coord_matches:
                        fact = anchor.get("fact") if isinstance(anchor, dict) else None
                        if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
                            continue
                        table_for = str((anchor or {}).get("table_for_type") or "none").strip().lower()
                        if table_for == item_type:
                            return anchor
                    for anchor in coord_matches:
                        fact = anchor.get("fact") if isinstance(anchor, dict) else None
                        if isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
                            if _stable_fair_table_supports_carry(fact, item_type, None, require_full_qty=False):
                                return anchor
                for anchor in coord_matches:
                    fact = anchor.get("fact") if isinstance(anchor, dict) else None
                    if isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
                        return anchor
            for item_type in mentioned_items:
                for anchor in coord_matches:
                    table_for = str((anchor or {}).get("table_for_type") or "none").strip().lower()
                    if table_for == item_type:
                        return anchor
                for anchor in coord_matches:
                    fact = anchor.get("fact") if isinstance(anchor, dict) else None
                    if isinstance(fact, dict) and str(fact.get("kind") or "") == "resource":
                        if str(fact.get("resource_type") or "").strip().lower() == item_type:
                            return anchor
                    if isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
                        if _stable_fair_table_supports_carry(fact, item_type, None, require_full_qty=False):
                            return anchor
            return coord_matches[0]
        if re.search(r"\b(?:need|needs|looking for|look for|report|find|check|tell me|do you know|can you|where)\b", low):
            for anchor in list(anchors or []):
                if str((anchor or {}).get("semantic_type") or "").upper() == "REQUEST":
                    return anchor
        if any(word in low for word in ("pick up", "picking up", "pickup")):
            for anchor in list(anchors or []):
                if str((anchor or {}).get("kind") or "") == "current_action_anchor":
                    return anchor
        if "resource" in low or any(item in low for item in ("wood", "sand", "stone")):
            for anchor in list(anchors or []):
                fact = anchor.get("fact") if isinstance(anchor, dict) else None
                if isinstance(fact, dict) and str(fact.get("kind") or "") == "resource":
                    resource_type = str(fact.get("resource_type") or "").lower()
                    if not resource_type or resource_type in low or "resource" in low:
                        return anchor
        if "table" in low:
            for anchor in list(anchors or []):
                fact = anchor.get("fact") if isinstance(anchor, dict) else None
                if isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
                    return anchor
        return default

    def _stable_fair_semantic_message_preserves_contract(
        self,
        message: str,
        anchor: dict[str, Any] | None,
        contract: dict[str, Any] | None,
        *,
        carry_type: str,
    ) -> bool:
        objective = _stable_fair_contract_objective(contract)
        if objective not in {"ask_missing_table", "share_table_for_carrier", "answer_request", "delivery_ready"}:
            return True
        line = str(message or "").strip().lower()
        anchor_type = str((anchor or {}).get("semantic_type") or "").strip().upper()
        fact = (anchor or {}).get("fact") if isinstance(anchor, dict) else None
        fact_kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
        item_type = str((contract or {}).get("item_type") or carry_type or "none").strip().lower()
        if objective == "ask_missing_table" and carry_type in {"wood", "sand", "stone"}:
            item_type = carry_type
        if item_type not in {"wood", "sand", "stone"}:
            item_type = str((contract or {}).get("latest_incoming_ask_type") or "none").strip().lower()

        if objective == "ask_missing_table":
            return (
                anchor_type == "REQUEST"
                and item_type in {"wood", "sand", "stone"}
                and item_type in line
                and "table" in line
                and bool(re.search(r"\b(?:need|report|find|check|tell me|do you know|can you)\b", line))
            )
        if objective == "share_table_for_carrier":
            return (
                anchor_type == "TABLE_FOR_CARRY"
                and fact_kind == "table"
                and item_type in {"wood", "sand", "stone"}
                and item_type in line
                and "table" in line
            )
        if objective == "answer_request":
            if item_type in {"wood", "sand", "stone"} and item_type not in line:
                return False
            return anchor_type in {"REPLY", "TABLE_FOR_CARRY", "TABLE", "RESOURCE"}
        if objective == "delivery_ready":
            return anchor_type in {"DELIVERY", "TABLE_FOR_CARRY", "CARRY"}
        return True

    def _stable_fair_semantic_message_safe_with_anchors(
        self,
        message: str,
        anchors: list[dict[str, Any]] | None,
        obs: dict[str, Any] | None,
        recipient: str | None,
        *,
        semantic_type: str = "",
        ask: str = "",
        selected_anchor: dict[str, Any] | None = None,
    ) -> bool:
        line = str(message or "").strip()
        if not line:
            return False
        data = dict(obs or {})
        semantic_norm = str(
            semantic_type
            or ((selected_anchor or {}).get("semantic_type") if isinstance(selected_anchor, dict) else "")
            or ""
        ).strip().upper()
        ask_norm = str(
            ask
            or ((selected_anchor or {}).get("ask") if isinstance(selected_anchor, dict) else "")
            or "none"
        ).strip().lower()
        sender_pos = _extract_xy(data.get("position"))
        allowed_coords: set[tuple[int, int]] = set()
        if sender_pos is not None:
            allowed_coords.add(sender_pos)
        table_facts: list[dict[str, Any]] = []
        resource_facts: list[dict[str, Any]] = []
        for anchor in list(anchors or []):
            fact = anchor.get("fact") if isinstance(anchor, dict) else None
            if not isinstance(fact, dict):
                continue
            pos = _extract_xy(fact.get("position"))
            if pos is not None:
                allowed_coords.add(pos)
            kind = str(fact.get("kind") or "").strip().lower()
            if kind == "table":
                table_facts.append(fact)
            elif kind == "resource":
                resource_facts.append(fact)
        mentioned_coords = _stable_fair_message_coords(line)
        if any(coord not in allowed_coords for coord in mentioned_coords):
            return False
        low = line.lower()
        sender_carry_type, sender_carry_qty = _inventory_resource_quantity(data)
        target_info = dict((data.get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_carry_type, _recipient_carry_qty = _candidate_resource_quantity(target_info)
        action_pickup_items: set[str] = set()
        for anchor in list(anchors or []):
            anchor_kind = str((anchor or {}).get("kind") or "")
            anchor_type = str((anchor or {}).get("semantic_type") or "").upper()
            if anchor_kind != "current_action_anchor" or anchor_type != "CARRY":
                continue
            action_msg = str((anchor or {}).get("message") or "").lower()
            match = re.search(r"\b(?:picking up|pick up|pickup)\s+(?:\d+\s+)?(wood|sand|stone)\b", action_msg)
            if match:
                action_pickup_items.add(str(match.group(1) or "").lower())
                continue
            fact = (anchor or {}).get("fact")
            if isinstance(fact, dict) and str(fact.get("kind") or "") == "resource":
                resource_type = str(fact.get("resource_type") or "").lower()
                if resource_type in {"wood", "sand", "stone"}:
                    action_pickup_items.add(resource_type)
        resource_available_for_pickup = semantic_norm == "RESOURCE" and bool(
            re.search(r"\bavailable\s+for\s+pickup\b", low)
        )
        if (
            any(word in low for word in ("picking up", "pick up", "pickup"))
            and not resource_available_for_pickup
            and sender_carry_type in {"wood", "sand", "stone"}
            and sender_carry_type not in action_pickup_items
        ):
            return False
        for match in re.finditer(r"\b(?:(my|your)\s+)?carried\s+(wood|sand|stone)\b", low):
            owner = str(match.group(1) or "").lower()
            item_type = str(match.group(2) or "").lower()
            if owner == "my":
                if sender_carry_type != item_type and item_type not in action_pickup_items:
                    return False
            elif owner == "your":
                if recipient_carry_type != item_type:
                    return False
            elif (
                sender_carry_type != item_type
                and recipient_carry_type != item_type
                and item_type not in action_pickup_items
            ):
                return False
        if "resource" in low:
            resource_claim = bool(
                re.search(r"\bresource\s+(?:wood|sand|stone)\b", low)
                or re.search(r"\b(?:wood|sand|stone)\s+resource\b", low)
                or re.search(r"\bresource\s+at\s*\(", low)
            )
            generic_table_capacity = "table" in low and re.search(r"\b(?:capacity|remaining|accepts?)\b", low)
            if (
                resource_claim
                and not generic_table_capacity
                and not resource_facts
                and not re.search(r"\b(?:report|find|check|tell me|do you know|can you)\b", low)
            ):
                return False
        if "table" in low:
            ask_match = re.search(r"table_for=(wood|sand|stone)", ask_norm)
            ask_item = str(ask_match.group(1) or "").lower() if ask_match else ""
            protocol_table_request = bool(
                semantic_norm == "REQUEST"
                and ask_item in {"wood", "sand", "stone"}
                and ask_item in low
                and re.search(
                    r"\b(?:need|needs|needed|looking for|look for|find|report|check|tell me|do you know|can you|where)\b",
                    low,
                )
            )
            is_request = bool(
                protocol_table_request
                or re.search(r"\b(?:report|find|check|tell me|do you know|can you)\b", low)
            )
            if is_request:
                return self._stable_fair_semantic_message_safe(line, None, data, recipient)
            if mentioned_coords:
                mentioned_tables = [
                    fact
                    for fact in table_facts
                    if (_extract_xy(fact.get("position")) in mentioned_coords)
                ]
            else:
                mentioned_tables = table_facts
            for item_type in ("wood", "sand", "stone"):
                if item_type not in low:
                    continue
                if not any(
                    _stable_fair_table_supports_carry(fact, item_type, None, require_full_qty=False)
                    for fact in mentioned_tables
                ):
                    if any(
                        _stable_fair_table_remaining_for_type(fact, item_type) == 0
                        for fact in mentioned_tables
                    ) and _stable_fair_negative_table_capacity_mention(low, item_type):
                        continue
                    return False
            delivery_match = re.search(
                r"\b(?:delivering|deliver|bringing|bring|putting down|put down)\s+(wood|sand|stone)\s*(?:[:=x]|\s+x\s+|\s+)?\s*(\d+)?",
                low,
            )
            if delivery_match:
                item_type = str(delivery_match.group(1) or "").lower()
                qty = _coerce_int(delivery_match.group(2))
                if sender_carry_type != item_type:
                    return False
                if sender_carry_qty is not None and qty is not None and int(sender_carry_qty) < int(qty):
                    return False
                if re.search(r"\b(?:delivering|putting down|put down)\b", low):
                    table_positions = {
                        _extract_xy(fact.get("position"))
                        for fact in mentioned_tables
                        if isinstance(fact, dict)
                    }
                    table_positions.discard(None)
                    if sender_pos is None or any(pos != sender_pos for pos in table_positions):
                        return False
                if not any(
                    _stable_fair_table_supports_carry(fact, item_type, qty, require_full_qty=True)
                    for fact in mentioned_tables
                ):
                    return False
        return self._stable_fair_semantic_message_safe(line, None, data, recipient)

    def _stable_fair_protocol_content_anchors(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> list[dict[str, Any]]:
        data = dict(obs or {})
        target_info = dict((data.get("agents_in_communication_range") or {}).get(recipient, {}) or {})
        recipient_carry_type, recipient_carry_qty = _candidate_resource_quantity(target_info)
        recent_request = bool(
            isinstance(feature, dict)
            and bool(feature.get("reply_recent"))
            and bool(feature.get("reply_info_request"))
        )
        recent_request_type = str((feature or {}).get("latest_incoming_ask_type") or "none").strip().lower()
        edge_reason = (
            str((feature or {}).get("edge_reason") or _stable_fair_routing_edge_reason_for_feature(feature)).strip()
            if isinstance(feature, dict)
            else "low_value_coverage"
        )
        routing_objective = _stable_fair_routing_objective_for_feature(feature)
        routing_hint = str((feature or {}).get("routing_hint") or (feature or {}).get("routing_intent") or "coverage").strip().lower()
        recipient_answerability = (
            str(
                (feature or {}).get("recipient_answerability")
                or _stable_fair_routing_recipient_answerability_for_feature(feature)
            ).strip()
            if isinstance(feature, dict)
            else "low_answerability"
        )
        send_risk = (
            str((feature or {}).get("send_risk") or _stable_fair_routing_send_risk_for_feature(feature)).strip()
            if isinstance(feature, dict)
            else "none"
        )
        payload_lifecycle = (
            str((feature or {}).get("payload_lifecycle") or "none").strip().lower()
            if isinstance(feature, dict)
            else "none"
        )
        suppress_static_lifecycle = payload_lifecycle in {
            "stale_static_table",
            "stale_static_resource",
            "exhausted_table",
            "request_cooldown",
        }
        suppressed_static_key: tuple[Any, ...] | None = None
        anchors: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def freshness_for_fact(fact: dict[str, Any] | None) -> str:
            if not isinstance(fact, dict):
                return "local"
            source = str(fact.get("source") or "").strip().lower()
            if source in {"visible", "memory"}:
                return source
            return "local"

        def add_anchor(
            anchor_id: str,
            anchor_kind: str,
            semantic_type: str,
            fact: dict[str, Any] | None,
            ask: str,
            reason: str,
            priority: int,
            *,
            table_for_type: str = "none",
            table_for_qty: int | None = None,
            recipient_fit: str = "memory_update",
            risk: str = "none",
            lifecycle: str = "none",
            context_source: str = "protocol",
            context_reason: str = "",
        ) -> None:
            semantic_norm = str(semantic_type or "STATUS").strip().upper()
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            ask_norm = self._normalize_stable_fair_protocol_ask(ask, carry_type)
            table_for_norm = str(table_for_type or "none").strip().lower()
            if semantic_norm == "TABLE_FOR_CARRY" and table_for_norm not in {"wood", "sand", "stone"}:
                return
            if semantic_norm == "REQUEST" and ask_norm == "none" and carry_type in {"wood", "sand", "stone"}:
                ask_norm = f"table_for={carry_type}"
            table_qty_norm = _coerce_int(table_for_qty)
            key = (
                str(anchor_kind),
                semantic_norm,
                self._stable_fair_protocol_candidate_fact_key(normalized),
                ask_norm,
                table_for_norm,
            )
            if key in seen:
                return
            seen.add(key)
            message_carry_type = table_for_norm if semantic_norm == "TABLE_FOR_CARRY" and table_for_norm != "none" else carry_type
            message_carry_qty = table_qty_norm if semantic_norm == "TABLE_FOR_CARRY" and table_for_norm != "none" else carry_qty
            message = self._stable_fair_semantic_fallback_message(
                semantic_type=semantic_norm,
                fact=normalized,
                carry_type=message_carry_type if semantic_norm in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else "none",
                carry_qty=message_carry_qty if semantic_norm in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else None,
                ask=ask_norm,
            )
            if not self._stable_fair_semantic_message_safe(message, normalized, data, recipient):
                return
            anchors.append(
                {
                    "id": str(anchor_id or "").strip().upper(),
                    "kind": str(anchor_kind or "").strip(),
                    "semantic_type": semantic_norm,
                    "fact": normalized,
                    "ask": ask_norm,
                    "table_for_type": table_for_norm if table_for_norm in {"wood", "sand", "stone"} else "none",
                    "table_for_qty": table_qty_norm if table_for_norm in {"wood", "sand", "stone"} else None,
                    "label": self._stable_fair_protocol_candidate_label(
                        normalized,
                        table_for_norm if table_for_norm in {"wood", "sand", "stone"} else carry_type,
                        table_qty_norm if table_for_norm in {"wood", "sand", "stone"} else carry_qty,
                    ),
                    "message": message,
                    "reason": str(reason or "").strip(),
                    "priority": int(priority),
                    "recipient_fit": str(recipient_fit or "memory_update"),
                    "freshness": freshness_for_fact(normalized),
                    "risk": str(risk or "none"),
                    "lifecycle": str(lifecycle or "none"),
                    "context_source": str(context_source or "protocol"),
                    "context_reason": str(context_reason or ""),
                }
            )

        recipient_table_fact = None
        if recipient_carry_type in {"wood", "sand", "stone"}:
            recipient_table_fact = self._stable_fair_best_table_fact(data, recipient_carry_type, recipient_carry_qty)
            if recipient_table_fact and not _stable_fair_table_supports_carry(
                recipient_table_fact,
                recipient_carry_type,
                recipient_carry_qty,
                require_full_qty=False,
            ):
                recipient_table_fact = None
        sender_table_fact = None
        if carry_type in {"wood", "sand", "stone"}:
            sender_table_fact = self._stable_fair_best_table_fact(data, carry_type, carry_qty)
            if sender_table_fact and not _stable_fair_table_supports_carry(
                sender_table_fact,
                carry_type,
                carry_qty,
                require_full_qty=False,
            ):
                sender_table_fact = None
        requested_table_fact = None
        if recent_request_type in {"wood", "sand", "stone"}:
            requested_table_fact = self._stable_fair_best_table_fact(data, recent_request_type, None)
            if requested_table_fact and not _stable_fair_table_supports_carry(
                requested_table_fact,
                recent_request_type,
                None,
                require_full_qty=False,
            ):
                requested_table_fact = None
        resource_fact = self._stable_fair_best_resource_fact(data)
        general_table_fact = self._stable_fair_best_table_fact(data, "none", None)
        round_num = int(data.get("round", 0) or 0)
        generic_random_context = bool(
            not isinstance(feature, dict)
            and carry_type not in {"wood", "sand", "stone"}
            and recipient_carry_type not in {"wood", "sand", "stone"}
            and not recent_request
        )
        resource_priority = 52
        table_priority = 68
        if generic_random_context:
            resource_priority = 58
            table_priority = 56
            resource_recent = self._stable_fair_fact_recent_message_count(
                data,
                resource_fact,
                round_num,
                window=max(4, int(self.DUPLICATE_FACT_LOOKBACK_ROUNDS)),
                include_received=True,
            )
            table_recent = self._stable_fair_fact_recent_message_count(
                data,
                general_table_fact,
                round_num,
                window=max(4, int(self.DUPLICATE_FACT_LOOKBACK_ROUNDS)),
                include_received=True,
            )
            resource_priority -= min(24, 6 * int(resource_recent))
            table_priority -= min(24, 6 * int(table_recent))
        active_sender_table_need = bool(
            carry_type in {"wood", "sand", "stone"}
            and not sender_table_fact
            and (
                edge_reason == "sender_needs_table_knowledge"
                or routing_objective == "ask_missing_table"
                or bool((feature or {}).get("need_table_for_type"))
            )
        )
        if (
            active_sender_table_need
            and general_table_fact
            and not _stable_fair_table_supports_carry(
                general_table_fact,
                carry_type,
                carry_qty,
                require_full_qty=False,
            )
        ):
            table_priority = min(table_priority, 44)

        context_fact = (feature or {}).get("best_fact") if isinstance(feature, dict) else None
        if isinstance(context_fact, dict):
            context_kind = str(context_fact.get("kind") or (feature or {}).get("best_fact_kind") or "").strip().lower()
            context_duplicate = bool((feature or {}).get("duplicate_fact_recent"))
            table_request_active = recent_request_type in {"wood", "sand", "stone", "table"}
            context_allowed = context_kind in {"resource", "table"}
            if suppress_static_lifecycle and context_kind in {"resource", "table"}:
                suppressed_static_key = self._stable_fair_protocol_candidate_fact_key(
                    self._normalize_stable_fair_protocol_candidate_fact(context_fact)
                )
                context_allowed = False
            if edge_reason == "sender_needs_table_knowledge":
                context_allowed = False
            if table_request_active and context_kind == "resource":
                context_allowed = False
            if context_duplicate and edge_reason not in {"answerable_recent_request", "recipient_needs_known_table"}:
                context_allowed = False
            if context_allowed:
                context_semantic_type = "RESOURCE" if context_kind == "resource" else "TABLE"
                context_table_for_type = "none"
                context_table_for_qty: int | None = None
                if context_kind == "table":
                    preferred_type = str((feature or {}).get("table_for_type") or "none").strip().lower()
                    if preferred_type not in {"wood", "sand", "stone"} and edge_reason == "recipient_needs_known_table":
                        preferred_type = recipient_carry_type
                    if (
                        preferred_type not in {"wood", "sand", "stone"}
                        and carry_type in {"wood", "sand", "stone"}
                        and _stable_fair_table_supports_carry(
                            context_fact,
                            carry_type,
                            carry_qty,
                            require_full_qty=False,
                        )
                    ):
                        preferred_type = carry_type
                    owner_preferred_type = "none"
                    if preferred_type == recipient_carry_type and preferred_type in {"wood", "sand", "stone"}:
                        owner_preferred_type = preferred_type
                    elif preferred_type == carry_type and preferred_type in {"wood", "sand", "stone"}:
                        owner_preferred_type = preferred_type
                    elif (
                        recipient_carry_type in {"wood", "sand", "stone"}
                        and _stable_fair_table_supports_carry(
                            context_fact,
                            recipient_carry_type,
                            recipient_carry_qty,
                            require_full_qty=False,
                        )
                    ):
                        owner_preferred_type = recipient_carry_type
                    elif (
                        carry_type in {"wood", "sand", "stone"}
                        and _stable_fair_table_supports_carry(
                            context_fact,
                            carry_type,
                            carry_qty,
                            require_full_qty=False,
                        )
                    ):
                        owner_preferred_type = carry_type
                    if owner_preferred_type in {"wood", "sand", "stone"}:
                        preferred_qty = (
                            recipient_carry_qty
                            if owner_preferred_type == recipient_carry_type
                            else carry_qty
                        )
                        if _stable_fair_table_supports_carry(
                            context_fact,
                            owner_preferred_type,
                            preferred_qty,
                            require_full_qty=False,
                        ):
                            context_semantic_type = "TABLE_FOR_CARRY"
                            context_table_for_type = owner_preferred_type
                            context_table_for_qty = preferred_qty
                    elif (
                        edge_reason == "recipient_needs_known_table"
                        and preferred_type in {"wood", "sand", "stone"}
                        and not _stable_fair_table_supports_carry(
                            context_fact,
                            preferred_type,
                            _coerce_int((feature or {}).get("table_required_qty")),
                            require_full_qty=False,
                        )
                    ):
                        context_allowed = False
                if context_allowed:
                    context_priority = 50
                    if edge_reason == "answerable_recent_request":
                        context_priority = 118
                    elif edge_reason == "recipient_needs_known_table":
                        context_priority = 116
                    elif edge_reason == "fresh_resource_coverage":
                        context_priority = 80
                    elif edge_reason == "fresh_table_coverage":
                        context_priority = 86
                    elif routing_objective in {"share_resource", "share_table"}:
                        context_priority = 84
                    if "static_low_value" in send_risk:
                        context_priority -= 6
                    add_anchor(
                        "RECIPIENT_CONTEXT_FACT",
                        "recipient_context_fact_anchor",
                        context_semantic_type,
                        context_fact,
                        "none",
                        f"selected recipient edge is {edge_reason}",
                        context_priority,
                        table_for_type=context_table_for_type,
                        table_for_qty=context_table_for_qty,
                        recipient_fit=(
                            "recipient_context_table_for_carry"
                            if context_semantic_type == "TABLE_FOR_CARRY"
                            else "recipient_context_fact"
                        ),
                        risk=send_risk or "none",
                        lifecycle=payload_lifecycle,
                        context_source="routing_recipient_context",
                        context_reason=(
                            f"edge_reason={edge_reason}; routing_objective={routing_objective}; "
                            f"routing_hint={routing_hint}; answerability={recipient_answerability}"
                        ),
                    )

        if recent_request:
            request_wants_table = recent_request_type in {"wood", "sand", "stone", "table"}
            if requested_table_fact:
                requested_recipient_carrying = recent_request_type == recipient_carry_type
                add_anchor(
                    "RECENT_REQUEST_TABLE_ANSWER",
                    "recent_request_table_answer_anchor",
                    "TABLE_FOR_CARRY" if requested_recipient_carrying else "TABLE",
                    requested_table_fact,
                    "none",
                    f"{recipient} recently asked for a table accepting {recent_request_type}",
                    124,
                    table_for_type=recent_request_type if requested_recipient_carrying else "none",
                    table_for_qty=recipient_carry_qty if requested_recipient_carrying else None,
                    recipient_fit="answers_table_request",
                )
            elif recipient_table_fact:
                add_anchor(
                    "RECENT_REQUEST_TABLE_FOR_RECIPIENT",
                    "recent_request_table_answer_anchor",
                    "TABLE_FOR_CARRY",
                    recipient_table_fact,
                    "none",
                    f"{recipient} recently asked; they carry {recipient_carry_type}",
                    112,
                    table_for_type=recipient_carry_type,
                    table_for_qty=recipient_carry_qty,
                    recipient_fit="answers_request_and_can_act",
                )
            elif sender_table_fact and recent_request_type in {"table", carry_type}:
                add_anchor(
                    "RECENT_REQUEST_TABLE_FOR_SENDER",
                    "recent_request_table_answer_anchor",
                    "TABLE",
                    sender_table_fact,
                    "none",
                    f"{recipient} recently asked; sender has compatible table context",
                    108,
                    recipient_fit="answers_request_with_table",
                )
            elif resource_fact and not request_wants_table:
                add_anchor(
                    "RECENT_REQUEST_RESOURCE",
                    "recent_request_resource_answer_anchor",
                    "RESOURCE",
                    resource_fact,
                    "none",
                    f"{recipient} recently asked; sender has a concrete resource fact",
                    104,
                    recipient_fit="answers_request_with_fact",
                )
            elif general_table_fact and recent_request_type not in {"wood", "sand", "stone"}:
                add_anchor(
                    "RECENT_REQUEST_TABLE",
                    "recent_request_table_answer_anchor",
                    "TABLE",
                    general_table_fact,
                    "none",
                    f"{recipient} recently asked; sender has a concrete table fact",
                    102,
                    recipient_fit="answers_request_with_fact",
                )

        if recipient_table_fact:
            add_anchor(
                "RECIPIENT_CARRY_TABLE",
                "recipient_carry_table_anchor",
                "TABLE_FOR_CARRY",
                recipient_table_fact,
                "none",
                f"{recipient} is carrying {recipient_carry_type}",
                96,
                table_for_type=recipient_carry_type,
                table_for_qty=recipient_carry_qty,
                recipient_fit="recipient_can_act_now",
            )

        if sender_table_fact:
            add_anchor(
                "SENDER_CARRY_TABLE",
                "sender_carry_table_anchor",
                "TABLE_FOR_CARRY",
                sender_table_fact,
                "none",
                f"sender is carrying {carry_type}",
                78,
                table_for_type=carry_type,
                table_for_qty=carry_qty,
                recipient_fit="sender_can_act_now",
                lifecycle="active_shared_table",
            )
        elif carry_type in {"wood", "sand", "stone"} and not (
            payload_lifecycle == "request_cooldown" or bool((feature or {}).get("request_cooldown"))
        ):
            sender_request_priority = 48
            if edge_reason == "sender_needs_table_knowledge" or routing_objective == "ask_missing_table":
                sender_request_priority = 104
                if recipient_answerability == "known_table_provider":
                    sender_request_priority += 8
                elif recipient_answerability == "recent_table_reporter":
                    sender_request_priority += 4
                elif recipient_answerability == "low_answerability":
                    sender_request_priority -= 12
            elif active_sender_table_need:
                sender_request_priority = 92
                if recipient_answerability == "known_table_provider":
                    sender_request_priority += 8
                elif recipient_answerability == "recent_table_reporter":
                    sender_request_priority += 4
                elif recipient_answerability == "low_answerability":
                    sender_request_priority -= 8
            add_anchor(
                "SENDER_TABLE_REQUEST",
                "sender_table_request_anchor",
                "REQUEST",
                {"kind": "status"},
                f"table_for={carry_type}",
                f"sender carries {carry_type} but lacks a compatible table",
                sender_request_priority,
                recipient_fit="may_answer_sender",
                lifecycle="active_request",
                context_source="routing_recipient_context" if sender_request_priority >= 90 else "protocol",
                context_reason=(
                    f"edge_reason={edge_reason}; routing_objective={routing_objective}; "
                    f"answerability={recipient_answerability}"
                ),
            )

        if resource_fact:
            resource_key = self._stable_fair_protocol_candidate_fact_key(
                self._normalize_stable_fair_protocol_candidate_fact(resource_fact)
            )
            if resource_key != suppressed_static_key:
                add_anchor(
                    "FRESH_RESOURCE",
                    "fresh_resource_anchor",
                    "RESOURCE",
                        resource_fact,
                        "none",
                        "fresh visible or remembered resource fact",
                        resource_priority,
                        recipient_fit="memory_update",
                        risk="static_fact",
                        lifecycle="fresh_resource",
                )
        if general_table_fact:
            table_key = self._stable_fair_protocol_candidate_fact_key(
                self._normalize_stable_fair_protocol_candidate_fact(general_table_fact)
            )
            if table_key != suppressed_static_key and payload_lifecycle not in {"request_cooldown", "exhausted_table"}:
                add_anchor(
                    "FRESH_TABLE",
                    "fresh_table_anchor",
                    "TABLE",
                        general_table_fact,
                        "none",
                        "fresh visible or remembered table fact",
                        table_priority,
                        recipient_fit="memory_update",
                        risk="static_fact",
                        lifecycle="fresh_static_table",
                )
        if carry_type in {"wood", "sand", "stone"}:
            add_anchor(
                "CARRY_STATUS",
                "carry_status_anchor",
                "CARRY",
                {"kind": "status"},
                "none",
                f"sender carrying {carry_type} may matter",
                42,
                recipient_fit="status_update",
                risk="movement_or_status_only",
            )
        add_anchor(
            "STATUS",
            "status_anchor",
            "STATUS",
            {"kind": "status"},
            "none",
            "fallback when no grounded anchor is useful",
            0,
            recipient_fit="none",
            risk="low_value",
        )
        anchors.sort(key=lambda item: int(item.get("priority") or 0), reverse=True)
        return anchors[:7]

    def _stable_fair_protocol_anchor_meta(self, anchor: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(anchor, dict):
            return {}
        return {
            "focus_id": str(anchor.get("focus_id") or ""),
            "id": str(anchor.get("id") or ""),
            "kind": str(anchor.get("kind") or ""),
            "semantic_type": str(anchor.get("semantic_type") or ""),
            "label": str(anchor.get("label") or ""),
            "ask": str(anchor.get("ask") or "none"),
            "table_for_type": str(anchor.get("table_for_type") or "none"),
            "table_for_qty": anchor.get("table_for_qty"),
            "recipient_fit": str(anchor.get("recipient_fit") or ""),
            "freshness": str(anchor.get("freshness") or ""),
            "risk": str(anchor.get("risk") or "none"),
            "lifecycle": str(anchor.get("lifecycle") or "none"),
            "reason": str(anchor.get("reason") or ""),
            "priority": int(anchor.get("priority") or 0),
            "context_source": str(anchor.get("context_source") or "protocol"),
            "context_reason": str(anchor.get("context_reason") or ""),
        }

    def _stable_fair_protocol_content_anchor_lines(self, anchors: list[dict[str, Any]] | None) -> list[str]:
        if not anchors:
            return ["- STATUS: kind=status_anchor; type=STATUS; label=status; ask=none; fit=none"]
        lines: list[str] = []
        for anchor in list(anchors or [])[:6]:
            meta = self._stable_fair_protocol_anchor_meta(anchor)
            lines.append(
                f"- {meta.get('id')}: kind={meta.get('kind')}; type={meta.get('semantic_type')}; "
                f"label={meta.get('label')}; ask={meta.get('ask')}; fit={meta.get('recipient_fit')}; "
                f"freshness={meta.get('freshness')}; reason={meta.get('reason')}"
            )
        return lines

    def _stable_fair_semantic_response_allowed(
        self,
        *,
        routing_intent: str,
        base_semantic_type: str,
        response_type: str,
        table_for_type: str = "none",
        routing_objective: str = "",
    ) -> bool:
        del routing_intent, routing_objective
        base_norm = str(base_semantic_type or "STATUS").strip().upper()
        resp_norm = str(response_type or base_norm or "STATUS").strip().upper()
        table_for_norm = str(table_for_type or "none").strip().lower()
        allowed = {"RESOURCE", "TABLE", "TABLE_FOR_CARRY", "CARRY", "DELIVERY", "DEPLETION", "REQUEST", "REPLY", "STATUS"}
        if resp_norm not in allowed:
            return False
        if base_norm == "RESOURCE":
            return resp_norm == "RESOURCE"
        if base_norm == "TABLE":
            return resp_norm == "TABLE"
        if base_norm == "REQUEST":
            return resp_norm in {"REQUEST", "TABLE", "RESOURCE", "STATUS"}
        if base_norm == "REPLY":
            return resp_norm in {"REPLY", "TABLE", "RESOURCE", "STATUS"}
        if resp_norm == "DELIVERY" and base_norm != "DELIVERY":
            return False
        if resp_norm == "TABLE_FOR_CARRY":
            return base_norm == "TABLE_FOR_CARRY" and table_for_norm in {"wood", "sand", "stone"}
        if base_norm == "TABLE_FOR_CARRY":
            return resp_norm in {"TABLE_FOR_CARRY", "TABLE"}
        if base_norm == "DELIVERY":
            return resp_norm in {"DELIVERY", "TABLE_FOR_CARRY", "CARRY", "REQUEST", "STATUS"}
        if base_norm == "CARRY":
            return resp_norm in {"CARRY", "REQUEST", "STATUS", "TABLE", "TABLE_FOR_CARRY", "RESOURCE"}
        return True

    def _stable_fair_action_override_allowed(
        self,
        *,
        routing_intent: str,
        semantic_type: str,
        action_type: str,
        action_message: str,
        action_is_strong: bool,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        routing_objective: str = "",
    ) -> bool:
        del routing_intent, routing_objective
        if not action_message:
            return False
        action_norm = str(action_type or "STATUS").strip().upper()
        base_norm = str(semantic_type or "STATUS").strip().upper()
        low = str(action_message or "").lower()
        if action_norm == "DELIVERY" and isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
            if not _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                return False
        if base_norm == "REQUEST":
            return False
        if base_norm == "TABLE_FOR_CARRY" and action_norm == "CARRY" and "moving" in low:
            return False
        if base_norm == "STATUS":
            return True
        return bool(action_is_strong)

    def _stable_fair_semantic_type(
        self,
        intent: str,
        fact: dict[str, Any] | None,
        carry_type: str,
        ask: str,
        carry_qty: int | None = None,
    ) -> str:
        intent_norm = str(intent or "").strip().lower()
        if intent_norm == "reply":
            return "REPLY"
        kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
        ask_norm = str(ask or "none").strip().lower()
        if intent_norm == "table_for_carry" and kind == "table":
            return "TABLE_FOR_CARRY"
        if intent_norm == "deliver" and kind == "table":
            if _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                return "DELIVERY"
            if ask_norm != "none":
                return "REQUEST"
            if carry_type in {"wood", "sand", "stone"}:
                return "CARRY"
            return "TABLE"
        if carry_type in {"wood", "sand", "stone"} and kind == "table":
            if _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=False):
                return "TABLE_FOR_CARRY"
            if ask_norm != "none":
                return "REQUEST"
            return "CARRY"
        if intent_norm == "deliver" and carry_type in {"wood", "sand", "stone"}:
            if ask_norm != "none":
                return "REQUEST"
            return "CARRY"
        if ask_norm != "none" or intent_norm == "request":
            return "REQUEST"
        if carry_type in {"wood", "sand", "stone"} or intent_norm == "carry":
            return "CARRY"
        if kind == "resource":
            return "RESOURCE"
        if kind == "table":
            return "TABLE"
        return "STATUS"

    def _stable_fair_semantic_fallback_message(
        self,
        *,
        semantic_type: str,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        ask: str,
    ) -> str:
        kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
        pos = _extract_xy((fact or {}).get("position") if isinstance(fact, dict) else None)
        qty_text = str(int(carry_qty)) if carry_qty is not None else "?"
        table_for_type = carry_type if carry_type in {"wood", "sand", "stone"} else "none"
        if semantic_type == "DELIVERY" and carry_type in {"wood", "sand", "stone"}:
            if kind == "table" and pos is not None:
                if not _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                    return f"I am carrying {carry_type}:{qty_text} and need a compatible table."
                return f"Delivering {carry_type}:{qty_text} to table ({pos[0]},{pos[1]}), which {_stable_fair_table_accepts_text(fact)}."
            return f"Carrying {carry_type}:{qty_text} for delivery; report a compatible table if you know one."
        if semantic_type == "TABLE_FOR_CARRY" and kind == "table" and pos is not None:
            return f"Table ({pos[0]},{pos[1]}) {_stable_fair_table_for_type_text(fact, table_for_type)} for carried {table_for_type}."
        if semantic_type == "CARRY" and carry_type in {"wood", "sand", "stone"}:
            if kind == "table" and pos is not None:
                if not _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                    return f"I am carrying {carry_type}:{qty_text} and need a compatible table."
                return f"Table ({pos[0]},{pos[1]}) {_stable_fair_table_for_type_text(fact, carry_type)} for carried {carry_type}."
            return f"I am carrying {carry_type}:{qty_text} and need a compatible table."
        if semantic_type == "REPLY" and kind == "resource" and pos is not None:
            resource_type = str(fact.get("resource_type") or "resource").strip().lower()
            qty = _coerce_int(fact.get("quantity"))
            return f"Reply: resource {resource_type} at ({pos[0]},{pos[1]}) has {int(qty) if qty is not None else '?'}."
        if semantic_type == "REPLY" and kind == "table" and pos is not None:
            return f"Reply: table ({pos[0]},{pos[1]}) {_stable_fair_table_accepts_text(fact)}."
        if semantic_type == "RESOURCE" and kind == "resource" and pos is not None:
            resource_type = str(fact.get("resource_type") or "resource").strip().lower()
            qty = _coerce_int(fact.get("quantity"))
            return f"Resource {resource_type} at ({pos[0]},{pos[1]}) has {int(qty) if qty is not None else '?'}."
        if semantic_type == "TABLE" and kind == "table" and pos is not None:
            return f"Table ({pos[0]},{pos[1]}) {_stable_fair_table_accepts_text(fact)}."
        ask_text = str(ask or "none").strip().lower()
        match = re.search(r"table_for=(wood|sand|stone)", ask_text)
        if semantic_type == "REQUEST" and match:
            ask_type = str(match.group(1) or "").lower()
            if carry_type == ask_type:
                return f"I carry {ask_type}:{qty_text} and need a table that accepts {ask_type}."
            return f"Can you report a table that accepts {ask_type}?"
        if semantic_type == "REQUEST":
            return "Can you report one useful nearby resource or table?"
        return "No new useful map fact from me; continuing my route."

    def _stable_fair_protocol_compound_fallback_message(
        self,
        fallback_message: str,
        selected_anchor: dict[str, Any] | None,
        anchors: list[dict[str, Any]] | None,
        *,
        carry_type: str,
        carry_qty: int | None,
    ) -> str:
        del carry_qty
        line = str(fallback_message or "").strip()
        if not line:
            return line
        low = line.lower()
        selected_kind = str((selected_anchor or {}).get("kind") or "")
        selected_type = str((selected_anchor or {}).get("semantic_type") or "").upper()
        if selected_kind != "current_action_anchor" and selected_type not in {"CARRY", "REQUEST"}:
            return line
        item_type = "none"
        match = re.search(r"\b(?:picking up|pick up|pickup|carry|carrying|need a table that accepts)\s+(?:\d+\s+)?(wood|sand|stone)\b", low)
        if match:
            item_type = str(match.group(1) or "").lower()
        if item_type not in {"wood", "sand", "stone"}:
            item_type = str(carry_type or "none").lower()
        if item_type not in {"wood", "sand", "stone"}:
            return line
        if "table" in low and "report" not in low and "need" not in low:
            return line
        best_table: dict[str, Any] | None = None
        best_key: tuple[int, int] | None = None
        for anchor in list(anchors or []):
            fact = anchor.get("fact") if isinstance(anchor, dict) else None
            if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
                continue
            if not _stable_fair_table_supports_carry(fact, item_type, None, require_full_qty=False):
                continue
            pos = _extract_xy(fact.get("position"))
            if pos is None:
                continue
            table_for = str(anchor.get("table_for_type") or "none").lower()
            remaining = _stable_fair_table_remaining_for_type(fact, item_type)
            key = (1 if table_for == item_type else 0, int(remaining or 0))
            if best_key is None or key > best_key:
                best_key = key
                best_table = fact
        pos = _extract_xy((best_table or {}).get("position") if isinstance(best_table, dict) else None)
        if best_table is None or pos is None:
            return line
        prefix = re.split(r";\s*(?:report|need|can you|do you know)\b", line, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ;")
        if not prefix:
            prefix = f"I carry {item_type}"
        table_text = _stable_fair_table_for_type_text(best_table, item_type)
        return f"{prefix}; table ({pos[0]},{pos[1]}) {table_text}."

    def _stable_fair_auto_state_line(
        self,
        obs: dict[str, Any],
        *,
        carry_type: str,
        carry_qty: int | None,
        goal: Any,
    ) -> str:
        data = dict(obs or {})
        pos = _extract_xy(data.get("position")) or (0, 0)
        carry_text = "none"
        if carry_type in {"wood", "sand", "stone"}:
            carry_text = f"{carry_type}:{int(carry_qty) if carry_qty is not None else '?'}"
        goal_pos = _extract_xy(goal) or _extract_xy(data.get("goal_position"))
        goal_text = f"({goal_pos[0]},{goal_pos[1]})" if goal_pos is not None else "none"
        action_info = dict(data.get("current_round_action_intent") or {})
        action = str(action_info.get("move") or data.get("action_text") or "unknown").strip()
        action = re.sub(r"[^A-Za-z0-9_:-]+", "_", action)[:24] or "unknown"
        return f"STATE: id={self.owner.name}; loc=({pos[0]},{pos[1]}); carry={carry_text}; goal={goal_text}; action={action}"

    def _stable_fair_auto_obs_line(self, obs: dict[str, Any], fact: dict[str, Any] | None) -> str:
        data = dict(obs or {})
        selected_fact = fact if isinstance(fact, dict) else None
        if not isinstance(selected_fact, dict):
            selected_fact = _stable_fair_current_tile_table_fact(data)
        if not isinstance(selected_fact, dict):
            selected_fact = _stable_fair_current_tile_resource_fact(data)
        if not isinstance(selected_fact, dict):
            return "OBS: none"
        kind = str(selected_fact.get("kind") or "").strip().lower()
        pos = _extract_xy(selected_fact.get("position"))
        source = str(selected_fact.get("source") or "protocol_anchor").strip().lower() or "protocol_anchor"
        source = re.sub(r"[^a-z0-9_:-]+", "_", source)[:32] or "protocol_anchor"
        if kind == "table" and pos is not None:
            display_fact = dict(selected_fact)
            rb = selected_fact.get("remaining_by_type")
            if isinstance(rb, dict):
                positive_rb = {
                    item_type: int(qty)
                    for item_type, qty in (
                        (str(k).lower(), _coerce_int(v))
                        for k, v in rb.items()
                    )
                    if item_type in {"wood", "sand", "stone"} and qty is not None and qty > 0
                }
                display_fact["remaining_by_type"] = positive_rb
            return f"OBS: table=({pos[0]},{pos[1]}); {_stable_fair_table_capacity_text(display_fact)}; src={source}"
        if kind == "resource" and pos is not None:
            resource_type = str(selected_fact.get("resource_type") or "resource").strip().lower()
            qty = _coerce_int(selected_fact.get("quantity"))
            return f"OBS: resource={resource_type}@({pos[0]},{pos[1]}):{int(qty) if qty is not None else 0}; src={source}"
        return "OBS: none"

    def _stable_fair_protocol_obs_summary_line(
        self,
        obs: dict[str, Any],
        *,
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        data = dict(obs or {})
        facts: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        pos = _extract_xy(data.get("position")) or (0, 0)

        def fact_key(fact: dict[str, Any]) -> tuple[Any, ...]:
            kind = str(fact.get("kind") or "status").strip().lower()
            fact_pos = _extract_xy(fact.get("position"))
            if kind == "resource" and fact_pos is not None:
                return ("resource", str(fact.get("resource_type") or ""), int(fact_pos[0]), int(fact_pos[1]))
            if kind == "table" and fact_pos is not None:
                return ("table", int(fact_pos[0]), int(fact_pos[1]))
            return ("status",)

        def add_fact(fact: dict[str, Any] | None) -> None:
            if not isinstance(fact, dict):
                return
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            kind = str(normalized.get("kind") or "").strip().lower()
            fact_pos = _extract_xy(normalized.get("position"))
            if kind not in {"resource", "table"} or fact_pos is None:
                return
            if kind == "table":
                rem = _coerce_int(normalized.get("remaining_capacity"))
                rb = normalized.get("remaining_by_type")
                if rem is not None and rem <= 0 and not any(
                    (_coerce_int(value) or 0) > 0 for value in (rb or {}).values()
                ):
                    return
            key = fact_key(normalized)
            if key in seen:
                return
            seen.add(key)
            facts.append(normalized)

        add_fact(_stable_fair_current_tile_table_fact(data))
        add_fact(_stable_fair_current_tile_resource_fact(data))
        for key in ("best_fact", "backup_fact"):
            add_fact((feature or {}).get(key) if isinstance(feature, dict) else None)
        add_fact(self._stable_fair_best_resource_fact(data))
        if carry_type in {"wood", "sand", "stone"}:
            add_fact(self._stable_fair_best_table_fact(data, carry_type, carry_qty))
        add_fact(self._stable_fair_best_table_fact(data, "none", None))

        def distance(fact: dict[str, Any]) -> int:
            return int(_manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0)

        resources = [fact for fact in facts if str(fact.get("kind") or "") == "resource"]
        tables = [fact for fact in facts if str(fact.get("kind") or "") == "table"]
        resources.sort(key=lambda fact: (str(fact.get("source") or "") == "current_tile", int(fact.get("quantity") or 0), -distance(fact)), reverse=True)
        tables.sort(
            key=lambda fact: (
                str(fact.get("source") or "") == "current_tile",
                int(_stable_fair_table_remaining_for_type(fact, carry_type) or 0) if carry_type in {"wood", "sand", "stone"} else 0,
                int(_coerce_int(fact.get("remaining_capacity")) or 0),
                -distance(fact),
            ),
            reverse=True,
        )
        selected_facts = resources[:2] + tables[:2]
        parts: list[str] = []
        for fact in selected_facts:
            kind = str(fact.get("kind") or "").strip().lower()
            fact_pos = _extract_xy(fact.get("position"))
            if fact_pos is None:
                continue
            source = str(fact.get("source") or "protocol_context").strip().lower()
            source = re.sub(r"[^a-z0-9_:-]+", "_", source)[:24] or "protocol_context"
            if kind == "resource":
                resource_type = str(fact.get("resource_type") or "resource").strip().lower()
                qty = _coerce_int(fact.get("quantity"))
                parts.append(f"resource={resource_type}@({fact_pos[0]},{fact_pos[1]}):{int(qty) if qty is not None else 0}; src={source}")
            elif kind == "table":
                rb_text = _stable_fair_remaining_by_type_text(fact.get("remaining_by_type"))
                capacity = _stable_fair_table_capacity_text(fact)
                if rb_text and "type_rem" not in capacity:
                    capacity = f"{capacity}; type_rem={rb_text}"
                parts.append(f"table=({fact_pos[0]},{fact_pos[1]}); {capacity}; src={source}")
        if not parts:
            return "OBS: none", []
        return "OBS: " + " | ".join(parts), selected_facts

    def _stable_fair_protocol_lane_candidate_facts(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        routing_contract: dict[str, Any] | None = None,
        carry_type: str,
        carry_qty: int | None,
    ) -> list[dict[str, Any]]:
        data = dict(obs or {})
        facts: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        pos = _extract_xy(data.get("position")) or (0, 0)
        target_info = dict((data.get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_carry_type, recipient_carry_qty = _candidate_resource_quantity(target_info)
        preferred_types = [
            item_type
            for item_type in (carry_type, recipient_carry_type)
            if item_type in {"wood", "sand", "stone"}
        ]
        objective = (
            str((routing_contract or {}).get("objective") or "").strip().lower()
            if isinstance(routing_contract, dict)
            else ""
        )
        if not objective and isinstance(feature, dict):
            objective = _stable_fair_routing_objective_for_feature(feature)
        table_need_type = (
            str((routing_contract or {}).get("item_type") or "").strip().lower()
            if isinstance(routing_contract, dict)
            else ""
        )
        if table_need_type not in {"wood", "sand", "stone"} and isinstance(feature, dict):
            table_need_type = str(
                feature.get("table_for_type") or feature.get("own_carrying") or feature.get("carrying") or ""
            ).strip().lower()
        if table_need_type not in {"wood", "sand", "stone"}:
            table_need_type = carry_type if carry_type in {"wood", "sand", "stone"} else recipient_carry_type
        table_need_active = objective == "ask_missing_table" and table_need_type in {"wood", "sand", "stone"}
        table_need_qty = (
            carry_qty
            if table_need_type == carry_type
            else recipient_carry_qty
            if table_need_type == recipient_carry_type
            else None
        )
        recipient_pos = _extract_xy(target_info.get("position"))
        priority_by_key: dict[tuple[Any, ...], int] = {}
        fact_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        use_roles = isinstance(feature, dict) or bool(routing_contract)
        role_rank = {"": 0, "LOCAL": 1, "ROUTING_BACKUP": 2, "ROUTING_FOCUS": 3}

        def add_fact(fact: dict[str, Any] | None, priority: int = 0, role: str = "") -> None:
            if not isinstance(fact, dict):
                return
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            kind = str(normalized.get("kind") or "").strip().lower()
            fact_pos = _extract_xy(normalized.get("position"))
            if kind not in {"resource", "table"} or fact_pos is None:
                return
            if kind == "table":
                rem = _coerce_int(normalized.get("remaining_capacity"))
                rb = normalized.get("remaining_by_type")
                has_type_space = isinstance(rb, dict) and any(
                    (_coerce_int(value) or 0) > 0 for value in rb.values()
                )
                if rem is not None and rem <= 0 and not has_type_space:
                    return
            key = self._stable_fair_protocol_candidate_fact_key(normalized)
            priority_by_key[key] = max(int(priority_by_key.get(key, 0)), int(priority))
            role_norm = str(role or "").strip().upper()
            if role_norm and use_roles:
                normalized["protocol_role"] = role_norm
            if key in seen:
                existing = fact_by_key.get(key)
                if isinstance(existing, dict) and role_norm and use_roles:
                    existing_role = str(existing.get("protocol_role") or "").strip().upper()
                    if role_rank.get(role_norm, 0) > role_rank.get(existing_role, 0):
                        existing["protocol_role"] = role_norm
                return
            seen.add(key)
            fact_by_key[key] = normalized
            facts.append(normalized)

        def is_compatible_table_for_need(fact: dict[str, Any] | None) -> bool:
            if not table_need_active or not isinstance(fact, dict):
                return False
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            if str(normalized.get("kind") or "").strip().lower() != "table":
                return False
            return _stable_fair_table_supports_carry(
                normalized,
                table_need_type,
                table_need_qty,
                require_full_qty=False,
            )

        def add_routing_fact(fact: dict[str, Any] | None, priority: int, role: str) -> None:
            if table_need_active:
                if is_compatible_table_for_need(fact):
                    add_fact(fact, priority=priority, role=role)
                elif isinstance(fact, dict):
                    add_fact(fact, priority=max(0, priority - 70), role="")
                return
            add_fact(fact, priority=priority, role=role)

        if isinstance(routing_contract, dict):
            add_routing_fact(routing_contract.get("primary_fact"), priority=124, role="ROUTING_FOCUS")
            add_routing_fact(routing_contract.get("backup_fact"), priority=114, role="ROUTING_BACKUP")
        if isinstance(feature, dict):
            add_routing_fact(feature.get("best_fact"), priority=120, role="ROUTING_FOCUS")
            add_routing_fact(feature.get("backup_fact"), priority=112, role="ROUTING_BACKUP")
        local_role = "LOCAL" if use_roles else ""
        add_fact(_stable_fair_current_tile_table_fact(data), priority=104, role=local_role)
        add_fact(_stable_fair_current_tile_resource_fact(data), priority=104, role=local_role)
        if table_need_active:
            add_fact(
                self._stable_fair_best_compatible_table_fact(
                    data,
                    table_need_type,
                    table_need_qty,
                    require_full_qty=False,
                    prefer_local=True,
                ),
                priority=130,
                role="ROUTING_FOCUS",
            )
        if carry_type in {"wood", "sand", "stone"}:
            add_fact(
                self._stable_fair_best_compatible_table_fact(
                    data,
                    carry_type,
                    carry_qty,
                    require_full_qty=False,
                    prefer_local=True,
                ),
                priority=98,
                role=local_role,
            )
        if recipient_carry_type in {"wood", "sand", "stone"}:
            add_fact(
                self._stable_fair_best_compatible_table_fact(
                    data,
                    recipient_carry_type,
                    recipient_carry_qty,
                    require_full_qty=False,
                    origin=recipient_pos,
                    prefer_local=True,
                ),
                priority=106,
                role=local_role,
            )
        for idx, resource_fact in enumerate(self._stable_fair_top_resource_facts(data, limit=3)):
            add_fact(resource_fact, priority=72 - idx * 4, role=local_role)
        for idx, table_fact in enumerate(self._stable_fair_top_table_facts(data, "none", None, limit=3)):
            add_fact(table_fact, priority=64 - idx * 4, role=local_role)

        def distance(fact: dict[str, Any]) -> int:
            return int(_manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0)

        def local_distance(fact: dict[str, Any]) -> int:
            fact_pos = _extract_xy(fact.get("position"))
            if fact_pos is None:
                return 999
            distances = [int(_manhattan_distance(pos, fact_pos) or 0)]
            if recipient_pos is not None:
                distances.append(int(_manhattan_distance(recipient_pos, fact_pos) or 0))
            return min(distances)

        def table_fit(fact: dict[str, Any]) -> int:
            best = 0
            for item_type in preferred_types:
                remaining = _stable_fair_table_remaining_for_type(fact, item_type)
                if remaining is not None:
                    best = max(best, int(remaining))
            if preferred_types:
                return best if best > 0 else -1
            if best <= 0:
                best = int(_coerce_int(fact.get("remaining_capacity")) or 0)
            return best

        def priority(fact: dict[str, Any]) -> int:
            return int(priority_by_key.get(self._stable_fair_protocol_candidate_fact_key(fact), 0))

        def local_priority(fact: dict[str, Any]) -> int:
            base = priority(fact)
            if str(fact.get("kind") or "") != "table":
                return base
            return max(0, base - max(0, local_distance(fact) - 6) * 5)

        ranked_resources = [fact for fact in facts if str(fact.get("kind") or "") == "resource"]
        ranked_tables = [fact for fact in facts if str(fact.get("kind") or "") == "table"]
        ranked_resources.sort(
            key=lambda fact: (
                priority(fact),
                str(fact.get("source") or "") == "current_tile",
                int(_coerce_int(fact.get("quantity")) or 0),
                -distance(fact),
            ),
            reverse=True,
        )
        ranked_tables.sort(
            key=lambda fact: (
                local_priority(fact),
                str(fact.get("source") or "") == "current_tile",
                -local_distance(fact),
                table_fit(fact),
                int(_coerce_int(fact.get("remaining_capacity")) or 0),
            ),
            reverse=True,
        )
        merged = ranked_resources[:3] + ranked_tables[:3]
        merged.sort(
            key=lambda fact: (
                local_priority(fact),
                str(fact.get("source") or "") == "current_tile",
                -local_distance(fact) if str(fact.get("kind") or "") == "table" else 0,
                int(_coerce_int(fact.get("quantity")) or 0)
                if str(fact.get("kind") or "") == "resource"
                else table_fit(fact),
                -distance(fact),
            ),
            reverse=True,
        )
        return merged[:6]

    def _stable_fair_protocol_lane_fact_lines(self, facts: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for fact in facts:
            kind = str((fact or {}).get("kind") or "").strip().lower()
            pos = _extract_xy((fact or {}).get("position"))
            source = str((fact or {}).get("source") or "context").strip().lower() or "context"
            role = str((fact or {}).get("protocol_role") or "").strip().upper()
            prefix = f"{role} " if role else ""
            if kind == "resource" and pos is not None:
                resource_type = str(fact.get("resource_type") or "resource").strip().lower()
                qty = _coerce_int(fact.get("quantity"))
                lines.append(f"- {prefix}RESOURCE {resource_type}@({pos[0]},{pos[1]}):{int(qty) if qty is not None else 0} src={source}")
            elif kind == "table" and pos is not None:
                rb = _stable_fair_positive_remaining_by_type(fact)
                can_deposit = ",".join(f"{item_type}:{rb[item_type]}" for item_type in ("wood", "sand", "stone") if item_type in rb)
                rem = _coerce_int(fact.get("remaining_capacity"))
                total_text = f" total={int(rem)}" if rem is not None else ""
                lines.append(f"- {prefix}TABLE ({pos[0]},{pos[1]}) can_deposit={can_deposit or 'unknown'}{total_text} src={source}")
        return lines or ["- none"]

    def _stable_fair_protocol_content_candidate_lines(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        routing_contract: dict[str, Any] | None,
        candidate_facts: list[dict[str, Any]],
        carry_type: str,
        carry_qty: int | None,
        recipient_evidence: bool = False,
    ) -> list[str]:
        data = dict(obs or {})
        sender_pos = _extract_xy(data.get("position")) or (0, 0)
        target_info = dict((data.get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_carry_type, recipient_carry_qty = _candidate_resource_quantity(target_info)
        objective = (
            str((routing_contract or {}).get("objective") or "").strip().lower()
            if isinstance(routing_contract, dict)
            else ""
        )
        if not objective and isinstance(feature, dict):
            objective = _stable_fair_routing_objective_for_feature(feature)
        if not objective:
            objective = "coverage"
        item_type = (
            str((routing_contract or {}).get("item_type") or "").strip().lower()
            if isinstance(routing_contract, dict)
            else ""
        )
        if item_type not in {"wood", "sand", "stone"} and isinstance(feature, dict):
            item_type = str(
                feature.get("table_for_type")
                or feature.get("own_carrying")
                or feature.get("carrying")
                or feature.get("best_fact_type")
                or ""
            ).strip().lower()
        if item_type not in {"wood", "sand", "stone"}:
            item_type = carry_type if carry_type in {"wood", "sand", "stone"} else recipient_carry_type
        item_qty = (
            carry_qty
            if item_type == carry_type
            else recipient_carry_qty
            if item_type == recipient_carry_type
            else None
        )
        normalized_facts = [
            self._normalize_stable_fair_protocol_candidate_fact(fact)
            for fact in list(candidate_facts or [])
            if isinstance(fact, dict)
        ]
        resources = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "resource"]
        tables = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "table"]
        lines: list[str] = []
        seen: set[str] = set()

        def add(line: str) -> None:
            clean = re.sub(r"\s+", " ", str(line or "").strip())
            if not clean or clean in seen:
                return
            seen.add(clean)
            lines.append(f"- {clean}")

        def table_line(fact: dict[str, Any], focus_type: str = "none") -> str:
            text = self._stable_fair_delivery_fact_text(fact, focus_type=focus_type)
            return f"TABLE {text}" if text else ""

        def resource_line(fact: dict[str, Any]) -> str:
            text = self._stable_fair_delivery_fact_text(fact)
            return f"RESOURCE {text}" if text else ""

        round_num = _coerce_int(data.get("round") or data.get("round_num")) or 0

        def fact_new_for_recipient(fact: dict[str, Any] | None) -> bool:
            if not isinstance(fact, dict):
                return False
            if self._stable_fair_fact_recently_sent_to_recipient(data, recipient, fact, int(round_num), window=6):
                return False
            if self._stable_fair_fact_recently_received_from_recipient(data, recipient, fact, int(round_num), window=6):
                return False
            return True

        def unique_ordered(facts: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
            ordered: list[dict[str, Any]] = []
            local_seen: set[tuple[Any, ...]] = set()
            for raw_fact in facts:
                fact = self._normalize_stable_fair_protocol_candidate_fact(raw_fact)
                kind = str(fact.get("kind") or "").strip().lower()
                if kind not in {"resource", "table"} or _extract_xy(fact.get("position")) is None:
                    continue
                key = self._stable_fair_protocol_candidate_fact_key(fact)
                if key in local_seen:
                    continue
                local_seen.add(key)
                ordered.append(fact)
            return ordered

        def resource_rank(fact: dict[str, Any] | None) -> tuple[int, int, int, int]:
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            if str(normalized.get("kind") or "") != "resource":
                return (0, 0, 0, 0)
            source = str(normalized.get("source") or "").strip().lower()
            source_value = 3 if source == "current_tile" else 2 if source == "visible" else 1
            is_new = 1 if fact_new_for_recipient(normalized) else 0
            qty = int(_coerce_int(normalized.get("quantity")) or 0)
            pos = _extract_xy(normalized.get("position"))
            dist = int(_manhattan_distance(sender_pos, pos) or 0) if pos is not None else 999
            return (source_value, is_new, qty, -dist)

        def order_resources(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
            ordered = unique_ordered(facts)
            ordered.sort(key=resource_rank, reverse=True)
            return ordered

        bundle_lines: list[str] = []

        def add_bundle(facts: list[dict[str, Any] | None]) -> None:
            ordered = unique_ordered(facts)[:2]
            if len(ordered) < 2:
                return
            parts: list[str] = []
            has_table = False
            resource_fact = next(
                (fact for fact in ordered if str(fact.get("kind") or "").strip().lower() == "resource"),
                None,
            )
            table_fact = next(
                (fact for fact in ordered if str(fact.get("kind") or "").strip().lower() == "table"),
                None,
            )
            bundle_focus_type = item_type
            if resource_fact is not None and table_fact is not None:
                bundle_focus_type = str(resource_fact.get("resource_type") or "").strip().lower()
                if bundle_focus_type not in {"wood", "sand", "stone"}:
                    return
                bundle_qty = _coerce_int(resource_fact.get("quantity"))
                if not _stable_fair_table_supports_carry(
                    table_fact,
                    bundle_focus_type,
                    bundle_qty,
                    require_full_qty=False,
                ):
                    return
            for fact in ordered:
                kind = str(fact.get("kind") or "").strip().lower()
                if kind == "table":
                    has_table = True
                    text = self._stable_fair_delivery_fact_text(fact, focus_type=bundle_focus_type)
                    if text:
                        parts.append(f"TABLE {text}")
                elif kind == "resource":
                    text = self._stable_fair_delivery_fact_text(fact)
                    if text:
                        parts.append(f"RESOURCE {text}")
            if len(parts) < 2:
                return
            lane = "TABLE" if has_table else "RESOURCE"
            clean = re.sub(r"\s+", " ", f"BUNDLE lane={lane} " + "; ".join(parts)).strip()
            if not clean or clean in seen:
                return
            seen.add(clean)
            bundle_lines.append(f"- {clean}")

        def compatible_table_for(item: str, qty: int | None) -> dict[str, Any] | None:
            item_norm = str(item or "none").strip().lower()
            if item_norm not in {"wood", "sand", "stone"}:
                return None
            compatible = [
                fact for fact in tables
                if _stable_fair_table_supports_carry(
                    fact,
                    item_norm,
                    qty,
                    require_full_qty=False,
                )
            ]
            return dict(compatible[0]) if compatible else None

        def primary_fact(kind: str) -> dict[str, Any] | None:
            kind_norm = str(kind or "").strip().lower()
            for source in (
                (routing_contract or {}).get("primary_fact") if isinstance(routing_contract, dict) else None,
                (feature or {}).get("best_fact") if isinstance(feature, dict) else None,
                (routing_contract or {}).get("backup_fact") if isinstance(routing_contract, dict) else None,
                (feature or {}).get("backup_fact") if isinstance(feature, dict) else None,
            ):
                normalized = self._normalize_stable_fair_protocol_candidate_fact(source)
                if str(normalized.get("kind") or "") == kind_norm and _extract_xy(normalized.get("position")) is not None:
                    return normalized
            pool = resources if kind_norm == "resource" else tables
            return dict(pool[0]) if pool else None

        sender_carry_active = carry_type in {"wood", "sand", "stone"}
        recipient_carry_active = recipient_carry_type in {"wood", "sand", "stone"}
        direct_delivery_context = bool(
            sender_carry_active
            or recipient_carry_active
            or objective in {"ask_missing_table", "share_table_for_carrier", "delivery_ready", "answer_request"}
        )

        if objective == "ask_missing_table" and item_type in {"wood", "sand", "stone"}:
            table = compatible_table_for(item_type, item_qty)
            if table:
                add(table_line(table, focus_type=item_type))
            else:
                qty_text = int(item_qty) if item_qty is not None else "?"
                add(f"REQUEST table_for={item_type}; carry={item_type}:{qty_text}; loc=({sender_pos[0]},{sender_pos[1]}); compatible_table=none")
            return lines[:3] or ["- STATUS no useful content candidate"]

        if objective in {"share_table_for_carrier", "delivery_ready"}:
            table = compatible_table_for(item_type, item_qty) or primary_fact("table")
            if table:
                add(table_line(table, focus_type=item_type))
        elif objective == "answer_request":
            table = compatible_table_for(item_type, item_qty)
            if table:
                add(table_line(table, focus_type=item_type))
            else:
                resource = primary_fact("resource")
                if resource:
                    add(resource_line(resource))
        elif objective == "share_resource":
            resource = primary_fact("resource")
            if resource:
                add(resource_line(resource))
        elif objective == "share_table":
            table = primary_fact("table")
            if table:
                add(table_line(table, focus_type=item_type))

        resource_first_lines: list[str] = []
        if not direct_delivery_context and recipient_evidence:
            resource_focus = primary_fact("resource")
            table_focus = primary_fact("table")
            fresh_resources = [fact for fact in resources if fact_new_for_recipient(fact)]
            fresh_tables = [fact for fact in tables if fact_new_for_recipient(fact)]
            ordered_resources = order_resources(
                [
                    resource_focus if fact_new_for_recipient(resource_focus) else None,
                    *fresh_resources,
                    resource_focus,
                    *resources,
                ]
            )
            ordered_tables = unique_ordered(
                [
                    table_focus if fact_new_for_recipient(table_focus) else None,
                    *fresh_tables,
                    table_focus,
                    *tables,
                ]
            )
            if ordered_resources and ordered_tables:
                add_bundle([ordered_resources[0], ordered_tables[0]])
            if len(ordered_resources) >= 2:
                add_bundle([ordered_resources[0], ordered_resources[1]])
            if len(ordered_tables) >= 2:
                add_bundle([ordered_tables[0], ordered_tables[1]])
            for fact in ordered_resources[:1]:
                clean = re.sub(r"\s+", " ", resource_line(fact)).strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    resource_first_lines.append(f"- {clean}")

        if not lines and not resource_first_lines and carry_type in {"wood", "sand", "stone"}:
            table = compatible_table_for(carry_type, carry_qty)
            if table:
                add(table_line(table, focus_type=carry_type))
            else:
                qty_text = int(carry_qty) if carry_qty is not None else "?"
                add(f"REQUEST table_for={carry_type}; carry={carry_type}:{qty_text}; loc=({sender_pos[0]},{sender_pos[1]}); compatible_table=none")

        if not lines and not resource_first_lines:
            resource = primary_fact("resource")
            table = primary_fact("table")
            if resource:
                add(resource_line(resource))
            elif table:
                add(table_line(table, focus_type=item_type))

        if not lines and not resource_first_lines:
            add(f"STATUS loc=({sender_pos[0]},{sender_pos[1]}); carry={carry_type}:{carry_qty if carry_qty is not None else '?'}")
        if not direct_delivery_context and resource_first_lines:
            final_lines = resource_first_lines + bundle_lines + lines
        elif not direct_delivery_context and bundle_lines and lines and any(line.startswith("- RESOURCE ") for line in lines):
            final_lines = lines + bundle_lines
        else:
            final_lines = (bundle_lines + lines) if not direct_delivery_context and bundle_lines else lines
        return final_lines[:3]

    def _parse_stable_fair_protocol_lane_response(self, response: str, max_chars: int) -> tuple[str, str]:
        text = str(response or "").strip()
        lane_match = re.search(r"^\s*LANE\s*:\s*([A-Za-z_ -]+)\s*$", text, re.IGNORECASE | re.MULTILINE)
        type_match = re.search(r"^\s*TYPE\s*:\s*([A-Za-z_ -]+)\s*$", text, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"^\s*MSG\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        msg = str(msg_match.group(1) or "").strip() if msg_match else _stable_fair_one_line_message(text)
        msg = self._sanitize_stable_fair_semantic_message_line(msg, max_chars)
        raw_lane = str((lane_match or type_match).group(1) if (lane_match or type_match) else "").strip().upper()
        raw_lane = raw_lane.replace(" ", "_").replace("-", "_")
        if raw_lane in {"TABLE_FOR_CARRY", "DELIVERY"}:
            raw_lane = "CARRY"
        if raw_lane not in {"RESOURCE", "TABLE", "CARRY", "REQUEST", "STATUS"}:
            semantic_type = self._infer_stable_fair_semantic_type(msg)
            if semantic_type == "RESOURCE":
                raw_lane = "RESOURCE"
            elif semantic_type == "TABLE":
                raw_lane = "TABLE"
            elif semantic_type in {"TABLE_FOR_CARRY", "DELIVERY", "CARRY"}:
                raw_lane = "CARRY"
            elif semantic_type in {"REQUEST", "REPLY"}:
                raw_lane = "REQUEST"
            else:
                raw_lane = "STATUS"
        return raw_lane, msg

    def _stable_fair_protocol_lane_for_message(self, message: str) -> str:
        semantic_type = self._infer_stable_fair_semantic_type(message)
        if semantic_type == "RESOURCE":
            return "RESOURCE"
        if semantic_type in {"TABLE", "TABLE_FOR_CARRY", "DELIVERY", "CARRY"}:
            return "TABLE" if "table" in str(message or "").lower() else "CARRY"
        if semantic_type in {"REQUEST", "REPLY"}:
            return "REQUEST"
        return "STATUS"

    def _stable_fair_message_has_observable_protocol_claim(self, lane: str, message: str) -> bool:
        low = str(message or "").lower()
        lane_norm = str(lane or "STATUS").strip().upper()
        semantic_type = self._infer_stable_fair_semantic_type(message)
        if lane_norm == "REQUEST" or semantic_type in {"REQUEST", "REPLY"}:
            return False
        if "table" in low or re.search(r"\b(?:deposit|accepts?|capacity|space|type_rem|remaining)\b", low):
            return True
        if lane_norm == "RESOURCE" or semantic_type == "RESOURCE":
            return True
        return False

    def _stable_fair_table_message_can_be_observed(self, message: str) -> bool:
        low = str(message or "").lower()
        if re.search(r"\b(?:pickup|pick\s+up|collect|gather)\b[^.]*\bfrom\s+(?:a\s+|the\s+)?table\b", low):
            return False
        if re.search(r"\btable\b[^.]*\bhas\b[^.]*\b(?:wood|sand|stone)\b", low) and not re.search(
            r"\b(?:can\s+deposit|can\s+accept|accepts?|deposit|remaining|capacity|space|slot|room|type_rem)\b",
            low,
        ):
            return False
        if "table" in low and re.search(r"\b(?:resources?|materials?|available\s+for\s+pickup)\b", low):
            return False
        return True

    def _stable_fair_message_table_claim_items(self, message: str) -> set[str]:
        low = str(message or "").lower()
        if not (
            "table" in low
            or re.search(r"\b(?:deposit|accepts?|capacity|space|slot|room|type_rem|remaining)\b", low)
        ):
            return set()
        claimed: set[str] = set()
        for item_type in ("wood", "sand", "stone"):
            if item_type not in low:
                continue
            patterns = (
                rf"\b(?:table|deposit|accepts?|capacity|space|slot|room|type_rem|remaining)\b[^.;]*\b{item_type}\b",
                rf"\b{item_type}\b[^.;]*\b(?:deposit|accepts?|capacity|space|slot|room|type_rem|remaining|to|toward|for)\b",
            )
            if any(re.search(pattern, low) for pattern in patterns):
                claimed.add(item_type)
        table_for_type = _stable_fair_message_table_for_type(low)
        if table_for_type in {"wood", "sand", "stone"}:
            claimed.add(table_for_type)
        return claimed

    def _stable_fair_table_fact_supports_message_claim(
        self,
        fact: dict[str, Any],
        message: str,
        *,
        carry_type: str,
        carry_qty: int | None,
    ) -> bool:
        if not self._stable_fair_table_message_can_be_observed(message):
            return False
        low = str(message or "").lower()
        claimed_items = self._stable_fair_message_table_claim_items(low)
        carry_norm = str(carry_type or "none").strip().lower()
        if (
            not claimed_items
            and carry_norm in {"wood", "sand", "stone"}
            and re.search(r"\b(?:deliver|delivering|bring|bringing|putting down|put down|deposit)\b", low)
        ):
            claimed_items.add(carry_norm)
        for item_type in claimed_items:
            if _stable_fair_table_supports_carry(
                fact,
                item_type,
                carry_qty if item_type == carry_norm else None,
                require_full_qty=False,
            ):
                continue
            if _stable_fair_negative_table_capacity_mention(low, item_type):
                continue
            return False
        return True

    def _stable_fair_protocol_obs_facts_for_plain_message(
        self,
        lane: str,
        message: str,
        facts: list[dict[str, Any]],
        *,
        carry_type: str = "none",
        carry_qty: int | None = None,
    ) -> list[dict[str, Any]]:
        line = str(message or "")
        low = line.lower()
        lane_norm = str(lane or "STATUS").strip().upper()
        semantic_type = self._infer_stable_fair_semantic_type(line)
        coords = _stable_fair_message_coords(line)
        mentioned_items = {item for item in ("wood", "sand", "stone") if item in low}
        wants_table = bool(
            "table" in low
            or lane_norm == "TABLE"
            or semantic_type in {"TABLE", "TABLE_FOR_CARRY", "DELIVERY"}
            or re.search(r"\b(?:deposit|accepts?|capacity|space|type_rem|remaining)\b", low)
        )
        if lane_norm == "REQUEST" or semantic_type in {"REQUEST", "REPLY"}:
            wants_table = False
        if wants_table and not self._stable_fair_table_message_can_be_observed(line):
            wants_table = False
        wants_resource = bool(
            lane_norm == "RESOURCE"
            or semantic_type == "RESOURCE"
            or (
                re.search(r"\b(?:resource|pickup|pick up|collect|gather|available|found)\b", low)
                and "table" not in low
            )
        )
        normalized_facts = [
            self._normalize_stable_fair_protocol_candidate_fact(fact)
            for fact in list(facts or [])
            if isinstance(fact, dict)
        ]
        resources = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "resource"]
        tables = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "table"]
        selected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add(fact: dict[str, Any] | None) -> None:
            if not isinstance(fact, dict):
                return
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            kind = str(normalized.get("kind") or "").strip().lower()
            if kind not in {"resource", "table"} or _extract_xy(normalized.get("position")) is None:
                return
            key = self._stable_fair_protocol_candidate_fact_key(normalized)
            if key in seen:
                return
            seen.add(key)
            selected.append(normalized)

        for coord in coords:
            for fact in resources:
                if _extract_xy(fact.get("position")) != coord:
                    continue
                resource_type = str(fact.get("resource_type") or "").strip().lower()
                if wants_resource or resource_type in mentioned_items:
                    add(fact)
            for fact in tables:
                if _extract_xy(fact.get("position")) == coord and wants_table and self._stable_fair_table_fact_supports_message_claim(
                    fact,
                    line,
                    carry_type=carry_type,
                    carry_qty=carry_qty,
                ):
                    add(fact)
            if len(selected) >= 2:
                return selected[:2]

        if not selected and re.search(r"\b(?:here|current tile|this table|this resource)\b", low):
            if wants_resource:
                for fact in resources:
                    resource_type = str(fact.get("resource_type") or "").strip().lower()
                    if str(fact.get("source") or "") == "current_tile" and (
                        not mentioned_items or resource_type in mentioned_items
                    ):
                        add(fact)
                        break
            if wants_table:
                for fact in tables:
                    if str(fact.get("source") or "") == "current_tile" and self._stable_fair_table_fact_supports_message_claim(
                        fact,
                        line,
                        carry_type=carry_type,
                        carry_qty=carry_qty,
                    ):
                        add(fact)
                        break
        return selected[:2]

    def _stable_fair_select_protocol_lane_fact(
        self,
        lane: str,
        message: str,
        facts: list[dict[str, Any]],
        *,
        carry_type: str,
        carry_qty: int | None,
    ) -> dict[str, Any]:
        lane_norm = str(lane or "STATUS").strip().upper()
        line = str(message or "")
        low = line.lower()
        coords = _stable_fair_message_coords(line)
        resources = [fact for fact in facts if str(fact.get("kind") or "") == "resource"]
        tables = [fact for fact in facts if str(fact.get("kind") or "") == "table"]

        def by_coord(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            for coord in coords:
                for fact in candidates:
                    if _extract_xy(fact.get("position")) == coord:
                        return dict(fact)
            return None

        def best_table_for(item_type: str) -> dict[str, Any] | None:
            if item_type in {"wood", "sand", "stone"}:
                compatible = [
                    fact for fact in tables
                    if _stable_fair_table_supports_carry(fact, item_type, carry_qty, require_full_qty=False)
                ]
                if compatible:
                    return dict(compatible[0])
            return dict(tables[0]) if tables else None

        table_for_type = _stable_fair_message_table_for_type(line)
        if table_for_type == "none" and carry_type in {"wood", "sand", "stone"}:
            table_for_type = carry_type
        inferred = self._infer_stable_fair_semantic_type(line)
        wants_table = lane_norm in {"TABLE", "CARRY"} or inferred in {"TABLE", "TABLE_FOR_CARRY", "DELIVERY", "CARRY"}
        wants_resource = lane_norm == "RESOURCE" or inferred == "RESOURCE"
        if "table" in low and not wants_resource:
            wants_table = True
        if any(word in low for word in ("resource", "pickup", "pick up", "collect", "available")) and "table" not in low:
            wants_resource = True

        if wants_table:
            matched_table = by_coord(tables)
            if matched_table is not None:
                return matched_table
            selected_table = best_table_for(table_for_type)
            if selected_table is not None:
                return selected_table
        if wants_resource:
            matched_resource = by_coord(resources)
            if matched_resource is not None:
                return matched_resource
            if resources:
                return dict(resources[0])
        matched_any = by_coord(tables) or by_coord(resources)
        if matched_any is not None:
            return matched_any
        if lane_norm == "REQUEST" and carry_type in {"wood", "sand", "stone"}:
            selected_table = best_table_for(carry_type)
            if selected_table is not None:
                return selected_table
        return {"kind": "status"}

    def _stable_fair_protocol_obs_facts_for_message(
        self,
        lane: str,
        message: str,
        facts: list[dict[str, Any]],
        *,
        carry_type: str,
        carry_qty: int | None,
    ) -> list[dict[str, Any]]:
        lane_norm = str(lane or "STATUS").strip().upper()
        line = str(message or "")
        low = line.lower()
        coords = _stable_fair_message_coords(line)
        mentioned_items = {item for item in ("wood", "sand", "stone") if item in low}
        inferred = self._infer_stable_fair_semantic_type(line)
        wants_table = bool(
            "table" in low
            or lane_norm in {"TABLE", "CARRY"}
            or inferred in {"TABLE", "TABLE_FOR_CARRY", "DELIVERY", "CARRY"}
        )
        wants_resource = bool(
            lane_norm == "RESOURCE"
            or inferred == "RESOURCE"
            or (
                any(word in low for word in ("resource", "pickup", "pick up", "collect", "available", "found"))
                and "table" not in low
            )
        )
        normalized_facts = [
            self._normalize_stable_fair_protocol_candidate_fact(fact)
            for fact in list(facts or [])
            if isinstance(fact, dict)
        ]
        selected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add(fact: dict[str, Any] | None) -> None:
            if not isinstance(fact, dict):
                return
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            kind = str(normalized.get("kind") or "").strip().lower()
            if kind not in {"resource", "table"} or _extract_xy(normalized.get("position")) is None:
                return
            key = self._stable_fair_protocol_candidate_fact_key(normalized)
            if key in seen:
                return
            seen.add(key)
            selected.append(normalized)

        resources = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "resource"]
        tables = [fact for fact in normalized_facts if str(fact.get("kind") or "") == "table"]
        for coord in coords:
            coord_resources = [fact for fact in resources if _extract_xy(fact.get("position")) == coord]
            coord_tables = [fact for fact in tables if _extract_xy(fact.get("position")) == coord]
            if wants_resource or any(str(fact.get("resource_type") or "").lower() in mentioned_items for fact in coord_resources):
                for fact in coord_resources:
                    add(fact)
            if wants_table or coord_tables:
                for fact in coord_tables:
                    add(fact)
            if len(selected) >= 2:
                return selected[:2]

        if not selected:
            primary = self._stable_fair_select_protocol_lane_fact(
                lane_norm,
                line,
                normalized_facts,
                carry_type=carry_type,
                carry_qty=carry_qty,
            )
            add(primary)

        if len(selected) == 1:
            first = selected[0]
            first_kind = str(first.get("kind") or "").strip().lower()
            if first_kind == "resource" and ("table" in low or any(word in low for word in ("deliver", "deposit", "accept"))):
                resource_type = str(first.get("resource_type") or "").strip().lower()
                for fact in tables:
                    if resource_type in {"wood", "sand", "stone"} and _stable_fair_table_supports_carry(
                        fact,
                        resource_type,
                        None,
                        require_full_qty=False,
                    ):
                        add(fact)
                        break
            elif first_kind == "table" and wants_resource:
                for fact in resources:
                    resource_type = str(fact.get("resource_type") or "").strip().lower()
                    if not mentioned_items or resource_type in mentioned_items:
                        add(fact)
                        break
        return selected[:2]

    def _stable_fair_protocol_obs_block_for_facts(
        self,
        lane: str,
        facts: list[dict[str, Any]],
    ) -> str:
        lines = [
            self._stable_fair_protocol_lane_obs_line(lane, fact, focus_type="none")
            for fact in list(facts or [])[:2]
            if isinstance(fact, dict)
        ]
        lines = [line for line in lines if line and line != "OBS_STATUS: none"]
        return "\n".join(lines) if lines else "OBS_STATUS: none"

    def _stable_fair_protocol_lane_obs_line(
        self,
        lane: str,
        fact: dict[str, Any] | None,
        *,
        focus_type: str = "none",
    ) -> str:
        del lane
        selected_fact = self._normalize_stable_fair_protocol_candidate_fact(fact)
        kind = str(selected_fact.get("kind") or "").strip().lower()
        pos = _extract_xy(selected_fact.get("position"))
        source = str(selected_fact.get("source") or "protocol_lane").strip().lower() or "protocol_lane"
        source = re.sub(r"[^a-z0-9_:-]+", "_", source)[:32] or "protocol_lane"
        if kind == "resource" and pos is not None:
            resource_type = str(selected_fact.get("resource_type") or "resource").strip().lower()
            qty = _coerce_int(selected_fact.get("quantity"))
            return f"OBS_RESOURCE: resource={resource_type}@({pos[0]},{pos[1]}):{int(qty) if qty is not None else 0}; src={source}"
        if kind == "table" and pos is not None:
            rem = _coerce_int(selected_fact.get("remaining_capacity"))
            rb = _stable_fair_positive_remaining_by_type(selected_fact)
            focus_norm = str(focus_type or "none").strip().lower()
            if focus_norm in {"wood", "sand", "stone"} and rb.get(focus_norm, 0) > 0:
                rb = {focus_norm: rb[focus_norm]}
            parts = [f"table=({pos[0]},{pos[1]})"]
            if rem is not None:
                parts.append(f"total_deposit_space={max(0, int(rem))}")
            if rb:
                parts.append(
                    "can_deposit="
                    + ",".join(f"{item_type}:{rb[item_type]}" for item_type in ("wood", "sand", "stone") if item_type in rb)
                )
            parts.append(f"src={source}")
            return "OBS_TABLE: " + "; ".join(parts)
        return "OBS_STATUS: none"

    def _stable_fair_protocol_safe_fallback_message(
        self,
        fact: dict[str, Any] | None,
        *,
        neutral_message: str,
    ) -> str:
        del fact
        return neutral_message

    def _stable_fair_delivery_fact_text(
        self,
        fact: dict[str, Any] | None,
        *,
        focus_type: str = "none",
    ) -> str:
        normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
        kind = str(normalized.get("kind") or "").strip().lower()
        pos = _extract_xy(normalized.get("position"))
        if kind == "resource" and pos is not None:
            resource_type = str(normalized.get("resource_type") or "resource").strip().lower()
            qty = _coerce_int(normalized.get("quantity"))
            qty_text = int(qty) if qty is not None else 0
            return f"{resource_type}@({pos[0]},{pos[1]}) qty={qty_text}"
        if kind == "table" and pos is not None:
            rb = _stable_fair_positive_remaining_by_type(normalized)
            focus_norm = str(focus_type or "none").strip().lower()
            if focus_norm in {"wood", "sand", "stone"} and rb.get(focus_norm, 0) > 0:
                rb = {focus_norm: rb[focus_norm]}
            can_deposit = ",".join(
                f"{item_type}:{rb[item_type]}"
                for item_type in ("wood", "sand", "stone")
                if item_type in rb
            )
            rem = _coerce_int(normalized.get("remaining_capacity"))
            total_text = f" total={int(rem)}" if rem is not None else ""
            return f"table@({pos[0]},{pos[1]}) can_deposit={can_deposit or 'unknown'}{total_text}"
        return ""

    def _stable_fair_delivery_context_meta(self, feature: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(feature, dict):
            return {}
        meta: dict[str, Any] = {}
        name = str(feature.get("name") or "").strip()
        if name:
            meta["recipient"] = name
        best_label = self._stable_fair_delivery_fact_text(feature.get("best_fact"))
        if best_label:
            meta["recipient_fact"] = best_label
        backup_label = self._stable_fair_delivery_fact_text(feature.get("backup_fact"))
        if backup_label and backup_label != best_label:
            meta["recipient_backup_fact"] = backup_label
        carrying = str(feature.get("carrying") or "none").strip().lower()
        if carrying in {"wood", "sand", "stone"}:
            qty = _coerce_int(feature.get("carrying_qty"))
            meta["recipient_carry"] = f"{carrying}:{int(qty) if qty is not None else '?'}"
        table_for_type = str(feature.get("table_for_type") or "").strip().lower()
        if table_for_type in {"wood", "sand", "stone"}:
            meta["table_request_type"] = table_for_type
        return meta

    def _stable_fair_protocol_has_recipient_evidence(
        self,
        feature: dict[str, Any] | None,
        routing_contract: dict[str, Any] | None,
    ) -> bool:
        if isinstance(feature, dict):
            if isinstance(feature.get("best_fact"), dict) or isinstance(feature.get("backup_fact"), dict):
                return True
            if str(feature.get("carrying") or "none").strip().lower() in {"wood", "sand", "stone"}:
                return True
            if str(feature.get("latest_incoming_ask_type") or "none").strip().lower() in {"wood", "sand", "stone"}:
                return True
            edge = str(feature.get("routing_edge") or "").strip().lower()
            intent = str(feature.get("routing_intent") or feature.get("routing_hint") or "").strip().lower()
            if edge and edge not in {"none", "coverage", "generic"}:
                return True
            if intent and intent not in {"none", "coverage", "generic"}:
                return True
        if isinstance(routing_contract, dict):
            if isinstance(routing_contract.get("primary_fact"), dict) or isinstance(routing_contract.get("backup_fact"), dict):
                return True
            if str(routing_contract.get("recipient_carrying") or "none").strip().lower() in {"wood", "sand", "stone"}:
                return True
            if str(routing_contract.get("latest_incoming_ask_type") or "none").strip().lower() in {"wood", "sand", "stone"}:
                return True
            ask = str(routing_contract.get("ask") or "none").strip().lower()
            objective = str(routing_contract.get("objective") or "").strip().lower()
            if ask not in {"", "none"}:
                return True
            if objective and objective not in {"coverage", "none"}:
                return True
        return False

    def _stable_fair_table_candidates(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        data = dict(obs or {})
        candidates: list[dict[str, Any]] = []
        try:
            for entry in list(data.get("tables_info") or []):
                pos = _extract_xy(entry.get("position"))
                if pos is None:
                    continue
                rem = _coerce_int(entry.get("remaining_capacity"))
                if rem is not None and rem <= 0:
                    continue
                candidates.append(
                    {
                        "kind": "table",
                        "position": pos,
                        "remaining_capacity": rem,
                        "remaining_by_type": dict(entry.get("remaining_by_type") or {}),
                        "source": str(entry.get("source") or "visible").strip().lower() or "visible",
                    }
                )
            for entry in list(data.get("kb_nearest_tables") or []):
                pos = _extract_xy(entry.get("position"))
                if pos is None:
                    continue
                rem = _coerce_int(entry.get("remaining_capacity"))
                if rem is not None and rem <= 0:
                    continue
                candidates.append(
                    {
                        "kind": "table",
                        "position": pos,
                        "remaining_capacity": rem,
                        "remaining_by_type": dict(entry.get("remaining_by_type") or {}),
                        "source": str(entry.get("source") or "memory").strip().lower() or "memory",
                    }
                )
        except Exception:
            return []
        return candidates

    def _stable_fair_best_compatible_table_fact(
        self,
        obs: dict[str, Any],
        item_type: str,
        item_qty: int | None,
        *,
        require_full_qty: bool = False,
        origin: Any = None,
        prefer_local: bool = False,
    ) -> dict[str, Any] | None:
        item_norm = str(item_type or "none").strip().lower()
        if item_norm not in {"wood", "sand", "stone"}:
            return None
        data = dict(obs or {})
        pos = _extract_xy(origin) or _extract_xy(data.get("position")) or (0, 0)
        candidates = [
            fact for fact in self._stable_fair_table_candidates(obs)
            if _stable_fair_table_supports_carry(
                fact,
                item_norm,
                item_qty,
                require_full_qty=require_full_qty,
            )
        ]
        if not candidates:
            return None

        def local_key(fact: dict[str, Any]) -> tuple[int, int, int, int, int]:
            type_remaining = _stable_fair_table_remaining_for_type(fact, item_norm)
            rem = _coerce_int(fact.get("remaining_capacity"))
            type_value = int(type_remaining) if type_remaining is not None else 0
            rem_value = int(rem) if rem is not None else 0
            qty_raw = _coerce_int(item_qty)
            qty_needed = int(qty_raw) if qty_raw is not None and qty_raw > 0 else None
            if type_remaining is not None and type_value > 0:
                fit_class = 2 if qty_needed is None or type_value >= qty_needed else 1
            elif rem_value > 0:
                fit_class = 1
            else:
                fit_class = 0
            distance = _manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0
            source = str(fact.get("source") or "").strip().lower()
            source_rank = 1 if source in {"current_tile", "visible"} else 0
            return (fit_class, -int(distance), source_rank, type_value, rem_value)

        def key(fact: dict[str, Any]) -> tuple[int, int, int]:
            type_remaining = _stable_fair_table_remaining_for_type(fact, item_norm)
            rem = _coerce_int(fact.get("remaining_capacity"))
            distance = _manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0
            return (
                int(type_remaining) if type_remaining is not None else 0,
                int(rem) if rem is not None else 0,
                -distance,
            )

        candidates.sort(key=local_key if prefer_local else key, reverse=True)
        return dict(candidates[0])

    def _stable_fair_delivery_context_lines(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> list[str]:
        data = dict(obs or {})
        recipient_lines: list[str] = []
        local_lines: list[str] = []
        sender_pos = _extract_xy(data.get("position")) or (0, 0)
        target_info = dict((data.get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_pos = _extract_xy(target_info.get("position"))
        action_info = dict(data.get("current_round_action_intent") or {})
        action_summary = _stable_fair_one_line_message(
            action_info.get("summary") or data.get("current_round_action_summary") or ""
        )

        def remote_table(fact: dict[str, Any] | None) -> bool:
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            if str(normalized.get("kind") or "") != "table":
                return False
            fact_pos = _extract_xy(normalized.get("position"))
            if fact_pos is None:
                return False
            distances = [int(_manhattan_distance(sender_pos, fact_pos) or 0)]
            if recipient_pos is not None:
                distances.append(int(_manhattan_distance(recipient_pos, fact_pos) or 0))
            return min(distances) > 8

        if isinstance(feature, dict):
            best_text = self._stable_fair_delivery_fact_text(feature.get("best_fact"))
            if best_text and not remote_table(feature.get("best_fact")):
                recipient_lines.append(f"- selected teammate fact: {best_text}")
            backup_text = self._stable_fair_delivery_fact_text(feature.get("backup_fact"))
            if backup_text and backup_text != best_text and not remote_table(feature.get("backup_fact")):
                recipient_lines.append(f"- paired teammate fact: {backup_text}")
            teammate_carry = str(feature.get("carrying") or "none").strip().lower()
            teammate_qty = _coerce_int(feature.get("carrying_qty"))
            if teammate_carry in {"wood", "sand", "stone"}:
                teammate_qty_text = int(teammate_qty) if teammate_qty is not None else "?"
                recipient_lines.append(f"- {recipient} carries {teammate_carry}:{teammate_qty_text}")
                teammate_table = self._stable_fair_best_compatible_table_fact(
                    data,
                    teammate_carry,
                    teammate_qty,
                    require_full_qty=False,
                    origin=recipient_pos,
                    prefer_local=True,
                )
                if teammate_table:
                    recipient_lines.append(
                        f"- useful table for {recipient}: {self._stable_fair_delivery_fact_text(teammate_table, focus_type=teammate_carry)}"
                    )

        if action_summary:
            local_lines.append(f"- current action: {action_summary[:160]}")

        carry_norm = str(carry_type or "none").strip().lower()
        qty_text = str(int(carry_qty)) if carry_qty is not None else "?"
        if carry_norm in {"wood", "sand", "stone"}:
            local_lines.append(f"- current load: {carry_norm}:{qty_text}")
            compatible = self._stable_fair_best_compatible_table_fact(
                data,
                carry_norm,
                carry_qty,
                require_full_qty=False,
                prefer_local=True,
            )
            if compatible:
                local_lines.append(
                    f"- usable table for current load: {self._stable_fair_delivery_fact_text(compatible, focus_type=carry_norm)}"
                )
            else:
                local_lines.append(f"- missing table: no shown table can deposit carried {carry_norm}; a useful message may ask for one")

        resource_fact = self._stable_fair_best_resource_fact(data)
        if resource_fact:
            resource_type = str(resource_fact.get("resource_type") or "").strip().lower()
            resource_text = self._stable_fair_delivery_fact_text(resource_fact)
            if resource_text:
                compatible = self._stable_fair_best_compatible_table_fact(
                    data,
                    resource_type,
                    None,
                    require_full_qty=False,
                    origin=_extract_xy(resource_fact.get("position")) or sender_pos,
                    prefer_local=True,
                )
                if compatible:
                    local_lines.append(
                        f"- possible resource-table pair: {resource_text} -> {self._stable_fair_delivery_fact_text(compatible, focus_type=resource_type)}"
                    )
                else:
                    local_lines.append(f"- visible resource: {resource_text}")

        deduped: list[str] = []
        seen: set[str] = set()
        for line in recipient_lines + local_lines:
            key = re.sub(r"\s+", " ", line.strip().lower())
            if key and key not in seen:
                seen.add(key)
                deduped.append(line)
            if len(deduped) >= 6:
                break
        return deduped

    def _stable_fair_repair_state_obs_message(
        self,
        lane: str,
        message: str,
        *,
        selected_facts: list[dict[str, Any]],
        candidate_facts: list[dict[str, Any]],
        state_line: str,
        obs: dict[str, Any],
        recipient: str,
        carry_type: str,
        carry_qty: int | None,
    ) -> tuple[str, str, list[dict[str, Any]], str, str] | None:
        carry_norm = str(carry_type or "none").strip().lower()
        if carry_norm not in {"wood", "sand", "stone"}:
            return None
        lane_norm = str(lane or "STATUS").strip().upper()
        low = str(message or "").lower()
        if not (
            lane_norm in {"CARRY", "DELIVERY", "TABLE", "REQUEST"}
            or any(word in low for word in ("table", "deliver", "deposit", "put down", "carrying", "accept"))
        ):
            return None
        pickup_match = re.search(
            r"\b(?:picking\s+up|pick\s+up|pickup|collect|gather)\s+(?:\d+\s+)?(wood|sand|stone)\b",
            low,
        )
        if pickup_match and str(pickup_match.group(1) or "").lower() != carry_norm:
            return None

        facts = [
            self._normalize_stable_fair_protocol_candidate_fact(fact)
            for fact in list(selected_facts or []) + list(candidate_facts or [])
            if isinstance(fact, dict)
        ]
        table_facts = [fact for fact in facts if str(fact.get("kind") or "") == "table"]
        compatible_tables = [
            fact for fact in table_facts
            if _stable_fair_table_supports_carry(
                fact,
                carry_norm,
                carry_qty,
                require_full_qty=False,
            )
        ]
        qty_text = str(int(carry_qty)) if carry_qty is not None else "?"
        if compatible_tables:
            sender_pos = _extract_xy(dict(obs or {}).get("position")) or (0, 0)
            qty_raw = _coerce_int(carry_qty)
            qty_needed = int(qty_raw) if qty_raw is not None and qty_raw > 0 else None

            def repair_table_key(fact: dict[str, Any]) -> tuple[int, int, int, int]:
                remaining = _stable_fair_table_remaining_for_type(fact, carry_norm)
                remaining_value = int(remaining) if remaining is not None else 0
                fit_class = 2 if qty_needed is None or remaining_value >= qty_needed else 1
                distance = int(_manhattan_distance(sender_pos, _extract_xy(fact.get("position"))) or 0)
                rem = int(_coerce_int(fact.get("remaining_capacity")) or 0)
                return (fit_class, -distance, remaining_value, rem)

            compatible_tables.sort(
                key=repair_table_key,
                reverse=True,
            )
            fact = dict(compatible_tables[0])
            pos = _extract_xy(fact.get("position"))
            if pos is None:
                return None
            remaining = _stable_fair_table_remaining_for_type(fact, carry_norm)
            if remaining is not None:
                repair_message = f"I carry {carry_norm}:{qty_text}; table ({pos[0]},{pos[1]}) has {carry_norm} space {int(remaining)}."
            else:
                repair_message = f"I carry {carry_norm}:{qty_text}; table ({pos[0]},{pos[1]}) can deposit {carry_norm}."
            repair_lane = "CARRY"
            repair_facts = [fact]
            repair_reason = "safe_current_load_table"
        else:
            repair_message = f"I carry {carry_norm}:{qty_text} and need a table that accepts {carry_norm}."
            repair_lane = "REQUEST"
            repair_facts = table_facts[:1]
            repair_reason = "safe_current_load_request"

        repair_obs_line = self._stable_fair_protocol_obs_block_for_facts(repair_lane, repair_facts)
        if not self._stable_fair_state_obs_message_safe(
            repair_message,
            state_line=state_line,
            obs_line=repair_obs_line,
            obs=obs,
            recipient=recipient,
            candidate_facts=candidate_facts,
        ):
            return None
        return repair_lane, repair_message, repair_facts, repair_obs_line, repair_reason

    def _build_stable_fair_state_obs_message_prompt(
        self,
        obs: dict[str, Any],
        recipient: str,
        *,
        state_line: str,
        content_candidate_lines: list[str],
        candidate_fact_lines: list[str],
        delivery_context_lines: list[str],
        feature: dict[str, Any] | None,
        routing_contract: dict[str, Any] | None,
        recipient_evidence: bool,
        carry_type: str,
        carry_qty: int | None,
    ) -> str:
        data = dict(obs or {})
        target_info = dict((data.get("agents_in_communication_range") or {}).get(recipient, {}) or {})
        target_pos = _extract_xy(target_info.get("position"))
        target_pos_text = f"({target_pos[0]},{target_pos[1]})" if target_pos is not None else "unknown"
        routing_reason_lines = self._stable_fair_protocol_routing_reason_lines(routing_contract)
        del feature
        limit = max(40, min(120, int(self.proto_v2_free_max or 120)))
        content_block = list(content_candidate_lines or ["- STATUS no useful content candidate"])[:3]
        facts_block = list(candidate_fact_lines or ["- none"])[:4]
        routing_block = list(routing_reason_lines or ["- none"])[:4]
        delivery_block = list(delivery_context_lines or ["- none"])[:3]
        opening = (
            f"You are {self.owner.name}. Send one useful message to {recipient}."
            if recipient_evidence
            else f"You are {self.owner.name}. Write one short teammate message."
        )
        audience_block = (
            [
                "Recipient:",
                f"- position={target_pos_text}; carrying={target_info.get('carrying_resource', 'unknown')}",
                f"- current_load={carry_type}:{carry_qty if carry_qty is not None else '?'}",
            ]
            if recipient_evidence
            else [
                "Audience:",
                f"- current_load={carry_type}:{carry_qty if carry_qty is not None else '?'}",
            ]
        )
        if not recipient_evidence:
            routing_block = ["- none"]
        recipient_guidance = (
            [
                "- Treat the Routing brief as recipient context, not as a required message type.",
                "- For a table need, ask for a compatible table if none is grounded; share the compatible TABLE if one is grounded.",
                "- When sharing RESOURCE/TABLE facts, keep them directly factual; do not describe an unrelated resource as useful for a different carried item.",
            ]
            if recipient_evidence
            else []
        )
        return "\n".join(
            [
                opening,
                "Protocol will attach STATE/OBS from the evidence facts that support your message.",
                "",
                "Output EXACTLY two lines:",
                "LANE: <RESOURCE|TABLE|CARRY|REQUEST|STATUS>",
                f"MSG: <one short grounded message under {limit} characters>",
                "",
                "STATE:",
                state_line,
                "",
                *audience_block,
                "",
                "Routing brief:",
                *routing_block,
                "",
                "Content candidates:",
                *content_block,
                "",
                "Evidence:",
                *facts_block,
                "",
                "Useful context:",
                *delivery_block,
                "",
                "Guidance:",
                "- Choose the content yourself; use a grounded fact when it helps the teammate act.",
                "- Choose one compact update from Content candidates; Evidence can support or qualify it.",
                "- A BUNDLE candidate may include up to two grounded facts when both help the teammate plan.",
                "- For BUNDLE, use RESOURCE lane for resource-only content and TABLE lane for table or resource+table content.",
                "- Turn candidate fields into a short teammate message.",
                "- Do not use Evidence as the main message when it is not listed as a Content candidate.",
                "- For REQUEST candidates, ask for the missing compatible fact; mention Evidence tables only as non-answer context unless a TABLE candidate lists them.",
                "- LANE names the message role: RESOURCE/TABLE/CARRY are factual updates; REQUEST is the only question or task-request lane.",
                "- TABLE may describe deposit space useful for a carried item.",
                "- If you or the recipient carries a resource, prefer a compatible table update or request over unrelated facts.",
                "- Write coordinates as (x,y), for example wood@(3,12):14 or table@(3,11):16.",
                "- If you share a RESOURCE or TABLE fact, include its type and coordinate.",
                *recipient_guidance,
                "- Prefer RESOURCE, TABLE, CARRY, or REQUEST over STATUS when a grounded fact or useful load/table context exists.",
                "- Answer a request with a concrete RESOURCE or TABLE fact when available.",
                "- For table facts, talk about deposit capacity or space.",
                "- Describe remote table/resource facts as map information; say you are picking up or depositing now only when STATE action and current grounding support it.",
                "- If no grounded fact is useful, send a brief STATUS; use REQUEST only when a missing table/resource answer would help.",
                "- Do not add explanation outside the two lines.",
                "",
                "Keep completion compact; no explanation.",
            ]
        )

    def _stable_fair_state_obs_message_safe(
        self,
        message: str,
        *,
        state_line: str,
        obs_line: str,
        obs: dict[str, Any],
        recipient: str,
        candidate_facts: list[dict[str, Any]] | None = None,
    ) -> bool:
        line = str(message or "").strip()
        if not line or "SF1|" in line.upper() or "SFPROTO:" in line.upper():
            return False
        if re.search(r"\bbuild\s+(?:one|a\s+table|the\s+table|table)\b", line, re.IGNORECASE):
            return False
        if re.search(r"\b(?:pick\s*up|pickup|collect|gather)\b[^.]*\bfrom\s+(?:a\s+|the\s+)?table\b", line, re.IGNORECASE):
            return False
        prefix = _stable_fair_parse_state_obs_prefix("\n".join([state_line, obs_line, f"MSG: {line}"]))
        state = dict(prefix.get("state") or {})
        obs_facts = [fact for fact in list(prefix.get("obs_facts") or []) if isinstance(fact, dict)]
        state_carry_type = str(state.get("carry_type") or "none").strip().lower()
        target_info = dict((dict(obs or {}).get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_carry_type = str(target_info.get("carrying_resource") or "none").strip().lower()
        if state_carry_type in {"wood", "sand", "stone"} and re.search(r"\b(?:picking\s+up|pick\s+up|pickup)\s+(?:\d+\s+)?(wood|sand|stone)\b", line, re.IGNORECASE):
            return False
        for match in re.finditer(r"\b(?:delivering|deliver|bringing|bring|putting down|put down|carrying)\s+(wood|sand|stone)\b", line, re.IGNORECASE):
            item_type = str(match.group(1) or "").lower()
            if item_type not in {state_carry_type, recipient_carry_type}:
                return False
        for match in re.finditer(r"\b(?:your\s+)?carried\s+(wood|sand|stone)\b", line, re.IGNORECASE):
            item_type = str(match.group(1) or "").lower()
            if item_type not in {state_carry_type, recipient_carry_type}:
                return False
        mentioned_coords = _stable_fair_message_coords(line)
        low = line.lower()
        if "table" in low:
            is_request = bool(re.search(r"\b(?:need|needs|looking for|look for|report|find|check|tell me|do you know|can you|where)\b", low))
            if not is_request:
                table_facts = [fact for fact in obs_facts if str(fact.get("kind") or "") == "table"]
                if mentioned_coords:
                    table_facts = [fact for fact in table_facts if _extract_xy(fact.get("position")) in mentioned_coords]
                if table_facts and re.search(r"\b(?:resources?|materials?|pickup|pick\s+up|collect|gather|available\s+for\s+pickup)\b", low):
                    return False
                if table_facts and re.search(r"\btable\b[^.]*\bhas\b[^.]*\b(?:wood|sand|stone)\b", low) and not re.search(
                    r"\b(?:can\s+deposit|can\s+accept|accepts?|deposit|remaining|capacity|space|slot|room)\b",
                    low,
                ):
                    return False
                for item_type in ("wood", "sand", "stone"):
                    if item_type not in low:
                        continue
                    has_positive = any(
                        _stable_fair_table_supports_carry(fact, item_type, None, require_full_qty=False)
                        for fact in table_facts
                    )
                    has_negative = any(
                        _stable_fair_table_remaining_for_type(fact, item_type) == 0
                        for fact in table_facts
                    ) and _stable_fair_negative_table_capacity_mention(low, item_type)
                    if not has_positive and not has_negative and re.search(r"\b(?:accepts?|remaining|capacity|space|slot|room|deliver|deposit)\b", low):
                        return False
        return self._stable_fair_semantic_message_safe(line, None, obs, recipient)

    def _stable_fair_semantic_nonredundant_message(
        self,
        message: str,
        *,
        semantic_type: str,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        ask: str,
    ) -> str:
        del carry_qty
        line = str(message or "").strip()
        if not line or not isinstance(fact, dict):
            return line
        semantic_norm = str(semantic_type or "STATUS").strip().upper()
        if semantic_norm in {"REQUEST", "REPLY"}:
            return line
        kind = str(fact.get("kind") or "").strip().lower()
        pos = _extract_xy(fact.get("position"))
        compact_line = re.sub(r"\s+", "", line.lower())
        repeats_pos = bool(pos is not None and f"({pos[0]},{pos[1]})" in compact_line)
        if not repeats_pos:
            return line
        ask_norm = str(ask or "none").strip().lower()
        table_for_type = _stable_fair_message_table_for_type(line)
        if table_for_type == "none" and carry_type in {"wood", "sand", "stone"}:
            table_for_type = carry_type
        if semantic_norm == "RESOURCE" and kind == "resource":
            resource_type = str(fact.get("resource_type") or "resource").strip().lower()
            return f"OBS resource is available for {resource_type}; pair it with a compatible table before committing."
        if semantic_norm == "TABLE_FOR_CARRY" and kind == "table" and table_for_type in {"wood", "sand", "stone"}:
            return f"OBS table fits carried {table_for_type}; move there before PUT_DOWN."
        if semantic_norm == "TABLE" and kind == "table":
            return "OBS table is a map fact; use it only if its type capacity matches the carried item."
        if semantic_norm == "DELIVERY" and kind == "table":
            return "I am on the OBS table and delivering my current load."
        if semantic_norm == "CARRY" and ask_norm.startswith("table_for="):
            return line
        if semantic_norm == "CARRY" and kind == "table" and table_for_type in {"wood", "sand", "stone"}:
            return f"OBS table may fit my carried {table_for_type}; I should move there before PUT_DOWN."
        return line

    def _stable_fair_action_semantic_candidate(
        self,
        obs: dict[str, Any],
        *,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> tuple[str, str, bool]:
        data = dict(obs or {})
        pos = _extract_xy(data.get("position")) or (0, 0)
        action_info = dict(data.get("current_round_action_intent") or {})
        summary = str(action_info.get("summary") or data.get("current_round_action_summary") or "").strip()
        low = summary.lower()
        qty_text = str(int(carry_qty)) if carry_qty is not None else "?"
        fact_kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
        fact_pos = _extract_xy((fact or {}).get("position") if isinstance(fact, dict) else None)

        def current_tile_resource() -> tuple[str, int | None]:
            items = data.get("tile_items_counts") or {}
            if isinstance(items, dict):
                best_type = "none"
                best_qty = 0
                for item_type in ("wood", "sand", "stone"):
                    value = _coerce_int(items.get(item_type))
                    if value is not None and value > best_qty:
                        best_type = item_type
                        best_qty = int(value)
                if best_type != "none":
                    return best_type, best_qty
            if isinstance(fact, dict) and str(fact.get("kind") or "") == "resource":
                resource_type = str(fact.get("resource_type") or "none").lower()
                qty = _coerce_int(fact.get("quantity"))
                if resource_type in {"wood", "sand", "stone"}:
                    return resource_type, qty
            return "none", None

        if "pick_up" in low or "pick up" in low or "pickup" in low:
            if carry_type in {"wood", "sand", "stone"}:
                return "", "", False
            match_type = re.search(r"\btype\s*=\s*(wood|sand|stone)\b", summary, re.IGNORECASE)
            match_qty = re.search(r"\bqty\s*=\s*(\d+)\b", summary, re.IGNORECASE)
            resource_type = str(match_type.group(1)).lower() if match_type else ""
            qty = _coerce_int(match_qty.group(1) if match_qty else None)
            if resource_type not in {"wood", "sand", "stone"}:
                resource_type, tile_qty = current_tile_resource()
                if qty is None:
                    qty = tile_qty
            if resource_type in {"wood", "sand", "stone"}:
                amount = f"{int(qty)} " if qty is not None and qty > 0 else ""
                return "CARRY", f"Picking up {amount}{resource_type} at ({pos[0]},{pos[1]}); report a compatible table.", True

        if "put_down" in low or "put down" in low or "putdown" in low:
            if carry_type in {"wood", "sand", "stone"}:
                if str(data.get("tile_terrain_type") or "").lower() == "table":
                    current_tile_fact = _stable_fair_current_tile_table_fact(data)
                    if not _stable_fair_table_supports_carry(
                        current_tile_fact,
                        carry_type,
                        carry_qty,
                        require_full_qty=True,
                    ):
                        return "CARRY", f"I am carrying {carry_type}:{qty_text} and need a compatible table.", False
                    if fact_kind == "table" and not _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                        return "CARRY", f"I am carrying {carry_type}:{qty_text} and need a compatible table.", False
                    return "DELIVERY", f"Delivering {carry_type}:{qty_text} to table ({pos[0]},{pos[1]}).", True
                return "", "", False

        if carry_type in {"wood", "sand", "stone"} and fact_kind == "table" and fact_pos is not None:
            if not _stable_fair_table_supports_carry(fact, carry_type, carry_qty, require_full_qty=True):
                return "CARRY", f"I am carrying {carry_type}:{qty_text} and need a compatible table.", False
            return (
                "TABLE_FOR_CARRY",
                f"Table ({fact_pos[0]},{fact_pos[1]}) {_stable_fair_table_for_type_text(fact, carry_type)} for carried {carry_type}.",
                False,
            )

        move_match = re.search(r"\bmove\s+([A-Z]+)\s+toward\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", summary, re.IGNORECASE)
        if move_match:
            direction = str(move_match.group(1) or "").upper()
            target = (int(move_match.group(2)), int(move_match.group(3)))
            if carry_type in {"wood", "sand", "stone"}:
                return "CARRY", f"Carrying {carry_type}:{qty_text}; moving {direction} from ({pos[0]},{pos[1]}) toward ({target[0]},{target[1]}).", False
            return "STATUS", f"Moving {direction} from ({pos[0]},{pos[1]}) toward ({target[0]},{target[1]}).", False

        return "", "", False

    def _parse_stable_fair_semantic_response(self, response: str, max_chars: int) -> tuple[str, str]:
        text = str(response or "").strip()
        type_match = re.search(r"^\s*TYPE\s*:\s*([A-Za-z_ -]+)\s*$", text, re.IGNORECASE | re.MULTILINE)
        msg_match = re.search(r"^\s*MSG\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        msg = str(msg_match.group(1) or "").strip() if msg_match else _stable_fair_one_line_message(text)
        msg = self._sanitize_stable_fair_semantic_message_line(msg, max_chars)
        msg_type = str(type_match.group(1) or "").strip().upper().replace(" ", "_").replace("-", "_") if type_match else ""
        if msg_type not in {"RESOURCE", "TABLE", "TABLE_FOR_CARRY", "CARRY", "DELIVERY", "DEPLETION", "REQUEST", "REPLY", "STATUS"}:
            msg_type = self._infer_stable_fair_semantic_type(msg)
        return msg_type, msg

    def _parse_stable_fair_semantic_decision_response(
        self,
        response: str,
        max_chars: int,
    ) -> tuple[str, str, str, str]:
        text = str(response or "").strip()

        def field(name: str) -> str:
            pattern = (
                rf"(?:^|[;\n|])\s*{name}\s*[:=]\s*"
                rf"(.+?)(?=\s*(?:[;\n|]\s*(?:FOCUS|FACT|TYPE|ASK|WHY|MSG)\s*[:=]|$))"
            )
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            return str(match.group(1) or "").strip() if match else ""

        focus = field("FOCUS") or field("FACT")
        focus = re.sub(r"[^A-Za-z0-9_:-]+", "", focus).upper()
        type_text = field("TYPE")
        ask_text = field("ASK")
        msg_text = field("MSG")
        if not msg_text:
            msg_text = _stable_fair_one_line_message(text)
        msg = self._sanitize_stable_fair_semantic_message_line(msg_text, max_chars)
        msg_type = str(type_text or "").strip().upper().replace(" ", "_").replace("-", "_")
        if msg_type not in {"RESOURCE", "TABLE", "TABLE_FOR_CARRY", "CARRY", "DELIVERY", "DEPLETION", "REQUEST", "REPLY", "STATUS"}:
            msg_type = self._infer_stable_fair_semantic_type(msg)
        ask = self._normalize_stable_fair_protocol_ask(ask_text or "none", "none")
        return focus, msg_type, ask, msg

    def _sanitize_stable_fair_semantic_message_line(self, text: str, max_chars: int) -> str:
        line = str(text or "").strip().strip('"').strip("'").strip()
        for _ in range(4):
            match = re.match(r"^\s*(?:SEND|TO|MSG|NOTE|PLAN|GOAL|HEADER|TYPE|LANE|STATE|OBS(?:_[A-Z]+)?)\s*:\s*(.*)$", line, re.IGNORECASE)
            if not match:
                break
            line = str(match.group(1) or "").strip()
        line = line.splitlines()[0].strip() if line else ""
        if not line or line.upper().startswith("SF1|") or "SFPROTO:" in line.upper():
            return ""
        line = re.sub(r"\s+", " ", line)
        limit = max(40, int(max_chars or 140))
        if len(line) > limit:
            line = line[:limit].rstrip(" ,;:")
        return line

    def _infer_stable_fair_semantic_type(self, message: str) -> str:
        low = str(message or "").lower()
        has_coord = bool(re.search(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", low))
        mentions_item = any(item in low for item in ("wood", "sand", "stone"))
        if any(word in low for word in ("report", "find", "check", "tell me", "do you know", "can you")) or (
            "table" in low and any(word in low for word in ("need", "needs", "looking for", "look for", "where"))
        ):
            return "REQUEST"
        if any(word in low for word in ("delivering", "delivery", "bringing", "bring")) and "table" in low:
            return "DELIVERY"
        if "table" in low and any(word in low for word in ("carried", "carrying")) and any(item in low for item in ("wood", "sand", "stone")):
            return "TABLE_FOR_CARRY"
        if "carrying" in low:
            return "CARRY"
        if "table" in low and any(word in low for word in ("accept", "remaining", "capacity", "deposit", "space")):
            return "TABLE"
        if "resource" in low or (
            has_coord
            and mentions_item
            and any(word in low for word in ("available", "pickup", "has", "units", "qty"))
        ):
            return "RESOURCE"
        return "STATUS"

    def _stable_fair_semantic_message_safe(
        self,
        message: str,
        fact: dict[str, Any] | None,
        obs: dict[str, Any] | None = None,
        recipient: str | None = None,
    ) -> bool:
        line = str(message or "").strip()
        if not line or "SF1|" in line.upper() or "SFPROTO:" in line.upper():
            return False
        data = dict(obs or {})
        sender_pos = _extract_xy(data.get("position"))
        target_info = dict((data.get("agents_in_communication_range") or {}).get(str(recipient or ""), {}) or {})
        recipient_pos = _extract_xy(target_info.get("position"))
        if sender_pos is not None:
            first_person_patterns = (
                r"^\s*(?:at|from)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
                r"\b(?:i\s+am|i'm|im|my\s+(?:position|location)\s+is)\s*(?:at\s*)?\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
                r"\bmoving\s+[A-Z]+\s+from\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
            )
            for pattern in first_person_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if not match:
                    continue
                msg_pos = (int(match.group(1)), int(match.group(2)))
                if msg_pos != sender_pos:
                    return False
                if recipient_pos is not None and msg_pos == recipient_pos and recipient_pos != sender_pos:
                    return False
        if isinstance(fact, dict):
            fact_kind = str(fact.get("kind") or "").strip().lower()
            fact_pos = _extract_xy(fact.get("position"))
            if fact_kind in {"resource", "table"} and fact_pos is not None:
                mentioned_coords = _stable_fair_message_coords(line)
                if mentioned_coords:
                    allowed_coords = {fact_pos}
                    if sender_pos is not None:
                        allowed_coords.add(sender_pos)
                    if any(coord not in allowed_coords for coord in mentioned_coords):
                        return False
        if not isinstance(fact, dict) or str(fact.get("kind") or "").strip().lower() != "table":
            return True
        rb = _stable_fair_positive_remaining_by_type(fact)
        if not rb:
            return True
        low = line.lower()
        if "table" not in low:
            return True
        claims_selected_table = bool(
            re.search(r"\btable(?:\s+at)?\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", low)
            or re.search(r"\b(?:accepts?|remaining|capacity)\b", low)
            or re.search(r"\b(?:deliver(?:ing|y)?|bring(?:ing)?|carrying)\b[^.]*\b(?:to|toward)\s+(?:the\s+)?table\b", low)
        )
        if not claims_selected_table:
            return True
        allowed = set(rb)
        if not any(item_type in low for item_type in allowed):
            return False
        delivery_match = re.search(
            r"\b(?:delivering|deliver|bringing|bring|carrying)\s+(wood|sand|stone)\s*(?:[:=x]|\s+x\s+|\s+)?\s*(\d+)?[^.]*\b(?:to|toward)\s+(?:the\s+)?table\b",
            low,
        )
        if delivery_match:
            msg_type = str(delivery_match.group(1) or "").lower()
            msg_qty = _coerce_int(delivery_match.group(2))
            if not _stable_fair_table_supports_carry(fact, msg_type, msg_qty, require_full_qty=True):
                return False
        for item_type in ("wood", "sand", "stone"):
            if item_type in allowed or item_type not in low:
                continue
            bad_patterns = (
                rf"\b(?:accepts?|needs?|remaining|capacity|bring(?:ing)?|deliver(?:ing|y)?|carrying)\b[^.]*\b{item_type}\b",
                rf"\b{item_type}\b[^.]*\b(?:to|toward|for|at)\s+(?:the\s+)?table\b",
            )
            if any(re.search(pattern, low) for pattern in bad_patterns):
                return False
        if re.search(r"\b(?:remaining\s+capacity|capacity)\s*(?:=|:|is)?\s*\d+\b", low) and not re.search(
            r"\b(?:accepts?|remaining)\b[^.]*\b(?:wood|sand|stone)\b",
            low,
        ):
            return False
        return True

    async def _compose_stable_fair_llm_selected_protocol(
        self,
        ctx: RoundContext,
        recipient: str,
    ) -> OutgoingMessage:
        header_message, protocol_meta = self._build_stable_fair_protocol_header(ctx.obs)
        protocol_meta = dict(protocol_meta)
        feature = self._routing_selected_feature(getattr(self, "_active_routing_meta", None), recipient)
        routing_context_meta = self._stable_fair_protocol_routing_context_meta(feature)
        protocol_meta.update(
            {
                "protocol_header_profile": self.PROTOCOL_PROFILE,
                "protocol_profile": self.PROMPT_PROTOCOL_PROFILE,
                "protocol_routing_context_used": bool(routing_context_meta),
                "protocol_routing_context": routing_context_meta,
            }
        )
        self._last_stable_fair_protocol_meta = dict(protocol_meta)

        final_message = header_message
        llm_generated_text = ""
        api_call_time = 0.0
        raw_response = "[STABLE_FAIR_PROTOCOL_MESSAGE_PROMPT]"
        prompt = ""
        free_resp = ""
        if self.proto_v2_free_enable and str(header_message or "").strip():
            free_prompt = self._build_proto_v2_free_prompt(self.owner.name, header_message, ctx.obs, self.proto_v2_free_max)
            prompt = free_prompt
            try:
                t0 = time.time()
                p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                free_resp = await self.owner.brain.generate(free_prompt)
                api_call_time += time.time() - t0
                p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
                c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
                self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
                self._jump_back()
            except Exception as exc:
                free_resp = f"[FREE_SLOT_ERROR] {exc}"
            free_line = self._sanitize_protocol_note_line(free_resp, self.proto_v2_free_max)
            if free_line:
                llm_generated_text = free_line
                final_message = self._append_free_slot_text(header_message, free_line)
            raw_response = f"{raw_response}\nFREE:{free_resp}"

        meta = dict(protocol_meta)
        meta.update(
            {
                "mode": str((getattr(self, "_active_routing_meta", None) or {}).get("mode") or self.mode or "basic"),
                "protocol_profile": self.PROMPT_PROTOCOL_PROFILE,
                "protocol_action": "original_protocol_v2_structured",
                "reason": "protocol_message_prompt_structured",
                "llm_generated_text": llm_generated_text,
                "deterministic_text": str(header_message or "").strip(),
                "protocol_free_enabled": bool(self.proto_v2_free_enable),
                "protocol_free_used": bool(llm_generated_text),
                "protocol_message_prompt": prompt,
                "protocol_message_response": str(free_resp or "").strip(),
                "protocol_message_prompt_used": bool(prompt),
            }
        )
        return OutgoingMessage(
            recipient=recipient,
            send=bool(str(final_message or "").strip()),
            prompt=prompt,
            raw_response=raw_response,
            message=str(final_message or "").strip(),
            meta=meta,
            api_call_time=api_call_time,
            tokens=self._snapshot_brain_usage(),
        )

    async def _build_stable_fair_llm_selected_protocol_header(
        self,
        obs: dict[str, Any] | None,
        recipient: str,
    ) -> tuple[str, dict[str, Any], str, str, float]:
        data = dict(obs or {})
        pos = _extract_xy(data.get("position")) or (0, 0)
        routing_meta = getattr(self, "_active_routing_meta", None)
        feature = self._routing_selected_feature(routing_meta, recipient)
        carry_type, carry_qty = _inventory_resource_quantity(data)
        candidates = self._stable_fair_protocol_content_candidates(
            data,
            recipient=recipient,
            feature=feature,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        select_prompt = self._build_stable_fair_protocol_content_select_prompt(
            data,
            recipient=recipient,
            candidates=candidates,
            carry_type=carry_type,
            carry_qty=carry_qty,
            feature=feature,
        )
        select_response = ""
        api_call_time = 0.0
        try:
            t0 = time.time()
            p0 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c0 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            select_response = await self.owner.brain.generate(select_prompt)
            api_call_time = time.time() - t0
            p1 = int(getattr(self.owner.brain, "usage_prompt", 0) or 0)
            c1 = int(getattr(self.owner.brain, "usage_completion", 0) or 0)
            self._accumulate_message_tokens(max(0, p1 - p0), max(0, c1 - c0))
            self._jump_back()
        except Exception as exc:
            select_response = f"[CONTENT_SELECT_ERROR] {exc}"

        selected, ask, valid_selection = self._parse_stable_fair_protocol_content_selection(
            select_response,
            candidates,
            carry_type,
        )
        fact = dict(selected.get("fact") or {"kind": "status"})
        if str(fact.get("kind") or "").strip().lower() == "table":
            ask = "none"
        selected_id = str(selected.get("id") or "").strip().upper()
        intent = self._stable_fair_protocol_intent(None, fact, carry_type)
        if selected_id == "TABLE_FOR_CARRY":
            intent = "table_for_carry"
        elif selected_id == "CARRY_STATUS" and ask != "none":
            intent = "request"
        goal = self._stable_fair_protocol_goal(data, fact, carry_type)
        message = self._stable_fair_protocol_compact_line(
            intent=intent,
            pos=pos,
            carry_type=carry_type,
            carry_qty=carry_qty,
            fact=fact,
            goal=goal,
            ask=ask,
        )
        candidate_meta = [
            {
                "id": str(candidate.get("id") or ""),
                "label": str(candidate.get("label") or ""),
                "ask": str(candidate.get("ask") or "none"),
                "reason": str(candidate.get("reason") or ""),
            }
            for candidate in candidates
        ]
        routing_context_meta = self._stable_fair_protocol_routing_context_meta(feature)
        return message, {
            "protocol_profile": self.LLM_SELECT_PROTOCOL_PROFILE,
            "protocol_intent": intent,
            "protocol_header": message,
            "protocol_carry_type": carry_type,
            "protocol_carry_qty": carry_qty,
            "protocol_goal": goal,
            "protocol_ask": ask,
            "protocol_fact": fact,
            "protocol_recipient": recipient,
            "protocol_used_routing_intent": False,
            "protocol_content_select_policy": "llm_select_structured_fact",
            "protocol_content_select_prompt": select_prompt,
            "protocol_content_select_response": str(select_response or "").strip(),
            "protocol_content_select_valid": bool(valid_selection),
            "protocol_selected_fact_id": str(selected.get("id") or ""),
            "protocol_selected_fact_label": str(selected.get("label") or ""),
            "protocol_selected_ask": ask,
            "protocol_content_candidates": candidate_meta,
            "protocol_routing_context_used": bool(routing_context_meta),
            "protocol_routing_context": routing_context_meta,
        }, select_prompt, str(select_response or "").strip(), api_call_time

    def _stable_fair_protocol_content_candidates(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_facts: set[tuple[Any, ...]] = set()
        status_added = False
        target_info = dict((obs.get("agents_in_communication_range") or {}).get(recipient, {}) or {})
        recipient_carry_type, recipient_carry_qty = _candidate_resource_quantity(target_info)

        def add_candidate(
            candidate_id: str,
            fact: dict[str, Any] | None,
            ask: str,
            reason: str,
            *,
            table_for_type: str = "none",
            table_for_qty: int | None = None,
        ) -> None:
            nonlocal status_added
            normalized = self._normalize_stable_fair_protocol_candidate_fact(fact)
            key = self._stable_fair_protocol_candidate_fact_key(normalized)
            if key == ("status",):
                if status_added:
                    return
                status_added = True
            elif key in seen_facts:
                return
            else:
                seen_facts.add(key)
            candidates.append(
                {
                    "id": str(candidate_id or "").strip().upper(),
                    "fact": normalized,
                    "ask": self._normalize_stable_fair_protocol_ask(ask, carry_type),
                    "label": self._stable_fair_protocol_candidate_label(
                        normalized,
                        table_for_type if table_for_type in {"wood", "sand", "stone"} else carry_type,
                        table_for_qty if table_for_type in {"wood", "sand", "stone"} else carry_qty,
                    ),
                    "reason": str(reason or "").strip(),
                    "table_for_type": table_for_type if table_for_type in {"wood", "sand", "stone"} else "none",
                    "table_for_qty": table_for_qty if table_for_type in {"wood", "sand", "stone"} else None,
                }
            )

        if recipient_carry_type in {"wood", "sand", "stone"}:
            table_fact = self._stable_fair_best_table_fact(obs, recipient_carry_type, recipient_carry_qty)
            if table_fact:
                add_candidate(
                    "TABLE_FOR_CARRY",
                    table_fact,
                    "none",
                    f"recipient is carrying {recipient_carry_type}",
                    table_for_type=recipient_carry_type,
                    table_for_qty=recipient_carry_qty,
                )

        if carry_type != "none":
            table_fact = self._stable_fair_best_table_fact(obs, carry_type, carry_qty)
            if table_fact:
                add_candidate(
                    "TABLE_FOR_CARRY",
                    table_fact,
                    "none",
                    f"sender is carrying {carry_type}",
                    table_for_type=carry_type,
                    table_for_qty=carry_qty,
                )
            else:
                add_candidate("CARRY_STATUS", {"kind": "status"}, f"table_for={carry_type}", f"sender is carrying {carry_type}")

        table_fact = self._stable_fair_best_table_fact(obs, carry_type, carry_qty)
        if table_fact:
            add_candidate("TABLE", table_fact, "none", "best visible or remembered table fact")

        resource_fact = self._stable_fair_best_resource_fact(obs)
        if resource_fact:
            add_candidate("RESOURCE", resource_fact, "none", "best visible or remembered resource fact")

        add_candidate("STATUS", {"kind": "status"}, "none", "fallback when no concrete fact is useful")
        return candidates[:6]

    def _normalize_stable_fair_protocol_candidate_fact(self, fact: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(fact, dict):
            return {"kind": "status"}
        normalized = dict(fact)
        kind = str(normalized.get("kind") or "status").strip().lower()
        if kind == "resource":
            pos = _extract_xy(normalized.get("position"))
            resource_type = str(normalized.get("resource_type") or normalized.get("type") or "").strip().lower()
            qty = _coerce_int(normalized.get("quantity"))
            if pos is None or resource_type not in {"wood", "sand", "stone"} or qty is None or qty <= 0:
                return {"kind": "status"}
            normalized["kind"] = "resource"
            normalized["resource_type"] = resource_type
            normalized["position"] = pos
            normalized["quantity"] = int(qty)
            return normalized
        if kind == "table":
            pos = _extract_xy(normalized.get("position"))
            if pos is None:
                return {"kind": "status"}
            rem = _coerce_int(normalized.get("remaining_capacity"))
            normalized["kind"] = "table"
            normalized["position"] = pos
            if rem is not None:
                normalized["remaining_capacity"] = int(rem)
            rb = normalized.get("remaining_by_type") or {}
            if isinstance(rb, dict):
                normalized["remaining_by_type"] = dict(rb)
            return normalized
        return {"kind": "status"}

    def _stable_fair_protocol_candidate_fact_key(self, fact: dict[str, Any]) -> tuple[Any, ...]:
        kind = str((fact or {}).get("kind") or "status").strip().lower()
        pos = _extract_xy((fact or {}).get("position"))
        if kind == "resource" and pos is not None:
            return ("resource", str(fact.get("resource_type") or ""), int(pos[0]), int(pos[1]))
        if kind == "table" and pos is not None:
            return ("table", int(pos[0]), int(pos[1]))
        return ("status",)

    def _stable_fair_protocol_candidate_label(
        self,
        fact: dict[str, Any],
        carry_type: str,
        carry_qty: int | None,
    ) -> str:
        fact_text = _stable_fair_protocol_fact_compact(fact)
        if fact_text:
            return fact_text
        if carry_type != "none":
            return f"carry={carry_type}:{int(carry_qty) if carry_qty is not None else '?'}"
        return "status"

    def _normalize_stable_fair_protocol_ask(self, ask: str, carry_type: str) -> str:
        text = str(ask or "none").strip().lower().replace(" ", "")
        if not text or text in {"none", "no", "n/a", "na"}:
            return "none"
        match = re.search(r"table_for=(wood|sand|stone)", text)
        if match and str(match.group(1) or "") == str(carry_type or ""):
            return f"table_for={match.group(1)}"
        return "none"

    def _build_stable_fair_protocol_content_select_prompt(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        candidates: list[dict[str, Any]],
        carry_type: str,
        carry_qty: int | None,
        feature: dict[str, Any] | None = None,
    ) -> str:
        pos = _extract_xy(obs.get("position")) or (0, 0)
        inventory_display = str((obs.get("inventory") or {}).get("display") or "empty")
        candidate_lines = []
        for candidate in candidates:
            ask = str(candidate.get("ask") or "none")
            candidate_lines.append(
                f"- {candidate.get('id')}: {candidate.get('label')} | ask={ask} | {candidate.get('reason')}"
            )
        allowed_asks = ["none"]
        if carry_type in {"wood", "sand", "stone"}:
            allowed_asks.append(f"table_for={carry_type}")
        routing_context_lines = self._stable_fair_protocol_routing_context_lines(recipient, feature)
        lines = [
            f"You are {self.owner.name}. Select content for one structured update to {recipient}.",
            "Your choice will be encoded automatically; do not write the teammate-facing message.",
            "Output EXACTLY two lines:",
            "FACT: <candidate id>",
            f"ASK: <{'|'.join(allowed_asks)}>",
            "",
            f"Current position: {pos[0]},{pos[1]}",
            f"Inventory: {inventory_display}",
            f"Carrying parsed by protocol: {carry_type}:{carry_qty if carry_qty is not None else '?'}",
            f"Visible resources: {_compact_visible_resource_summary(obs, max_lines=3)}",
            f"Visible tables: {_compact_visible_table_summary(obs, max_lines=3)}",
            f"Memory hints: {_compact_memory_hint_summary(obs, max_entries=3)}",
            "",
            "Routing context for selected recipient:",
            *routing_context_lines,
            "",
            "Candidate facts:",
            *candidate_lines,
            "",
            "Selection rules:",
            "- Pick the single candidate that will be most useful for coordination this round.",
            "- Routing already chose the recipient; do not change or second-guess TO.",
            "- Use recipient context as weak audience/need context; choose content from current facts, memory, inventory, and recent messages.",
            "- Prefer concrete table/resource/carry/request content when it helps this recipient's next actions.",
            "- Use CARRY_STATUS when carrying is important but no table fact should be asserted.",
            "- Use STATUS only when the concrete candidates are not useful.",
            "- Use ASK only when the allowed table_for request matches what you are carrying.",
        ]
        return "\n".join(lines)

    def _stable_fair_protocol_routing_context_meta(self, feature: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(feature, dict):
            return {}
        fields = (
            "name",
            "reply_recent",
            "reply_info_request",
            "carrying",
            "carrying_qty",
            "in_view",
            "contact_last_round",
            "rolling_recent_contact_count",
            "routing_hint",
            "routing_objective",
            "need_table_for_type",
            "table_for_type",
            "latest_incoming_ask_type",
            "best_fact_kind",
            "best_fact_type",
            "fact_area_key",
            "coverage_novelty",
            "duplicate_fact_recent",
            "fact_recent_message_count",
            "fact_area_recent_message_count",
            "safe_table_delivery",
            "request_cooldown",
            "routing_edge",
            "payload_state",
            "audience_fit",
        )
        meta = {field: feature.get(field) for field in fields if field in feature}
        meta["recipient_signal"] = _stable_fair_recipient_signal_for_feature(feature)
        meta["routing_edge"] = str(meta.get("routing_edge") or _stable_fair_routing_edge_class_for_feature(feature))
        meta["payload_state"] = str(meta.get("payload_state") or _stable_fair_routing_payload_state_for_feature(feature))
        meta["audience_fit"] = str(meta.get("audience_fit") or _stable_fair_routing_audience_fit_for_feature(feature))
        meta["routing_objective"] = str(meta.get("routing_objective") or _stable_fair_routing_objective_for_feature(feature))
        meta["routing_hint"] = str(meta.get("routing_hint") or feature.get("routing_intent") or "coverage")
        return meta

    def _stable_fair_protocol_routing_context_lines(
        self,
        recipient: str,
        feature: dict[str, Any] | None,
    ) -> list[str]:
        meta = self._stable_fair_protocol_routing_context_meta(feature)
        if not meta:
            return ["- no routing feature was available; choose from candidate facts only."]

        def yn(value: Any) -> str:
            return "yes" if bool(value) else "no"

        return [
            (
                f"- selected_recipient={recipient}; edge={meta.get('routing_edge', 'weak')}; "
                f"payload_state={meta.get('payload_state', 'blocked')}; "
                f"audience_fit={meta.get('audience_fit', 'weak')} "
                "(weak audience cue, not fixed message content)"
            ),
            (
                f"- routing_hint={meta.get('routing_hint', 'coverage')}; "
                f"routing_objective={meta.get('routing_objective', 'coverage')}; "
                f"latest_request={meta.get('latest_incoming_ask_type', 'none')}; "
                f"table_for_type={meta.get('table_for_type', 'none')}; "
                f"can_use_table={yn(meta.get('safe_table_delivery'))}; "
                f"needs_table_from_recipient={yn(meta.get('need_table_for_type'))}"
            ),
            (
                f"- best_fact_kind={meta.get('best_fact_kind', 'none')}; "
                f"coverage_novelty={meta.get('coverage_novelty', 'none')}; "
                f"area={meta.get('fact_area_key', 'unknown')}; "
                f"duplicate_fact_recent={yn(meta.get('duplicate_fact_recent'))}; "
                f"fact_recent_count={meta.get('fact_recent_message_count', 0)}; "
                f"request_cooldown={yn(meta.get('request_cooldown'))}; "
                f"recent_contact_count={meta.get('rolling_recent_contact_count', 0)}; "
                f"contact_last_round={yn(meta.get('contact_last_round'))}"
            ),
            (
                f"- recipient_carrying={meta.get('carrying', 'none')}; "
                f"recipient_carrying_qty={meta.get('carrying_qty', 'unknown')}; "
                f"recent_request={yn(meta.get('reply_info_request'))}; "
                f"in_view={yn(meta.get('in_view'))}"
            ),
        ]

    def _stable_fair_semantic_candidate_lines(
        self,
        obs: dict[str, Any],
        *,
        recipient: str,
        feature: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        semantic_type: str,
        fallback_message: str,
        action_type: str = "",
        action_message: str = "",
        action_is_strong: bool = False,
        content_anchors: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        base_fact = fact if isinstance(fact, dict) else None
        fallback_display = self._stable_fair_semantic_nonredundant_message(
            fallback_message,
            semantic_type=semantic_type,
            fact=base_fact,
            carry_type=carry_type,
            carry_qty=carry_qty,
            ask="none",
        )
        lines = [f"- Recommended: TYPE={semantic_type}; MSG={fallback_display}"]
        seen = {fallback_display}
        action_allowed = self._stable_fair_action_override_allowed(
            routing_intent="",
            semantic_type=semantic_type,
            action_type=action_type,
            action_message=action_message,
            action_is_strong=action_is_strong,
            fact=base_fact,
            carry_type=carry_type,
            carry_qty=carry_qty,
        )
        if (
            action_allowed
            and action_message
            and action_message not in seen
            and self._stable_fair_semantic_message_safe(action_message, base_fact, obs, recipient)
        ):
            lines.append(f"- Current action: TYPE={action_type or 'STATUS'}; MSG={action_message}")
            seen.add(action_message)
        for anchor in list(content_anchors or [])[1:]:
            anchor_type = str((anchor or {}).get("semantic_type") or "STATUS").strip().upper()
            anchor_message = str((anchor or {}).get("message") or "").strip()
            anchor_fact = anchor.get("fact") if isinstance(anchor, dict) else None
            anchor_id = str((anchor or {}).get("id") or "ANCHOR").strip().upper()
            anchor_table_for_type = str((anchor or {}).get("table_for_type") or carry_type or "none").strip().lower()
            anchor_message = self._stable_fair_semantic_nonredundant_message(
                anchor_message,
                semantic_type=anchor_type,
                fact=anchor_fact if isinstance(anchor_fact, dict) else None,
                carry_type=anchor_table_for_type,
                carry_qty=_coerce_int((anchor or {}).get("table_for_qty")) or carry_qty,
                ask=str((anchor or {}).get("ask") or "none"),
            )
            if (
                not anchor_message
                or anchor_message in seen
                or not self._stable_fair_semantic_message_safe(anchor_message, anchor_fact, obs, recipient)
            ):
                continue
            seen.add(anchor_message)
            lines.append(f"- Anchor {anchor_id}: TYPE={anchor_type}; MSG={anchor_message}")
            if len(lines) >= 5:
                return lines
        try:
            candidates = self._stable_fair_protocol_content_candidates(
                obs,
                recipient=recipient,
                feature=feature,
                carry_type=carry_type,
                carry_qty=carry_qty,
            )
        except Exception:
            candidates = []
        for candidate in candidates:
            candidate_id = str((candidate or {}).get("id") or "").upper()
            candidate_fact = candidate.get("fact") if isinstance(candidate, dict) else None
            ask = str((candidate or {}).get("ask") or "none")
            kind = str((candidate_fact or {}).get("kind") or "").strip().lower() if isinstance(candidate_fact, dict) else ""
            if candidate_id == "TABLE_FOR_CARRY":
                cand_type = "TABLE_FOR_CARRY"
            else:
                cand_type = self._stable_fair_semantic_type(
                    "table_for_carry" if kind == "table" and carry_type in {"wood", "sand", "stone"} else "discover",
                    candidate_fact if isinstance(candidate_fact, dict) else None,
                    carry_type if kind == "table" else "none",
                    ask,
                    carry_qty if kind == "table" else None,
                )
            if candidate_id == "CARRY_STATUS":
                cand_type = "REQUEST" if ask != "none" else ("CARRY" if carry_type in {"wood", "sand", "stone"} else "STATUS")
            message_carry_type = carry_type
            message_carry_qty = carry_qty
            if cand_type == "TABLE_FOR_CARRY":
                candidate_table_for_type = str((candidate or {}).get("table_for_type") or "").strip().lower()
                if candidate_table_for_type in {"wood", "sand", "stone"}:
                    message_carry_type = candidate_table_for_type
                    message_carry_qty = _coerce_int((candidate or {}).get("table_for_qty"))
            message = self._stable_fair_semantic_fallback_message(
                semantic_type=cand_type,
                fact=candidate_fact if isinstance(candidate_fact, dict) else None,
                carry_type=message_carry_type if cand_type in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else "none",
                carry_qty=message_carry_qty if cand_type in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else None,
                ask=ask,
            )
            message = self._stable_fair_semantic_nonredundant_message(
                message,
                semantic_type=cand_type,
                fact=candidate_fact if isinstance(candidate_fact, dict) else None,
                carry_type=message_carry_type if cand_type in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else "none",
                carry_qty=message_carry_qty if cand_type in {"CARRY", "DELIVERY", "REQUEST", "TABLE_FOR_CARRY"} else None,
                ask=ask,
            )
            if not message or message in seen or not self._stable_fair_semantic_message_safe(message, candidate_fact, obs, recipient):
                continue
            seen.add(message)
            lines.append(f"- Option: TYPE={cand_type}; MSG={message}")
            if len(lines) >= 5:
                break
        return lines

    def _stable_fair_protocol_routing_reason_lines(self, contract: dict[str, Any] | None) -> list[str]:
        if not isinstance(contract, dict) or not contract:
            return ["- none"]
        fact = contract.get("primary_fact")
        backup_fact = contract.get("backup_fact")
        objective = str(contract.get("objective") or "coverage").strip().lower()
        item_type = str(contract.get("item_type") or "none").strip().lower()
        if item_type not in {"wood", "sand", "stone"}:
            item_type = "none"

        def fact_kind_text(value: Any, fallback: str = "fact") -> tuple[str, str]:
            fact_dict = value if isinstance(value, dict) else None
            compact = _stable_fair_protocol_fact_compact(fact_dict)
            if not compact:
                return fallback.upper(), "none"
            kind = str((fact_dict or {}).get("kind") or fallback or "fact").strip().upper()
            return kind, compact

        def carry_text(label: str, item: Any, qty: Any) -> str | None:
            carried = str(item or "none").strip().lower()
            if carried not in {"wood", "sand", "stone"}:
                return None
            return f"{label} carrying {carried}:{qty if qty is not None else '?'}"

        reason_by_objective = {
            "answer_request": "answer a recent information need",
            "ask_missing_table": "missing compatible-table information",
            "share_table_for_carrier": "share a compatible table for carried load",
            "delivery_ready": "share delivery-ready table context",
            "share_resource": "share a fresh resource fact",
            "share_table": "share a fresh table fact",
            "coverage": "share useful map coverage",
        }
        reason = reason_by_objective.get(objective, "share useful coordination context")
        if item_type != "none" and objective in {"answer_request", "ask_missing_table", "share_table_for_carrier", "delivery_ready"}:
            reason = f"{reason} for {item_type}"

        lines = [f"- selected reason: {reason}."]
        focus_kind, focus_text = fact_kind_text(fact, str(contract.get("fact_kind") or "fact"))
        if focus_text != "none":
            lines.append(f"- selected {focus_kind} fact: {focus_text}.")
        backup_kind, backup_text = fact_kind_text(backup_fact, str(contract.get("backup_fact_kind") or "fact"))
        if backup_text != "none" and backup_text != focus_text:
            lines.append(f"- related {backup_kind} fact: {backup_text}.")

        carry_parts = [
            part
            for part in (
                carry_text("sender", contract.get("sender_carrying"), contract.get("sender_carrying_qty")),
                carry_text("recipient", contract.get("recipient_carrying"), contract.get("recipient_carrying_qty")),
            )
            if part
        ]
        if carry_parts:
            lines.append("- carry context: " + "; ".join(carry_parts) + ".")
        return lines or ["- none"]

    def _stable_fair_protocol_contract_lines(self, contract: dict[str, Any] | None) -> list[str]:
        if not isinstance(contract, dict) or not contract:
            return ["- objective=coverage; primary_fact=none; ask=none"]
        fact = contract.get("primary_fact")
        fact_text = _stable_fair_protocol_fact_compact(fact if isinstance(fact, dict) else None) or "none"
        backup_fact = contract.get("backup_fact")
        backup_text = _stable_fair_protocol_fact_compact(backup_fact if isinstance(backup_fact, dict) else None)
        parts = [
            f"objective={contract.get('objective', 'coverage')}",
            f"item_type={contract.get('item_type', 'none')}",
            f"ask={contract.get('ask', 'none')}",
            f"primary_fact={fact_text}",
            f"coverage_novelty={contract.get('coverage_novelty', 'none')}",
            f"area={contract.get('fact_area_key', 'unknown')}",
            f"sender_carrying={contract.get('sender_carrying', 'none')}:{contract.get('sender_carrying_qty', '?')}",
            f"recipient_carrying={contract.get('recipient_carrying', 'none')}:{contract.get('recipient_carrying_qty', '?')}",
        ]
        if backup_text:
            parts.append(f"backup_fact={backup_text}")
        return ["- " + "; ".join(parts)]

    def _build_stable_fair_semantic_protocol_prompt(
        self,
        obs: dict[str, Any],
        recipient: str,
        *,
        semantic_type: str,
        fallback_message: str,
        fact: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
        ask: str,
        feature: dict[str, Any] | None,
        action_type: str = "",
        action_message: str = "",
        action_is_strong: bool = False,
        content_anchors: list[dict[str, Any]] | None = None,
        state_line: str = "",
        obs_line: str = "",
        routing_contract: dict[str, Any] | None = None,
    ) -> str:
        del semantic_type, fallback_message, fact, ask, action_type, action_message, action_is_strong, state_line, obs_line
        data = dict(obs or {})
        sender_pos = _extract_xy(data.get("position")) or (0, 0)
        inventory_display = str((data.get("inventory") or {}).get("display") or "empty")
        tile_items = _format_items_compact(data.get("tile_items_counts"))
        tile_type = str(data.get("tile_terrain_type") or "unknown")
        tile_free = data.get("tile_placeable_free_capacity")
        tile_type_remaining = _stable_fair_remaining_by_type_text(data.get("tile_remaining_by_type"))
        tile_capacity_text = f"total_free_capacity={tile_free}"
        if tile_type_remaining:
            tile_capacity_text = f"{tile_capacity_text}; type_remaining={tile_type_remaining}"
        action_block = _current_round_action_prompt_block(data).strip()
        action_summary = re.sub(r"\s+", " ", action_block)[:180] if action_block else "unknown"
        target_info = dict((data.get("agents_in_communication_range") or {}).get(recipient, {}) or {})
        target_pos = _extract_xy(target_info.get("position"))
        target_pos_text = f"({target_pos[0]},{target_pos[1]})" if target_pos is not None else "unknown"
        facts_block = _compact_current_facts_block(data)
        anchor_lines = self._stable_fair_protocol_brief_anchor_lines(content_anchors)
        routing_context = self._stable_fair_protocol_simple_routing_context(recipient, feature)
        contract_lines = self._stable_fair_protocol_contract_lines(routing_contract)
        limit = max(40, min(120, int(self.proto_v2_free_max or 120)))
        lines = [
            f"You are {self.owner.name}. Recipient is fixed: {recipient}; do not choose another recipient.",
            "Protocol is a small grounded brief; decide what content to say by choosing one FOCUS.",
            "Runtime will attach deterministic STATE and one matching OBS before MSG.",
            "",
            "Output EXACTLY one line:",
            f"FOCUS=<F-id>; TYPE=<RESOURCE|TABLE|TABLE_FOR_CARRY|CARRY|DELIVERY|REQUEST|REPLY|STATUS>; ASK=<none|table_for=wood|table_for=sand|table_for=stone>; MSG=<one sentence under {limit} chars>",
            "",
            "Routing audience cue (weak; do not force the message type):",
            *contract_lines,
            "",
            "Grounded brief:",
            f"- routing: {routing_context}",
            *anchor_lines,
            "",
            "Writing rules:",
            "- Choose exactly one FOCUS id from the Grounded brief; this is the protocol content decision.",
            "- Treat routing as recipient/audience context; choose the most useful grounded content for this recipient.",
            "- Use only sender state, recipient status, and the chosen FOCUS fact.",
            "- Current local facts are background only; do not take message coordinates from them unless listed as the chosen FOCUS.",
            "- You may report a fact, ask for missing table information, or explain current carry/action.",
            "- Do not invent coordinates, resource quantities, table capacity, or carried items.",
            "- If a table has type_remaining, say it accepts only types with positive remaining count.",
            "- A table request is a request for deposit capacity, not a pickup task.",
            "- Output only the requested FOCUS line; do not add extra headings or explanation.",
            "- If you mention a coordinate, it must appear in the grounded brief or be my own position.",
            "- If no concrete fact is useful, choose the STATUS focus.",
            "",
            "Sender status:",
            f"- name={self.owner.name}; my_position=({sender_pos[0]},{sender_pos[1]}); inventory={inventory_display}",
            f"- current_tile=terrain={tile_type}; items={tile_items}; {tile_capacity_text}",
            f"- current_action={action_summary}",
            "",
            f"Recipient status: name={recipient}; position={target_pos_text}; carrying={target_info.get('carrying_resource', 'unknown')}",
            f"Current carrying parsed by protocol: {carry_type}:{carry_qty if carry_qty is not None else '?'}",
            "",
            facts_block,
            "",
            "Keep completion compact; no explanation.",
        ]
        return "\n".join(lines)

    def format_history_response(
        self,
        recipient: str,
        route_response: str,
        outgoing: OutgoingMessage,
        decision_meta: dict[str, Any] | None = None,
    ) -> str:
        meta = dict(decision_meta or {})
        if str(meta.get("protocol_action") or "") == "stable_fair_semantic_protocol":
            outgoing_message = str(getattr(outgoing, "message", "") or "").strip()
            if outgoing_message:
                if outgoing_message.upper().startswith("STATE:"):
                    return f"SEND: yes\nTO: {recipient}\n{outgoing_message}\n[SEMANTIC_PROTOCOL_APPLIED]"
                return f"SEND: yes\nTO: {recipient}\nMSG: {outgoing_message}\n[SEMANTIC_PROTOCOL_APPLIED]"
        return self._reference_plain_history_response(recipient, route_response, outgoing, decision_meta)

    def _parse_stable_fair_protocol_content_selection(
        self,
        response: str,
        candidates: list[dict[str, Any]],
        carry_type: str,
    ) -> tuple[dict[str, Any], str, bool]:
        candidates_by_id = {str(candidate.get("id") or "").upper(): candidate for candidate in candidates}
        text = str(response or "").strip()
        fact_id = ""
        ask_text = ""
        fact_match = re.search(r"^\s*FACT\s*:\s*([A-Za-z0-9_ -]+)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if fact_match:
            fact_id = str(fact_match.group(1) or "").strip().upper().replace("-", "_").replace(" ", "_")
        ask_match = re.search(r"^\s*ASK\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
        if ask_match:
            ask_text = str(ask_match.group(1) or "").strip()
        if not fact_id:
            upper_text = text.upper()
            for candidate_id in candidates_by_id:
                if re.search(rf"\b{re.escape(candidate_id)}\b", upper_text):
                    fact_id = candidate_id
                    break
        selected = candidates_by_id.get(fact_id)
        valid_selection = selected is not None
        if selected is None:
            selected = candidates[0] if candidates else {"id": "STATUS", "fact": {"kind": "status"}, "ask": "none", "label": "status"}
        ask = self._normalize_stable_fair_protocol_ask(ask_text, carry_type)
        if ask == "none":
            ask = self._normalize_stable_fair_protocol_ask(str(selected.get("ask") or "none"), carry_type)
        return selected, ask, valid_selection

    def _build_original_protocol_v2_message(self, obs: dict) -> str:
        message, meta = self._build_stable_fair_protocol_header(obs)
        self._last_stable_fair_protocol_meta = dict(meta)
        return message

    def _append_free_slot_text(self, header_msg: str, free_line: str) -> str:
        header = str(header_msg or "").strip()
        note = str(free_line or "").strip()
        if not note:
            return header
        return f"{header}\nNOTE:{note}"

    def _routing_selected_feature(
        self,
        routing_meta: dict[str, Any] | None,
        recipient: str,
    ) -> dict[str, Any] | None:
        target = str(recipient or "").strip()
        if not target:
            return None
        reorder_meta = dict((routing_meta or {}).get("candidate_reorder_meta") or {})
        for key in ("features", "features_full"):
            for feature in list(reorder_meta.get(key) or []):
                if isinstance(feature, dict) and str(feature.get("name") or "").strip() == target:
                    return feature
        return None

    def _build_stable_fair_protocol_header(self, obs: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        data = dict(obs or {})
        pos = _extract_xy(data.get("position")) or (0, 0)
        round_num = _coerce_int(data.get("round"))
        routing_meta = getattr(self, "_active_routing_meta", None)
        recipient = str(getattr(self, "_active_routing_recipient", "") or "")
        feature = self._routing_selected_feature(routing_meta, recipient)
        carry_type, carry_qty = _inventory_resource_quantity(data)
        fact = self._stable_fair_protocol_fact(data, feature, carry_type, carry_qty)
        intent = self._stable_fair_protocol_intent(feature, fact, carry_type)
        goal = self._stable_fair_protocol_goal(data, fact, carry_type)
        ask = self._stable_fair_protocol_ask(intent, carry_type, fact)

        message = self._stable_fair_protocol_compact_line(
            intent=intent,
            pos=pos,
            carry_type=carry_type,
            carry_qty=carry_qty,
            fact=fact,
            goal=goal,
            ask=ask,
        )
        return message, {
            "protocol_profile": self.PROTOCOL_PROFILE,
            "protocol_intent": intent,
            "protocol_header": message,
            "protocol_carry_type": carry_type,
            "protocol_carry_qty": carry_qty,
            "protocol_goal": goal,
            "protocol_ask": ask,
            "protocol_fact": fact,
            "protocol_recipient": recipient,
            "protocol_used_routing_intent": False,
            "protocol_recipient_context_used": bool(feature),
        }

    def _stable_fair_protocol_intent(
        self,
        feature: dict[str, Any] | None,
        fact: dict[str, Any] | None,
        carry_type: str,
    ) -> str:
        del feature
        kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""
        if (
            carry_type in {"wood", "sand", "stone"}
            and kind == "table"
            and _stable_fair_table_supports_carry(fact, carry_type, None, require_full_qty=False)
        ):
            return "table_for_carry"
        if carry_type in {"wood", "sand", "stone"} and kind != "table":
            return "request"
        if carry_type != "none":
            return "carry"
        if kind in {"resource", "table"}:
            return "discover"
        return "status"

    def _stable_fair_protocol_compact_line(
        self,
        *,
        intent: str,
        pos: tuple[int, int],
        carry_type: str,
        carry_qty: int | None,
        fact: dict[str, Any] | None,
        goal: tuple[int, int] | None,
        ask: str,
    ) -> str:
        intent_code = _STABLE_FAIR_INTENT_TO_CODE.get(str(intent or "status").strip().lower(), "S")
        parts = ["SF1", intent_code]
        if intent_code in {"D", "T", "C", "H", "Q", "R", "S"}:
            parts.append(f"l={int(pos[0])},{int(pos[1])}")
        if carry_type != "none":
            parts.append(f"c={carry_type}:{int(carry_qty) if carry_qty is not None else '?'}")
        fact_text = _stable_fair_protocol_fact_compact(fact)
        if fact_text:
            parts.append(f"f={fact_text}")
        need_text = _stable_fair_protocol_need_compact(fact)
        if need_text:
            parts.append(f"n={need_text}")
        if goal is not None:
            parts.append(f"g={int(goal[0])},{int(goal[1])}")
        ask_text = str(ask or "none").strip()
        if ask_text and ask_text.lower() != "none":
            parts.append(f"a={ask_text}")
        return "|".join(parts)

    def _protocol_note_context(self, header_msg: str, obs: dict[str, Any]) -> str:
        proto = _parse_stable_fair_compact_protocol(header_msg) or {}
        lines = [
            f"Header: {header_msg}",
            f"Intent: {proto.get('intent', 'status')}",
        ]
        carry_type = str(proto.get("carry_type") or "none")
        if carry_type != "none":
            lines.append(f"Carrying: {carry_type}:{proto.get('carry_qty') if proto.get('carry_qty') is not None else '?'}")
        fact = proto.get("fact")
        if isinstance(fact, dict):
            lines.append(f"Fact: {_stable_fair_protocol_memory_summary(header_msg)}")
        ask = str(proto.get("ask") or "none")
        if ask and ask.lower() != "none":
            lines.append(f"Ask: {ask}")

        data = dict(obs or {})
        lines.append(f"Current facts: inventory={str((data.get('inventory') or {}).get('display') or 'empty')}")
        visible_resources = _compact_visible_resource_summary(data, max_lines=2)
        visible_tables = _compact_visible_table_summary(data, max_lines=2)
        memory_hints = _compact_memory_hint_summary(data, max_entries=2)
        if visible_resources != "none":
            lines.append(f"Visible resources: {visible_resources}")
        if visible_tables != "none":
            lines.append(f"Visible tables: {visible_tables}")
        if memory_hints != "none":
            lines.append(f"Memory hints: {memory_hints}")
        recipient = str(getattr(self, "_active_routing_recipient", "") or "").strip()
        if recipient:
            lines.append(f"Recipient: {recipient}")
        return "\n".join(lines)

    def _protocol_note_policy(self, header_msg: str) -> str:
        proto = _parse_stable_fair_compact_protocol(header_msg) or {}
        intent = str(proto.get("intent") or "status").strip().lower()
        carry_type = str(proto.get("carry_type") or "none").strip().lower()
        fact = proto.get("fact")
        fact_kind = str((fact or {}).get("kind") or "").strip().lower() if isinstance(fact, dict) else ""

        rules = [
            "Intent-specific rules:",
            "- Do not ask anyone to pick up from a table; tables are deposit-only.",
            "- Do not turn table need fields into a resource assignment unless the header says someone is already carrying that resource.",
        ]
        if fact_kind == "table" and carry_type == "none" and intent in {"discover", "reply", "status"}:
            rules.extend(
                [
                    "- This is a table status/need fact, not a pickup task.",
                    "- Do not say 'I will handle wood/sand/stone' or 'you take wood/sand/stone' from this table need list.",
                    "- Allowed: say the table can receive resources, ask for current carrying/resource status, or ask the recipient to check nearby resource piles.",
                ]
            )
        elif fact_kind == "resource" and intent == "discover":
            resource_type = str((fact or {}).get("resource_type") or "resource")
            rules.extend(
                [
                    f"- This is a {resource_type} resource fact. You may say you will check or collect that resource if grounded by the header/current facts.",
                    "- You may ask the recipient to check a different nearby area, table capacity, or another resource type.",
                ]
            )
        elif intent in {"deliver", "carry"} or carry_type in {"wood", "sand", "stone"}:
            rules.extend(
                [
                    "- This is grounded carrying/delivery status. You may state your current load, delivery target, or ask for complementary resource/table status.",
                    "- Keep the recipient request mild; do not redirect them from a visible pickup unless the header gives a concrete reason.",
                ]
            )
        else:
            rules.append("- Keep the note as a status/request; avoid making unsupported task assignments.")
        return "\n".join(rules)

    def _stable_fair_protocol_fact(
        self,
        obs: dict[str, Any],
        feature: dict[str, Any] | None,
        carry_type: str,
        carry_qty: int | None,
    ) -> dict[str, Any]:
        recipient_carry_type = "none"
        recipient_carry_qty = None
        if isinstance(feature, dict):
            recipient_carry_type = str(feature.get("carrying") or "none").strip().lower()
            recipient_carry_qty = _coerce_int(feature.get("carrying_qty"))
        if recipient_carry_type in {"wood", "sand", "stone"}:
            recipient_table_fact = self._stable_fair_best_table_fact(obs, recipient_carry_type, recipient_carry_qty)
            if recipient_table_fact:
                return recipient_table_fact
        if carry_type != "none":
            table_fact = self._stable_fair_best_table_fact(obs, carry_type, carry_qty)
            if table_fact:
                return table_fact
            return {"kind": "status"}
        resource_fact = self._stable_fair_best_resource_fact(obs)
        if resource_fact:
            return resource_fact
        table_fact = self._stable_fair_best_table_fact(obs, carry_type, carry_qty)
        if table_fact:
            return table_fact
        return {"kind": "status"}

    def _stable_fair_best_table_fact(
        self,
        obs: dict[str, Any],
        carry_type: str,
        carry_qty: int | None,
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        try:
            for entry in list(obs.get("tables_info") or []):
                pos = _extract_xy(entry.get("position"))
                if pos is None:
                    continue
                rem = _coerce_int(entry.get("remaining_capacity"))
                if rem is not None and rem <= 0:
                    continue
                candidates.append(
                    {
                        "kind": "table",
                        "position": pos,
                        "remaining_capacity": rem,
                        "remaining_by_type": dict(entry.get("remaining_by_type") or {}),
                        "source": "visible",
                    }
                )
            for entry in list(obs.get("kb_nearest_tables") or []):
                pos = _extract_xy(entry.get("position"))
                if pos is None:
                    continue
                rem = _coerce_int(entry.get("remaining_capacity"))
                if rem is not None and rem <= 0:
                    continue
                candidates.append(
                    {
                        "kind": "table",
                        "position": pos,
                        "remaining_capacity": rem,
                        "remaining_by_type": dict(entry.get("remaining_by_type") or {}),
                        "source": "memory",
                    }
                )
        except Exception:
            candidates = []
        if not candidates:
            return None
        pos = _extract_xy(obs.get("position")) or (0, 0)

        def key(fact: dict[str, Any]) -> tuple[int, int, int]:
            rem = _coerce_int(fact.get("remaining_capacity"))
            rem_value = int(rem) if rem is not None else 0
            rb = fact.get("remaining_by_type") or {}
            type_fit = 0
            if carry_type in {"wood", "sand", "stone"} and isinstance(rb, dict):
                need = _coerce_int(rb.get(carry_type))
                if need is not None and need > 0:
                    type_fit = 2 if carry_qty is None or need >= int(carry_qty) else 1
            elif rem is None or rem_value > 0:
                type_fit = 1
            distance = _manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0
            return (type_fit, rem_value, -distance)

        candidates.sort(key=key, reverse=True)
        return dict(candidates[0])

    def _stable_fair_resource_candidates(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        data = dict(obs or {})
        cap = data.get("capacity_info") or {}
        details = (cap.get("_resource_details") if isinstance(cap, dict) else {}) or {}
        candidates: list[dict[str, Any]] = []
        for rec in details.values():
            pos = _extract_xy(rec.get("position"))
            if pos is None or str(rec.get("terrain_type") or "").strip().lower() == "table":
                continue
            for item in list(rec.get("items") or []):
                if not isinstance(item, dict):
                    continue
                resource_type = str(item.get("type") or "").strip().lower()
                qty = _coerce_int(item.get("quantity"))
                if resource_type not in {"wood", "sand", "stone"} or qty is None or qty <= 0:
                    continue
                candidates.append(
                    {
                        "kind": "resource",
                        "resource_type": resource_type,
                        "position": pos,
                        "quantity": int(qty),
                        "source": "visible",
                    }
                )
        for rec in list(data.get("kb_nearest_items") or []):
            pos = _extract_xy(rec.get("position"))
            if pos is None:
                continue
            raw_items = rec.get("items") or {}
            iterator = raw_items.items() if isinstance(raw_items, dict) else []
            for item_type, qty_raw in iterator:
                resource_type = str(item_type or "").strip().lower()
                qty = _coerce_int(qty_raw)
                if resource_type in {"wood", "sand", "stone"} and qty is not None and qty > 0:
                    candidates.append(
                        {
                            "kind": "resource",
                            "resource_type": resource_type,
                            "position": pos,
                            "quantity": int(qty),
                            "source": "memory",
                        }
                    )
        by_key: dict[tuple[str, tuple[int, int]], dict[str, Any]] = {}

        def source_rank(fact: dict[str, Any]) -> int:
            source = str(fact.get("source") or "").strip().lower()
            if source == "current_tile":
                return 3
            if source == "visible":
                return 2
            return 1

        for fact in candidates:
            pos = _extract_xy(fact.get("position"))
            resource_type = str(fact.get("resource_type") or "").strip().lower()
            if pos is None or resource_type not in {"wood", "sand", "stone"}:
                continue
            key = (resource_type, pos)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = dict(fact)
                continue
            if (
                source_rank(fact),
                int(_coerce_int(fact.get("quantity")) or 0),
            ) > (
                source_rank(existing),
                int(_coerce_int(existing.get("quantity")) or 0),
            ):
                by_key[key] = dict(fact)
        return list(by_key.values())

    def _stable_fair_top_resource_facts(self, obs: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
        data = dict(obs or {})
        candidates = self._stable_fair_resource_candidates(data)
        if not candidates:
            return []
        pos = _extract_xy(data.get("position")) or (0, 0)

        def source_rank(fact: dict[str, Any]) -> int:
            source = str(fact.get("source") or "").strip().lower()
            if source == "current_tile":
                return 3
            if source == "visible":
                return 2
            return 1

        candidates.sort(
            key=lambda fact: (
                source_rank(fact),
                int(fact.get("quantity") or 0),
                -int(_manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0),
            ),
            reverse=True,
        )
        return [dict(fact) for fact in candidates[: max(1, int(limit))]]

    def _stable_fair_table_rank_tuple(
        self,
        fact: dict[str, Any],
        obs: dict[str, Any],
        carry_type: str,
        carry_qty: int | None,
    ) -> tuple[int, int, int, int]:
        source = str(fact.get("source") or "").strip().lower()
        source_value = 3 if source == "current_tile" else 2 if source == "visible" else 1
        rb = fact.get("remaining_by_type") or {}
        positive_values: list[int] = []
        if isinstance(rb, dict):
            for value in rb.values():
                amount = _coerce_int(value)
                if amount is not None and int(amount) > 0:
                    positive_values.append(int(amount))
        rem = _coerce_int(fact.get("remaining_capacity"))
        remaining_value = int(rem) if rem is not None else sum(positive_values)
        fit = 0
        carry_norm = str(carry_type or "").strip().lower()
        if carry_norm in {"wood", "sand", "stone"}:
            fit_amount = _stable_fair_table_remaining_for_type(fact, carry_norm)
            if fit_amount is not None and int(fit_amount) > 0:
                fit = 2 if carry_qty is None or int(fit_amount) >= int(carry_qty) else 1
        else:
            fit = min(3, len(positive_values))
        pos = _extract_xy(obs.get("position")) or (0, 0)
        distance = int(_manhattan_distance(pos, _extract_xy(fact.get("position"))) or 0)
        return (source_value, fit, remaining_value, -distance)

    def _stable_fair_top_table_facts(
        self,
        obs: dict[str, Any],
        carry_type: str = "none",
        carry_qty: int | None = None,
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        data = dict(obs or {})
        by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        for raw_fact in self._stable_fair_table_candidates(data):
            fact = self._normalize_stable_fair_protocol_candidate_fact(raw_fact)
            if str(fact.get("kind") or "").strip().lower() != "table":
                continue
            pos = _extract_xy(fact.get("position"))
            if pos is None:
                continue
            rb = fact.get("remaining_by_type")
            rem = _coerce_int(fact.get("remaining_capacity"))
            has_type_space = isinstance(rb, dict) and any((_coerce_int(value) or 0) > 0 for value in rb.values())
            if rem is not None and rem <= 0 and not has_type_space:
                continue
            key = self._stable_fair_protocol_candidate_fact_key(fact)
            existing = by_key.get(key)
            if existing is None or self._stable_fair_table_rank_tuple(
                fact,
                data,
                carry_type,
                carry_qty,
            ) > self._stable_fair_table_rank_tuple(
                existing,
                data,
                carry_type,
                carry_qty,
            ):
                by_key[key] = dict(fact)
        candidates = list(by_key.values())
        candidates.sort(
            key=lambda fact: self._stable_fair_table_rank_tuple(fact, data, carry_type, carry_qty),
            reverse=True,
        )
        return [dict(fact) for fact in candidates[: max(1, int(limit))]]

    def _stable_fair_best_resource_fact(self, obs: dict[str, Any]) -> dict[str, Any] | None:
        candidates = self._stable_fair_top_resource_facts(obs, limit=1)
        return dict(candidates[0]) if candidates else None

    def _stable_fair_protocol_fact_line(self, fact: dict[str, Any] | None) -> str:
        if not isinstance(fact, dict):
            return "FACT:status"
        kind = str(fact.get("kind") or "status").strip().lower()
        pos = _extract_xy(fact.get("position"))
        if kind == "resource" and pos is not None:
            resource_type = str(fact.get("resource_type") or "unknown").strip().lower()
            qty = _coerce_int(fact.get("quantity"))
            return f"FACT:resource:{resource_type}@({pos[0]},{pos[1]}) qty={int(qty) if qty is not None else 0}"
        if kind == "table" and pos is not None:
            rem = _coerce_int(fact.get("remaining_capacity"))
            if rem is None:
                rb = fact.get("remaining_by_type") or {}
                if isinstance(rb, dict):
                    rem = sum(max(0, int(v)) for v in rb.values() if _coerce_int(v) is not None)
            return f"FACT:table@({pos[0]},{pos[1]}) rem={int(rem) if rem is not None else 0}"
        return "FACT:status"

    def _stable_fair_protocol_need_line(self, fact: dict[str, Any] | None) -> str:
        if not isinstance(fact, dict) or str(fact.get("kind") or "") != "table":
            return ""
        rb = fact.get("remaining_by_type") or {}
        if not isinstance(rb, dict) or not rb:
            return ""
        parts = []
        for item_type in ("wood", "sand", "stone"):
            value = _coerce_int(rb.get(item_type))
            if value is not None:
                parts.append(f"{item_type}:{max(0, int(value))}")
        return f"NEED:{','.join(parts)}" if parts else ""

    def _stable_fair_protocol_goal(
        self,
        obs: dict[str, Any],
        fact: dict[str, Any] | None,
        carry_type: str,
    ) -> tuple[int, int] | None:
        if isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
            pos = _extract_xy(fact.get("position"))
            if pos is not None and carry_type != "none" and _stable_fair_table_supports_carry(fact, carry_type, None, require_full_qty=False):
                return pos
        for key in ("goal_position", "goal"):
            pos = _extract_xy(obs.get(key))
            if pos is not None:
                return pos
        return None

    def _stable_fair_protocol_ask(
        self,
        intent: str,
        carry_type: str,
        fact: dict[str, Any] | None,
    ) -> str:
        if str(intent) == "table_for_carry":
            return "none"
        if str(intent) == "deliver":
            if carry_type in {"wood", "sand", "stone"} and isinstance(fact, dict) and str(fact.get("kind") or "") == "table":
                if not _stable_fair_table_supports_carry(fact, carry_type, None, require_full_qty=False):
                    return f"table_for={carry_type}"
            return "none"
        if str(intent) == "reply":
            return "none"
        if carry_type != "none" and (not isinstance(fact, dict) or str(fact.get("kind") or "") != "table"):
            return f"table_for={carry_type}"
        return "none"

    def parse_incoming(
        self,
        ctx: RoundContext,
        sender: str,
        raw_message: str,
    ) -> ParsedIncomingMessage:
        if self._disabled_by_preset():
            del ctx
            return ParsedIncomingMessage(sender=sender, raw_message=str(raw_message or ""), parsed_facts={})

        state_obs_prefix = _stable_fair_parse_state_obs_prefix(raw_message)
        proto = self._parse_stable_fair_protocol_header(raw_message)
        semantic = False
        if not proto:
            proto = _parse_stable_fair_semantic_message(raw_message)
            semantic = bool(proto)
        if not proto:
            parsed = self._reference_parse_incoming(ctx, sender, raw_message)
            return parsed
        protocol_grounded = bool(state_obs_prefix) or not semantic
        if protocol_grounded:
            parsed = ParsedIncomingMessage(
                sender=sender,
                raw_message=str(raw_message or ""),
                parsed_facts={"tables": [], "resources": [], "goal": None},
                meta={
                    "tables_count": 0,
                    "resources_count": 0,
                    "has_goal": False,
                    "protocol_parse_boundary": "protocol_fields_only",
                },
            )
        else:
            parsed = self._reference_parse_incoming(ctx, sender, raw_message)
        note = _stable_fair_protocol_note(raw_message)
        if note:
            proto["note"] = note
        facts = dict(parsed.parsed_facts or {})
        facts.setdefault("tables", [])
        facts.setdefault("resources", [])
        protocol_facts: list[dict[str, Any]] = []
        for obs_fact in list(proto.get("obs_facts") or []):
            if isinstance(obs_fact, dict):
                protocol_facts.append(obs_fact)
        fact = proto.get("fact")
        if isinstance(fact, dict):
            protocol_facts.append(fact)
        seen_protocol_facts: set[tuple[Any, ...]] = set()
        for fact in protocol_facts:
            kind = str(fact.get("kind") or "").strip().lower()
            pos = _extract_xy(fact.get("position"))
            fact_key = (
                kind,
                pos,
                str(fact.get("resource_type") or ""),
                tuple(sorted((fact.get("remaining_by_type") or {}).items()))
                if isinstance(fact.get("remaining_by_type"), dict)
                else (),
            )
            if fact_key in seen_protocol_facts:
                continue
            seen_protocol_facts.add(fact_key)
            if kind == "table" and pos is not None:
                table = {
                    "x": int(pos[0]),
                    "y": int(pos[1]),
                    "remaining_capacity": fact.get("remaining_capacity"),
                }
                rb = fact.get("remaining_by_type")
                if isinstance(rb, dict) and rb:
                    table["remaining_by_type"] = dict(rb)
                facts["tables"].append(table)
            elif kind == "resource" and pos is not None:
                facts["resources"].append(
                    {
                        "x": int(pos[0]),
                        "y": int(pos[1]),
                        "type": str(fact.get("resource_type") or ""),
                        "quantity": fact.get("quantity"),
                    }
                )
        goal = proto.get("goal")
        if isinstance(goal, tuple) and len(goal) == 2 and facts.get("goal") is None:
            facts["goal"] = {"x": int(goal[0]), "y": int(goal[1])}
        if semantic:
            facts["stable_fair_semantic"] = proto
        else:
            facts["stable_fair_protocol"] = proto
        parsed.parsed_facts = facts
        meta = dict(parsed.meta or {})
        meta["protocol_profile"] = self.SEMANTIC_PROTOCOL_PROFILE if semantic else self.PROTOCOL_PROFILE
        meta["stable_fair_protocol_intent"] = str(proto.get("intent") or "status")
        parsed.meta = meta
        return parsed

    def _parse_stable_fair_protocol_header(self, raw_message: str) -> dict[str, Any] | None:
        text = str(raw_message or "")
        compact = _parse_stable_fair_compact_protocol(text)
        if compact is not None:
            return compact
        if "SFPROTO:" not in text:
            return None

        def line_value(name: str) -> str:
            match = re.search(rf"^\s*{re.escape(name)}\s*:\s*(.+?)\s*$", text, re.IGNORECASE | re.MULTILINE)
            return str(match.group(1) or "").strip() if match else ""

        proto: dict[str, Any] = {
            "version": line_value("SFPROTO") or "1",
            "id": line_value("ID"),
            "intent": (line_value("INTENT") or "status").lower(),
            "ask": line_value("ASK") or "none",
        }
        loc = _extract_xy_from_text(line_value("LOC"))
        if loc is not None:
            proto["loc"] = loc
        round_num = _coerce_int(line_value("ROUND"))
        if round_num is not None:
            proto["round"] = int(round_num)
        carry_line = line_value("CARRY")
        carry_type, carry_qty = _parse_resource_quantity(carry_line)
        proto["carry_type"] = carry_type
        proto["carry_qty"] = carry_qty
        fact_line = line_value("FACT")
        fact = self._parse_stable_fair_fact_line(fact_line)
        need = self._parse_stable_fair_need_line(line_value("NEED"))
        if fact and str(fact.get("kind") or "") == "table" and need:
            fact["remaining_by_type"] = need
        if fact:
            proto["fact"] = fact
        goal = _extract_xy_from_text(line_value("GOAL"))
        if goal is not None:
            proto["goal"] = goal
        return proto

    def _parse_stable_fair_fact_line(self, fact_line: str) -> dict[str, Any] | None:
        text = str(fact_line or "").strip()
        if not text or text.lower() == "status":
            return None
        table_match = re.search(r"table@\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*rem\s*=\s*(\d+)", text, re.IGNORECASE)
        if table_match:
            return {
                "kind": "table",
                "position": (int(table_match.group(1)), int(table_match.group(2))),
                "remaining_capacity": int(table_match.group(3)),
            }
        resource_match = re.search(
            r"resource\s*:\s*(wood|sand|stone)@\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*qty\s*=\s*(\d+)",
            text,
            re.IGNORECASE,
        )
        if resource_match:
            return {
                "kind": "resource",
                "resource_type": str(resource_match.group(1)).lower(),
                "position": (int(resource_match.group(2)), int(resource_match.group(3))),
                "quantity": int(resource_match.group(4)),
            }
        return None

    def _parse_stable_fair_need_line(self, need_line: str) -> dict[str, int]:
        text = str(need_line or "").strip()
        if not text:
            return {}
        needs: dict[str, int] = {}
        for item_type, value in re.findall(r"\b(wood|sand|stone)\s*:\s*(\d+)", text, re.IGNORECASE):
            needs[str(item_type).lower()] = int(value)
        return needs

    def _plain_routing_payload_prompt_block(self, target_name: str, obs: dict[str, Any] | None = None) -> str:
        routing_meta = getattr(self, "_active_routing_meta", None)
        mode = str((routing_meta or {}).get("mode") or self.mode or "").strip().lower()
        if mode != "gate_llm":
            return ""
        feature = self._routing_selected_feature(routing_meta, target_name)
        if not isinstance(feature, dict):
            return ""
        data = dict(obs or {})
        carry_type, carry_qty = _inventory_resource_quantity(data)
        context_lines: list[str] = []
        best_text = self._stable_fair_delivery_fact_text(feature.get("best_fact"))
        if best_text:
            context_lines.append(f"- selected teammate fact: {best_text}")
        if carry_type in {"wood", "sand", "stone"}:
            qty_text = int(carry_qty) if carry_qty is not None else "?"
            context_lines.append(f"- current load: {carry_type}:{qty_text}")
        context_lines = context_lines[:2]
        if not context_lines:
            return ""
        return (
            "Selected teammate context:\n"
            + "\n".join(context_lines)
            + "\n"
        )

    def _build_original_random_target_prompt(self, obs: dict, target_name: str, target_info: dict | None = None) -> str:
        prompt = self._reference_plain_target_prompt(obs, target_name, target_info)
        marker = "\nOutput EXACTLY one line:\n"
        routing_evidence = self._plain_routing_payload_prompt_block(target_name, obs or {})
        if not routing_evidence:
            return prompt
        extra = f"\n{routing_evidence}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"

    def _build_proto_v2_free_prompt(self, agent_name: str, header_msg: str, obs: dict, max_chars: int) -> str:
        del agent_name, max_chars
        target_name = str(getattr(self, "_active_routing_recipient", "") or "").strip() or "selected teammate"
        proto = _parse_stable_fair_semantic_message(header_msg) or _parse_stable_fair_compact_protocol(header_msg) or {}
        fact = proto.get("fact") if isinstance(proto, dict) else None
        carry_type = str((proto or {}).get("carry_type") or "none").strip().lower()
        carry_qty = (proto or {}).get("carry_qty")
        ask = str((proto or {}).get("ask") or "none")
        semantic_type = self._stable_fair_semantic_type(
            str((proto or {}).get("intent") or "status"),
            fact,
            carry_type,
            ask,
            _coerce_int(carry_qty),
        )
        fallback_message = str((proto or {}).get("note") or "").strip()
        if not fallback_message:
            fallback_message = self._stable_fair_semantic_fallback_message(
                semantic_type=semantic_type,
                fact=fact if isinstance(fact, dict) else None,
                carry_type=carry_type,
                carry_qty=carry_qty if _coerce_int(carry_qty) is not None else None,
                ask=ask,
            )
        action_type, action_message, _action_is_strong = self._stable_fair_action_semantic_candidate(
            obs or {},
            fact=fact if isinstance(fact, dict) else None,
            carry_type=carry_type,
            carry_qty=carry_qty if _coerce_int(carry_qty) is not None else None,
        )
        feature = self._routing_selected_feature(getattr(self, "_active_routing_meta", None), target_name)
        return self._build_stable_fair_semantic_protocol_prompt(
            obs or {},
            target_name,
            semantic_type=semantic_type,
            fallback_message=fallback_message,
            fact=fact if isinstance(fact, dict) else None,
            carry_type=carry_type,
            carry_qty=carry_qty if _coerce_int(carry_qty) is not None else None,
            ask=ask,
            feature=feature,
            action_type=action_type,
            action_message=action_message,
        )

    def _build_stable_fair_strong_message_prompt(
        self,
        obs: dict | None,
        target_name: str,
        target_info: dict | None = None,
    ) -> str:
        prompt = self._reference_plain_target_prompt(obs or {}, target_name, target_info)
        marker = "\nOutput EXACTLY one line:\n"
        facts_block = _compact_current_facts_block(obs)
        routing_reason = self._routing_decision_reason_prompt_block(target_name)
        message_guidance = _direct_actionable_message_guidance()
        extra = f"\n{facts_block}{routing_reason}{message_guidance}"
        if marker in prompt:
            return prompt.replace(marker, f"{extra}{marker}", 1)
        return f"{prompt}{extra}"

    def _build_stable_fair_protocol_message_prompt(
        self,
        obs: dict | None,
        target_name: str,
        target_info: dict | None,
        header_msg: str,
        max_chars: int,
    ) -> str:
        prompt = self._build_stable_fair_strong_message_prompt(obs, target_name, target_info)
        marker = "\nOutput EXACTLY one line:\n"
        format_block = self._stable_fair_protocol_message_format_block(header_msg, obs or {}, max_chars)
        if marker in prompt:
            return prompt.replace(marker, f"\n{format_block}{marker}", 1)
        return f"{prompt}\n{format_block}"

    def _stable_fair_protocol_message_format_block(
        self,
        header_msg: str,
        obs: dict[str, Any],
        max_chars: int,
    ) -> str:
        limit = int(max_chars or 140)
        return "\n".join(
            [
                "Message note requirements:",
                "- A structured update is already prepared; write only the natural-language note.",
                "- Use the provided intent and fact as ground truth, and do not contradict them.",
                "- Output only the requested MSG line; do not add extra headings or explanation.",
                f"- Keep the message text under {limit} characters.",
                "- The final line must be MSG: <short natural-language note>.",
                self._protocol_note_policy(header_msg),
                self._protocol_note_decoded_context(header_msg, obs),
            ]
        )

    def _protocol_note_decoded_context(self, header_msg: str, obs: dict[str, Any]) -> str:
        proto = _parse_stable_fair_compact_protocol(header_msg) or {}
        lines = [
            "Decoded protocol context:",
            f"- Summary: {_stable_fair_protocol_memory_summary(header_msg)}",
            f"- Intent: {proto.get('intent', 'status')}",
        ]
        carry_type = str(proto.get("carry_type") or "none")
        if carry_type != "none":
            lines.append(f"- Carrying: {carry_type}:{proto.get('carry_qty') if proto.get('carry_qty') is not None else '?'}")
        ask = str(proto.get("ask") or "none")
        if ask and ask.lower() != "none":
            lines.append(f"- Ask: {ask}")

        data = dict(obs or {})
        lines.append(f"- Inventory: {str((data.get('inventory') or {}).get('display') or 'empty')}")
        visible_resources = _compact_visible_resource_summary(data, max_lines=2)
        visible_tables = _compact_visible_table_summary(data, max_lines=2)
        memory_hints = _compact_memory_hint_summary(data, max_entries=2)
        if visible_resources != "none":
            lines.append(f"- Visible resources: {visible_resources}")
        if visible_tables != "none":
            lines.append(f"- Visible tables: {visible_tables}")
        if memory_hints != "none":
            lines.append(f"- Memory hints: {memory_hints}")
        recipient = str(getattr(self, "_active_routing_recipient", "") or "").strip()
        if recipient:
            lines.append(f"- Recipient: {recipient}")
        return "\n".join(lines)

    def _sanitize_protocol_note_line(self, text: str, max_chars: int) -> str:
        parsed = self._parse_msg_only(text)
        lines = str(text or "").splitlines()
        line = parsed
        if not line:
            for raw in lines:
                if re.match(r"^\s*(?:MSG|NOTE|PLAN|GOAL)\s*:", raw, re.IGNORECASE):
                    line = raw
                    break
        if not line:
            for raw in lines:
                if (
                    raw.strip()
                    and not raw.strip().upper().startswith("SF1|")
                    and not re.match(r"^\s*(?:STATE|OBS)\s*:", raw, re.IGNORECASE)
                ):
                    line = raw
                    break
        if not line:
            line = lines[0] if lines else ""
        for _ in range(3):
            stripped = line.strip()
            match = re.match(r"^(?:SEND|TO|MSG|NOTE|PLAN|GOAL|HEADER|STATE|OBS)\s*:\s*(.*)$", stripped, re.IGNORECASE)
            if not match:
                break
            line = str(match.group(1) or "").strip()
        line = re.sub(r"\bSF1\s*\|[^\s]+", "", line, flags=re.IGNORECASE)
        line = " ".join(line.strip().split())
        try:
            limit = int(max_chars or 0)
        except Exception:
            limit = 0
        if limit > 0 and len(line) > limit:
            line = line[:limit].rstrip()
        return line

    def _sanitize_free_line(self, text: str, max_chars: int) -> str:
        return self._sanitize_protocol_note_line(text, max_chars)

    def _routing_focus_feature(
        self,
        routing_meta: dict[str, Any] | None,
        recipient: str,
    ) -> dict[str, Any] | None:
        target = str(recipient or "").strip()
        if not target:
            return None
        reorder_meta = dict((routing_meta or {}).get("candidate_reorder_meta") or {})
        seen: set[int] = set()
        for key in ("features", "features_full"):
            for feature in list(reorder_meta.get(key) or []):
                if not isinstance(feature, dict):
                    continue
                ident = id(feature)
                if ident in seen:
                    continue
                seen.add(ident)
                if str(feature.get("name") or "").strip() != target:
                    continue
                if self._routing_focus_fact_is_usable(feature):
                    return feature
        return None

    def _routing_decision_reason_prompt_block(self, target_name: str) -> str:
        routing_meta = getattr(self, "_active_routing_meta", None)
        feature = self._routing_selected_feature(routing_meta, target_name)
        if not isinstance(feature, dict):
            return ""
        mode = str((routing_meta or {}).get("mode") or self.mode or "unknown")
        send_policy = str((routing_meta or {}).get("routing_send_policy") or "unknown")
        select_policy = str((routing_meta or {}).get("routing_select_policy") or "unknown")
        variant = str((routing_meta or {}).get("route_strategy_variant") or "unknown")
        context_lines = self._stable_fair_protocol_routing_context_lines(target_name, feature)
        return "\n".join(
            [
                "Routing-selected recipient context:",
                f"- routing_mode={mode}; send_policy={send_policy}; select_policy={select_policy}; strategy={variant}",
                "- recipient choice is already made; this block describes audience need, not a fixed message type.",
                *context_lines,
                "- Choose content from current facts, inventory, current action, recent messages, and table compatibility.",
                "- Treat edge, payload_state, and audience_fit as weak audience context for grounded anchors, not as a required payload.",
                "",
            ]
        )

    def _recipient_context_prompt_block(self, target_name: str) -> str:
        routing_meta = getattr(self, "_active_routing_meta", None)
        feature = self._routing_selected_feature(routing_meta, target_name)
        if not isinstance(feature, dict):
            return ""
        context_lines = self._stable_fair_protocol_routing_context_lines(target_name, feature)
        return "\n".join(
            [
                "Selected recipient context:",
                *context_lines,
                "- Use this only as audience context.",
                "",
            ]
        )

    def _routing_focus_fact_is_usable(self, feature: dict[str, Any] | None) -> bool:
        if not isinstance(feature, dict):
            return False
        fact = feature.get("best_fact")
        if not isinstance(fact, dict):
            return False
        kind = str(feature.get("best_fact_kind") or fact.get("kind") or "").strip().lower()
        if kind not in {"resource", "table"}:
            return False
        if bool(feature.get("duplicate_fact_recent")):
            return False
        if kind == "resource":
            resource_type = str(feature.get("best_fact_type") or fact.get("resource_type") or "").strip().lower()
            if resource_type not in {"wood", "sand", "stone"}:
                return False
            return float(feature.get("target_need", 1.0) or 1.0) >= 0.75
        objective = _stable_fair_routing_objective_for_feature(feature)
        if not (bool(feature.get("safe_table_delivery")) or objective == "share_table_for_carrier" or str(feature.get("routing_intent") or "") == "table_for_carry"):
            return False
        remaining_capacity = _coerce_int(fact.get("remaining_capacity"))
        return remaining_capacity is None or remaining_capacity > 0

    def _routing_fact_label(self, fact: dict[str, Any] | None) -> str:
        if not isinstance(fact, dict):
            return "none"
        pos = _extract_xy(fact.get("position"))
        if pos is None:
            return "none"
        if str(fact.get("kind") or "") == "table":
            return f"table@({pos[0]},{pos[1]})"
        resource_type = str(fact.get("resource_type") or "unknown").strip().lower() or "unknown"
        return f"{resource_type}@({pos[0]},{pos[1]})"


__all__ = [
    "StableFairMemoryMechanism",
    "StableFairRoutingMechanism",
    "StableFairProtocolMechanism",
]
