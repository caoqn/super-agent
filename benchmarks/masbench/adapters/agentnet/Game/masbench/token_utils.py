from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[\u4e00-\u9fff]|[^\w\s]", re.UNICODE)


def estimate_text_tokens(text: Any) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    return len(_TOKEN_PATTERN.findall(raw))


def _looks_like_protocol_v2_message(text: str) -> bool:
    lines = [str(line or "").strip() for line in str(text or "").splitlines() if str(line or "").strip()]
    if not lines:
        return False
    has_id = any(line.startswith("ID:") for line in lines)
    has_loc = any(line.startswith("LOC:") for line in lines)
    return has_id and has_loc


def split_sent_message_parts(message: Any) -> dict[str, str]:
    item = message if isinstance(message, dict) else {}
    text = str(item.get("message") if isinstance(item, dict) else message or "").strip()
    if not text:
        return {
            "full_text": "",
            "llm_generated_text": "",
            "deterministic_text": "",
        }

    llm_text = ""
    deterministic_text = ""
    if isinstance(item, dict):
        parts = item.get("message_parts")
        if isinstance(parts, dict):
            llm_text = str(parts.get("llm_generated_text") or "").strip()
            deterministic_text = str(parts.get("deterministic_text") or "").strip()
        if not llm_text and not deterministic_text:
            llm_text = str(item.get("llm_generated_text") or "").strip()
            deterministic_text = str(item.get("deterministic_text") or "").strip()

    if llm_text or deterministic_text:
        return {
            "full_text": text,
            "llm_generated_text": llm_text,
            "deterministic_text": deterministic_text,
        }

    if not _looks_like_protocol_v2_message(text):
        return {
            "full_text": text,
            "llm_generated_text": text,
            "deterministic_text": "",
        }

    lines = [str(line or "").strip() for line in text.splitlines() if str(line or "").strip()]
    note_body = ""
    deterministic_lines: list[str] = []
    for line in lines:
        if line.startswith("NOTE:") and not note_body:
            note_body = line.split(":", 1)[1].strip() if ":" in line else ""
            deterministic_lines.append("NOTE:")
        else:
            deterministic_lines.append(line)
    return {
        "full_text": text,
        "llm_generated_text": note_body,
        "deterministic_text": "\n".join(deterministic_lines).strip(),
    }


def collect_message_exchange_stats(messages: list[dict[str, Any]] | None) -> dict[str, Any]:
    stats = {
        "method": "text_estimate_v1",
        "total": 0,
        "estimated_total": 0,
        "llm_generated_total": 0,
        "estimated_llm_generated_total": 0,
        "deterministic_total": 0,
        "estimated_deterministic_total": 0,
        "messages_count": 0,
        "deliveries_count": 0,
        "unique_pairs_count": 0,
        "by_agent": {},
    }
    unique_pairs: set[tuple[str, str]] = set()
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        sender = str(item.get("agent_name") or "").strip()
        text = str(item.get("message") or "").strip()
        recipients = item.get("recipients")
        recipient_list = [str(r).strip() for r in recipients] if isinstance(recipients, list) and recipients else [""]
        recipient_list = [r for r in recipient_list if r]
        recipient_count = len(recipient_list) if recipient_list else 1
        parts = split_sent_message_parts(item)
        token_count = estimate_text_tokens(parts["full_text"])
        llm_generated_count = estimate_text_tokens(parts["llm_generated_text"])
        deterministic_count = estimate_text_tokens(parts["deterministic_text"])
        if (llm_generated_count + deterministic_count) != token_count:
            deterministic_count = max(0, token_count - llm_generated_count)
        weighted = token_count * max(1, int(recipient_count))
        weighted_llm = llm_generated_count * max(1, int(recipient_count))
        weighted_deterministic = deterministic_count * max(1, int(recipient_count))
        stats["messages_count"] += 1
        stats["deliveries_count"] += max(1, int(recipient_count))
        stats["total"] += weighted
        stats["estimated_total"] += weighted
        stats["llm_generated_total"] += weighted_llm
        stats["estimated_llm_generated_total"] += weighted_llm
        stats["deterministic_total"] += weighted_deterministic
        stats["estimated_deterministic_total"] += weighted_deterministic
        if sender:
            for recipient in recipient_list:
                if recipient:
                    unique_pairs.add((sender, recipient))
        if not sender:
            continue
        agent_stats = stats["by_agent"].setdefault(
            sender,
            {
                "estimated_total": 0,
                "estimated_llm_generated_total": 0,
                "estimated_deterministic_total": 0,
                "messages_count": 0,
                "deliveries_count": 0,
            },
        )
        agent_stats["estimated_total"] += weighted
        agent_stats["estimated_llm_generated_total"] += weighted_llm
        agent_stats["estimated_deterministic_total"] += weighted_deterministic
        agent_stats["messages_count"] += 1
        agent_stats["deliveries_count"] += max(1, int(recipient_count))
    stats["unique_pairs_count"] = len(unique_pairs)
    return stats


def _iter_game_log_messages(records: Any):
    if not isinstance(records, list):
        return
    seen: set[str] = set()
    for row in records:
        if not isinstance(row, dict):
            continue
        for msg in row.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            try:
                key = json.dumps(
                    {
                        "round": msg.get("round"),
                        "agent_name": msg.get("agent_name"),
                        "message": msg.get("message"),
                        "recipients": msg.get("recipients"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
            except Exception:
                key = repr(
                    (
                        msg.get("round"),
                        msg.get("agent_name"),
                        msg.get("message"),
                        tuple(msg.get("recipients") or []),
                    )
                )
            if key in seen:
                continue
            seen.add(key)
            yield msg


def load_message_exchange_stats_from_run_dir(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir)
    if not path.is_dir():
        return None
    messages: list[dict[str, Any]] = []
    for game_log_path in sorted(path.glob("game_log_*.json")):
        try:
            with game_log_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            continue
        for msg in _iter_game_log_messages(records):
            messages.append(dict(msg))
    if not messages:
        return None
    return collect_message_exchange_stats(messages)
