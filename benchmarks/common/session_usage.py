"""Recover provider-reported token usage from locally persisted DSH Sessions."""
from pathlib import Path
import json
import io


def read_session_usage(session_id, sessions_root=None):
    """Return usage totals for one worker session, or None if unavailable."""
    if not session_id:
        return None
    try:
        import zstandard
    except ImportError:
        return None
    root = Path(sessions_root or Path.home() / '.dsh/sessions')
    matches = list(root.glob(f'**/{session_id}/session.v4.jsonl.zstd'))
    if not matches:
        return None
    keys = ('inputTokens', 'outputTokens', 'totalTokens', 'cacheReadTokens', 'cacheWriteTokens')
    totals = {key: 0 for key in keys}
    events = 0
    with matches[0].open('rb') as source:
        reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(source), encoding='utf-8')
        for line in reader:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            usage = event.get('data', {}).get('usage')
            if event.get('type') != 'assistant/message' or not usage:
                continue
            events += 1
            for key in keys:
                totals[key] += int(usage.get(key, 0) or 0)
    if not events:
        return None
    return {'usage_events': events, **totals}
