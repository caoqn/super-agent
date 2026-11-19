import os
import builtins
from typing import Any

_original_print = builtins.print
_silencer_enabled = False

def _filtered_print(*args: Any, **kwargs: Any):
    try:
        s = " ".join(str(a) for a in args)
        if "DEBUG:" in s:
            return
    except Exception:
        # If anything goes wrong, fall back to original printing
        pass
    return _original_print(*args, **kwargs)

def silence_debug_prints(enable: bool = True) -> None:
    """Globally silence print outputs that contain 'DEBUG:'.

    Use env var `ENABLE_DEBUG_PRINTS=1` to force-enable debug prints regardless.
    """
    global _silencer_enabled
    if str(os.getenv("ENABLE_DEBUG_PRINTS", "")).strip() in ("1", "true", "True"):
        return
    if enable and not _silencer_enabled:
        builtins.print = _filtered_print  # type: ignore
        _silencer_enabled = True
    elif not enable and _silencer_enabled:
        builtins.print = _original_print  # type: ignore
        _silencer_enabled = False
