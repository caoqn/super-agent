"""External Evoteam tests are opt-in; unrelated method tests run independently."""
import os
from pathlib import Path

collect_ignore = []
if not os.environ.get("EVOTEAM_ROOT") and not (Path(__file__).resolve().parents[1] / "external/EvoTeam").exists():
    collect_ignore = ["test_macbench_adapter.py", "test_masbench_adapter.py", "test_evoteam_budget.py", "test_evoteam_training.py"]
os.environ.setdefault("DEV_API_BASE", "https://example.invalid/v1")
os.environ.setdefault("DEV_API_KEY", "test-placeholder")
