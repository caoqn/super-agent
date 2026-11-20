"""External Evoteam tests are opt-in; unrelated method tests run independently."""
import os

collect_ignore = []
if not os.environ.get("EVOTEAM_ROOT"):
    collect_ignore = ["test_macbench_adapter.py", "test_masbench_adapter.py"]
os.environ.setdefault("DEV_API_BASE", "https://example.invalid/v1")
os.environ.setdefault("DEV_API_KEY", "test-placeholder")
