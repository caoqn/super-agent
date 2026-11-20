"""GAIA split loading and historical normalized-exact-match scoring."""
import os, json, re
from pathlib import Path

DATA_DIR = Path(os.environ.get("GAIA_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "gaia"))
VAL_FILES_DIR = DATA_DIR / "val_files"

SPLIT_MAP = {
    "all": ("val_new", None),
    "level1": ("val_new", {1}),
    "level2": ("val_new", {2}),
    "level3": ("val_new", {3}),
    "train_20": ("train_20", None),
    "test_100": ("test_100", None),
}

def load_gaia_data(
    split: str = "all",
    max_items: int | None = None,
) -> list[dict]:
    if split not in SPLIT_MAP:
        raise ValueError(f"Unknown split: {split!r}. Valid: {list(SPLIT_MAP)}")

    file_kind, level_filter = SPLIT_MAP[split]
    file_suffix = {
        "val_new": "val_new",
        "train_20": "train_20",
        "test_100": "test_100",
    }[file_kind]

    items = []
    for level in [1, 2, 3]:
        if level_filter is not None and level not in level_filter:
            continue

        data_file = DATA_DIR / f"level_{level}_{file_suffix}.json"
        if not data_file.exists():
            raise FileNotFoundError(
                f"GAIA data file not found: {data_file}\n"
                "Provision the official GAIA inputs, then run scripts/prepare_evoteam_gaia.py "
                "or set GAIA_DATA_DIR to your prepared split directory."
            )

        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["_level"] = level
            items.append(item)

    if max_items:
        items = items[:max_items]

    return items

def normalize_answer(s: str) -> str:
    s = s.lower()
    return re.sub(r'[^a-z0-9]', '', s)

def extract_final_answer(model_output: str) -> str:
    parts = re.split(r'(?i)final\s*answer\s*:', model_output, maxsplit=1)
    if len(parts) > 1:
        return parts[1].strip()
    return model_output.strip()

def check_answer(model_output: str, standard_answer: str) -> bool:
    extracted = extract_final_answer(model_output)
    return normalize_answer(extracted) == normalize_answer(standard_answer)
