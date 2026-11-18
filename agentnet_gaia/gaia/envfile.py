"""Tiny .env loader (no python-dotenv dependency)."""
import os


def load_env(paths):
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                elif " #" in v or "\t#" in v:
                    v = v.split("#", 1)[0].strip()  # inline comment
                # real environment variables win over the file
                os.environ.setdefault(k, v)
