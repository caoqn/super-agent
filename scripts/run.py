"""Launch one benchmark in an isolated method process; extra flags override defaults."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--config", required=True, type=Path)
args, extra = p.parse_known_args()
config = json.loads(args.config.read_text())
benchmark = config["benchmark"]
if benchmark not in {"gaia", "macbench", "masbench"}:
    p.error("benchmark must be gaia, macbench, or masbench")
env = os.environ.copy()
# Config supplies defaults; the external method may load env-file overrides.
env.update({str(k): str(v) for k,v in config.get("env", {}).items()})
command = [sys.executable, str(ROOT / "methods/Evoteam" / benchmark / "run.py"),
           *config.get("args", []), *extra]
raise SystemExit(subprocess.call(command, env=env, cwd=ROOT))
