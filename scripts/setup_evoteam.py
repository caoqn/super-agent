"""Download the pinned public runtime; no Git or GitHub login is required."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify_source(root, lock):
    for name, expected in lock["files"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"EvoTeam source mismatch: {name}; use a fresh destination.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "external/EvoTeam")
    parser.add_argument("--no-install", action="store_true", help="Only download and verify source")
    args = parser.parse_args()
    lock = json.loads((ROOT / "methods/Evoteam/source-lock.json").read_text())
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        verify_source(destination, lock)
    else:
        url = f'https://codeload.github.com/{lock["repository"]}/tar.gz/{lock["revision"]}'
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as stage:
            unpacked = Path(stage) / "source"
            unpacked.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                for member in tar:
                    parts = Path(member.name).parts[1:]
                    if not parts:
                        continue
                    if member.issym() or member.islnk() or ".." in parts or Path(*parts).is_absolute():
                        raise RuntimeError("Unsafe source archive member")
                    target = unpacked.joinpath(*parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as source:
                            target.write_bytes(source.read())
                    else:
                        raise RuntimeError("Unsupported source archive member")
            verify_source(unpacked, lock)
            shutil.move(str(unpacked), destination)
    if not args.no_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(ROOT / "requirements/base.txt"), "-r",
                        str(destination / "requirements.txt")], check=True)
    print(f'Verified EvoTeam {lock["revision"]} at {destination}')
    if destination != (ROOT / "external/EvoTeam").resolve():
        print(f'Export EVOTEAM_ROOT="{destination}" before running adapters.')
    print("Next: configure your API credentials and benchmark data; see methods/Evoteam/README.md.")


if __name__ == "__main__":
    main()
