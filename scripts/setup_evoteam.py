"""Download or verify the pinned runtime from a configurable source repository."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
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
    parser.add_argument("--source-repository", default=os.environ.get("EVOTEAM_SOURCE_REPOSITORY", ""),
                        help="GitHub owner/repository containing the pinned runtime commit "
                             "(defaults to EVOTEAM_SOURCE_REPOSITORY)")
    args = parser.parse_args()
    lock = json.loads((ROOT / "methods/Evoteam/source-lock.json").read_text())
    destination = args.destination.expanduser().resolve()
    if destination.exists():
        verify_source(destination, lock)
    else:
        repository = args.source_repository.strip()
        if not repository:
            parser.error("A fresh download requires --source-repository owner/repository or "
                         "EVOTEAM_SOURCE_REPOSITORY pointing to the pinned runtime. "
                         "Alternatively, use --destination with an existing runtime checkout; "
                         "see methods/Evoteam/README.md.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            parser.error("--source-repository must be a GitHub owner/repository")
        url = f'https://codeload.github.com/{repository}/tar.gz/{lock["revision"]}'
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
