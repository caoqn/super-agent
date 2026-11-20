"""JSON-lines bridge to an isolated process running the upstream Game."""
from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import sys
import threading


class GameRPC:
    def __init__(self, source: Path, map_path: Path, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.log = (out / "worker.log").open("a", encoding="utf-8")
        self.lock = threading.Lock()
        command = [sys.executable, str(Path(__file__).with_name("worker.py")),
                   str(source), str(map_path), str(out.resolve())]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, encoding="utf-8", bufsize=1,
            )
        except OSError:
            self.log.close()
            raise
        try:
            self.actors = self._read()["actors"]
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict:
        if not select.select([self.process.stdout], [], [], 90)[0]:
            raise TimeoutError("MASBench Game RPC timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MASBench Game worker exited; inspect {self.log.name}")
        response = json.loads(line)
        if not response.get("ok"):
            raise ValueError(response.get("error"))
        return response

    def call(self, op: str, **kwargs):
        with self.lock:
            self.process.stdin.write(json.dumps({"op": op, **kwargs}) + "\n")
            self.process.stdin.flush()
            return self._read()["result"]

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.log.close()
