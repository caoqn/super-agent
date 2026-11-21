"""Run MAC-Bench engineering tasks through the local Docker CLI.

This backend replaces ModalSandboxBackend with calls to the local `docker` CLI.
Engineering tasks use standard Docker Hub images (such as python:3.11-slim) and
ordinary shell setup scripts, so they can run locally.

Inject it in run_macbench.py with `open_task(rec, dir, backend=DockerSandboxBackend())`.
Requires Docker with its daemon running (Docker Desktop or Colima).
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import tarfile
import time
import uuid
from pathlib import Path
from typing import Mapping

from mac_bench.runtime.sandbox.base import (
    CommandResult,
    SandboxBackend,
    SandboxHandle,
)


def _docker(args, *, input_bytes=None, timeout=None, check=False):
    """Run a Docker CLI command and return its CompletedProcess."""
    cmd = ["docker", *args]
    try:
        p = subprocess.run(
            cmd, input=input_bytes, capture_output=True, timeout=timeout or 600
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"docker {' '.join(args[:2])} timed out")
    if check and p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", "replace")[:500]
        raise RuntimeError(f"docker {' '.join(args)} failed: {err}")
    return p


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class DockerSandboxBackend(SandboxBackend):
    """Implement MAC-Bench SandboxBackend using local Docker containers."""

    def __init__(self, timeout: int = 600):
        self.timeout = timeout

    # ---------------------------------------------------------------- create
    def create(
        self,
        *,
        name: str,
        snapshot: str | None = None,
        image: str | None = None,
        network_blocked: bool = False,
    ) -> SandboxHandle:
        img = snapshot or image
        if not img:
            raise ValueError("create requires image or snapshot")
        # Pull missing images; snapshot images are already available locally.
        if image and _docker(["image", "inspect", image]).returncode != 0:
            _docker(["pull", image], check=True)
        cid = "mb-" + uuid.uuid4().hex[:12]
        args = ["run", "-d", "--name", cid]
        if network_blocked:
            args += ["--network", "none"]
        args += [img, "sh", "-c", "exec sleep infinity"]
        _docker(args, check=True)
        # Create only /work; write_file/upload_tree create subdirectories on demand.
        # Do not precreate /work/project: consumer validation uses its absence to check a clean environment.
        _docker(["exec", cid, "sh", "-c", "mkdir -p /work"], check=True)
        # Rewrite apt sources to HTTPS because proxy fake-IP handling is unreliable on HTTP.
        # HTTPS SNI lets the proxy route apt-get and pip traffic to the correct host.
        _docker(
            ["exec", cid, "sh", "-c",
             "for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do "
             "[ -f \"$f\" ] && sed -i 's|http://|https://|g' \"$f\"; done; true"],
            check=True,
        )
        return SandboxHandle(id=cid, provider="docker", raw=cid)

    # ---------------------------------------------------------------- exec
    def exec(
        self,
        sandbox: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> CommandResult:
        started = time.monotonic()
        full = command
        if env:
            prefix = " ".join(f"export {k}={shlex.quote(v)};" for k, v in env.items())
            full = prefix + command
        if cwd:
            full = f"cd {shlex.quote(cwd)} && " + full
        try:
            p = _docker(["exec", sandbox.id, "bash", "-lc", full], timeout=timeout)
            stdout = (p.stdout + p.stderr).decode("utf-8", errors="replace")
            exit_code = p.returncode
        except RuntimeError as e:
            stdout = f"[docker exec failed: {e}]"
            exit_code = 125
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            duration_seconds=time.monotonic() - started,
        )

    # ---------------------------------------------------------------- files
    def write_file(self, sandbox: SandboxHandle, *, path: str, content: bytes) -> None:
        b64 = _b64encode(content)
        parent = os.path.dirname(path)
        prefix = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        script = prefix + f"base64 -d > {shlex.quote(path)}"
        _docker(
            ["exec", "-i", sandbox.id, "sh", "-c", script],
            input_bytes=b64.encode("ascii"),
            check=True,
        )

    def write_files(self, sandbox: SandboxHandle, *, files: Mapping[str, bytes]) -> None:
        for path, content in files.items():
            self.write_file(sandbox, path=path, content=content)

    def read_file(self, sandbox: SandboxHandle, *, path: str) -> bytes:
        p = _docker(
            ["exec", sandbox.id, "sh", "-c", f"base64 -w0 {shlex.quote(path)}"],
            check=True,
        )
        return base64.b64decode(p.stdout.strip())

    def delete_file(self, sandbox: SandboxHandle, *, path: str) -> None:
        _docker(["exec", sandbox.id, "rm", "-f", "--", path], check=False)

    def upload_tree(
        self, sandbox: SandboxHandle, *, local_dir: Path, remote_dir: str
    ) -> None:
        local_dir = Path(local_dir).resolve()
        import io
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(local_dir, arcname=".")
        data = buf.getvalue()
        # Extract stdin with exec/tar because docker cp from stdin requires a directory target.
        script = f"mkdir -p {shlex.quote(remote_dir)} && tar xzf - -C {shlex.quote(remote_dir)}"
        _docker(
            ["exec", "-i", sandbox.id, "sh", "-c", script],
            input_bytes=data,
            check=True,
        )

    # ---------------------------------------------------------------- snapshot / clone
    def snapshot_exists(self, *, name: str) -> bool:
        return _docker(["image", "inspect", name]).returncode == 0

    def create_snapshot(self, sandbox: SandboxHandle, *, name: str, timeout: int = 600) -> None:
        _docker(["commit", sandbox.id, name], timeout=timeout, check=True)

    def clone(
        self,
        sandbox: SandboxHandle,
        *,
        url: str,
        path: str,
        revision: str | None = None,
    ) -> None:
        cmd = f"git clone {shlex.quote(url)} {shlex.quote(path)}"
        if revision:
            cmd += f" && git -C {shlex.quote(path)} checkout {shlex.quote(revision)}"
        _docker(["exec", sandbox.id, "sh", "-c", cmd], check=True)

    # ---------------------------------------------------------------- network / delete
    def set_network_blocked(self, sandbox: SandboxHandle, *, blocked: bool) -> None:
        action = "disconnect" if blocked else "connect"
        _docker(["network", action, "bridge", sandbox.id], check=False)

    def delete(self, sandbox: SandboxHandle) -> None:
        _docker(["rm", "-f", sandbox.id], check=False)
