"""本地 Docker 沙盒后端：让 MAC-Bench 工程题在本机 Docker 容器里跑，替代 Modal 云。

把 ModalSandboxBackend 换成对本地 `docker` CLI 的调用。工程题的镜像就是标准
Docker Hub 镜像（如 python:3.11-slim），setup 是普通 shell 脚本，本地完全可执行。

用法：在 run_macbench.py 里 `open_task(rec, dir, backend=DockerSandboxBackend())` 注入。
依赖：本机安装 Docker 并启动 daemon（Docker Desktop 或 colima）。
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
    """跑一条 docker 命令，返回 CompletedProcess。"""
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
    """实现 MAC-Bench 的 SandboxBackend 接口，用本地 docker 容器执行。"""

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
        # 首次使用先确保本地有该镜像（snapshot 是本地 commit 的镜像，必然在；image 可能要 pull）
        if image and _docker(["image", "inspect", image]).returncode != 0:
            _docker(["pull", image], check=True)
        cid = "mb-" + uuid.uuid4().hex[:12]
        args = ["run", "-d", "--name", cid]
        if network_blocked:
            args += ["--network", "none"]
        args += [img, "sh", "-c", "exec sleep infinity"]
        _docker(args, check=True)
        # 只建最基础的 /work；子目录（/work/project、/work/grader 等）按需由 write_file/upload_tree
        # 的 mkdir -p 创建。切勿预建 /work/project：consumer 验证器依赖它不存在来证明环境干净。
        _docker(["exec", cid, "sh", "-c", "mkdir -p /work"], check=True)
        # 代理的 fake-ip 对 HTTP 明文（80）不稳定，对 HTTPS 稳定：把 apt 源统一转成 HTTPS。
        # setup 脚本里的 apt-get / pip 都会受益；HTTPS 有 SNI，代理能正确识别真实域名。
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
        # docker cp - 从 stdin 复制要求目标必须是目录，这里改用 exec + tar 从 stdin 解压
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
