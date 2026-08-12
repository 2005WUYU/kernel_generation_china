from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfileRun:
    argv: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    output_root: str


class MsprofRunner:
    """Runs one profiler process without involving a shell."""

    def __init__(self, executable: str = "msprof") -> None:
        self.executable = executable

    def available(self) -> bool:
        return shutil.which(self.executable) is not None

    def build_argv(
        self,
        *,
        output_root: Path,
        python_executable: str,
        script: Path,
        kernel_name: str | None,
        extra_args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        argv = [self.executable, "op"]
        if kernel_name:
            argv.append(f"--kernel-name={kernel_name}")
        argv.append(f"--output={output_root}")
        argv.extend(extra_args)
        argv.extend((python_executable, str(script)))
        return tuple(argv)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
        output_root: Path,
    ) -> ProfileRun:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        output_root.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        timed_out = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait(timeout=10)
        return ProfileRun(
            argv=tuple(argv),
            returncode=returncode,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - start,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            output_root=str(output_root),
        )
