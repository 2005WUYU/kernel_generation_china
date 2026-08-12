"""Resource-bounded subprocess execution for untrusted candidate stages."""

from __future__ import annotations

import contextlib
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_SAFE_ENV_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "TZ",
    }
)
_SAFE_ENV_PREFIXES = (
    "ASCEND_",
    "CANN_",
    "DEVICE_ID",
    "HCCL_",
    "NPU_",
    "RANK_",
    "SOC_VERSION",
    "TBE_",
    "TRITON_",
)
_SENSITIVE_PARTS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "PROXY",
    "SECRET",
    "TOKEN",
)


def _safe_environment_name(name: str) -> bool:
    upper = name.upper()
    if any(part in upper for part in _SENSITIVE_PARTS):
        return False
    return name in _SAFE_ENV_NAMES or any(name.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES)


def clean_environment(
    source: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Retain runtime paths while dropping credentials and all proxy variables."""

    clean: dict[str, str] = {}
    for name, value in (source or os.environ).items():
        if _safe_environment_name(name) and "\x00" not in name and "\x00" not in value:
            clean[name] = value
    for name, value in (extra or {}).items():
        if not _safe_environment_name(name):
            raise ValueError(f"unsafe environment variable requested for stage: {name}")
        if "\x00" in name or "\x00" in value:
            raise ValueError("environment names and values must not contain NUL")
        clean[name] = value
    clean.setdefault("PATH", os.defpath)
    clean.setdefault("LANG", "C.UTF-8")
    return clean


@dataclass(frozen=True)
class ResourceLimits:
    """POSIX limits applied in the child immediately before exec."""

    address_space_bytes: int = 16 * 1024**3
    # Triton-Ascend/FlagTree can generate a precompiled C++ header well above
    # 256 MiB before launching even a small kernel.
    file_size_bytes: int = 2 * 1024**3
    open_files: int = 256
    # RLIMIT_NPROC counts every process owned by the Unix user on several
    # platforms. Keep this above a normal shared development account's baseline;
    # the AST policy and dedicated deployment user provide the tighter boundary.
    processes: int = 512
    core_size_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class StageProcessResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    timed_out: bool = False
    output_limit_exceeded: bool = False
    terminated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.returncode == 0
            and not self.timed_out
            and not self.output_limit_exceeded
        )

    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


class StageRunner:
    """Execute an argv directly, with a clean env and whole-process-group cleanup."""

    def __init__(
        self,
        *,
        maximum_output_bytes: int = 2 * 1024**2,
        maximum_stdin_bytes: int = 8 * 1024**2,
        termination_grace_seconds: float = 2.0,
        limits: ResourceLimits | None = None,
    ) -> None:
        if maximum_output_bytes < 1 or maximum_stdin_bytes < 1:
            raise ValueError("stage byte limits must be positive")
        if not math.isfinite(termination_grace_seconds) or termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be finite and positive")
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_stdin_bytes = maximum_stdin_bytes
        self.termination_grace_seconds = termination_grace_seconds
        self.limits = limits or ResourceLimits()
        self._process_lock = threading.Lock()
        self._current_process: subprocess.Popen[bytes] | None = None

    def cancel_current(self) -> bool:
        """Terminate the currently running process group, if any."""

        with self._process_lock:
            process = self._current_process
        if process is None:
            return False
        self._terminate(process)
        return True

    @staticmethod
    def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
        result = tuple(argv)
        if not result or any(not item or "\x00" in item for item in result):
            raise ValueError("argv must contain non-empty, NUL-free strings")
        return result

    def _launcher_argv(
        self, command: tuple[str, ...], timeout_seconds: float
    ) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            "ascend_kernel_lab.worker.launcher",
            "--address-space",
            str(self.limits.address_space_bytes),
            "--file-size",
            str(self.limits.file_size_bytes),
            "--open-files",
            str(self.limits.open_files),
            "--processes",
            str(self.limits.processes),
            "--core-size",
            str(self.limits.core_size_bytes),
            "--cpu-seconds",
            str(math.ceil(timeout_seconds) + 1),
            "--",
            *command,
        )

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            # Some managed macOS sandboxes reject signaling a group after its
            # leader exits. The direct child can still be reaped below; the
            # production worker additionally runs under a dedicated OS sandbox.
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.send_signal(sig)

    @staticmethod
    def _group_exists(process: subprocess.Popen[bytes]) -> bool:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        self._signal_group(process, signal.SIGTERM)
        deadline = time.monotonic() + self.termination_grace_seconds
        while self._group_exists(process) and time.monotonic() < deadline:
            if process.poll() is None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(0.02, max(0.0, deadline - time.monotonic())))
            else:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        if self._group_exists(process):
            self._signal_group(process, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=self.termination_grace_seconds)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        payload: bytes = b"",
        env: Mapping[str, str] | None = None,
        timeout_seconds: float,
    ) -> StageProcessResult:
        """Run one stage without a shell and return capped output bytes."""

        command = self._validate_argv(argv)
        workdir = Path(cwd).resolve()
        if not workdir.is_dir():
            raise ValueError("stage cwd must be an existing directory")
        if len(payload) > self.maximum_stdin_bytes:
            raise ValueError("stage stdin payload exceeds the configured limit")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")

        started = time.monotonic()
        stdin_file: BinaryIO
        with tempfile.TemporaryFile(mode="w+b", dir=workdir) as stdin_file:
            stdin_file.write(payload)
            stdin_file.seek(0)
            process = subprocess.Popen(
                self._launcher_argv(command, timeout_seconds),
                cwd=workdir,
                env=clean_environment(extra=env),
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
            with self._process_lock:
                self._current_process = process
            assert process.stdout is not None and process.stderr is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            buffers = {"stdout": bytearray(), "stderr": bytearray()}
            timed_out = False
            too_much_output = False
            terminated = False
            deadline = started + timeout_seconds

            total_output = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    terminated = True
                    self._terminate(process)
                    for registered in tuple(selector.get_map().values()):
                        stream = registered.fileobj
                        selector.unregister(stream)
                        if hasattr(stream, "close"):
                            stream.close()
                    break
                events = selector.select(timeout=max(0.0, min(0.1, remaining)))
                for key, _mask in events:
                    stream = key.fileobj
                    chunk = os.read(key.fd, 65_536)
                    if not chunk:
                        selector.unregister(stream)
                        if hasattr(stream, "close"):
                            stream.close()
                        continue
                    name = str(key.data)
                    buffer = buffers[name]
                    remaining_bytes = self.maximum_output_bytes - total_output
                    if remaining_bytes > 0:
                        kept = chunk[:remaining_bytes]
                        buffer.extend(kept)
                        total_output += len(kept)
                    if len(chunk) > remaining_bytes:
                        too_much_output = True
                        terminated = True
                        self._terminate(process)
                        for registered in tuple(selector.get_map().values()):
                            stream = registered.fileobj
                            selector.unregister(stream)
                            if hasattr(stream, "close"):
                                stream.close()
                        break

            selector.close()
            if process.poll() is None:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminated = True
                    self._terminate(process)
            returncode = process.returncode
            with self._process_lock:
                if self._current_process is process:
                    self._current_process = None

        return StageProcessResult(
            argv=command,
            returncode=returncode,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            output_limit_exceeded=too_much_output,
            terminated=terminated,
        )
