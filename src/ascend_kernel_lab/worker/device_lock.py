"""Cross-process exclusive leases for physical Ascend devices."""

from __future__ import annotations

import errno
import fcntl
import json
import math
import os
import re
import time
from pathlib import Path
from types import TracebackType


class DeviceLockTimeout(TimeoutError):
    """Raised when another process keeps the requested device busy."""


class DeviceLock:
    """An advisory flock used as the final single-device concurrency fence."""

    def __init__(
        self,
        device: str,
        *,
        lock_root: Path | str = "/tmp/ascend-kernel-lab-locks",
        timeout_seconds: float = 600.0,
        poll_seconds: float = 0.05,
    ) -> None:
        if not device.strip() or "\x00" in device:
            raise ValueError("device must be non-empty and NUL-free")
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be finite and non-negative")
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be finite and positive")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", device)
        self.device = device
        self.lock_root = Path(lock_root)
        self.path = self.lock_root / f"{safe_name}.lock"
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._fd: int | None = None

    @property
    def acquired(self) -> bool:
        return self._fd is not None

    def acquire(self) -> DeviceLock:
        if self._fd is not None:
            raise RuntimeError("device lock is already acquired")
        self.lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o600)
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    if time.monotonic() >= deadline:
                        raise DeviceLockTimeout(
                            f"timed out waiting for exclusive lock on {self.device}"
                        ) from exc
                    time.sleep(min(self.poll_seconds, max(0.0, deadline - time.monotonic())))
            owner = json.dumps(
                {"device": self.device, "pid": os.getpid(), "acquired_at": time.time()},
                sort_keys=True,
            ).encode("utf-8")
            os.ftruncate(fd, 0)
            os.write(fd, owner)
            os.fsync(fd)
            self._fd = fd
            return self
        except BaseException:
            os.close(fd)
            raise

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> DeviceLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()
