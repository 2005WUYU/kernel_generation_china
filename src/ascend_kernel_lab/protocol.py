"""Cross-process release identity for controller/worker protocol fencing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ascend_kernel_lab import __version__

EVALUATION_JOB_PROTOCOL = "ascend_eval_job_v1"


def _installed_python_files(root: Path) -> tuple[tuple[str, Path], ...]:
    values: list[tuple[str, Path]] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("installed harness Python files must be regular non-symlinks")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise RuntimeError("installed harness Python file escapes its package root")
        values.append((relative, path))
    if not values:
        raise RuntimeError("installed harness package contains no Python files")
    return tuple(sorted(values))


def harness_digest() -> str:
    """Hash executable protocol code so mixed releases fail before execution."""

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(
        f"ascend-harness-v2\0{__version__}\0{EVALUATION_JOB_PROTOCOL}\0".encode()
    )
    for relative, path in _installed_python_files(root):
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = ["EVALUATION_JOB_PROTOCOL", "harness_digest"]
