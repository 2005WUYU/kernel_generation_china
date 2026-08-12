"""Permission contracts for controller/worker shared durable state."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from pathlib import Path

SHARED_DIRECTORY_MODE = 0o2770
SHARED_FILE_MODE = 0o660


def validate_shared_directory_mode(mode: int) -> int:
    """Require owner/group collaboration, setgid inheritance, and no other access."""

    if mode != SHARED_DIRECTORY_MODE:
        raise ValueError("shared directory mode must be 02770")
    return mode


def validate_shared_file_mode(mode: int) -> int:
    """Require owner/group read-write access and no execute/other access."""

    if mode != SHARED_FILE_MODE:
        raise ValueError("shared file mode must be 0660")
    return mode


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def _chmod_if_needed(path: Path, mode: int) -> None:
    if _mode(path) != mode:
        try:
            os.chmod(path, mode, follow_symlinks=False)
        except PermissionError as exc:
            raise PermissionError(
                f"cannot repair shared state permissions for {path}; "
                "stop services and run the documented root migration"
            ) from exc


def ensure_shared_directory(
    path: Path,
    *,
    mode: int = SHARED_DIRECTORY_MODE,
) -> Path:
    """Create or repair one trusted shared directory without accepting symlinks."""

    validate_shared_directory_mode(mode)
    missing: list[Path] = []
    current_path = path
    while not os.path.lexists(current_path):
        missing.append(current_path)
        parent = current_path.parent
        if parent == current_path:
            raise ValueError(f"cannot locate an existing parent for shared state: {path}")
        current_path = parent
    current = current_path.lstat()
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise ValueError(f"shared state path must be below a real directory: {current_path}")
    for candidate in reversed(missing):
        with suppress(FileExistsError):
            candidate.mkdir(mode=mode)
        created = candidate.lstat()
        if not stat.S_ISDIR(created.st_mode) or stat.S_ISLNK(created.st_mode):
            raise ValueError(
                f"shared state path changed to an unsafe component: {candidate}"
            )
        _chmod_if_needed(candidate, mode)
    current = path.lstat()
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise ValueError(f"shared state path must be a real directory: {path}")
    _chmod_if_needed(path, mode)
    return path


def ensure_shared_regular_file(
    path: Path,
    *,
    mode: int = SHARED_FILE_MODE,
    create: bool = False,
) -> bool:
    """Create or repair a shared regular file.

    Returns ``False`` when the file is absent and ``create`` is false. Creation
    uses ``O_EXCL`` so an existing path is never truncated.
    """

    validate_shared_file_mode(mode)
    if not os.path.lexists(path) and create:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, mode)
        except FileExistsError:
            pass
        else:
            try:
                os.fchmod(descriptor, mode)
            finally:
                os.close(descriptor)

    if not os.path.lexists(path):
        return False
    current = path.lstat()
    if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise ValueError(f"shared state path must be a regular file: {path}")
    _chmod_if_needed(path, mode)
    return True
