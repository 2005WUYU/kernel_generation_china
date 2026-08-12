"""Filesystem artifact store with atomic visibility and SHA-256 verification."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from ascend_kernel_lab.domain import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactMetadata,
)

from .database import canonical_json_dumps
from .permissions import (
    SHARED_DIRECTORY_MODE,
    SHARED_FILE_MODE,
    ensure_shared_directory,
    ensure_shared_regular_file,
    validate_shared_directory_mode,
    validate_shared_file_mode,
)


class AtomicArtifactStore:
    """Commit artifacts atomically within a confined root directory.

    Writers flush and fsync a sibling temporary file first. Immutable writes use
    ``link`` for atomic create-if-absent semantics; explicit replacements use
    ``os.replace``. A formal path is therefore never partially visible.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_artifact_bytes: int | None = 2 * 1024 * 1024 * 1024,
        file_mode: int = SHARED_FILE_MODE,
        directory_mode: int = SHARED_DIRECTORY_MODE,
    ) -> None:
        if max_artifact_bytes is not None and max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive or None")
        file_mode = validate_shared_file_mode(file_mode)
        directory_mode = validate_shared_directory_mode(directory_mode)
        requested_root = Path(root).expanduser()
        if requested_root.exists() and (
            requested_root.is_symlink() or not requested_root.is_dir()
        ):
            raise ValueError("artifact root must be a real directory, not a symlink")
        ensure_shared_directory(requested_root, mode=directory_mode)
        self.root = requested_root.resolve()
        self.max_artifact_bytes = max_artifact_bytes
        self.file_mode = file_mode
        self.directory_mode = directory_mode

    @staticmethod
    def _normalize(relative_path: str | PurePosixPath) -> str:
        raw = str(relative_path)
        if not raw or "\x00" in raw or "\\" in raw:
            raise ValueError("artifact path must be a non-empty POSIX relative path")
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact path must remain below the artifact root")
        normalized = path.as_posix()
        if normalized != raw:
            raise ValueError("artifact path must already be normalized")
        return normalized

    def path_for(self, relative_path: str | PurePosixPath) -> Path:
        """Resolve a safe artifact path without creating the artifact."""

        normalized = self._normalize(relative_path)
        target = self.root.joinpath(*PurePosixPath(normalized).parts)
        current = self.root
        for part in PurePosixPath(normalized).parent.parts:
            if part == ".":
                continue
            current /= part
            if os.path.lexists(current) and (
                current.is_symlink() or not current.is_dir()
            ):
                raise ValueError(
                    "artifact parent contains a symlink or non-directory component"
                )
            try:
                ensure_shared_directory(current, mode=self.directory_mode)
            except ValueError as exc:
                raise ValueError(
                    "artifact parent changed to an unsafe component"
                ) from exc
            try:
                current.resolve().relative_to(self.root)
            except ValueError as exc:
                raise ValueError("artifact parent escapes the artifact root") from exc
        return target

    @staticmethod
    def _hash_file(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        try:
            descriptor = os.open(directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _metadata_for(
        target: Path,
        relative_path: str,
        sha256: str,
        size_bytes: int,
        media_type: str,
    ) -> ArtifactMetadata:
        created_at = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc)
        return ArtifactMetadata(
            relative_path=relative_path,
            sha256=sha256,
            size_bytes=size_bytes,
            media_type=media_type,
            created_at=created_at,
        )

    def put_stream(
        self,
        relative_path: str | PurePosixPath,
        stream: BinaryIO,
        *,
        media_type: str = "application/octet-stream",
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> ArtifactMetadata:
        """Atomically persist bytes read from a binary stream."""

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not media_type or not media_type.strip():
            raise ValueError("media_type must be non-empty")
        normalized = self._normalize(relative_path)
        target = self.path_for(normalized)
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise TypeError("artifact stream must return bytes")
                    chunk_bytes = bytes(chunk)
                    size += len(chunk_bytes)
                    if (
                        self.max_artifact_bytes is not None
                        and size > self.max_artifact_bytes
                    ):
                        raise ValueError("artifact exceeds max_artifact_bytes")
                    digest.update(chunk_bytes)
                    temporary.write(chunk_bytes)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_path, self.file_mode)
            sha256 = digest.hexdigest()

            if overwrite:
                if target.exists() and target.is_dir():
                    raise ArtifactConflictError("artifact target is a directory")
                os.replace(temporary_path, target)
                temporary_path = None
            else:
                try:
                    os.link(temporary_path, target)
                except FileExistsError:
                    if target.is_symlink() or not target.is_file():
                        raise ArtifactConflictError(
                            "artifact target is not a regular file"
                        ) from None
                    existing_sha, existing_size = self._hash_file(target)
                    if existing_sha != sha256 or existing_size != size:
                        raise ArtifactConflictError(
                            f"artifact {normalized!r} already has different contents"
                        ) from None
                    ensure_shared_regular_file(target, mode=self.file_mode)
                else:
                    self._fsync_directory(target.parent)

            ensure_shared_regular_file(target, mode=self.file_mode)
            self._fsync_directory(target.parent)
            return self._metadata_for(
                target, normalized, sha256, size, media_type
            )
        finally:
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()

    def put_bytes(
        self,
        relative_path: str | PurePosixPath,
        data: bytes | bytearray | memoryview,
        *,
        media_type: str = "application/octet-stream",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self.put_stream(
            relative_path,
            io.BytesIO(bytes(data)),
            media_type=media_type,
            overwrite=overwrite,
        )

    def put_text(
        self,
        relative_path: str | PurePosixPath,
        text: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        return self.put_bytes(
            relative_path,
            text.encode("utf-8"),
            media_type=media_type,
            overwrite=overwrite,
        )

    def put_json(
        self,
        relative_path: str | PurePosixPath,
        value: Any,
        *,
        overwrite: bool = False,
    ) -> ArtifactMetadata:
        """Commit canonical strict JSON so its digest is reproducible."""

        return self.put_text(
            relative_path,
            canonical_json_dumps(value) + "\n",
            media_type="application/json",
            overwrite=overwrite,
        )

    def put_content_addressed(
        self,
        data: bytes | bytearray | memoryview,
        *,
        suffix: str = "",
        media_type: str = "application/octet-stream",
    ) -> ArtifactMetadata:
        """Store immutable content below ``sha256/<prefix>/<digest>``."""

        if suffix and ("/" in suffix or "\\" in suffix or not suffix.startswith(".")):
            raise ValueError("suffix must be empty or a filename extension")
        payload = bytes(data)
        digest = hashlib.sha256(payload).hexdigest()
        relative_path = f"sha256/{digest[:2]}/{digest}{suffix}"
        metadata = self.put_bytes(
            relative_path,
            payload,
            media_type=media_type,
            overwrite=False,
        )
        if metadata.sha256 != digest:
            raise ArtifactIntegrityError("content-addressed path digest mismatch")
        return metadata

    def verify(self, metadata: ArtifactMetadata) -> bool:
        """Verify path type, byte length, and digest; raise on corruption."""

        target = self.path_for(metadata.relative_path)
        if target.is_symlink() or not target.is_file():
            raise ArtifactIntegrityError("artifact is missing or not a regular file")
        sha256, size = self._hash_file(target)
        if sha256 != metadata.sha256 or size != metadata.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact integrity check failed for {metadata.relative_path!r}"
            )
        return True

    def read_bytes(
        self,
        metadata: ArtifactMetadata,
        *,
        verify: bool = True,
    ) -> bytes:
        if verify:
            self.verify(metadata)
        return self.path_for(metadata.relative_path).read_bytes()
