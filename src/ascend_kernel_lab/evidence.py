"""Integrity verification for worker-published public execution evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_MANIFEST_NAME = re.compile(r"^artifact_manifest\.([0-9a-f]{64})\.json$")
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_FILES = 50_000
_MAX_TOTAL_BYTES = 32 * 1024**3


class EvidenceIntegrityError(ValueError):
    """Published stage evidence is missing, unsafe, or no longer immutable."""


@dataclass(frozen=True)
class EvidenceSummary:
    manifest_path: Path
    manifest_sha256: str
    file_count: int
    total_bytes: int


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            if size > _MAX_TOTAL_BYTES:
                raise EvidenceIntegrityError("one evidence file exceeds the size limit")
            digest.update(block)
    return digest.hexdigest(), size


def _safe_descendant(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise EvidenceIntegrityError("evidence manifest references a missing file") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceIntegrityError("evidence path contains a symlink")
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvidenceIntegrityError("evidence path escapes its published directory") from exc
    return current


def verify_stage_artifact_manifest(
    manifest_path: Path | str,
    *,
    artifact_root: Path | str,
) -> EvidenceSummary:
    """Verify the content-addressed manifest and every file it exclusively owns."""

    lexical_root = Path(os.path.abspath(Path(artifact_root).expanduser()))
    root = lexical_root.resolve(strict=True)
    requested = Path(manifest_path).expanduser()
    absolute = requested if requested.is_absolute() else lexical_root / requested
    try:
        relative_manifest = absolute.relative_to(lexical_root)
    except ValueError as exc:
        # Accept platform aliases such as macOS /var -> /private/var while the
        # component-by-component walk below still rejects user-controlled links.
        try:
            if absolute.is_symlink():
                raise EvidenceIntegrityError("evidence manifest must not be a symlink")
            relative_manifest = absolute.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            raise EvidenceIntegrityError("evidence manifest escapes artifact_root") from exc
    manifest = _safe_descendant(root, PurePosixPath(relative_manifest.as_posix()))
    match = _MANIFEST_NAME.fullmatch(manifest.name)
    if match is None:
        raise EvidenceIntegrityError("evidence manifest is not content-addressed")
    metadata = manifest.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_MANIFEST_BYTES:
        raise EvidenceIntegrityError("evidence manifest is not a bounded regular file")
    payload = manifest.read_bytes()
    manifest_digest = hashlib.sha256(payload).hexdigest()
    if manifest_digest != match.group(1):
        raise EvidenceIntegrityError("evidence manifest digest disagrees with its filename")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceIntegrityError("evidence manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "files"}:
        raise EvidenceIntegrityError("evidence manifest envelope is invalid")
    if value["schema_version"] != "ascend_stage_artifact_manifest_v1":
        raise EvidenceIntegrityError("evidence manifest schema is unsupported")
    files = value["files"]
    if (
        not isinstance(files, Sequence)
        or isinstance(files, (str, bytes, bytearray))
        or not files
        or len(files) > _MAX_FILES
    ):
        raise EvidenceIntegrityError("evidence manifest file list is invalid")

    published_root = manifest.parent
    declared: set[str] = set()
    total_bytes = 0
    for raw in files:
        if not isinstance(raw, Mapping) or set(raw) != {
            "relative_path",
            "sha256",
            "size_bytes",
            "type",
        }:
            raise EvidenceIntegrityError("evidence manifest entry is invalid")
        relative_raw = raw["relative_path"]
        if not isinstance(relative_raw, str) or not relative_raw or "\\" in relative_raw:
            raise EvidenceIntegrityError("evidence relative path is invalid")
        relative = PurePosixPath(relative_raw)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise EvidenceIntegrityError("evidence relative path is not normalized")
        if relative_raw in declared:
            raise EvidenceIntegrityError("evidence manifest contains a duplicate path")
        declared.add(relative_raw)
        expected_digest = raw["sha256"]
        expected_size = raw["size_bytes"]
        media_type = raw["type"]
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(media_type, str)
            or not media_type.strip()
        ):
            raise EvidenceIntegrityError("evidence manifest metadata is invalid")
        path = _safe_descendant(published_root, relative)
        file_metadata = path.lstat()
        if not stat.S_ISREG(file_metadata.st_mode):
            raise EvidenceIntegrityError("evidence entry is not a regular file")
        digest, size = _sha256(path)
        if digest != expected_digest or size != expected_size:
            raise EvidenceIntegrityError("published evidence hash or size changed")
        total_bytes += size
        if total_bytes > _MAX_TOTAL_BYTES:
            raise EvidenceIntegrityError("published evidence exceeds the total size limit")

    actual: set[str] = set()
    for path in published_root.rglob("*"):
        entry_metadata = path.lstat()
        if stat.S_ISLNK(entry_metadata.st_mode):
            raise EvidenceIntegrityError("published evidence contains a symlink")
        if stat.S_ISDIR(entry_metadata.st_mode):
            continue
        if not stat.S_ISREG(entry_metadata.st_mode):
            raise EvidenceIntegrityError("published evidence contains a special file")
        if path != manifest:
            actual.add(path.relative_to(published_root).as_posix())
        if len(actual) > _MAX_FILES:
            raise EvidenceIntegrityError("published evidence contains too many files")
    if actual != declared:
        raise EvidenceIntegrityError("published evidence files disagree with the manifest")
    return EvidenceSummary(manifest, manifest_digest, len(files), total_bytes)


def validate_artifact_map(
    artifacts: Mapping[str, Any],
    *,
    artifact_root: Path | str,
) -> EvidenceSummary | None:
    """Validate one StageResult artifact map and its optional evidence manifest."""

    raw_manifest = artifacts.get("artifact_manifest")
    if raw_manifest is None:
        return None
    if not isinstance(raw_manifest, str):
        raise EvidenceIntegrityError("artifact_manifest path must be a string")
    summary = verify_stage_artifact_manifest(raw_manifest, artifact_root=artifact_root)
    published_root = summary.manifest_path.parent
    for name, raw in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw, str):
            raise EvidenceIntegrityError("artifact map must contain string pairs")
        if name == "artifact_manifest":
            continue
        candidate = Path(raw)
        absolute = candidate if candidate.is_absolute() else Path(artifact_root) / candidate
        try:
            absolute.resolve(strict=True).relative_to(published_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise EvidenceIntegrityError("artifact map path is outside its manifest tree") from exc
    return summary


__all__ = [
    "EvidenceIntegrityError",
    "EvidenceSummary",
    "validate_artifact_map",
    "verify_stage_artifact_manifest",
]
