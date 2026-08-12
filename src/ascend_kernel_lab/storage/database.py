"""SQLite connection policy, crash-safe migrations, and transaction helpers."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .permissions import (
    SHARED_DIRECTORY_MODE,
    SHARED_FILE_MODE,
    ensure_shared_directory,
    ensure_shared_regular_file,
    validate_shared_directory_mode,
    validate_shared_file_mode,
)


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="initial_metadata_and_queue_schema",
        statements=(
            """
            CREATE TABLE experiments (
                experiment_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1)
            )
            """,
            """
            CREATE TABLE tasks (
                experiment_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_version INTEGER NOT NULL DEFAULT 1 CHECK (task_version >= 1),
                state TEXT NOT NULL,
                current_round INTEGER NOT NULL DEFAULT 0 CHECK (current_round >= 0),
                best_candidate_id TEXT,
                task_spec_sha256 TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
                PRIMARY KEY (experiment_id, task_id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE rounds (
                experiment_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_number INTEGER NOT NULL CHECK (round_number >= 1),
                state TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
                PRIMARY KEY (experiment_id, task_id, round_number),
                FOREIGN KEY (experiment_id, task_id)
                    REFERENCES tasks(experiment_id, task_id)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_number INTEGER NOT NULL CHECK (round_number >= 1),
                source_sha256 TEXT NOT NULL,
                source_artifact_path TEXT NOT NULL,
                response_sha256 TEXT,
                created_at REAL NOT NULL,
                UNIQUE (experiment_id, task_id, round_number, candidate_id),
                FOREIGN KEY (experiment_id, task_id, round_number)
                    REFERENCES rounds(experiment_id, task_id, round_number)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE candidate_scores (
                candidate_id TEXT PRIMARY KEY,
                round_number INTEGER NOT NULL CHECK (round_number >= 1),
                compile_passed INTEGER NOT NULL CHECK (compile_passed IN (0, 1)),
                correctness_passed INTEGER NOT NULL CHECK (correctness_passed IN (0, 1)),
                anti_bypass_passed INTEGER NOT NULL CHECK (anti_bypass_passed IN (0, 1)),
                hidden_correctness_passed INTEGER
                    CHECK (hidden_correctness_passed IS NULL OR hidden_correctness_passed IN (0, 1)),
                minimum_speedup REAL,
                geomean_speedup REAL,
                candidate_kernel_coverage REAL,
                stability_cv REAL,
                reward_json TEXT,
                updated_at REAL NOT NULL,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                    ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                task_id TEXT,
                round_number INTEGER,
                relative_path TEXT NOT NULL UNIQUE,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                media_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                CHECK (round_number IS NULL OR round_number >= 1)
            )
            """,
            """
            CREATE TABLE evaluation_jobs (
                job_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                round_number INTEGER NOT NULL CHECK (round_number >= 1),
                candidate_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
                max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
                available_at REAL NOT NULL,
                lease_owner TEXT,
                lease_token TEXT,
                lease_expires_at REAL,
                heartbeat_at REAL,
                result_json TEXT,
                last_error_json TEXT,
                idempotency_key TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                CHECK (
                    (status = 'LEASED' AND lease_owner IS NOT NULL
                        AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL
                        AND heartbeat_at IS NOT NULL)
                    OR
                    (status <> 'LEASED' AND lease_owner IS NULL
                        AND lease_token IS NULL AND lease_expires_at IS NULL
                        AND heartbeat_at IS NULL)
                )
            )
            """,
            """
            CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                aggregate_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                experiment_id TEXT,
                task_id TEXT,
                round_number INTEGER,
                payload_json TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                CHECK (round_number IS NULL OR round_number >= 1)
            )
            """,
        ),
    ),
    Migration(
        version=2,
        name="queue_indexes_and_immutable_events",
        statements=(
            """
            CREATE UNIQUE INDEX evaluation_jobs_idempotency_key_uq
            ON evaluation_jobs(idempotency_key)
            WHERE idempotency_key IS NOT NULL
            """,
            """
            CREATE INDEX evaluation_jobs_claim_idx
            ON evaluation_jobs(status, available_at, priority DESC, created_at)
            """,
            """
            CREATE INDEX evaluation_jobs_lease_expiry_idx
            ON evaluation_jobs(lease_expires_at)
            WHERE status = 'LEASED'
            """,
            """
            CREATE INDEX events_aggregate_idx
            ON events(aggregate_type, aggregate_id, sequence)
            """,
            """
            CREATE INDEX events_experiment_task_idx
            ON events(experiment_id, task_id, sequence)
            """,
            """
            CREATE TRIGGER events_are_immutable_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are immutable');
            END
            """,
            """
            CREATE TRIGGER events_are_immutable_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are immutable');
            END
            """,
        ),
    ),
)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return value
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("JSON datetime values must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            converted[key] = _json_compatible(item)
        return converted
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_dumps(value: Any) -> str:
    """Serialize deterministic strict JSON suitable for hashes and equality."""

    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_loads_object(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("stored JSON value is not an object")
    return parsed


def datetime_to_timestamp(value: datetime) -> float:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.timestamp()


def timestamp_to_datetime(value: float) -> datetime:
    return datetime.fromtimestamp(value, timezone.utc)


class SQLiteDatabase:
    """Open consistently configured SQLite connections and apply migrations."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_seconds: float = 30.0,
        synchronous: str = "NORMAL",
        directory_mode: int = SHARED_DIRECTORY_MODE,
        file_mode: int = SHARED_FILE_MODE,
    ) -> None:
        if not math.isfinite(busy_timeout_seconds) or busy_timeout_seconds <= 0:
            raise ValueError("busy_timeout_seconds must be finite and positive")
        synchronous = synchronous.upper()
        if synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError("unsupported SQLite synchronous mode")
        self._busy_timeout_ms = max(1, round(busy_timeout_seconds * 1000))
        self._synchronous = synchronous
        self._directory_mode = validate_shared_directory_mode(directory_mode)
        self._file_mode = validate_shared_file_mode(file_mode)
        self._migration_lock = threading.Lock()
        self._anchor: sqlite3.Connection | None = None

        raw_path = str(path)
        if raw_path == ":memory:":
            self.path: Path | None = None
            self._target = f"file:ascend-kernel-lab-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._uri = True
            self._anchor = self._open_connection()
        else:
            requested_path = Path(path).expanduser()
            if os.path.lexists(requested_path):
                current = requested_path.lstat()
                if not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode):
                    raise ValueError(
                        f"shared state path must be a regular file: {requested_path}"
                    )
            ensure_shared_directory(
                requested_path.parent,
                mode=self._directory_mode,
            )
            self.path = requested_path.resolve()
            ensure_shared_directory(
                self.path.parent,
                mode=self._directory_mode,
            )
            ensure_shared_regular_file(
                self.path,
                mode=self._file_mode,
                create=True,
            )
            self._target = str(self.path)
            self._uri = False
        self.initialize()

    @property
    def latest_schema_version(self) -> int:
        return MIGRATIONS[-1].version if MIGRATIONS else 0

    def _repair_file_modes(self) -> None:
        if self.path is None:
            return
        ensure_shared_directory(self.path.parent, mode=self._directory_mode)
        ensure_shared_regular_file(self.path, mode=self._file_mode, create=True)
        for suffix in ("-wal", "-shm"):
            ensure_shared_regular_file(
                Path(f"{self.path}{suffix}"),
                mode=self._file_mode,
            )

    def _open_connection(self) -> sqlite3.Connection:
        self._repair_file_modes()
        connection = sqlite3.connect(
            self._target,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
            uri=self._uri,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute(f"PRAGMA synchronous = {self._synchronous}")
            connection.execute("PRAGMA journal_mode = WAL")
            self._repair_file_modes()
            return connection
        except BaseException:
            connection.close()
            raise

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
        finally:
            try:
                self._repair_file_modes()
            finally:
                connection.close()
                self._repair_file_modes()

    @contextmanager
    def transaction(
        self,
        connection: sqlite3.Connection,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """Run a transaction, using savepoints when already nested."""

        if connection.in_transaction:
            savepoint = f"sp_{uuid.uuid4().hex}"
            connection.execute(f"SAVEPOINT {savepoint}")
            try:
                yield connection
            except BaseException:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            else:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            return

        connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield connection
        except BaseException:
            connection.rollback()
            self._repair_file_modes()
            raise
        else:
            connection.commit()
            self._repair_file_modes()

    def initialize(self) -> None:
        """Apply every pending migration exactly once."""

        with self._migration_lock, self.connection() as connection:
            with self.transaction(connection, immediate=True):
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at REAL NOT NULL
                    )
                    """
                )
            rows = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            applied = {int(row["version"]) for row in rows}
            unknown = applied.difference(migration.version for migration in MIGRATIONS)
            if unknown:
                versions = ", ".join(str(version) for version in sorted(unknown))
                raise RuntimeError(f"database schema is newer than this code: {versions}")

            expected_previous = 0
            for migration in MIGRATIONS:
                if migration.version != expected_previous + 1:
                    raise RuntimeError("migration versions must be contiguous")
                expected_previous = migration.version
                if migration.version in applied:
                    continue
                with self.transaction(connection, immediate=True):
                    for statement in migration.statements:
                        connection.execute(statement)
                    connection.execute(
                        """
                        INSERT INTO schema_migrations(version, name, applied_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            migration.version,
                            migration.name,
                            datetime.now(timezone.utc).timestamp(),
                        ),
                    )
                    connection.execute(f"PRAGMA user_version = {migration.version}")

    def close(self) -> None:
        """Close the anchor used by shared in-memory databases."""

        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def __enter__(self) -> SQLiteDatabase:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
