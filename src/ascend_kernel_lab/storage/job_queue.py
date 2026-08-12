"""Crash-resilient SQLite evaluation queue with expiring worker leases."""

from __future__ import annotations

import math
import secrets
import sqlite3
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from ascend_kernel_lab.domain import (
    EvaluationJob,
    EvaluationStage,
    IdempotencyConflictError,
    JobStatus,
    LeasedEvaluationJob,
    LeaseLostError,
    StoredEvaluationJob,
    utc_now,
)

from .database import (
    SQLiteDatabase,
    canonical_json_dumps,
    datetime_to_timestamp,
    json_loads_object,
    timestamp_to_datetime,
)
from .event_log import SQLiteEventLog


def _validate_duration(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero")


def _timestamp_or_none(value: float | None) -> datetime | None:
    return timestamp_to_datetime(float(value)) if value is not None else None


class SQLiteEvaluationJobQueue:
    """Coordinate workers without relying on process-local queue state."""

    def __init__(
        self,
        database: SQLiteDatabase,
        event_log: SQLiteEventLog | None = None,
    ) -> None:
        self.database = database
        self.event_log = event_log or SQLiteEventLog(database)

    @staticmethod
    def _stored_from_row(row: sqlite3.Row) -> StoredEvaluationJob:
        payload = json_loads_object(row["payload_json"])
        assert payload is not None
        values: dict[str, Any] = {
            "job_id": str(row["job_id"]),
            "experiment_id": str(row["experiment_id"]),
            "task_id": str(row["task_id"]),
            "round_number": int(row["round_number"]),
            "candidate_id": str(row["candidate_id"]),
            "stage": EvaluationStage(str(row["stage"])),
            "payload": payload,
            "priority": int(row["priority"]),
            "status": JobStatus(str(row["status"])),
            "attempt_count": int(row["attempt_count"]),
            "max_attempts": int(row["max_attempts"]),
            "available_at": timestamp_to_datetime(float(row["available_at"])),
            "lease_owner": row["lease_owner"],
            "lease_token": row["lease_token"],
            "lease_expires_at": _timestamp_or_none(row["lease_expires_at"]),
            "heartbeat_at": _timestamp_or_none(row["heartbeat_at"]),
            "result": json_loads_object(row["result_json"]),
            "last_error": json_loads_object(row["last_error_json"]),
            "idempotency_key": row["idempotency_key"],
            "created_at": timestamp_to_datetime(float(row["created_at"])),
            "updated_at": timestamp_to_datetime(float(row["updated_at"])),
        }
        if values["status"] is JobStatus.LEASED:
            return LeasedEvaluationJob(**values)
        return StoredEvaluationJob(**values)

    @staticmethod
    def _select_job(
        connection: sqlite3.Connection, job_id: str
    ) -> sqlite3.Row | None:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM evaluation_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return row

    def get(self, job_id: str) -> StoredEvaluationJob | None:
        with self.database.connection() as connection:
            row = self._select_job(connection, job_id)
        return self._stored_from_row(row) if row is not None else None

    @staticmethod
    def _same_enqueue_request(row: sqlite3.Row, job: EvaluationJob) -> bool:
        return (
            str(row["experiment_id"]) == job.experiment_id
            and str(row["task_id"]) == job.task_id
            and int(row["round_number"]) == job.round_number
            and str(row["candidate_id"]) == job.candidate_id
            and str(row["stage"]) == job.stage.value
            and str(row["payload_json"]) == canonical_json_dumps(job.payload)
            and int(row["priority"]) == job.priority
            and int(row["max_attempts"]) == job.max_attempts
            and float(row["available_at"])
            == datetime_to_timestamp(job.available_at)
            and row["idempotency_key"] == job.idempotency_key
        )

    def enqueue(self, job: EvaluationJob) -> StoredEvaluationJob:
        """Persist a job, safely replaying identical idempotent requests."""

        now = utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = self._select_job(connection, job.job_id)
                if existing is None and job.idempotency_key is not None:
                    existing = connection.execute(
                        """
                        SELECT * FROM evaluation_jobs WHERE idempotency_key = ?
                        """,
                        (job.idempotency_key,),
                    ).fetchone()
                if existing is not None:
                    if not self._same_enqueue_request(existing, job):
                        raise IdempotencyConflictError(
                            "job ID or idempotency key is already used by a "
                            "different evaluation request"
                        )
                    return self._stored_from_row(existing)

                connection.execute(
                    """
                    INSERT INTO evaluation_jobs(
                        job_id, experiment_id, task_id, round_number,
                        candidate_id, stage, payload_json, priority, status,
                        attempt_count, max_attempts, available_at,
                        idempotency_key, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.experiment_id,
                        job.task_id,
                        job.round_number,
                        job.candidate_id,
                        job.stage.value,
                        canonical_json_dumps(job.payload),
                        job.priority,
                        JobStatus.QUEUED.value,
                        job.max_attempts,
                        datetime_to_timestamp(job.available_at),
                        job.idempotency_key,
                        datetime_to_timestamp(now),
                        datetime_to_timestamp(now),
                    ),
                )
                self.event_log._append_in_transaction(
                    connection,
                    event_type="EVALUATION_JOB_ENQUEUED",
                    aggregate_type="evaluation_job",
                    aggregate_id=job.job_id,
                    experiment_id=job.experiment_id,
                    task_id=job.task_id,
                    round_number=job.round_number,
                    payload={"stage": job.stage.value, "priority": job.priority},
                )
                row = self._select_job(connection, job.job_id)
                assert row is not None
                return self._stored_from_row(row)

    def _sweep_expired_in_transaction(
        self,
        connection: sqlite3.Connection,
        now: datetime,
    ) -> list[StoredEvaluationJob]:
        now_timestamp = datetime_to_timestamp(now)
        expired_rows = connection.execute(
            """
            SELECT * FROM evaluation_jobs
            WHERE status = ? AND lease_expires_at <= ?
            ORDER BY lease_expires_at, job_id
            """,
            (JobStatus.LEASED.value, now_timestamp),
        ).fetchall()
        if not expired_rows:
            return []

        error_json = canonical_json_dumps(
            {
                "reason": "lease_expired",
                "detected_at": now,
            }
        )
        expired: list[StoredEvaluationJob] = []
        for old_row in expired_rows:
            terminal = int(old_row["attempt_count"]) >= int(old_row["max_attempts"])
            new_status = JobStatus.DEAD if terminal else JobStatus.RETRY_WAIT
            connection.execute(
                """
                UPDATE evaluation_jobs
                SET status = ?, available_at = ?, lease_owner = NULL,
                    lease_token = NULL, lease_expires_at = NULL,
                    heartbeat_at = NULL, last_error_json = ?, updated_at = ?
                WHERE job_id = ? AND status = ? AND lease_expires_at <= ?
                """,
                (
                    new_status.value,
                    now_timestamp,
                    error_json,
                    now_timestamp,
                    old_row["job_id"],
                    JobStatus.LEASED.value,
                    now_timestamp,
                ),
            )
            current = self._select_job(connection, str(old_row["job_id"]))
            assert current is not None
            stored = self._stored_from_row(current)
            expired.append(stored)
            self.event_log._append_in_transaction(
                connection,
                event_type="EVALUATION_JOB_LEASE_EXPIRED",
                aggregate_type="evaluation_job",
                aggregate_id=stored.job_id,
                experiment_id=stored.experiment_id,
                task_id=stored.task_id,
                round_number=stored.round_number,
                payload={"status": stored.status.value},
            )
        return expired

    def sweep_expired_leases(
        self, *, now: datetime | None = None
    ) -> list[StoredEvaluationJob]:
        """Requeue abandoned work, or dead-letter it after the last attempt."""

        now = now or utc_now()
        datetime_to_timestamp(now)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                return self._sweep_expired_in_transaction(connection, now)

    def claim(
        self,
        worker_id: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> LeasedEvaluationJob | None:
        """Atomically claim the highest-priority ready job."""

        if not worker_id or not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        _validate_duration(lease_seconds, "lease_seconds")
        now = now or utc_now()
        now_timestamp = datetime_to_timestamp(now)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        lease_token = secrets.token_urlsafe(32)

        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                self._sweep_expired_in_transaction(connection, now)
                selected = connection.execute(
                    """
                    SELECT job_id FROM evaluation_jobs
                    WHERE status IN (?, ?)
                      AND available_at <= ?
                      AND attempt_count < max_attempts
                    ORDER BY priority DESC, available_at ASC,
                             created_at ASC, job_id ASC
                    LIMIT 1
                    """,
                    (
                        JobStatus.QUEUED.value,
                        JobStatus.RETRY_WAIT.value,
                        now_timestamp,
                    ),
                ).fetchone()
                if selected is None:
                    return None
                job_id = str(selected["job_id"])
                cursor = connection.execute(
                    """
                    UPDATE evaluation_jobs
                    SET status = ?, attempt_count = attempt_count + 1,
                        lease_owner = ?, lease_token = ?, lease_expires_at = ?,
                        heartbeat_at = ?, updated_at = ?
                    WHERE job_id = ? AND status IN (?, ?)
                      AND available_at <= ? AND attempt_count < max_attempts
                    """,
                    (
                        JobStatus.LEASED.value,
                        worker_id,
                        lease_token,
                        datetime_to_timestamp(lease_expires_at),
                        now_timestamp,
                        now_timestamp,
                        job_id,
                        JobStatus.QUEUED.value,
                        JobStatus.RETRY_WAIT.value,
                        now_timestamp,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("evaluation job claim lost inside transaction")
                row = self._select_job(connection, job_id)
                assert row is not None
                leased = self._stored_from_row(row)
                assert isinstance(leased, LeasedEvaluationJob)
                self.event_log._append_in_transaction(
                    connection,
                    event_type="EVALUATION_JOB_LEASED",
                    aggregate_type="evaluation_job",
                    aggregate_id=job_id,
                    experiment_id=leased.experiment_id,
                    task_id=leased.task_id,
                    round_number=leased.round_number,
                    payload={
                        "worker_id": worker_id,
                        "attempt": leased.attempt_count,
                        "lease_expires_at": lease_expires_at,
                    },
                )
                return leased

    @staticmethod
    def _raise_lease_lost(
        connection: sqlite3.Connection,
        job_id: str,
        worker_id: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT status, lease_owner, lease_expires_at
            FROM evaluation_jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            detail = "job does not exist"
        else:
            detail = (
                f"status={row['status']}, owner={row['lease_owner']!r}, "
                f"worker={worker_id!r}"
            )
        raise LeaseLostError(f"lease for job {job_id!r} was lost ({detail})")

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        *,
        lease_seconds: float,
        now: datetime | None = None,
    ) -> LeasedEvaluationJob:
        """Extend a live lease; stale lease tokens can never revive a job."""

        _validate_duration(lease_seconds, "lease_seconds")
        now = now or utc_now()
        now_timestamp = datetime_to_timestamp(now)
        new_expiry = now + timedelta(seconds=lease_seconds)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                cursor = connection.execute(
                    """
                    UPDATE evaluation_jobs
                    SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                    WHERE job_id = ? AND status = ? AND lease_owner = ?
                      AND lease_token = ? AND lease_expires_at > ?
                    """,
                    (
                        now_timestamp,
                        datetime_to_timestamp(new_expiry),
                        now_timestamp,
                        job_id,
                        JobStatus.LEASED.value,
                        worker_id,
                        lease_token,
                        now_timestamp,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_lease_lost(connection, job_id, worker_id)
                row = self._select_job(connection, job_id)
                assert row is not None
                leased = self._stored_from_row(row)
                assert isinstance(leased, LeasedEvaluationJob)
                return leased

    def complete(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        result: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> StoredEvaluationJob:
        """Commit a result only while the caller still owns a live lease."""

        now = now or utc_now()
        now_timestamp = datetime_to_timestamp(now)
        result_json = canonical_json_dumps(result)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                cursor = connection.execute(
                    """
                    UPDATE evaluation_jobs
                    SET status = ?, result_json = ?, lease_owner = NULL,
                        lease_token = NULL, lease_expires_at = NULL,
                        heartbeat_at = NULL, updated_at = ?
                    WHERE job_id = ? AND status = ? AND lease_owner = ?
                      AND lease_token = ? AND lease_expires_at > ?
                    """,
                    (
                        JobStatus.SUCCEEDED.value,
                        result_json,
                        now_timestamp,
                        job_id,
                        JobStatus.LEASED.value,
                        worker_id,
                        lease_token,
                        now_timestamp,
                    ),
                )
                if cursor.rowcount != 1:
                    self._raise_lease_lost(connection, job_id, worker_id)
                row = self._select_job(connection, job_id)
                assert row is not None
                stored = self._stored_from_row(row)
                self.event_log._append_in_transaction(
                    connection,
                    event_type="EVALUATION_JOB_SUCCEEDED",
                    aggregate_type="evaluation_job",
                    aggregate_id=job_id,
                    experiment_id=stored.experiment_id,
                    task_id=stored.task_id,
                    round_number=stored.round_number,
                    payload={"attempt": stored.attempt_count},
                )
                return stored

    def fail(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        error: Mapping[str, Any],
        *,
        retryable: bool = True,
        retry_delay_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> StoredEvaluationJob:
        """Release a failed lease into retry-wait or the dead-letter state."""

        if (
            not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        now = now or utc_now()
        now_timestamp = datetime_to_timestamp(now)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                leased_row = connection.execute(
                    """
                    SELECT * FROM evaluation_jobs
                    WHERE job_id = ? AND status = ? AND lease_owner = ?
                      AND lease_token = ? AND lease_expires_at > ?
                    """,
                    (
                        job_id,
                        JobStatus.LEASED.value,
                        worker_id,
                        lease_token,
                        now_timestamp,
                    ),
                ).fetchone()
                if leased_row is None:
                    self._raise_lease_lost(connection, job_id, worker_id)
                assert leased_row is not None
                attempts_exhausted = int(leased_row["attempt_count"]) >= int(
                    leased_row["max_attempts"]
                )
                new_status = (
                    JobStatus.DEAD
                    if attempts_exhausted or not retryable
                    else JobStatus.RETRY_WAIT
                )
                available_at = now + timedelta(seconds=retry_delay_seconds)
                connection.execute(
                    """
                    UPDATE evaluation_jobs
                    SET status = ?, available_at = ?, last_error_json = ?,
                        lease_owner = NULL, lease_token = NULL,
                        lease_expires_at = NULL, heartbeat_at = NULL,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    (
                        new_status.value,
                        datetime_to_timestamp(available_at),
                        canonical_json_dumps(error),
                        now_timestamp,
                        job_id,
                    ),
                )
                row = self._select_job(connection, job_id)
                assert row is not None
                stored = self._stored_from_row(row)
                self.event_log._append_in_transaction(
                    connection,
                    event_type=(
                        "EVALUATION_JOB_RETRY_SCHEDULED"
                        if new_status is JobStatus.RETRY_WAIT
                        else "EVALUATION_JOB_DEAD"
                    ),
                    aggregate_type="evaluation_job",
                    aggregate_id=job_id,
                    experiment_id=stored.experiment_id,
                    task_id=stored.task_id,
                    round_number=stored.round_number,
                    payload={
                        "attempt": stored.attempt_count,
                        "retryable": retryable,
                        "error": error,
                    },
                )
                return stored

    def cancel(self, job_id: str, *, now: datetime | None = None) -> bool:
        """Cancel work that has not been leased; return whether it changed."""

        now = now or utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                cursor = connection.execute(
                    """
                    UPDATE evaluation_jobs
                    SET status = ?, updated_at = ?
                    WHERE job_id = ? AND status IN (?, ?)
                    """,
                    (
                        JobStatus.CANCELLED.value,
                        datetime_to_timestamp(now),
                        job_id,
                        JobStatus.QUEUED.value,
                        JobStatus.RETRY_WAIT.value,
                    ),
                )
                if cursor.rowcount != 1:
                    return False
                row = self._select_job(connection, job_id)
                assert row is not None
                stored = self._stored_from_row(row)
                self.event_log._append_in_transaction(
                    connection,
                    event_type="EVALUATION_JOB_CANCELLED",
                    aggregate_type="evaluation_job",
                    aggregate_id=job_id,
                    experiment_id=stored.experiment_id,
                    task_id=stored.task_id,
                    round_number=stored.round_number,
                    payload={},
                )
                return True

    def list(
        self,
        *,
        statuses: set[JobStatus] | None = None,
        limit: int = 1000,
    ) -> list[StoredEvaluationJob]:
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        parameters: list[Any] = []
        query = "SELECT * FROM evaluation_jobs"
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            parameters.extend(status.value for status in sorted(statuses, key=str))
        query += " ORDER BY created_at, job_id LIMIT ?"
        parameters.append(limit)
        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._stored_from_row(row) for row in rows]
