"""Append-only SQLite event log used as durable completion evidence."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ascend_kernel_lab.domain import (
    EventRecord,
    IdempotencyConflictError,
    utc_now,
)

from .database import (
    SQLiteDatabase,
    canonical_json_dumps,
    datetime_to_timestamp,
    json_loads_object,
    timestamp_to_datetime,
)


class SQLiteEventLog:
    """Store immutable, monotonically sequenced domain events."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> EventRecord:
        payload = json_loads_object(row["payload_json"])
        assert payload is not None
        return EventRecord(
            sequence=int(row["sequence"]),
            event_id=str(row["event_id"]),
            event_type=str(row["event_type"]),
            aggregate_type=str(row["aggregate_type"]),
            aggregate_id=str(row["aggregate_id"]),
            experiment_id=row["experiment_id"],
            task_id=row["task_id"],
            round_number=row["round_number"],
            payload=payload,
            occurred_at=timestamp_to_datetime(float(row["occurred_at"])),
        )

    def _append_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        payload: Mapping[str, Any],
        experiment_id: str | None = None,
        task_id: str | None = None,
        round_number: int | None = None,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> EventRecord:
        event_id = event_id or uuid.uuid4().hex
        requested_occurred_at = occurred_at
        occurred_at = occurred_at or utc_now()
        payload_json = canonical_json_dumps(payload)

        existing = connection.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            record = self._record_from_row(existing)
            same = (
                record.event_type == event_type
                and record.aggregate_type == aggregate_type
                and record.aggregate_id == aggregate_id
                and record.experiment_id == experiment_id
                and record.task_id == task_id
                and record.round_number == round_number
                and canonical_json_dumps(record.payload) == payload_json
                and (
                    requested_occurred_at is None
                    or record.occurred_at == requested_occurred_at
                )
            )
            if not same:
                raise IdempotencyConflictError(
                    f"event_id {event_id!r} already refers to a different event"
                )
            return record

        cursor = connection.execute(
            """
            INSERT INTO events(
                event_id, event_type, aggregate_type, aggregate_id,
                experiment_id, task_id, round_number, payload_json, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                aggregate_type,
                aggregate_id,
                experiment_id,
                task_id,
                round_number,
                payload_json,
                datetime_to_timestamp(occurred_at),
            ),
        )
        row = connection.execute(
            "SELECT * FROM events WHERE sequence = ?", (cursor.lastrowid,)
        ).fetchone()
        assert row is not None
        return self._record_from_row(row)

    def append(
        self,
        *,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        payload: Mapping[str, Any],
        experiment_id: str | None = None,
        task_id: str | None = None,
        round_number: int | None = None,
        event_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> EventRecord:
        """Append an event atomically; identical explicit IDs are idempotent."""

        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                return self._append_in_transaction(
                    connection,
                    event_type=event_type,
                    aggregate_type=aggregate_type,
                    aggregate_id=aggregate_id,
                    payload=payload,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    round_number=round_number,
                    event_id=event_id,
                    occurred_at=occurred_at,
                )

    def read(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 1000,
        aggregate_type: str | None = None,
        aggregate_id: str | None = None,
        experiment_id: str | None = None,
        task_id: str | None = None,
    ) -> list[EventRecord]:
        """Read events in durable sequence order using an exclusive cursor."""

        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if (aggregate_type is None) != (aggregate_id is None):
            raise ValueError(
                "aggregate_type and aggregate_id must be provided together"
            )

        predicates = ["sequence > ?"]
        parameters: list[Any] = [after_sequence]
        if aggregate_type is not None:
            predicates.extend(["aggregate_type = ?", "aggregate_id = ?"])
            parameters.extend([aggregate_type, aggregate_id])
        if experiment_id is not None:
            predicates.append("experiment_id = ?")
            parameters.append(experiment_id)
        if task_id is not None:
            predicates.append("task_id = ?")
            parameters.append(task_id)
        parameters.append(limit)
        query = (
            "SELECT * FROM events WHERE "
            + " AND ".join(predicates)
            + " ORDER BY sequence LIMIT ?"
        )
        with self.database.connection() as connection:
            rows: Sequence[sqlite3.Row] = connection.execute(
                query, parameters
            ).fetchall()
        return [self._record_from_row(row) for row in rows]
