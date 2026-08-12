"""Transactional metadata store for experiments, tasks, rounds, and candidates."""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ascend_kernel_lab.domain import (
    ArtifactConflictError,
    ArtifactMetadata,
    CandidateRecord,
    CandidateScore,
    ConcurrentUpdateError,
    ExperimentRecord,
    ExperimentState,
    InvalidTransitionError,
    RoundRecord,
    RoundState,
    TaskRecord,
    TaskState,
    compute_reward,
    round_state_machine,
    select_best_candidate,
    task_state_machine,
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
from .job_queue import SQLiteEvaluationJobQueue

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256(value: str | None, field_name: str) -> None:
    if value is not None and _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


class SQLiteStateStore:
    """High-level durable state facade used by the controller."""

    def __init__(
        self,
        database: SQLiteDatabase | str | Path,
        *,
        busy_timeout_seconds: float = 30.0,
    ) -> None:
        self.database = (
            database
            if isinstance(database, SQLiteDatabase)
            else SQLiteDatabase(
                database, busy_timeout_seconds=busy_timeout_seconds
            )
        )
        self.events = SQLiteEventLog(self.database)
        self.jobs = SQLiteEvaluationJobQueue(self.database, self.events)

    @staticmethod
    def _experiment_from_row(row: sqlite3.Row) -> ExperimentRecord:
        config = json_loads_object(row["config_json"])
        assert config is not None
        return ExperimentRecord(
            experiment_id=str(row["experiment_id"]),
            state=ExperimentState(str(row["state"])),
            config=config,
            version=int(row["version"]),
            created_at=timestamp_to_datetime(float(row["created_at"])),
            updated_at=timestamp_to_datetime(float(row["updated_at"])),
        )

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            experiment_id=str(row["experiment_id"]),
            task_id=str(row["task_id"]),
            state=TaskState(str(row["state"])),
            current_round=int(row["current_round"]),
            best_candidate_id=row["best_candidate_id"],
            version=int(row["version"]),
            created_at=timestamp_to_datetime(float(row["created_at"])),
            updated_at=timestamp_to_datetime(float(row["updated_at"])),
        )

    @staticmethod
    def _round_from_row(row: sqlite3.Row) -> RoundRecord:
        return RoundRecord(
            experiment_id=str(row["experiment_id"]),
            task_id=str(row["task_id"]),
            round_number=int(row["round_number"]),
            state=RoundState(str(row["state"])),
            version=int(row["version"]),
            created_at=timestamp_to_datetime(float(row["created_at"])),
            updated_at=timestamp_to_datetime(float(row["updated_at"])),
        )

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=str(row["candidate_id"]),
            experiment_id=str(row["experiment_id"]),
            task_id=str(row["task_id"]),
            round_number=int(row["round_number"]),
            source_sha256=str(row["source_sha256"]),
            source_artifact_path=str(row["source_artifact_path"]),
            response_sha256=row["response_sha256"],
            created_at=timestamp_to_datetime(float(row["created_at"])),
        )

    @staticmethod
    def _score_from_row(row: sqlite3.Row) -> CandidateScore:
        hidden = row["hidden_correctness_passed"]
        return CandidateScore(
            candidate_id=str(row["candidate_id"]),
            round_number=int(row["round_number"]),
            compile_passed=bool(row["compile_passed"]),
            correctness_passed=bool(row["correctness_passed"]),
            anti_bypass_passed=bool(row["anti_bypass_passed"]),
            hidden_correctness_passed=None if hidden is None else bool(hidden),
            minimum_speedup=row["minimum_speedup"],
            geomean_speedup=row["geomean_speedup"],
            candidate_kernel_coverage=row["candidate_kernel_coverage"],
            stability_cv=row["stability_cv"],
        )

    def create_experiment(
        self,
        experiment_id: str,
        config: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> ExperimentRecord:
        """Create an experiment; an identical replay returns the existing row."""

        now = now or utc_now()
        timestamp = datetime_to_timestamp(now)
        config_json = canonical_json_dumps(config)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["config_json"]) != config_json:
                        raise ConcurrentUpdateError(
                            f"experiment {experiment_id!r} already has different config"
                        )
                    return self._experiment_from_row(existing)
                connection.execute(
                    """
                    INSERT INTO experiments(
                        experiment_id, state, config_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        ExperimentState.CREATED.value,
                        config_json,
                        timestamp,
                        timestamp,
                    ),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type="EXPERIMENT_CREATED",
                    aggregate_type="experiment",
                    aggregate_id=experiment_id,
                    experiment_id=experiment_id,
                    payload={},
                    occurred_at=now,
                )
                row = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                assert row is not None
                return self._experiment_from_row(row)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._experiment_from_row(row) if row is not None else None

    def transition_experiment(
        self,
        experiment_id: str,
        target: ExperimentState,
        *,
        expected_version: int | None = None,
        now: datetime | None = None,
    ) -> ExperimentRecord:
        transitions = {
            ExperimentState.CREATED: {
                ExperimentState.RUNNING,
                ExperimentState.FAILED,
                ExperimentState.CANCELLED,
            },
            ExperimentState.RUNNING: {
                ExperimentState.FINISHED,
                ExperimentState.FAILED,
                ExperimentState.CANCELLED,
            },
            ExperimentState.FINISHED: set(),
            ExperimentState.FAILED: set(),
            ExperimentState.CANCELLED: set(),
        }
        now = now or utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                row = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(experiment_id)
                current = ExperimentState(str(row["state"]))
                if target not in transitions[current]:
                    raise InvalidTransitionError(
                        f"invalid transition {current.value} -> {target.value}"
                    )
                version = int(row["version"])
                if expected_version is not None and version != expected_version:
                    raise ConcurrentUpdateError(
                        f"expected experiment version {expected_version}, found {version}"
                    )
                cursor = connection.execute(
                    """
                    UPDATE experiments SET state = ?, updated_at = ?, version = version + 1
                    WHERE experiment_id = ? AND version = ?
                    """,
                    (
                        target.value,
                        datetime_to_timestamp(now),
                        experiment_id,
                        version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentUpdateError("experiment changed concurrently")
                self.events._append_in_transaction(
                    connection,
                    event_type=f"EXPERIMENT_{target.value}",
                    aggregate_type="experiment",
                    aggregate_id=experiment_id,
                    experiment_id=experiment_id,
                    payload={"previous_state": current.value},
                    occurred_at=now,
                )
                updated = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                assert updated is not None
                return self._experiment_from_row(updated)

    def create_task(
        self,
        experiment_id: str,
        task_id: str,
        *,
        task_version: int = 1,
        task_spec_sha256: str | None = None,
        now: datetime | None = None,
    ) -> TaskRecord:
        if task_version < 1:
            raise ValueError("task_version must be positive")
        _validate_sha256(task_spec_sha256, "task_spec_sha256")
        now = now or utc_now()
        timestamp = datetime_to_timestamp(now)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = connection.execute(
                    """
                    SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                if existing is not None:
                    if (
                        int(existing["task_version"]) != task_version
                        or existing["task_spec_sha256"] != task_spec_sha256
                    ):
                        raise ConcurrentUpdateError(
                            "task identity already exists with different metadata"
                        )
                    return self._task_from_row(existing)
                connection.execute(
                    """
                    INSERT INTO tasks(
                        experiment_id, task_id, task_version, state,
                        task_spec_sha256, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        task_id,
                        task_version,
                        TaskState.TASK_CREATED.value,
                        task_spec_sha256,
                        timestamp,
                        timestamp,
                    ),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type=TaskState.TASK_CREATED.value,
                    aggregate_type="task",
                    aggregate_id=f"{experiment_id}/{task_id}",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    payload={"task_version": task_version},
                    occurred_at=now,
                )
                row = connection.execute(
                    """
                    SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                assert row is not None
                return self._task_from_row(row)

    def get_task(self, experiment_id: str, task_id: str) -> TaskRecord | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                """,
                (experiment_id, task_id),
            ).fetchone()
        return self._task_from_row(row) if row is not None else None

    def transition_task(
        self,
        experiment_id: str,
        task_id: str,
        target: TaskState,
        *,
        expected_version: int | None = None,
        now: datetime | None = None,
    ) -> TaskRecord:
        now = now or utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                row = connection.execute(
                    """
                    SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                if row is None:
                    raise KeyError((experiment_id, task_id))
                current = TaskState(str(row["state"]))
                task_state_machine.transition(current, target)
                version = int(row["version"])
                if expected_version is not None and version != expected_version:
                    raise ConcurrentUpdateError(
                        f"expected task version {expected_version}, found {version}"
                    )
                cursor = connection.execute(
                    """
                    UPDATE tasks SET state = ?, updated_at = ?, version = version + 1
                    WHERE experiment_id = ? AND task_id = ? AND version = ?
                    """,
                    (
                        target.value,
                        datetime_to_timestamp(now),
                        experiment_id,
                        task_id,
                        version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentUpdateError("task changed concurrently")
                self.events._append_in_transaction(
                    connection,
                    event_type=target.value,
                    aggregate_type="task",
                    aggregate_id=f"{experiment_id}/{task_id}",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    payload={"previous_state": current.value},
                    occurred_at=now,
                )
                updated = connection.execute(
                    """
                    SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                assert updated is not None
                return self._task_from_row(updated)

    def create_round(
        self,
        experiment_id: str,
        task_id: str,
        round_number: int,
        *,
        now: datetime | None = None,
    ) -> RoundRecord:
        if round_number < 1:
            raise ValueError("round_number must be positive")
        now = now or utc_now()
        timestamp = datetime_to_timestamp(now)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = connection.execute(
                    """
                    SELECT * FROM rounds
                    WHERE experiment_id = ? AND task_id = ? AND round_number = ?
                    """,
                    (experiment_id, task_id, round_number),
                ).fetchone()
                if existing is not None:
                    return self._round_from_row(existing)
                task_row = connection.execute(
                    """
                    SELECT * FROM tasks WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                if task_row is None:
                    raise KeyError((experiment_id, task_id))
                if TaskState(str(task_row["state"])) is not TaskState.ROUNDS_RUNNING:
                    raise InvalidTransitionError(
                        "rounds can only be created while task is ROUNDS_RUNNING"
                    )
                expected_round = int(task_row["current_round"]) + 1
                if round_number != expected_round:
                    raise InvalidTransitionError(
                        f"expected round {expected_round}, got {round_number}"
                    )
                if round_number > 1:
                    previous = connection.execute(
                        """
                        SELECT state FROM rounds
                        WHERE experiment_id = ? AND task_id = ?
                          AND round_number = ?
                        """,
                        (experiment_id, task_id, round_number - 1),
                    ).fetchone()
                    if (
                        previous is None
                        or RoundState(str(previous["state"]))
                        is not RoundState.ROUND_FINISHED
                    ):
                        raise InvalidTransitionError(
                            "the previous round must be finished first"
                        )
                connection.execute(
                    """
                    INSERT INTO rounds(
                        experiment_id, task_id, round_number, state,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        task_id,
                        round_number,
                        RoundState.ROUND_CREATED.value,
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    UPDATE tasks
                    SET current_round = ?, updated_at = ?, version = version + 1
                    WHERE experiment_id = ? AND task_id = ?
                    """,
                    (round_number, timestamp, experiment_id, task_id),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type=RoundState.ROUND_CREATED.value,
                    aggregate_type="round",
                    aggregate_id=f"{experiment_id}/{task_id}/{round_number}",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    round_number=round_number,
                    payload={},
                    occurred_at=now,
                )
                row = connection.execute(
                    """
                    SELECT * FROM rounds
                    WHERE experiment_id = ? AND task_id = ? AND round_number = ?
                    """,
                    (experiment_id, task_id, round_number),
                ).fetchone()
                assert row is not None
                return self._round_from_row(row)

    def get_round(
        self, experiment_id: str, task_id: str, round_number: int
    ) -> RoundRecord | None:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM rounds
                WHERE experiment_id = ? AND task_id = ? AND round_number = ?
                """,
                (experiment_id, task_id, round_number),
            ).fetchone()
        return self._round_from_row(row) if row is not None else None

    def transition_round(
        self,
        experiment_id: str,
        task_id: str,
        round_number: int,
        target: RoundState,
        *,
        expected_version: int | None = None,
        event_payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> RoundRecord:
        """Compare-and-swap a round and append its checkpoint event atomically."""

        now = now or utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                row = connection.execute(
                    """
                    SELECT * FROM rounds
                    WHERE experiment_id = ? AND task_id = ? AND round_number = ?
                    """,
                    (experiment_id, task_id, round_number),
                ).fetchone()
                if row is None:
                    raise KeyError((experiment_id, task_id, round_number))
                current = RoundState(str(row["state"]))
                round_state_machine.transition(current, target)
                version = int(row["version"])
                if expected_version is not None and version != expected_version:
                    raise ConcurrentUpdateError(
                        f"expected round version {expected_version}, found {version}"
                    )
                cursor = connection.execute(
                    """
                    UPDATE rounds SET state = ?, updated_at = ?, version = version + 1
                    WHERE experiment_id = ? AND task_id = ?
                      AND round_number = ? AND version = ?
                    """,
                    (
                        target.value,
                        datetime_to_timestamp(now),
                        experiment_id,
                        task_id,
                        round_number,
                        version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConcurrentUpdateError("round changed concurrently")
                payload = {"previous_state": current.value}
                payload.update(event_payload or {})
                self.events._append_in_transaction(
                    connection,
                    event_type=target.value,
                    aggregate_type="round",
                    aggregate_id=f"{experiment_id}/{task_id}/{round_number}",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    round_number=round_number,
                    payload=payload,
                    occurred_at=now,
                )
                updated = connection.execute(
                    """
                    SELECT * FROM rounds
                    WHERE experiment_id = ? AND task_id = ? AND round_number = ?
                    """,
                    (experiment_id, task_id, round_number),
                ).fetchone()
                assert updated is not None
                return self._round_from_row(updated)

    def register_candidate(
        self,
        *,
        candidate_id: str,
        experiment_id: str,
        task_id: str,
        round_number: int,
        source_sha256: str,
        source_artifact_path: str,
        response_sha256: str | None = None,
        now: datetime | None = None,
    ) -> CandidateRecord:
        _validate_sha256(source_sha256, "source_sha256")
        _validate_sha256(response_sha256, "response_sha256")
        now = now or utc_now()
        values = (
            experiment_id,
            task_id,
            round_number,
            source_sha256,
            source_artifact_path,
            response_sha256,
        )
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = connection.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if existing is not None:
                    existing_values = tuple(
                        existing[name]
                        for name in (
                            "experiment_id",
                            "task_id",
                            "round_number",
                            "source_sha256",
                            "source_artifact_path",
                            "response_sha256",
                        )
                    )
                    if existing_values != values:
                        raise ConcurrentUpdateError(
                            f"candidate {candidate_id!r} already has different metadata"
                        )
                    return self._candidate_from_row(existing)
                connection.execute(
                    """
                    INSERT INTO candidates(
                        candidate_id, experiment_id, task_id, round_number,
                        source_sha256, source_artifact_path, response_sha256,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (candidate_id, *values, datetime_to_timestamp(now)),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type="CANDIDATE_REGISTERED",
                    aggregate_type="candidate",
                    aggregate_id=candidate_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    round_number=round_number,
                    payload={
                        "source_sha256": source_sha256,
                        "source_artifact_path": source_artifact_path,
                    },
                    occurred_at=now,
                )
                row = connection.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                assert row is not None
                return self._candidate_from_row(row)

    def save_candidate_score(
        self,
        score: CandidateScore,
        *,
        now: datetime | None = None,
    ) -> CandidateScore:
        now = now or utc_now()
        reward = compute_reward(score)
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                candidate = connection.execute(
                    "SELECT * FROM candidates WHERE candidate_id = ?",
                    (score.candidate_id,),
                ).fetchone()
                if candidate is None:
                    raise KeyError(score.candidate_id)
                if int(candidate["round_number"]) != score.round_number:
                    raise ValueError("score round does not match candidate round")
                existing_score = connection.execute(
                    "SELECT * FROM candidate_scores WHERE candidate_id = ?",
                    (score.candidate_id,),
                ).fetchone()
                if (
                    existing_score is not None
                    and self._score_from_row(existing_score) == score
                ):
                    return score
                connection.execute(
                    """
                    INSERT INTO candidate_scores(
                        candidate_id, round_number, compile_passed,
                        correctness_passed, anti_bypass_passed,
                        hidden_correctness_passed, minimum_speedup,
                        geomean_speedup, candidate_kernel_coverage,
                        stability_cv, reward_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id) DO UPDATE SET
                        round_number = excluded.round_number,
                        compile_passed = excluded.compile_passed,
                        correctness_passed = excluded.correctness_passed,
                        anti_bypass_passed = excluded.anti_bypass_passed,
                        hidden_correctness_passed = excluded.hidden_correctness_passed,
                        minimum_speedup = excluded.minimum_speedup,
                        geomean_speedup = excluded.geomean_speedup,
                        candidate_kernel_coverage = excluded.candidate_kernel_coverage,
                        stability_cv = excluded.stability_cv,
                        reward_json = excluded.reward_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        score.candidate_id,
                        score.round_number,
                        int(score.compile_passed),
                        int(score.correctness_passed),
                        int(score.anti_bypass_passed),
                        (
                            None
                            if score.hidden_correctness_passed is None
                            else int(score.hidden_correctness_passed)
                        ),
                        score.minimum_speedup,
                        score.geomean_speedup,
                        score.candidate_kernel_coverage,
                        score.stability_cv,
                        canonical_json_dumps(reward),
                        datetime_to_timestamp(now),
                    ),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type="CANDIDATE_SCORE_COMMITTED",
                    aggregate_type="candidate",
                    aggregate_id=score.candidate_id,
                    experiment_id=str(candidate["experiment_id"]),
                    task_id=str(candidate["task_id"]),
                    round_number=score.round_number,
                    payload={"reward": reward},
                    occurred_at=now,
                )
                return score

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        """Return one immutable candidate identity, if it exists."""

        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._candidate_from_row(row) if row is not None else None

    def get_candidate_score(self, candidate_id: str) -> CandidateScore | None:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT * FROM candidate_scores WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._score_from_row(row) if row is not None else None

    def list_candidate_scores(
        self, experiment_id: str, task_id: str
    ) -> list[CandidateScore]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT score.* FROM candidate_scores AS score
                JOIN candidates AS candidate USING (candidate_id)
                WHERE candidate.experiment_id = ? AND candidate.task_id = ?
                ORDER BY score.round_number, score.candidate_id
                """,
                (experiment_id, task_id),
            ).fetchall()
        return [self._score_from_row(row) for row in rows]

    def select_and_set_best_candidate(
        self,
        experiment_id: str,
        task_id: str,
        *,
        require_hidden_correctness: bool = False,
        now: datetime | None = None,
    ) -> CandidateScore | None:
        """Rank all historical candidates and atomically persist the winner."""

        now = now or utc_now()
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                rows = connection.execute(
                    """
                    SELECT score.* FROM candidate_scores AS score
                    JOIN candidates AS candidate USING (candidate_id)
                    WHERE candidate.experiment_id = ? AND candidate.task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchall()
                best = select_best_candidate(
                    (self._score_from_row(row) for row in rows),
                    require_hidden_correctness=require_hidden_correctness,
                )
                if best is None:
                    return None
                task_row = connection.execute(
                    """
                    SELECT best_candidate_id FROM tasks
                    WHERE experiment_id = ? AND task_id = ?
                    """,
                    (experiment_id, task_id),
                ).fetchone()
                if task_row is None:
                    raise KeyError((experiment_id, task_id))
                if task_row["best_candidate_id"] == best.candidate_id:
                    return best
                cursor = connection.execute(
                    """
                    UPDATE tasks SET best_candidate_id = ?, updated_at = ?,
                                     version = version + 1
                    WHERE experiment_id = ? AND task_id = ?
                    """,
                    (
                        best.candidate_id,
                        datetime_to_timestamp(now),
                        experiment_id,
                        task_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError((experiment_id, task_id))
                self.events._append_in_transaction(
                    connection,
                    event_type="BEST_CANDIDATE_SELECTED",
                    aggregate_type="task",
                    aggregate_id=f"{experiment_id}/{task_id}",
                    experiment_id=experiment_id,
                    task_id=task_id,
                    payload={
                        "candidate_id": best.candidate_id,
                        "round_number": best.round_number,
                        "require_hidden_correctness": require_hidden_correctness,
                    },
                    occurred_at=now,
                )
                return best

    def register_artifact(
        self,
        metadata: ArtifactMetadata,
        *,
        artifact_id: str | None = None,
        experiment_id: str | None = None,
        task_id: str | None = None,
        round_number: int | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register an already committed artifact and its verified digest."""

        artifact_id = artifact_id or uuid.uuid4().hex
        with (
            self.database.connection() as connection,
            self.database.transaction(connection, immediate=True),
        ):
                existing = connection.execute(
                    "SELECT * FROM artifacts WHERE relative_path = ?",
                    (metadata.relative_path,),
                ).fetchone()
                if existing is not None:
                    expected_metadata = canonical_json_dumps(extra_metadata or {})
                    if (
                        str(existing["sha256"]) != metadata.sha256
                        or int(existing["size_bytes"]) != metadata.size_bytes
                        or str(existing["media_type"]) != metadata.media_type
                        or existing["experiment_id"] != experiment_id
                        or existing["task_id"] != task_id
                        or existing["round_number"] != round_number
                        or str(existing["metadata_json"]) != expected_metadata
                    ):
                        raise ArtifactConflictError(
                            "artifact path is registered with different contents or ownership"
                        )
                    return str(existing["artifact_id"])
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        artifact_id, experiment_id, task_id, round_number,
                        relative_path, sha256, size_bytes, media_type,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        experiment_id,
                        task_id,
                        round_number,
                        metadata.relative_path,
                        metadata.sha256,
                        metadata.size_bytes,
                        metadata.media_type,
                        canonical_json_dumps(extra_metadata or {}),
                        datetime_to_timestamp(metadata.created_at),
                    ),
                )
                self.events._append_in_transaction(
                    connection,
                    event_type="ARTIFACT_REGISTERED",
                    aggregate_type="artifact",
                    aggregate_id=artifact_id,
                    experiment_id=experiment_id,
                    task_id=task_id,
                    round_number=round_number,
                    payload={
                        "relative_path": metadata.relative_path,
                        "sha256": metadata.sha256,
                        "size_bytes": metadata.size_bytes,
                    },
                    occurred_at=metadata.created_at,
                )
                return artifact_id

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> SQLiteStateStore:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
