"""Typed domain values shared by controller, worker, and persistence code."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, TypeAlias

from .enums import (
    EvaluationStage,
    ExperimentState,
    JobStatus,
    RoundState,
    TaskState,
)

JSONValue: TypeAlias = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)
JSONObject = dict[str, JSONValue]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _require_identifier(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Protect a top-level metadata mapping from accidental mutation."""

    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class BenchmarkSample:
    """One shape/dtype speedup used in aggregate scoring."""

    case_id: str
    speedup: float
    weight: float = 1.0
    coefficient_of_variation: float | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.case_id, "case_id")
        if not math.isfinite(self.speedup) or self.speedup <= 0:
            raise ValueError("speedup must be finite and greater than zero")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and greater than zero")
        if self.coefficient_of_variation is not None and (
            not math.isfinite(self.coefficient_of_variation)
            or self.coefficient_of_variation < 0
        ):
            raise ValueError(
                "coefficient_of_variation must be finite and non-negative"
            )


@dataclass(frozen=True)
class CandidateScore:
    """Raw, recomposable candidate metrics.

    Performance values remain optional because failed source/compile/correctness
    stages do not produce trustworthy benchmark measurements.
    """

    candidate_id: str
    round_number: int
    compile_passed: bool
    correctness_passed: bool
    anti_bypass_passed: bool
    hidden_correctness_passed: bool | None = None
    minimum_speedup: float | None = None
    geomean_speedup: float | None = None
    candidate_kernel_coverage: float | None = None
    stability_cv: float | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, "candidate_id")
        if self.round_number < 1:
            raise ValueError("round_number must be at least 1")
        for field_name in ("minimum_speedup", "geomean_speedup"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{field_name} must be finite and greater than zero")
        if self.candidate_kernel_coverage is not None and (
            not math.isfinite(self.candidate_kernel_coverage)
            or not 0 <= self.candidate_kernel_coverage <= 1
        ):
            raise ValueError("candidate_kernel_coverage must be between 0 and 1")
        if self.stability_cv is not None and (
            not math.isfinite(self.stability_cv) or self.stability_cv < 0
        ):
            raise ValueError("stability_cv must be finite and non-negative")

    @property
    def is_publicly_valid(self) -> bool:
        """Whether the candidate is eligible during public optimization rounds."""

        return (
            self.compile_passed
            and self.correctness_passed
            and self.anti_bypass_passed
            and self.minimum_speedup is not None
            and self.geomean_speedup is not None
        )

    @property
    def is_finally_valid(self) -> bool:
        """Whether the candidate also passed the hidden correctness suite."""

        return self.is_publicly_valid and self.hidden_correctness_passed is True


@dataclass(frozen=True)
class EvaluationJob:
    """A durable unit of work before it is claimed by a worker."""

    job_id: str
    experiment_id: str
    task_id: str
    round_number: int
    candidate_id: str
    stage: EvaluationStage
    payload: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0
    max_attempts: int = 3
    available_at: datetime = field(default_factory=utc_now)
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        for name in ("job_id", "experiment_id", "task_id", "candidate_id"):
            _require_identifier(getattr(self, name), name)
        if self.round_number < 1:
            raise ValueError("round_number must be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        _require_aware(self.available_at, "available_at")
        if self.idempotency_key is not None:
            _require_identifier(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))


@dataclass(frozen=True)
class StoredEvaluationJob:
    """Current persisted view of an evaluation job."""

    job_id: str
    experiment_id: str
    task_id: str
    round_number: int
    candidate_id: str
    stage: EvaluationStage
    payload: Mapping[str, Any]
    priority: int
    status: JobStatus
    attempt_count: int
    max_attempts: int
    available_at: datetime
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    heartbeat_at: datetime | None
    result: Mapping[str, Any] | None
    last_error: Mapping[str, Any] | None
    idempotency_key: str | None
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))
        if self.result is not None:
            object.__setattr__(self, "result", _immutable_mapping(self.result))
        if self.last_error is not None:
            object.__setattr__(self, "last_error", _immutable_mapping(self.last_error))

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            JobStatus.SUCCEEDED,
            JobStatus.DEAD,
            JobStatus.CANCELLED,
        }


@dataclass(frozen=True)
class LeasedEvaluationJob(StoredEvaluationJob):
    """A job view whose lease fields are guaranteed to be populated."""

    lease_owner: str
    lease_token: str
    lease_expires_at: datetime
    heartbeat_at: datetime

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.status is not JobStatus.LEASED:
            raise ValueError("a leased job must have LEASED status")
        _require_identifier(self.lease_owner, "lease_owner")
        _require_identifier(self.lease_token, "lease_token")


@dataclass(frozen=True)
class EventRecord:
    """An immutable event-log entry."""

    sequence: int
    event_id: str
    event_type: str
    aggregate_type: str
    aggregate_id: str
    payload: Mapping[str, Any]
    occurred_at: datetime
    experiment_id: str | None = None
    task_id: str | None = None
    round_number: int | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "aggregate_type", "aggregate_id"):
            _require_identifier(getattr(self, name), name)
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if self.round_number is not None and self.round_number < 1:
            raise ValueError("round_number must be positive")
        _require_aware(self.occurred_at, "occurred_at")
        object.__setattr__(self, "payload", _immutable_mapping(self.payload))


@dataclass(frozen=True)
class ArtifactMetadata:
    """Integrity metadata returned for an atomically committed artifact."""

    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    created_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.relative_path, "relative_path")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        _require_identifier(self.media_type, "media_type")
        _require_aware(self.created_at, "created_at")


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    state: ExperimentState
    config: Mapping[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _immutable_mapping(self.config))


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    experiment_id: str
    task_id: str
    round_number: int
    source_sha256: str
    source_artifact_path: str
    response_sha256: str | None
    created_at: datetime


@dataclass(frozen=True)
class RoundRecord:
    experiment_id: str
    task_id: str
    round_number: int
    state: RoundState
    version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TaskRecord:
    experiment_id: str
    task_id: str
    state: TaskState
    current_round: int
    best_candidate_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime
