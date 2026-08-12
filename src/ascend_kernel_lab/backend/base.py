"""Backend contract and transport-safe stage results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec


class StageStatus(str, Enum):
    """Portable outcome categories shared by real and fake backends."""

    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    TIMEOUT = "timeout"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class StageResult:
    """One immutable, JSON-serializable backend-stage outcome."""

    stage: EvaluationStage | str
    status: StageStatus
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime = field(default_factory=_utc_now)
    details: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        _aware(self.started_at, "started_at")
        _aware(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        object.__setattr__(self, "details", _mapping(self.details))
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))
        if self.error is not None:
            object.__setattr__(self, "error", _mapping(self.error))
        if self.status not in {StageStatus.ERROR, StageStatus.TIMEOUT} and self.retryable:
            raise ValueError("only error and timeout results may be retryable")

    @property
    def passed(self) -> bool:
        return self.status is StageStatus.PASS

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        stage = self.stage.value if isinstance(self.stage, EvaluationStage) else self.stage
        return {
            "schema_version": "ascend_stage_result_v1",
            "stage": stage,
            "status": self.status.value,
            "passed": self.passed,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "details": dict(self.details),
            "artifacts": dict(self.artifacts),
            "error": dict(self.error) if self.error is not None else None,
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StageResult:
        raw_stage = str(value["stage"])
        try:
            stage: EvaluationStage | str = EvaluationStage(raw_stage)
        except ValueError:
            stage = raw_stage
        started = datetime.fromisoformat(str(value["started_at"]))
        finished = datetime.fromisoformat(str(value["finished_at"]))
        details = value.get("details", {})
        artifacts = value.get("artifacts", {})
        error = value.get("error")
        if not isinstance(details, Mapping) or not isinstance(artifacts, Mapping):
            raise ValueError("stage result details and artifacts must be mappings")
        if error is not None and not isinstance(error, Mapping):
            raise ValueError("stage result error must be a mapping or null")
        return cls(
            stage=stage,
            status=StageStatus(str(value["status"])),
            started_at=started,
            finished_at=finished,
            details={str(key): item for key, item in details.items()},
            artifacts={str(key): str(item) for key, item in artifacts.items()},
            error=(
                {str(key): item for key, item in error.items()}
                if isinstance(error, Mapping)
                else None
            ),
            retryable=bool(value.get("retryable", False)),
        )

    @classmethod
    def success(
        cls,
        stage: EvaluationStage | str,
        *,
        started_at: datetime | None = None,
        details: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, str] | None = None,
    ) -> StageResult:
        return cls(
            stage=stage,
            status=StageStatus.PASS,
            started_at=started_at or _utc_now(),
            finished_at=_utc_now(),
            details=details or {},
            artifacts=artifacts or {},
        )

    @classmethod
    def failure(
        cls,
        stage: EvaluationStage | str,
        *,
        started_at: datetime | None = None,
        details: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, str] | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> StageResult:
        return cls(
            stage=stage,
            status=StageStatus.FAIL,
            started_at=started_at or _utc_now(),
            finished_at=_utc_now(),
            details=details or {},
            artifacts=artifacts or {},
            error=error,
        )

    @classmethod
    def infrastructure_error(
        cls,
        stage: EvaluationStage | str,
        *,
        message: str,
        error_type: str = "BackendError",
        retryable: bool = True,
        timed_out: bool = False,
        started_at: datetime | None = None,
        details: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, str] | None = None,
    ) -> StageResult:
        return cls(
            stage=stage,
            status=StageStatus.TIMEOUT if timed_out else StageStatus.ERROR,
            started_at=started_at or _utc_now(),
            finished_at=_utc_now(),
            details=details or {},
            artifacts=artifacts or {},
            error={"type": error_type, "message": message},
            retryable=retryable,
        )


@runtime_checkable
class Backend(Protocol):
    """Execution interface consumed by the orchestrator and durable worker."""

    def source_check(self, candidate_path: Path, task: TaskSpec) -> StageResult: ...

    def compile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult: ...

    def check_correctness(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult: ...

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
    ) -> StageResult: ...

    def profile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult: ...

    def health_check(self) -> StageResult: ...

    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]: ...


def finite_positive(value: float, name: str) -> float:
    """Validate backend duration/config values without duplicating edge cases."""

    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result
