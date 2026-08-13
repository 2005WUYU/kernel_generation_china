"""Synchronous controller adapter backed by the durable SQLite worker queue.

The evaluation orchestrator deliberately exposes a synchronous five-stage
interface.  This adapter preserves that interface while making every stage a
durable, idempotent job.  It never executes candidate code in the controller.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ascend_kernel_lab.backend.base import Backend, StageResult, StageStatus
from ascend_kernel_lab.domain import (
    EvaluationJob,
    EvaluationStage,
    JobStatus,
    StoredEvaluationJob,
)
from ascend_kernel_lab.evidence import EvidenceIntegrityError, validate_artifact_map
from ascend_kernel_lab.protocol import EVALUATION_JOB_PROTOCOL, harness_digest
from ascend_kernel_lab.storage import SQLiteEvaluationJobQueue, canonical_json_dumps
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec
from ascend_kernel_lab.tasks.runtime import hidden_suite_commitment

from .errors import (
    QueueBackendConfigurationError,
    QueueJobCancelled,
    QueueJobFailed,
    QueueProtocolError,
    QueueWaitTimeout,
)

JOB_PROTOCOL_VERSION: Final = EVALUATION_JOB_PROTOCOL
RESULT_PROTOCOL_VERSION: Final = "ascend_stage_result_v1"
PATH_MODE: Final = "artifact_root_relative"
_READY_AT: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)
_RESULT_KEYS: Final = frozenset(
    {
        "schema_version",
        "stage",
        "status",
        "passed",
        "started_at",
        "finished_at",
        "duration_seconds",
        "details",
        "artifacts",
        "error",
        "retryable",
    }
)
_HIDDEN_GENERATOR: Final = "hidden-v1"
_HIDDEN_CASE_ID_PREFIX: Final = "hidden_"
_PRIVATE_RESULT_FORBIDDEN_KEYS: Final = frozenset(
    {
        "address_offset",
        "dimensions",
        "distribution",
        "dtype",
        "noncontiguous",
        "params",
        "per_case",
        "seed",
        "shape",
        "shapes",
        "stderr",
        "stdout",
        "traceback",
    }
)


def _identifier(value: str, name: str) -> str:
    if not value or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty, NUL-free string")
    return value


def _safe_segment(value: str, name: str) -> str:
    _identifier(value, name)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be one safe artifact-path segment")
    return value


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _task_snapshot(task: TaskSpec) -> dict[str, Any]:
    return {
        "id": task.id,
        "version": task.version,
        "name": task.name,
        "description": task.description,
        "entry_point": task.entry_point,
        "inputs": [dict(item) for item in task.inputs],
        "outputs": [dict(item) for item in task.outputs],
        "semantics": dict(task.semantics),
        "correctness": dict(task.correctness),
        "benchmark": dict(task.benchmark),
        "restrictions": dict(task.restrictions),
        "public_cases": [case.to_dict() for case in task.public_cases],
    }


class QueueEvaluationBackend(Backend):
    """Submit one durable job per evaluation stage and wait for its result.

    ``experiment_id`` binds this adapter to one controller run.  Candidate IDs
    and round numbers are resolved from the controller's persisted candidate
    records, so public rounds and the copied final-best candidate both retain
    their original identity without trusting path names supplied by a worker.

    Waiting timeouts are transport deadlines, not candidate-stage outcomes.
    They therefore raise :class:`QueueWaitTimeout`.  By default the durable job
    remains live for a later controller resume; callers may opt into cancelling
    jobs that have not yet been leased.
    """

    def __init__(
        self,
        queue: SQLiteEvaluationJobQueue,
        *,
        artifact_root: Path | str,
        experiment_id: str,
        wait_timeout_seconds: float
        | Mapping[EvaluationStage, float] = 1_800.0,
        poll_interval_seconds: float = 0.25,
        maximum_job_attempts: int = 3,
        priority: int = 0,
        cancel_on_timeout: bool = False,
        idempotency_namespace: str = "default",
        maximum_candidate_bytes: int = 4 * 1024 * 1024,
        maximum_payload_bytes: int = 8 * 1024 * 1024,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.queue = queue
        configured_root = Path(artifact_root).expanduser()
        if not configured_root.is_absolute():
            configured_root = configured_root.absolute()
        # Keep the lexical spelling as well as the canonical root.  macOS often
        # exposes /var paths whose canonical spelling starts with /private/var;
        # resolving an entire candidate path merely to handle that alias would
        # accidentally dereference (and thereby hide) a final symlink.
        self._lexical_artifact_root = configured_root
        self.artifact_root = configured_root.resolve()
        if not self.artifact_root.is_dir():
            raise ValueError("artifact_root must be an existing directory")
        self.experiment_id = _safe_segment(experiment_id, "experiment_id")
        self.poll_interval_seconds = _positive_finite(
            poll_interval_seconds, "poll_interval_seconds"
        )
        if maximum_job_attempts < 1:
            raise ValueError("maximum_job_attempts must be positive")
        if maximum_candidate_bytes < 1 or maximum_payload_bytes < 1:
            raise ValueError("candidate and payload byte limits must be positive")
        _identifier(idempotency_namespace, "idempotency_namespace")
        self.maximum_job_attempts = maximum_job_attempts
        self.priority = int(priority)
        self.cancel_on_timeout = bool(cancel_on_timeout)
        self.idempotency_namespace = idempotency_namespace
        self.maximum_candidate_bytes = maximum_candidate_bytes
        self.maximum_payload_bytes = maximum_payload_bytes
        self._monotonic = monotonic
        self._sleeper = sleeper
        if isinstance(wait_timeout_seconds, Mapping):
            self._stage_timeouts = {
                EvaluationStage(stage): _positive_finite(value, f"timeout for {stage}")
                for stage, value in wait_timeout_seconds.items()
            }
            self._default_timeout = 1_800.0
        else:
            self._stage_timeouts = {}
            self._default_timeout = _positive_finite(
                wait_timeout_seconds, "wait_timeout_seconds"
            )

    def _timeout(self, stage: EvaluationStage) -> float:
        return self._stage_timeouts.get(stage, self._default_timeout)

    def _reject_symlink_components(self, relative: PurePosixPath) -> None:
        current = self.artifact_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"artifact path contains a symlink component: {relative}")

    def _relative_path(
        self,
        path: Path | str,
        *,
        name: str,
        require_file: bool = False,
        require_directory_if_present: bool = False,
    ) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            try:
                relative_path = candidate.relative_to(self.artifact_root)
            except ValueError:
                try:
                    relative_path = candidate.relative_to(self._lexical_artifact_root)
                except ValueError as exc:
                    raise ValueError(f"{name} escapes artifact_root") from exc
        else:
            relative_path = candidate
        raw = relative_path.as_posix()
        if (
            not raw
            or raw == "."
            or "\x00" in raw
            or "\\" in raw
            or len(raw.encode("utf-8")) > 4096
        ):
            raise ValueError(f"{name} is not a safe relative artifact path")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError(f"{name} is not normalized below artifact_root")
        expected_prefix = PurePosixPath(self.experiment_id, "tasks")
        if relative.parts[: len(expected_prefix.parts)] != expected_prefix.parts:
            raise ValueError(
                f"{name} must remain below the bound experiment task directory"
            )
        self._reject_symlink_components(relative)
        target = self.artifact_root.joinpath(*relative.parts)
        try:
            target.resolve(strict=False).relative_to(self.artifact_root)
        except ValueError as exc:
            raise ValueError(f"{name} resolves outside artifact_root") from exc
        if require_file and (target.is_symlink() or not target.is_file()):
            raise ValueError(f"{name} must be a regular non-symlink file")
        if (
            require_directory_if_present
            and target.exists()
            and (target.is_symlink() or not target.is_dir())
        ):
            raise ValueError(f"{name} must be a directory when it exists")
        return relative.as_posix()

    def _candidate_digest(self, candidate_path: Path, relative_path: str) -> str:
        target = self.artifact_root.joinpath(*PurePosixPath(relative_path).parts)
        before = target.stat()
        if before.st_size > self.maximum_candidate_bytes:
            raise ValueError("candidate exceeds maximum_candidate_bytes")
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > self.maximum_candidate_bytes:
                    raise ValueError("candidate exceeds maximum_candidate_bytes")
                digest.update(chunk)
        after = target.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise QueueBackendConfigurationError(
                f"candidate changed while it was being hashed: {candidate_path}"
            )
        return digest.hexdigest()

    def _resolve_identity(
        self,
        *,
        task: TaskSpec,
        candidate_relative: str,
        candidate_sha256: str,
    ) -> tuple[int, str]:
        task_digest = task.digest()
        with self.queue.database.connection() as connection:
            task_row = connection.execute(
                """
                SELECT task_version, task_spec_sha256, best_candidate_id
                FROM tasks WHERE experiment_id = ? AND task_id = ?
                """,
                (self.experiment_id, task.id),
            ).fetchone()
            if task_row is None:
                raise QueueBackendConfigurationError(
                    "task must be persisted before submitting evaluation work"
                )
            if (
                int(task_row["task_version"]) != task.version
                or str(task_row["task_spec_sha256"] or "") != task_digest
            ):
                raise QueueBackendConfigurationError(
                    "persisted task version/digest does not match the submitted task"
                )
            row = connection.execute(
                """
                SELECT candidate_id, round_number, source_sha256
                FROM candidates
                WHERE experiment_id = ? AND task_id = ?
                  AND source_artifact_path = ?
                """,
                (self.experiment_id, task.id, candidate_relative),
            ).fetchone()
            if row is None and task_row["best_candidate_id"] is not None:
                row = connection.execute(
                    """
                    SELECT candidate_id, round_number, source_sha256
                    FROM candidates
                    WHERE experiment_id = ? AND task_id = ? AND candidate_id = ?
                    """,
                    (self.experiment_id, task.id, task_row["best_candidate_id"]),
                ).fetchone()
        if row is None:
            raise QueueBackendConfigurationError(
                "candidate must be registered (or selected as best) before evaluation"
            )
        if str(row["source_sha256"]) != candidate_sha256:
            raise QueueBackendConfigurationError(
                "candidate bytes do not match the immutable persisted digest"
            )
        round_number = int(row["round_number"])
        if round_number < 1:
            raise QueueProtocolError("persisted candidate has an invalid round number")
        return round_number, str(row["candidate_id"])

    @staticmethod
    def _validate_cases(stage: EvaluationStage, cases: Sequence[CaseSpec]) -> None:
        if stage is EvaluationStage.SOURCE_CHECK:
            if cases:
                raise ValueError("SOURCE_CHECK must not carry cases")
            return
        if not cases:
            raise ValueError(f"{stage.value} requires at least one case")
        identifiers = [case.id for case in cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evaluation cases must have unique IDs")
        if stage in {EvaluationStage.COMPILE, EvaluationStage.CORRECTNESS} and any(
            case.kind != "correctness" for case in cases
        ):
            raise ValueError(f"{stage.value} accepts only correctness cases")
        if stage is EvaluationStage.FULL_EVALUATION:
            if not any(case.kind == "correctness" for case in cases) or not any(
                case.kind == "benchmark" for case in cases
            ):
                raise ValueError(
                    "FULL_EVALUATION requires correctness and benchmark cases"
                )
            return
        if stage is EvaluationStage.BENCHMARK and any(
            case.kind != "benchmark" for case in cases
        ):
            raise ValueError("BENCHMARK accepts only benchmark cases")
        if stage is EvaluationStage.PROFILE and any(
            case.kind not in {"profile", "benchmark"} for case in cases
        ):
            raise ValueError("PROFILE accepts only profile or benchmark cases")

    def _hidden_case_set(
        self,
        stage: EvaluationStage,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
    ) -> dict[str, Any] | None:
        """Describe a hidden suite without serializing any derived case data."""

        hidden = [case.id.startswith(_HIDDEN_CASE_ID_PREFIX) for case in cases]
        if not any(hidden):
            return None
        if not all(hidden):
            raise ValueError("public and hidden cases must never share one queue job")
        if stage not in {
            EvaluationStage.COMPILE,
            EvaluationStage.CORRECTNESS,
            EvaluationStage.BENCHMARK,
            EvaluationStage.PROFILE,
        }:
            raise ValueError(f"{stage.value} does not accept a hidden case set")
        if stage in {EvaluationStage.COMPILE, EvaluationStage.CORRECTNESS}:
            kind = "correctness"
            expected_prefix = "hidden_correctness_"
            maximum = 20
        elif stage is EvaluationStage.BENCHMARK:
            kind = "benchmark"
            expected_prefix = "hidden_benchmark_"
            maximum = 6
        else:
            kind = "profile"
            expected_prefix = "hidden_benchmark_"
            maximum = 6
        if any(not case.id.startswith(expected_prefix) for case in cases):
            raise ValueError(
                f"hidden case IDs do not match the {stage.value} suite"
            )
        if len(cases) > maximum:
            raise ValueError(
                f"hidden {kind} count exceeds the trusted {_HIDDEN_GENERATOR} suite"
            )
        return {
            "visibility": "hidden",
            "generator": _HIDDEN_GENERATOR,
            "kind": kind,
            "count": len(cases),
            "suite_commitment": hidden_suite_commitment(
                cases,
                experiment_id=self.experiment_id,
                task_id=task.id,
                generator=_HIDDEN_GENERATOR,
                kind=kind,
            ),
        }

    @staticmethod
    def _detached_json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
        try:
            encoded = canonical_json_dumps(value)
            detached = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{name} must be a strict JSON object: {exc}") from exc
        if not isinstance(detached, dict):  # pragma: no cover - Mapping always encodes object
            raise ValueError(f"{name} must be a JSON object")
        return detached

    def _payload(
        self,
        *,
        stage: EvaluationStage,
        task: TaskSpec,
        candidate_relative: str,
        candidate_sha256: str,
        artifact_relative: str,
        cases: Sequence[CaseSpec],
        baseline_snapshot: Mapping[str, Any] | None,
        benchmark_settings: Mapping[str, Any] | None = None,
        incumbent_relative: str | None = None,
        incumbent_sha256: str | None = None,
    ) -> dict[str, Any]:
        hidden_case_set = self._hidden_case_set(stage, task, cases)
        payload: dict[str, Any] = {
            "protocol_version": JOB_PROTOCOL_VERSION,
            "harness_digest": harness_digest(),
            "path_mode": PATH_MODE,
            "candidate_path": candidate_relative,
            "artifact_dir": artifact_relative,
            "candidate_sha256": candidate_sha256,
            "task_digest": task.digest(),
            "task_bundle_digest": task.bundle_digest(),
            "task_version": task.version,
            "task": _task_snapshot(task),
        }
        if hidden_case_set is None:
            payload["cases"] = [case.to_dict() for case in cases]
        else:
            payload["case_set"] = hidden_case_set
        if baseline_snapshot is not None:
            if not isinstance(baseline_snapshot, Mapping):
                raise TypeError("baseline_snapshot must be a mapping or None")
            payload["baseline_snapshot"] = dict(baseline_snapshot)
        if benchmark_settings is not None:
            payload["benchmark_settings"] = dict(benchmark_settings)
        if incumbent_relative is not None:
            payload["incumbent_path"] = incumbent_relative
            payload["incumbent_sha256"] = incumbent_sha256
        payload = self._detached_json_object(payload, "evaluation payload")
        if len(canonical_json_dumps(payload).encode("utf-8")) > self.maximum_payload_bytes:
            raise ValueError("evaluation payload exceeds maximum_payload_bytes")
        return payload

    def _make_job(
        self,
        *,
        stage: EvaluationStage,
        task: TaskSpec,
        candidate_path: Path,
        artifact_dir: Path,
        cases: Sequence[CaseSpec],
        baseline_snapshot: Mapping[str, Any] | None = None,
        benchmark_settings: Mapping[str, Any] | None = None,
        incumbent_path: Path | None = None,
    ) -> EvaluationJob:
        self._validate_cases(stage, cases)
        candidate_relative = self._relative_path(
            candidate_path, name="candidate_path", require_file=True
        )
        expected_task_prefix = PurePosixPath(self.experiment_id, "tasks", task.id)
        candidate_parts = PurePosixPath(candidate_relative).parts
        if candidate_parts[: len(expected_task_prefix.parts)] != expected_task_prefix.parts:
            raise ValueError("candidate_path does not belong to the submitted task")
        artifact_relative = self._relative_path(
            artifact_dir,
            name="artifact_dir",
            require_directory_if_present=True,
        )
        artifact_parts = PurePosixPath(artifact_relative).parts
        if artifact_parts[: len(expected_task_prefix.parts)] != expected_task_prefix.parts:
            raise ValueError("artifact_dir does not belong to the submitted task")
        candidate_sha256 = self._candidate_digest(candidate_path, candidate_relative)
        incumbent_relative = None
        incumbent_sha256 = None
        if incumbent_path is not None:
            incumbent_relative = self._relative_path(
                incumbent_path, name="incumbent_path", require_file=True
            )
            incumbent_sha256 = self._candidate_digest(
                incumbent_path, incumbent_relative
            )
        round_number, candidate_id = self._resolve_identity(
            task=task,
            candidate_relative=candidate_relative,
            candidate_sha256=candidate_sha256,
        )
        payload = self._payload(
            stage=stage,
            task=task,
            candidate_relative=candidate_relative,
            candidate_sha256=candidate_sha256,
            artifact_relative=artifact_relative,
            cases=cases,
            baseline_snapshot=baseline_snapshot,
            benchmark_settings=benchmark_settings,
            incumbent_relative=incumbent_relative,
            incumbent_sha256=incumbent_sha256,
        )
        request_identity = {
            "protocol_version": JOB_PROTOCOL_VERSION,
            "namespace": self.idempotency_namespace,
            "experiment_id": self.experiment_id,
            "task_id": task.id,
            "round_number": round_number,
            "candidate_id": candidate_id,
            "stage": stage.value,
            "payload": payload,
            "priority": self.priority,
            "maximum_job_attempts": self.maximum_job_attempts,
        }
        digest = hashlib.sha256(
            canonical_json_dumps(request_identity).encode("utf-8")
        ).hexdigest()
        return EvaluationJob(
            job_id=f"eval-{digest}",
            experiment_id=self.experiment_id,
            task_id=task.id,
            round_number=round_number,
            candidate_id=candidate_id,
            stage=stage,
            payload=payload,
            priority=self.priority,
            max_attempts=self.maximum_job_attempts,
            available_at=_READY_AT,
            idempotency_key=f"akg-eval-v1:{digest}",
        )

    def _validate_result_artifact(self, raw: str) -> None:
        if not raw or "\x00" in raw or "\\" in raw:
            raise QueueProtocolError("worker result contains an unsafe artifact path")
        path = Path(raw)
        if any(part in {"", ".", ".."} for part in path.parts):
            raise QueueProtocolError("worker result artifact path is not normalized")
        absolute = path if path.is_absolute() else self.artifact_root / path
        try:
            relative = absolute.relative_to(self.artifact_root)
        except ValueError as exc:
            try:
                relative = absolute.relative_to(self._lexical_artifact_root)
            except ValueError:
                raise QueueProtocolError(
                    "worker result artifact path escapes artifact_root"
                ) from exc
        relative_posix = PurePosixPath(relative.as_posix())
        self._reject_symlink_components(relative_posix)
        try:
            absolute.resolve(strict=False).relative_to(self.artifact_root)
        except ValueError as exc:
            raise QueueProtocolError(
                "worker result artifact resolves outside artifact_root"
            ) from exc
        if not absolute.exists():
            raise QueueProtocolError("worker result references a missing artifact")

    def _parse_result(
        self,
        stored: StoredEvaluationJob,
        expected_stage: EvaluationStage,
    ) -> StageResult:
        value = stored.result
        if value is None:
            raise QueueProtocolError(
                f"successful evaluation job {stored.job_id!r} has no result"
            )
        keys = set(value)
        if keys != _RESULT_KEYS:
            raise QueueProtocolError(
                "worker stage result keys disagree with protocol; "
                f"missing={sorted(_RESULT_KEYS - keys)}, unknown={sorted(keys - _RESULT_KEYS)}"
            )
        if value.get("schema_version") != RESULT_PROTOCOL_VERSION:
            raise QueueProtocolError("unsupported worker stage-result schema version")
        if value.get("stage") != expected_stage.value:
            raise QueueProtocolError(
                f"worker result stage mismatch: expected {expected_stage.value}, "
                f"got {value.get('stage')!r}"
            )
        if not isinstance(value.get("passed"), bool):
            raise QueueProtocolError("worker result passed field must be boolean")
        if not isinstance(value.get("retryable"), bool):
            raise QueueProtocolError("worker result retryable field must be boolean")
        details = value.get("details")
        artifacts = value.get("artifacts")
        error = value.get("error")
        if not isinstance(details, Mapping) or not isinstance(artifacts, Mapping):
            raise QueueProtocolError("worker result details/artifacts must be objects")
        if error is not None and not isinstance(error, Mapping):
            raise QueueProtocolError("worker result error must be an object or null")
        if any(not isinstance(key, str) or not isinstance(item, str) for key, item in artifacts.items()):
            raise QueueProtocolError("worker result artifact map must contain string pairs")
        if "case_set" in stored.payload:
            if artifacts:
                raise QueueProtocolError(
                    "hidden worker result must not expose persistent artifacts"
                )

            def inspect_private(item: Any) -> None:
                if isinstance(item, Mapping):
                    for key, child in item.items():
                        if str(key).lower() in _PRIVATE_RESULT_FORBIDDEN_KEYS:
                            raise QueueProtocolError(
                                "hidden worker result contains private case metadata"
                            )
                        inspect_private(child)
                elif isinstance(item, Sequence) and not isinstance(
                    item, (str, bytes, bytearray)
                ):
                    for child in item:
                        inspect_private(child)

            inspect_private(details)
            inspect_private(error)
        for artifact in artifacts.values():
            self._validate_result_artifact(artifact)
        try:
            evidence = validate_artifact_map(
                artifacts,
                artifact_root=self.artifact_root,
            )
        except EvidenceIntegrityError as exc:
            raise QueueProtocolError(f"worker evidence integrity failed: {exc}") from exc
        if (
            "case_set" not in stored.payload
            and value.get("passed") is True
            and expected_stage is not EvaluationStage.SOURCE_CHECK
            and evidence is None
        ):
            raise QueueProtocolError(
                "passed public worker stage has no content-addressed evidence manifest"
            )
        duration = value.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise QueueProtocolError("worker result duration_seconds must be numeric")
        duration_float = float(duration)
        if not math.isfinite(duration_float) or duration_float < 0:
            raise QueueProtocolError("worker result duration_seconds is invalid")
        try:
            result = StageResult.from_dict(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise QueueProtocolError(f"invalid worker stage result: {exc}") from exc
        if not isinstance(result.stage, EvaluationStage) or result.stage is not expected_stage:
            raise QueueProtocolError("worker result stage was not a known expected stage")
        if bool(value["passed"]) is not (result.status is StageStatus.PASS):
            raise QueueProtocolError("worker result passed flag disagrees with status")
        if not math.isclose(
            duration_float,
            result.duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise QueueProtocolError("worker result duration disagrees with timestamps")
        return result

    def _terminal_result(
        self,
        stored: StoredEvaluationJob,
        expected_stage: EvaluationStage,
    ) -> StageResult:
        if stored.status is JobStatus.SUCCEEDED:
            return self._parse_result(stored, expected_stage)
        terminal_error = stored.last_error
        if "case_set" in stored.payload and terminal_error is not None:
            terminal_error = {
                "type": "HiddenWorkerFailure",
                "message": "hidden worker job failed; diagnostic details redacted",
            }
        if stored.status is JobStatus.CANCELLED:
            raise QueueJobCancelled(
                stored.job_id,
                status=stored.status,
                error=terminal_error,
            )
        if stored.status is JobStatus.DEAD:
            if (
                terminal_error is not None
                and terminal_error.get("type") == "StageTimeout"
            ):
                # Older workers dead-lettered a valid timeout StageResult after
                # its final retry.  Materialize that known stage outcome so a
                # durable Controller resume can finish the round and retain the
                # timeout in its trajectory.  Other dead letters remain hard
                # transport failures.
                return StageResult(
                    stage=expected_stage,
                    status=StageStatus.TIMEOUT,
                    started_at=stored.updated_at,
                    finished_at=stored.updated_at,
                    details={
                        "recovered_from_dead": True,
                        "attempt_count": stored.attempt_count,
                        "max_attempts": stored.max_attempts,
                        "queue_status": stored.status.value,
                        "failure_origin": "infrastructure",
                        "failure_type": "StageTimeout",
                    },
                    error={str(key): value for key, value in terminal_error.items()},
                    retryable=False,
                )
            raise QueueJobFailed(
                stored.job_id,
                status=stored.status,
                error=terminal_error,
            )
        raise QueueProtocolError(
            f"job {stored.job_id!r} was treated as terminal in {stored.status.value}"
        )

    def _wait(self, job: EvaluationJob) -> StageResult:
        stored = self.queue.enqueue(job)
        timeout = self._timeout(job.stage)
        deadline = self._monotonic() + timeout
        while True:
            if stored.is_terminal:
                return self._terminal_result(stored, job.stage)
            # A controller can make progress even if the worker that held the
            # lease disappeared and no replacement worker has claimed yet.
            self.queue.sweep_expired_leases()
            refreshed_after_sweep = self.queue.get(job.job_id)
            if refreshed_after_sweep is None:
                raise QueueProtocolError(
                    f"evaluation job {job.job_id!r} disappeared from durable storage"
                )
            stored = refreshed_after_sweep
            if stored.is_terminal:
                return self._terminal_result(stored, job.stage)
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                # Check and cancel by ID under queue transactions.  A concurrent
                # completion wins safely; leased work cannot be cancelled by a
                # controller that does not own its fencing token.
                cancelled = self.queue.cancel(job.job_id) if self.cancel_on_timeout else False
                latest = self.queue.get(job.job_id)
                if latest is None:
                    raise QueueProtocolError(
                        f"evaluation job {job.job_id!r} disappeared during timeout handling"
                    )
                if latest.status is JobStatus.SUCCEEDED:
                    return self._parse_result(latest, job.stage)
                raise QueueWaitTimeout(
                    job.job_id,
                    timeout_seconds=timeout,
                    cancelled=cancelled,
                )
            self._sleeper(min(self.poll_interval_seconds, remaining))
            refreshed = self.queue.get(job.job_id)
            if refreshed is None:
                raise QueueProtocolError(
                    f"evaluation job {job.job_id!r} disappeared from durable storage"
                )
            stored = refreshed

    def _run(
        self,
        stage: EvaluationStage,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
        benchmark_settings: Mapping[str, Any] | None = None,
        incumbent_path: Path | None = None,
    ) -> StageResult:
        return self._wait(
            self._make_job(
                stage=stage,
                task=task,
                candidate_path=candidate_path,
                artifact_dir=artifact_dir,
                cases=cases,
                baseline_snapshot=baseline_snapshot,
                benchmark_settings=benchmark_settings,
                incumbent_path=incumbent_path,
            )
        )

    def source_check(self, candidate_path: Path, task: TaskSpec) -> StageResult:
        return self._run(
            EvaluationStage.SOURCE_CHECK,
            candidate_path,
            task,
            (),
            Path(candidate_path).parent,
        )

    def compile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._run(
            EvaluationStage.COMPILE, candidate_path, task, cases, artifact_dir
        )

    def check_correctness(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._run(
            EvaluationStage.CORRECTNESS,
            candidate_path,
            task,
            cases,
            artifact_dir,
        )

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
        benchmark_settings: Mapping[str, Any] | None = None,
    ) -> StageResult:
        return self._run(
            EvaluationStage.BENCHMARK,
            candidate_path,
            task,
            cases,
            artifact_dir,
            baseline_snapshot,
            benchmark_settings,
        )

    def candidate_evaluation(
        self,
        candidate_path: Path,
        task: TaskSpec,
        correctness_cases: Sequence[CaseSpec],
        benchmark_cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
        benchmark_settings: Mapping[str, Any] | None = None,
        incumbent_path: Path | None = None,
    ) -> StageResult:
        return self._run(
            EvaluationStage.FULL_EVALUATION,
            candidate_path,
            task,
            (*correctness_cases, *benchmark_cases),
            artifact_dir,
            baseline_snapshot,
            benchmark_settings,
            incumbent_path,
        )

    def profile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._run(
            EvaluationStage.PROFILE, candidate_path, task, cases, artifact_dir
        )

    def health_check(self) -> StageResult:
        raise NotImplementedError(
            "worker health is queried independently; it is not candidate evaluation work"
        )

    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]:
        del task, cases, artifact_dir
        raise NotImplementedError(
            "baseline measurement uses a dedicated durable baseline workflow"
        )


__all__ = [
    "JOB_PROTOCOL_VERSION",
    "PATH_MODE",
    "RESULT_PROTOCOL_VERSION",
    "QueueEvaluationBackend",
]
