"""Durable SQLite worker with lease heartbeats and strict payload validation."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import secrets
import shutil
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from ascend_kernel_lab.backend import Backend, StageResult, StageStatus
from ascend_kernel_lab.domain import EvaluationStage, LeasedEvaluationJob, LeaseLostError
from ascend_kernel_lab.protocol import EVALUATION_JOB_PROTOCOL, harness_digest
from ascend_kernel_lab.storage import SQLiteEvaluationJobQueue, canonical_json_dumps
from ascend_kernel_lab.storage.permissions import (
    SHARED_DIRECTORY_MODE,
    ensure_shared_directory,
)
from ascend_kernel_lab.tasks import CaseSpec, TaskRegistry, TaskSpec
from ascend_kernel_lab.tasks.runtime import (
    hidden_cases_from_template,
    hidden_suite_commitment,
    validate_hidden_seed,
)


class WorkerPayloadError(ValueError):
    """A non-retryable queue payload or artifact integrity violation."""


_PAYLOAD_KEYS = frozenset(
    {
        "protocol_version",
        "harness_digest",
        "path_mode",
        "candidate_path",
        "artifact_dir",
        "candidate_sha256",
        "task_digest",
        "task_bundle_digest",
        "task_version",
        "task",
        "cases",
        "case_set",
        "baseline_snapshot",
        "benchmark_settings",
        "incumbent_path",
        "incumbent_sha256",
        "run_profile",
        "profile_coverage_required",
        "minimum_kernel_coverage",
        "hidden",
    }
)
_REQUIRED_PAYLOAD_KEYS = frozenset(
    {
        "protocol_version",
        "harness_digest",
        "path_mode",
        "candidate_path",
        "artifact_dir",
        "candidate_sha256",
        "task_digest",
        "task_bundle_digest",
        "task_version",
        "task",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class _ValidatedJob:
    task: TaskSpec
    candidate_path: Path
    artifact_dir: Path
    candidate_sha256: str
    cases: tuple[CaseSpec, ...]
    baseline_snapshot: Mapping[str, Any] | None
    benchmark_settings: Mapping[str, Any] | None
    incumbent_path: Path | None
    incumbent_sha256: str | None
    run_profile: bool
    profile_coverage_required: bool
    minimum_kernel_coverage: float
    hidden: bool
    private_cases: bool


class _Heartbeat:
    def __init__(
        self,
        *,
        queue: SQLiteEvaluationJobQueue,
        job: LeasedEvaluationJob,
        worker_id: str,
        heartbeat_seconds: float,
        lease_seconds: float,
        on_lost: Any,
    ) -> None:
        self.queue = queue
        self.job = job
        self.worker_id = worker_id
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds
        self.on_lost = on_lost
        self.stop_event = threading.Event()
        self.lost_error: BaseException | None = None
        self._finalize_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run,
            name=f"lease-heartbeat-{job.job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.heartbeat_seconds):
            try:
                with self._finalize_lock:
                    if self.stop_event.is_set():
                        return
                    self.queue.heartbeat(
                        self.job.job_id,
                        self.worker_id,
                        self.job.lease_token,
                        lease_seconds=self.lease_seconds,
                    )
            except BaseException as exc:
                self.lost_error = exc
                self.stop_event.set()
                with contextlib.suppress(BaseException):
                    self.on_lost()
                return

    def finalize(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.heartbeat_seconds * 2))
        if self.thread.is_alive():
            raise RuntimeError("worker lease heartbeat thread did not stop")
        if self.lost_error is not None:
            raise LeaseLostError(
                f"lease was lost while evaluating {self.job.job_id}: {self.lost_error}"
            )
        with self._finalize_lock:
            self.queue.heartbeat(
                self.job.job_id,
                self.worker_id,
                self.job.lease_token,
                lease_seconds=self.lease_seconds,
            )


class WorkerService:
    """Claim durable jobs, maintain leases, and execute backend stages."""

    def __init__(
        self,
        queue: SQLiteEvaluationJobQueue,
        backend: Backend,
        registry: TaskRegistry,
        artifact_root: Path,
        *,
        worker_id: str,
        lease_seconds: float = 120.0,
        heartbeat_seconds: float = 10.0,
        poll_seconds: float = 1.0,
        retry_delay_seconds: float = 5.0,
        hidden_seed_env: str = "AKG_HIDDEN_SEED",
        allow_insecure_hidden_seed_for_testing: bool = False,
    ) -> None:
        if not worker_id.strip() or "\x00" in worker_id:
            raise ValueError("worker_id must be non-empty and NUL-free")
        self.queue = queue
        self.backend = backend
        self.registry = registry
        root = Path(artifact_root)
        if root.exists() and root.is_symlink():
            raise ValueError("artifact_root must not be a symlink")
        ensure_shared_directory(root)
        self.artifact_root = root.resolve()
        self.worker_id = worker_id
        self.lease_seconds = _finite_positive(lease_seconds, "lease_seconds")
        self.heartbeat_seconds = _finite_positive(
            heartbeat_seconds, "heartbeat_seconds"
        )
        self.poll_seconds = _finite_positive(poll_seconds, "poll_seconds")
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        self.retry_delay_seconds = float(retry_delay_seconds)
        if not hidden_seed_env or not hidden_seed_env.replace("_", "A").isalnum():
            raise ValueError("hidden_seed_env must be a non-empty environment name")
        self.hidden_seed_env = hidden_seed_env
        self.allow_insecure_hidden_seed_for_testing = bool(
            allow_insecure_hidden_seed_for_testing
        )
        if self.lease_seconds < self.heartbeat_seconds * 3:
            raise ValueError("lease_seconds must be at least three heartbeat intervals")
        self._stop_event = threading.Event()
        self._quarantine_lock = threading.Lock()
        self._quarantined = False
        self._quarantine_reason: Mapping[str, Any] | None = None

    @property
    def quarantined(self) -> bool:
        with self._quarantine_lock:
            return self._quarantined

    @property
    def quarantine_reason(self) -> Mapping[str, Any] | None:
        with self._quarantine_lock:
            return self._quarantine_reason

    def _quarantine(self, health_status: str) -> dict[str, Any]:
        reason = {
            "type": "DeviceQuarantined",
            "message": "post-failure device health check did not pass",
            "health_status": health_status,
        }
        with self._quarantine_lock:
            self._quarantined = True
            self._quarantine_reason = MappingProxyType(dict(reason))
        return reason

    def stop(self) -> None:
        self._stop_event.set()
        cancel = getattr(self.backend, "cancel_current", None)
        if callable(cancel):
            cancel()

    @staticmethod
    def _requires_health_check(value: Any) -> bool:
        if isinstance(value, Mapping):
            status = str(value.get("status", "")).lower()
            if status in {StageStatus.ERROR.value, StageStatus.TIMEOUT.value}:
                return True
            for key in ("output_limit_exceeded", "terminated", "timed_out"):
                if value.get(key) is True:
                    return True
            return any(WorkerService._requires_health_check(child) for child in value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return any(WorkerService._requires_health_check(child) for child in value)
        return False

    def _post_failure_health_error(
        self, result: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if not self._requires_health_check(result):
            return None
        try:
            health = self.backend.health_check()
        except BaseException:
            return self._quarantine("error")
        if health.status is StageStatus.PASS:
            return None
        return self._quarantine(health.status.value)

    @staticmethod
    def _relative(value: Any, name: str) -> PurePosixPath:
        if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
            raise WorkerPayloadError(f"{name} must be a non-empty POSIX relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or path.as_posix() != value or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise WorkerPayloadError(f"{name} is not a canonical relative path")
        return path

    def _resolve_existing_file(self, value: Any, name: str) -> Path:
        relative = self._relative(value, name)
        lexical = self.artifact_root.joinpath(*relative.parts)
        if lexical.is_symlink() or not lexical.is_file():
            raise WorkerPayloadError(f"{name} must identify a regular non-symlink file")
        resolved = lexical.resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise WorkerPayloadError(f"{name} escapes artifact_root")
        current = self.artifact_root
        for part in relative.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise WorkerPayloadError(f"{name} traverses a symlink")
        return resolved

    def _resolve_output_dir(self, value: Any, job: LeasedEvaluationJob) -> Path:
        relative = self._relative(value, "artifact_dir")
        base = self.artifact_root
        for part in relative.parts:
            base /= part
            if base.exists() and base.is_symlink():
                raise WorkerPayloadError("artifact_dir traverses a symlink")
        resolved_parent = base.resolve(strict=False)
        if not resolved_parent.is_relative_to(self.artifact_root):
            raise WorkerPayloadError("artifact_dir escapes artifact_root")
        job_digest = hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()
        try:
            ensure_shared_directory(base)
            worker_jobs = ensure_shared_directory(base / "worker_jobs")
            job_root = ensure_shared_directory(worker_jobs / job_digest)
        except ValueError as exc:
            raise WorkerPayloadError(f"unsafe shared artifact directory: {exc}") from exc
        attempt = job_root / f"attempt_{job.attempt_count:03d}"
        try:
            attempt.mkdir(mode=SHARED_DIRECTORY_MODE, exist_ok=False)
            attempt.chmod(SHARED_DIRECTORY_MODE)
            ensure_shared_directory(job_root)
            ensure_shared_directory(worker_jobs)
            ensure_shared_directory(base)
        except FileExistsError:
            raise WorkerPayloadError("worker attempt directory already exists") from None
        resolved = attempt.resolve()
        if not resolved.is_relative_to(self.artifact_root):
            raise WorkerPayloadError("worker attempt directory escapes artifact_root")
        return resolved

    @staticmethod
    def _parse_cases(value: Any) -> tuple[CaseSpec, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise WorkerPayloadError("cases must be an array")
        cases: list[CaseSpec] = []
        seen: set[str] = set()
        for raw in value:
            if not isinstance(raw, Mapping):
                raise WorkerPayloadError("each case must be an object")
            try:
                case = CaseSpec.from_dict(raw)
            except (TypeError, ValueError) as exc:
                raise WorkerPayloadError(f"invalid case: {exc}") from exc
            if case.id in seen:
                raise WorkerPayloadError(f"duplicate case ID: {case.id}")
            seen.add(case.id)
            cases.append(case)
        return tuple(cases)

    def _hidden_cases(
        self,
        task: TaskSpec,
        value: Any,
        *,
        experiment_id: str,
    ) -> tuple[CaseSpec, ...]:
        if not isinstance(value, Mapping) or set(value) != {
            "visibility",
            "generator",
            "kind",
            "count",
            "suite_commitment",
        }:
            raise WorkerPayloadError(
                "case_set contains invalid fields"
            )
        if value["visibility"] != "hidden":
            raise WorkerPayloadError("only hidden case_set descriptors are supported")
        if value["generator"] != "hidden-v1":
            raise WorkerPayloadError("hidden case_set generator is unsupported")
        kind = str(value["kind"])
        if kind not in {"correctness", "benchmark", "profile"}:
            raise WorkerPayloadError("hidden case_set kind is unsupported")
        try:
            count = int(value["count"])
        except (TypeError, ValueError) as exc:
            raise WorkerPayloadError("hidden case_set count must be an integer") from exc
        maximum = 20 if kind == "correctness" else 6
        if isinstance(value["count"], bool) or not 1 <= count <= maximum:
            raise WorkerPayloadError(
                f"hidden {kind} case_set count must be between 1 and {maximum}"
            )
        raw_seed = os.environ.get(self.hidden_seed_env)
        if raw_seed is None:
            raise WorkerPayloadError(
                f"hidden evaluation requires worker-only {self.hidden_seed_env}"
            )
        try:
            secret_seed = validate_hidden_seed(
                raw_seed,
                allow_insecure_for_testing=self.allow_insecure_hidden_seed_for_testing,
            )
        except ValueError as exc:
            raise WorkerPayloadError(
                "hidden evaluation secret does not meet deployment policy"
            ) from exc
        if task.root is None:
            raise WorkerPayloadError("hidden evaluation requires a registry-backed task root")
        derived = hidden_cases_from_template(
            task.root,
            secret_seed=secret_seed,
            count_correctness=20,
            count_benchmark=6,
        )
        derived_kind = "benchmark" if kind == "profile" else kind
        selected = tuple(case for case in derived if case.kind == derived_kind)[:count]
        if len(selected) != count:
            raise WorkerPayloadError("hidden case derivation returned an unexpected count")
        commitment = value.get("suite_commitment")
        if not isinstance(commitment, str) or not re_full_sha256(commitment):
            raise WorkerPayloadError("hidden suite commitment is invalid")
        expected_commitment = hidden_suite_commitment(
            selected,
            experiment_id=experiment_id,
            task_id=task.id,
            generator="hidden-v1",
            kind=kind,
        )
        if not secrets.compare_digest(commitment, expected_commitment):
            raise WorkerPayloadError("hidden suite commitment mismatch")
        return selected

    def _validate(self, job: LeasedEvaluationJob) -> _ValidatedJob:
        payload = job.payload
        unknown = set(payload) - _PAYLOAD_KEYS
        missing = _REQUIRED_PAYLOAD_KEYS - set(payload)
        if unknown or missing:
            raise WorkerPayloadError(
                f"invalid payload keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["protocol_version"] != EVALUATION_JOB_PROTOCOL:
            raise WorkerPayloadError("unsupported evaluation job protocol")
        requested_harness = str(payload["harness_digest"])
        if not re_full_sha256(requested_harness) or not secrets.compare_digest(
            requested_harness, harness_digest()
        ):
            raise WorkerPayloadError(
                "controller and worker harness releases do not match"
            )
        if payload["path_mode"] != "artifact_root_relative":
            raise WorkerPayloadError("worker accepts only artifact-root-relative paths")
        has_cases = "cases" in payload
        has_case_set = "case_set" in payload
        if has_cases == has_case_set:
            raise WorkerPayloadError("payload must contain exactly one of cases or case_set")
        task = self.registry.load(job.task_id)
        try:
            requested_version = int(payload["task_version"])
        except (TypeError, ValueError) as exc:
            raise WorkerPayloadError("task_version must be an integer") from exc
        if requested_version != task.version:
            raise WorkerPayloadError("task version differs from the worker registry")
        digest = str(payload["task_digest"])
        if digest != task.digest():
            raise WorkerPayloadError("task digest differs from the worker registry")
        bundle_digest = str(payload["task_bundle_digest"])
        if not re_full_sha256(bundle_digest) or not secrets.compare_digest(
            bundle_digest, task.bundle_digest()
        ):
            raise WorkerPayloadError(
                "trusted task bundle differs from the controller release"
            )
        snapshot = payload["task"]
        if not isinstance(snapshot, Mapping):
            raise WorkerPayloadError("task snapshot must be an object")
        expected_snapshot = {
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
        try:
            snapshot_json = canonical_json_dumps(snapshot)
        except (TypeError, ValueError) as exc:
            raise WorkerPayloadError(f"task snapshot is not strict JSON: {exc}") from exc
        if snapshot_json != canonical_json_dumps(expected_snapshot):
            raise WorkerPayloadError("task snapshot differs from the worker registry")
        candidate_path = self._resolve_existing_file(payload["candidate_path"], "candidate_path")
        requested_sha = str(payload["candidate_sha256"])
        if not re_full_sha256(requested_sha) or _sha256(candidate_path) != requested_sha:
            raise WorkerPayloadError("candidate SHA-256 mismatch")
        incumbent_path = None
        incumbent_sha = None
        if payload.get("incumbent_path") is not None:
            incumbent_path = self._resolve_existing_file(
                payload["incumbent_path"], "incumbent_path"
            )
            incumbent_sha = str(payload.get("incumbent_sha256", ""))
            if not re_full_sha256(incumbent_sha) or _sha256(
                incumbent_path
            ) != incumbent_sha:
                raise WorkerPayloadError("incumbent SHA-256 mismatch")
        private_cases = has_case_set
        if private_cases:
            raw_case_set = payload["case_set"]
            assert isinstance(raw_case_set, Mapping)
            expected_kind = {
                EvaluationStage.COMPILE: "correctness",
                EvaluationStage.CORRECTNESS: "correctness",
                EvaluationStage.FINAL_HIDDEN_CORRECTNESS: "correctness",
                EvaluationStage.BENCHMARK: "benchmark",
                EvaluationStage.FINAL_BENCHMARK: "benchmark",
                EvaluationStage.PROFILE: "profile",
                EvaluationStage.FINAL_FULL_PROFILE: "profile",
            }.get(job.stage)
            if expected_kind is None or raw_case_set.get("kind") != expected_kind:
                raise WorkerPayloadError(
                    f"hidden case_set kind does not match stage {job.stage.value}"
                )
        cases = (
            self._hidden_cases(
                task,
                payload["case_set"],
                experiment_id=job.experiment_id,
            )
            if private_cases
            else self._parse_cases(payload["cases"])
        )
        baseline = payload.get("baseline_snapshot")
        if baseline is not None and not isinstance(baseline, Mapping):
            raise WorkerPayloadError("baseline_snapshot must be an object or null")
        benchmark_settings = payload.get("benchmark_settings")
        if benchmark_settings is not None and not isinstance(
            benchmark_settings, Mapping
        ):
            raise WorkerPayloadError("benchmark_settings must be an object or null")
        minimum = float(payload.get("minimum_kernel_coverage", 0.90))
        if not math.isfinite(minimum) or not 0 <= minimum <= 1:
            raise WorkerPayloadError("minimum_kernel_coverage must be between zero and one")
        artifact_dir = self._resolve_output_dir(payload["artifact_dir"], job)
        return _ValidatedJob(
            task=task,
            candidate_path=candidate_path,
            artifact_dir=artifact_dir,
            candidate_sha256=requested_sha,
            cases=cases,
            baseline_snapshot=dict(baseline) if isinstance(baseline, Mapping) else None,
            benchmark_settings=(
                dict(benchmark_settings)
                if isinstance(benchmark_settings, Mapping)
                else None
            ),
            incumbent_path=incumbent_path,
            incumbent_sha256=incumbent_sha,
            run_profile=bool(payload.get("run_profile", True)),
            profile_coverage_required=bool(payload.get("profile_coverage_required", True)),
            minimum_kernel_coverage=minimum,
            hidden=private_cases or bool(payload.get("hidden", False)),
            private_cases=private_cases,
        )

    @staticmethod
    def _cases_for(
        stage: EvaluationStage, task: TaskSpec, supplied: tuple[CaseSpec, ...]
    ) -> tuple[CaseSpec, ...]:
        if supplied:
            return supplied
        if stage in {
            EvaluationStage.COMPILE,
            EvaluationStage.CORRECTNESS,
            EvaluationStage.FINAL_HIDDEN_CORRECTNESS,
        }:
            return task.correctness_cases
        if stage in {EvaluationStage.BENCHMARK, EvaluationStage.FINAL_BENCHMARK}:
            return task.benchmark_cases
        if stage in {EvaluationStage.PROFILE, EvaluationStage.FINAL_FULL_PROFILE}:
            return task.profile_cases
        return ()

    def _execute(self, job: LeasedEvaluationJob) -> Mapping[str, Any]:
        validated = self._validate(job)
        try:
            task = validated.task
            cases = self._cases_for(job.stage, task, validated.cases)
            if job.stage is EvaluationStage.SOURCE_CHECK:
                result = self.backend.source_check(validated.candidate_path, task)
            elif job.stage is EvaluationStage.COMPILE:
                result = self.backend.compile(
                    validated.candidate_path, task, cases, validated.artifact_dir
                )
            elif job.stage in {
                EvaluationStage.CORRECTNESS,
                EvaluationStage.FINAL_HIDDEN_CORRECTNESS,
            }:
                result = self.backend.check_correctness(
                    validated.candidate_path, task, cases, validated.artifact_dir
                )
            elif job.stage in {
                EvaluationStage.BENCHMARK,
                EvaluationStage.FINAL_BENCHMARK,
            }:
                result = self.backend.benchmark(
                    validated.candidate_path,
                    task,
                    cases,
                    validated.artifact_dir,
                    validated.baseline_snapshot,
                    validated.benchmark_settings,
                )
            elif job.stage in {
                EvaluationStage.PROFILE,
                EvaluationStage.FINAL_FULL_PROFILE,
            }:
                result = self.backend.profile(
                    validated.candidate_path, task, cases, validated.artifact_dir
                )
            elif job.stage is EvaluationStage.FULL_EVALUATION:
                correctness = (
                    tuple(case for case in cases if case.kind == "correctness")
                )
                benchmark = (
                    tuple(case for case in cases if case.kind == "benchmark")
                )
                result = self.backend.candidate_evaluation(
                    validated.candidate_path,
                    task,
                    correctness,
                    benchmark,
                    validated.artifact_dir,
                    validated.baseline_snapshot,
                    validated.benchmark_settings,
                    validated.incumbent_path,
                )
            else:  # pragma: no cover - enum exhaustiveness guard
                raise WorkerPayloadError(
                    f"worker does not support stage {job.stage.value}"
                )
            if _sha256(validated.candidate_path) != validated.candidate_sha256:
                raise WorkerPayloadError(
                    "candidate changed while it was being evaluated"
                )
            if (
                validated.incumbent_path is not None
                and _sha256(validated.incumbent_path)
                != validated.incumbent_sha256
            ):
                raise WorkerPayloadError(
                    "incumbent changed while it was being evaluated"
                )
            if result.stage != job.stage:
                result = StageResult(
                    stage=job.stage,
                    status=result.status,
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                    details=result.details,
                    artifacts=result.artifacts,
                    error=result.error,
                    retryable=result.retryable,
                )
            result_mapping = result.to_dict()
            return (
                _sanitize_hidden_mapping(result_mapping)
                if validated.private_cases
                else result_mapping
            )
        finally:
            if validated.private_cases:
                _remove_private_attempt(validated.artifact_dir)

    def _cancel_backend(self) -> None:
        cancel = getattr(self.backend, "cancel_current", None)
        if callable(cancel):
            cancel()

    def run_once(self) -> bool:
        if self.quarantined or self._stop_event.is_set():
            return False
        job = self.queue.claim(self.worker_id, lease_seconds=self.lease_seconds)
        if job is None:
            return False
        heartbeat = _Heartbeat(
            queue=self.queue,
            job=job,
            worker_id=self.worker_id,
            heartbeat_seconds=self.heartbeat_seconds,
            lease_seconds=self.lease_seconds,
            on_lost=self._cancel_backend,
        )
        heartbeat.start()
        result: Mapping[str, Any] | None = None
        error: dict[str, Any] | None = None
        retryable = False
        retryable_stage_result = False
        private_job = "case_set" in job.payload
        try:
            result = self._execute(job)
            quarantine_error = self._post_failure_health_error(result)
            if quarantine_error is not None:
                error = quarantine_error
                retryable = True
            else:
                status = str(result.get("status", ""))
                retryable = bool(result.get("retryable", False)) and status in {
                    StageStatus.ERROR.value,
                    StageStatus.TIMEOUT.value,
                }
                if retryable:
                    retryable_stage_result = True
                    raw_error = result.get("error")
                    error = (
                        {str(key): value for key, value in raw_error.items()}
                        if isinstance(raw_error, Mapping)
                        else {"type": "RetryableStageError", "message": status}
                    )
        except WorkerPayloadError as exc:
            error = (
                {
                    "type": "HiddenWorkerPayloadError",
                    "message": "hidden evaluation payload was rejected; details redacted",
                }
                if private_job
                else {"type": type(exc).__name__, "message": str(exc)}
            )
            retryable = False
        except BaseException as exc:
            error = (
                {
                    "type": "HiddenInfrastructureError",
                    "message": "hidden evaluation infrastructure failed; details redacted",
                }
                if private_job
                else {"type": type(exc).__name__, "message": str(exc)[:16_384]}
            )
            retryable = True
        try:
            heartbeat.finalize()
        except LeaseLostError:
            return True
        if error is not None:
            if retryable_stage_result and job.attempt_count >= job.max_attempts:
                assert result is not None
                # The worker transport succeeded and returned a valid stage
                # outcome.  Once retries are exhausted, persist that outcome
                # for the Controller instead of turning it into an opaque dead
                # letter that leaves the round without evaluation_result.json.
                self.queue.complete(
                    job.job_id,
                    self.worker_id,
                    job.lease_token,
                    result,
                )
            else:
                self.queue.fail(
                    job.job_id,
                    self.worker_id,
                    job.lease_token,
                    error,
                    retryable=retryable,
                    retry_delay_seconds=(
                        self.retry_delay_seconds if retryable else 0.0
                    ),
                )
        else:
            assert result is not None
            self.queue.complete(job.job_id, self.worker_id, job.lease_token, result)
        return True

    def serve_forever(
        self,
        stop_event: threading.Event | None = None,
        max_jobs: int | None = None,
    ) -> int:
        if max_jobs is not None and max_jobs < 0:
            raise ValueError("max_jobs must be non-negative")
        external_stop = stop_event or threading.Event()
        completed = 0
        while (
            not self._stop_event.is_set()
            and not external_stop.is_set()
            and not self.quarantined
        ):
            if max_jobs is not None and completed >= max_jobs:
                break
            if self.run_once():
                completed += 1
                continue
            wait_for = min(self.poll_seconds, 0.1) if max_jobs is not None else self.poll_seconds
            if self._stop_event.wait(wait_for) or external_stop.is_set():
                break
        return completed


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


_HIDDEN_FORBIDDEN_KEYS = frozenset(
    {
        "address_offset",
        "artifacts",
        "case_id",
        "case_results",
        "cases",
        "distribution",
        "dtype",
        "error",
        "maximum_error_flat_index",
        "measurement_attempts",
        "noncontiguous",
        "params",
        "per_case",
        "raw_samples_us",
        "seed",
        "shape",
        "source_files",
        "stderr_tail",
        "traceback",
        "executed_candidate_kernels",
        "kernel_name",
    }
)


def _sanitize_hidden_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively remove fields that could reveal hidden cases or candidate logs."""

    def clean(item: Any, key: str | None = None) -> Any:
        if key in _HIDDEN_FORBIDDEN_KEYS:
            return None
        if isinstance(item, Mapping):
            return {
                str(child_key): cleaned
                for child_key, child_value in item.items()
                if child_key not in _HIDDEN_FORBIDDEN_KEYS
                and (cleaned := clean(child_value, str(child_key))) is not None
            }
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [clean(child) for child in item]
        return item

    details = clean(value.get("details", {}))
    if not isinstance(details, dict):
        details = {}
    details["hidden_case_details_redacted"] = True
    raw_error = value.get("error")
    error = (
        None
        if raw_error is None
        else {
            "type": "HiddenStageError",
            "message": "hidden evaluation stage failed; diagnostic details redacted",
        }
    )
    return {
        "schema_version": value.get("schema_version"),
        "stage": value.get("stage"),
        "status": value.get("status"),
        "passed": value.get("passed"),
        "started_at": value.get("started_at"),
        "finished_at": value.get("finished_at"),
        "duration_seconds": value.get("duration_seconds"),
        "details": details,
        "artifacts": {},
        "error": error,
        "retryable": value.get("retryable", False),
    }


def _remove_private_attempt(path: Path) -> None:
    """Remove only the precise attempt directory created for this leased job."""

    candidate = Path(path)
    if candidate.name.startswith("attempt_") and candidate.parent.parent.name == "worker_jobs":
        shutil.rmtree(candidate)
