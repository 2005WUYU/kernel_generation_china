"""Real Triton-Ascend backend implemented through isolated stage processes."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.evaluation.benchmark import weighted_geometric_mean
from ascend_kernel_lab.evaluation.source_guard import (
    SourceGuard,
    candidate_kernel_pattern,
)
from ascend_kernel_lab.profiling import MsprofParser, MsprofRunner
from ascend_kernel_lab.storage.permissions import (
    SHARED_DIRECTORY_MODE,
    SHARED_FILE_MODE,
    ensure_shared_directory,
)
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec
from ascend_kernel_lab.worker.device_lock import DeviceLock, DeviceLockTimeout
from ascend_kernel_lab.worker.health import AscendHealthChecker
from ascend_kernel_lab.worker.stage_runner import StageProcessResult, StageRunner, clean_environment

from ..base import Backend, StageResult, StageStatus, finite_positive


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _serialized_task(task: TaskSpec) -> dict[str, Any]:
    return {
        "id": task.id,
        "version": task.version,
        "name": task.name,
        "description": task.description,
        "entry_point": task.entry_point,
        "inputs": [dict(value) for value in task.inputs],
        "outputs": [dict(value) for value in task.outputs],
        "semantics": dict(task.semantics),
        "correctness": dict(task.correctness),
        "benchmark": dict(task.benchmark),
        "restrictions": dict(task.restrictions),
    }


class AscendTritonBackend(Backend):
    """Compile and evaluate candidates on one exclusively locked Ascend device."""

    def __init__(
        self,
        *,
        device: str = "npu:0",
        python_executable: str = sys.executable,
        runner: StageRunner | None = None,
        source_guard: SourceGuard | None = None,
        msprof_runner: MsprofRunner | None = None,
        health_checker: AscendHealthChecker | None = None,
        lock_root: Path | str = "/tmp/ascend-kernel-lab-locks",
        lock_timeout_seconds: float = 600.0,
        compile_timeout_seconds: float = 180.0,
        correctness_case_timeout_seconds: float = 30.0,
        benchmark_timeout_seconds: float = 300.0,
        profile_timeout_seconds: float = 600.0,
        benchmark_settings: Mapping[str, Any] | None = None,
        profile_settings: Mapping[str, Any] | None = None,
        device_environment: Mapping[str, str] | None = None,
    ) -> None:
        if not re.fullmatch(r"npu:\d+", device):
            raise ValueError("device must use npu:<index> syntax")
        self.device = device
        self.device_index = device.removeprefix("npu:")
        self.python_executable = python_executable
        self.runner = runner or StageRunner()
        self.source_guard = source_guard or SourceGuard()
        self.msprof_runner = msprof_runner or MsprofRunner()
        self.health_checker = health_checker or AscendHealthChecker(
            runner=self.runner,
            python_executable=python_executable,
            device=device,
        )
        self.lock_root = Path(lock_root)
        self.lock_timeout_seconds = finite_positive(lock_timeout_seconds, "lock_timeout_seconds")
        self.compile_timeout_seconds = finite_positive(
            compile_timeout_seconds, "compile_timeout_seconds"
        )
        self.correctness_case_timeout_seconds = finite_positive(
            correctness_case_timeout_seconds, "correctness_case_timeout_seconds"
        )
        self.benchmark_timeout_seconds = finite_positive(
            benchmark_timeout_seconds, "benchmark_timeout_seconds"
        )
        self.profile_timeout_seconds = finite_positive(
            profile_timeout_seconds, "profile_timeout_seconds"
        )
        self.benchmark_settings = dict(benchmark_settings or {})
        self.profile_settings = dict(profile_settings or {})
        self.device_environment = dict(device_environment or {})

    def _lock(self) -> DeviceLock:
        return DeviceLock(
            self.device,
            lock_root=self.lock_root,
            timeout_seconds=self.lock_timeout_seconds,
        )

    def cancel_current(self) -> bool:
        """Best-effort cancellation used when a durable queue lease is lost."""

        return self.runner.cancel_current()

    @staticmethod
    def _attempt_dir(artifact_dir: Path, stage_name: str) -> Path:
        requested = Path(artifact_dir)
        if requested.exists() and requested.is_symlink():
            raise ValueError("artifact directory must not be a symlink")
        ensure_shared_directory(requested)
        root = requested.resolve()
        for number in range(1, 10_000):
            name = stage_name if number == 1 else f"{stage_name}_attempt_{number:02d}"
            candidate = root / name
            try:
                candidate.mkdir(mode=0o700)
                candidate.chmod(0o700)
                ensure_shared_directory(root)
                return candidate
            except FileExistsError:
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ValueError(
                        f"unsafe existing stage artifact path: {candidate}"
                    ) from None
        raise RuntimeError(f"too many {stage_name} attempts in {root}")

    @staticmethod
    def _copy_candidate(candidate_path: Path, work_dir: Path) -> Path:
        source = Path(candidate_path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("candidate must be a regular non-symlink file")
        destination = work_dir / "candidate.py"
        value = source.read_bytes()
        with destination.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o400)
        return destination

    def _prepare(
        self,
        artifact_dir: Path,
        stage_name: str,
        candidate_path: Path,
    ) -> tuple[Path, Path, dict[str, str]]:
        attempt = self._attempt_dir(artifact_dir, stage_name)
        work = attempt / "work"
        work.mkdir(mode=0o700)
        cache = work / "triton-cache"
        dump = work / "ir"
        temporary = work / "tmp"
        home = work / "home"
        for path in (cache, dump, temporary, home):
            path.mkdir(mode=0o700)
        self._copy_candidate(candidate_path, work)
        env = {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TRITON_CACHE_DIR": str(cache),
            "TRITON_DUMP_DIR": str(dump),
            "DEVICE_ID": self.device_index,
            **self.device_environment,
        }
        return attempt, work, env

    @staticmethod
    def _payload(
        stage_name: str,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        *,
        device: str,
        settings: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not cases:
            raise ValueError(f"{stage_name} requires at least one case")
        return {
            "schema_version": "ascend_isolated_stage_request_v1",
            "stage": stage_name,
            "task": _serialized_task(task),
            "cases": [case.to_dict() for case in cases],
            "candidate": "candidate.py",
            "result": "stage_result.json",
            "device": device,
            "settings": dict(settings or {}),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _copy_public_file(source: Path, destination: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"stage artifact is not a regular file: {source}")
            with (
                os.fdopen(descriptor, "rb", closefd=False) as input_stream,
                destination.open("xb") as output_stream,
            ):
                shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            destination.chmod(SHARED_FILE_MODE)
        finally:
            os.close(descriptor)

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

    @classmethod
    def _copy_public_tree(cls, source: Path, destination: Path) -> None:
        source_metadata = source.lstat()
        if not stat.S_ISDIR(source_metadata.st_mode) or stat.S_ISLNK(
            source_metadata.st_mode
        ):
            raise ValueError(f"stage artifact is not a real directory: {source}")
        destination.mkdir(mode=0o700)
        entry_count = 0
        pending = [(source, destination)]
        created_directories = [destination]
        while pending:
            current_source, current_destination = pending.pop()
            with os.scandir(current_source) as entries:
                for entry in entries:
                    entry_count += 1
                    if entry_count > 50_000:
                        raise ValueError("stage artifact tree exceeds 50000 entries")
                    entry_source = Path(entry.path)
                    entry_destination = current_destination / entry.name
                    metadata = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(metadata.st_mode):
                        raise ValueError(
                            f"stage artifact tree contains a symlink: {entry_source}"
                        )
                    if stat.S_ISDIR(metadata.st_mode):
                        entry_destination.mkdir(mode=0o700)
                        created_directories.append(entry_destination)
                        pending.append((entry_source, entry_destination))
                    elif stat.S_ISREG(metadata.st_mode):
                        cls._copy_public_file(entry_source, entry_destination)
                    else:
                        raise ValueError(
                            f"stage artifact tree contains a special file: {entry_source}"
                        )
        for directory in reversed(created_directories):
            directory.chmod(SHARED_DIRECTORY_MODE)

    @staticmethod
    def _public_manifest(staging: Path) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for path in sorted(staging.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"published artifact contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"published artifact contains a special file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            relative = path.relative_to(staging).as_posix()
            media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            files.append(
                {
                    "relative_path": relative,
                    "sha256": digest.hexdigest(),
                    "size_bytes": metadata.st_size,
                    "type": media_type,
                }
            )
        return {
            "schema_version": "ascend_stage_artifact_manifest_v1",
            "files": files,
        }

    @classmethod
    def _publish_artifacts(cls, attempt: Path, work: Path) -> dict[str, str]:
        """Atomically expose trusted evidence outside the private candidate cwd."""

        shared_parent = attempt.parent
        published_parent = ensure_shared_directory(shared_parent / "published")
        final = published_parent / attempt.name
        if os.path.lexists(final):
            raise ValueError(f"published stage artifact already exists: {final}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{attempt.name}.publish.", dir=published_parent)
        )
        artifacts: dict[str, str] = {"attempt_dir": str(final)}
        selected = {
            "candidate": work / "candidate.py",
            "stdout": attempt / "stdout.log",
            "stderr": attempt / "stderr.log",
            "stage_result": work / "stage_result.json",
            "triton_cache": work / "triton-cache",
            "ir": work / "ir",
            "profile_raw": attempt / "profile_raw",
            "profile_summary": attempt / "profile_summary.json",
        }
        try:
            for key, source in selected.items():
                if not os.path.lexists(source):
                    continue
                destination = staging / source.name
                metadata = source.lstat()
                if stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                ):
                    cls._copy_public_file(source, destination)
                elif stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
                    metadata.st_mode
                ):
                    cls._copy_public_tree(source, destination)
                else:
                    raise ValueError(
                        f"stage artifact is a symlink or special file: {source}"
                    )
                artifacts[key] = str(final / destination.name)
            manifest_value = cls._public_manifest(staging)
            manifest_bytes = cls._manifest_bytes(manifest_value)
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            manifest = staging / f"artifact_manifest.{manifest_digest}.json"
            cls._atomic_public_manifest(manifest, manifest_bytes)
            artifacts["artifact_manifest"] = str(final / manifest.name)
            staging.chmod(SHARED_DIRECTORY_MODE)
            os.rename(staging, final)
            final.chmod(SHARED_DIRECTORY_MODE)
            cls._fsync_directory(published_parent)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return artifacts

    @staticmethod
    def _manifest_bytes(value: Mapping[str, Any]) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _atomic_public_manifest(path: Path, value: bytes) -> None:
        _atomic_bytes(path, value)
        path.chmod(SHARED_FILE_MODE)

    @staticmethod
    def _read_result(path: Path, expected_stage: str) -> dict[str, Any]:
        if path.is_symlink() or not path.is_file():
            raise ValueError("isolated stage did not commit a regular result file")
        raw = path.read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("isolated stage result exceeds 16 MiB")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("isolated stage result must be a JSON object")
        actual_stage = value.get("stage")
        if actual_stage not in {expected_stage, "unknown"}:
            raise ValueError(
                f"isolated stage result mismatch: expected {expected_stage}, got {actual_stage}"
            )
        if not isinstance(value.get("details", {}), Mapping):
            raise ValueError("isolated stage details must be an object")
        return value

    def _process_result(
        self,
        stage: EvaluationStage | str,
        stage_name: str,
        started: datetime,
        process: StageProcessResult,
        attempt: Path,
        work: Path,
        *,
        private_cases: bool = False,
    ) -> StageResult:
        _atomic_bytes(attempt / "stdout.log", b"" if private_cases else process.stdout)
        _atomic_bytes(attempt / "stderr.log", b"" if private_cases else process.stderr)
        artifacts = {} if private_cases else self._publish_artifacts(attempt, work)
        common = {
            "returncode": process.returncode,
            "process_duration_seconds": process.duration_seconds,
            "timed_out": process.timed_out,
            "output_limit_exceeded": process.output_limit_exceeded,
            "terminated": process.terminated,
        }
        if process.timed_out:
            return StageResult.infrastructure_error(
                stage,
                message=f"{stage_name} stage exceeded its wall-clock timeout",
                error_type="StageTimeout",
                retryable=True,
                timed_out=True,
                started_at=started,
                details=common,
                artifacts=artifacts,
            )
        if process.output_limit_exceeded:
            return StageResult.failure(
                stage,
                started_at=started,
                details=common,
                artifacts=artifacts,
                error={
                    "type": "OutputLimitExceeded",
                    "message": "candidate stage exceeded the stdout/stderr limit",
                },
            )
        try:
            raw = self._read_result(work / "stage_result.json", stage_name)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return StageResult.infrastructure_error(
                stage,
                message=f"invalid isolated stage result: {exc}",
                error_type="StageProtocolError",
                retryable=True,
                started_at=started,
                details={**common, "stderr_tail": process.stderr_text()[-8192:]},
                artifacts=artifacts,
            )
        details = dict(raw.get("details", {}))
        details.update(common)
        error = raw.get("error")
        normalized_error = (
            {str(key): item for key, item in error.items()}
            if isinstance(error, Mapping)
            else None
        )
        if bool(raw.get("passed")) and process.returncode == 0:
            return StageResult.success(
                stage,
                started_at=started,
                details=details,
                artifacts=artifacts,
            )
        if bool(raw.get("passed")) != (process.returncode == 0):
            return StageResult.infrastructure_error(
                stage,
                message="isolated stage exit code disagrees with its result file",
                error_type="StageProtocolError",
                retryable=True,
                started_at=started,
                details=details,
                artifacts=artifacts,
            )
        return StageResult.failure(
            stage,
            started_at=started,
            details=details,
            artifacts=artifacts,
            error=normalized_error,
        )

    def _run_isolated(
        self,
        stage: EvaluationStage,
        stage_name: str,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        *,
        timeout_seconds: float,
        settings: Mapping[str, Any] | None = None,
        compile_debug: bool = False,
    ) -> StageResult:
        started = datetime.now(timezone.utc)
        try:
            attempt, work, env = self._prepare(artifact_dir, stage_name, candidate_path)
            if compile_debug:
                env.update({"TRITON_DEBUG": "1", "TRITON_ALWAYS_COMPILE": "1"})
            private_cases = bool(cases) and all(
                case.id.startswith("hidden_") for case in cases
            )
            effective_settings = dict(settings or {})
            if private_cases:
                effective_settings["redact_case_details"] = True
            payload = self._payload(
                stage_name,
                task,
                cases,
                device=self.device,
                settings=effective_settings,
            )
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            with self._lock():
                process = self.runner.run(
                    (
                        self.python_executable,
                        "-m",
                        "ascend_kernel_lab.worker.stage_entry",
                    ),
                    cwd=work,
                    payload=encoded,
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
            return self._process_result(
                stage,
                stage_name,
                started,
                process,
                attempt,
                work,
                private_cases=private_cases,
            )
        except DeviceLockTimeout as exc:
            return StageResult.infrastructure_error(
                stage,
                message=str(exc),
                error_type="DeviceLockTimeout",
                retryable=True,
                timed_out=True,
                started_at=started,
            )
        except (OSError, ValueError, TypeError) as exc:
            return StageResult.infrastructure_error(
                stage,
                message=str(exc),
                error_type=type(exc).__name__,
                retryable=isinstance(exc, OSError),
                started_at=started,
            )

    def source_check(self, candidate_path: Path, task: TaskSpec) -> StageResult:
        del task
        started = datetime.now(timezone.utc)
        result = self.source_guard.check_path(candidate_path)
        details = result.to_dict()
        if result.passed:
            return StageResult.success(
                EvaluationStage.SOURCE_CHECK,
                started_at=started,
                details=details,
                artifacts={"candidate": str(Path(candidate_path).resolve())},
            )
        return StageResult.failure(
            EvaluationStage.SOURCE_CHECK,
            started_at=started,
            details=details,
            artifacts={"candidate": str(Path(candidate_path).absolute())},
        )

    def compile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._run_isolated(
            EvaluationStage.COMPILE,
            "compile",
            candidate_path,
            task,
            cases,
            artifact_dir,
            timeout_seconds=self.compile_timeout_seconds,
            compile_debug=True,
        )

    def check_correctness(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._run_isolated(
            EvaluationStage.CORRECTNESS,
            "correctness",
            candidate_path,
            task,
            cases,
            artifact_dir,
            timeout_seconds=self.correctness_case_timeout_seconds * max(1, len(cases)),
        )

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
    ) -> StageResult:
        settings = dict(self.benchmark_settings)
        if baseline_snapshot:
            # Kept out of the child timing loop; only its digestable metadata is
            # attached for traceability.  The eager baseline is always measured
            # in the same paired session as the candidate.
            settings["baseline_snapshot_available"] = True
        result = self._run_isolated(
            EvaluationStage.BENCHMARK,
            "benchmark",
            candidate_path,
            task,
            cases,
            artifact_dir,
            timeout_seconds=self.benchmark_timeout_seconds,
            settings=settings,
        )
        return self._attach_strong_baselines(result, baseline_snapshot)

    @staticmethod
    def _attach_strong_baselines(
        result: StageResult,
        baseline_snapshot: Mapping[str, Any] | None,
    ) -> StageResult:
        """Attach B1/B2 comparisons without putting snapshots in candidate cwd."""

        if not result.passed or not baseline_snapshot:
            return result
        measured = result.details.get("per_case")
        reference = baseline_snapshot.get("per_case")
        if not isinstance(measured, Sequence) or isinstance(measured, (str, bytes)):
            return result
        if not isinstance(reference, Sequence) or isinstance(reference, (str, bytes)):
            return result
        by_id = {
            str(item.get("case_id")): item
            for item in reference
            if isinstance(item, Mapping) and item.get("case_id") is not None
        }
        enriched: list[dict[str, Any]] = []
        for raw in measured:
            if not isinstance(raw, Mapping):
                return result
            item = dict(raw)
            stats = item.get("candidate")
            candidate_us: float | None = None
            if isinstance(stats, Mapping):
                try:
                    raw_median = stats.get("median_us")
                    if raw_median is not None:
                        value = float(raw_median)
                        candidate_us = (
                            value if math.isfinite(value) and value > 0 else None
                        )
                except (TypeError, ValueError):
                    candidate_us = None
            baseline = by_id.get(str(item.get("case_id")))
            for label, source_key in (
                ("compile", "torch_compile_us"),
                ("official", "official_us"),
            ):
                latency: float | None = None
                if baseline is not None and baseline.get(source_key) is not None:
                    try:
                        parsed = float(baseline[source_key])
                        latency = parsed if math.isfinite(parsed) and parsed > 0 else None
                    except (TypeError, ValueError):
                        latency = None
                item[source_key] = latency
                item[f"speedup_vs_{label}"] = (
                    latency / candidate_us
                    if latency is not None and candidate_us is not None
                    else None
                )
            enriched.append(item)

        details = dict(result.details)
        details["per_case"] = enriched
        details["baseline_identity_sha256"] = baseline_snapshot.get("identity_sha256")
        for label in ("compile", "official"):
            key = f"speedup_vs_{label}"
            valid = [item for item in enriched if item.get(key) is not None]
            speeds = [float(item[key]) for item in valid]
            weights = [float(item.get("weight", 1.0)) for item in valid]
            details[f"geomean_{key}"] = (
                weighted_geometric_mean(speeds, weights) if speeds else None
            )
            details[f"minimum_{key}"] = min(speeds) if speeds else None
            details[f"maximum_{key}"] = max(speeds) if speeds else None
            details[f"{label}_comparison_case_count"] = len(speeds)
        return StageResult(
            stage=result.stage,
            status=result.status,
            started_at=result.started_at,
            finished_at=result.finished_at,
            details=details,
            artifacts=result.artifacts,
            error=result.error,
            retryable=result.retryable,
        )

    @staticmethod
    def _kernel_names(candidate_path: Path) -> tuple[str, ...]:
        try:
            tree = ast.parse(Path(candidate_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ValueError(
                "candidate source cannot provide safe profiler identities"
            ) from exc
        names: list[str] = []
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Attribute) and target.attr == "jit":
                    names.append(node.name)
        if not names:
            raise ValueError("candidate source declares no attributable Triton kernel")
        unique_names = tuple(dict.fromkeys(names))
        for name in unique_names:
            candidate_kernel_pattern(name)
        return unique_names

    @classmethod
    def _kernel_patterns(cls, candidate_path: Path) -> tuple[str, ...]:
        return tuple(
            candidate_kernel_pattern(name)
            for name in cls._kernel_names(candidate_path)
        )

    @staticmethod
    def _missing_profile_groups(
        summary: Mapping[str, Any], mandatory_groups: Sequence[str]
    ) -> tuple[str, ...]:
        def finite(value: Any, *, positive: bool = False) -> bool:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return False
            return math.isfinite(number) and (number > 0 if positive else number >= 0)

        pipeline = summary.get("pipeline")
        memory = summary.get("memory")
        scheduling = summary.get("scheduling")
        available = {
            "task_time": (
                int(summary.get("kernel_count", 0) or 0) >= 1
                and finite(summary.get("candidate_kernel_coverage"), positive=True)
                and isinstance(scheduling, Mapping)
                and finite(scheduling.get("device_execution_us"), positive=True)
            ),
            "pipe_utilization": (
                isinstance(pipeline, Mapping)
                and any(finite(value) for value in pipeline.values())
            ),
            "memory": (
                isinstance(memory, Mapping)
                and any(
                    finite(memory.get(name))
                    for name in (
                        "gm_read_gbps",
                        "gm_write_gbps",
                        "ub_read_gbps",
                        "ub_write_gbps",
                    )
                )
            ),
            "l2_cache": (
                isinstance(memory, Mapping) and finite(memory.get("l2_hit_rate"))
            ),
            "resource_conflict": False,
        }
        return tuple(group for group in mandatory_groups if not available.get(group, False))

    def profile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        started = datetime.now(timezone.utc)
        if not self.msprof_runner.available():
            return StageResult(
                stage=EvaluationStage.PROFILE,
                status=StageStatus.UNAVAILABLE,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                details={
                    "profile_available": False,
                    "unavailable_reason": "msprof executable was not found",
                },
            )
        try:
            kernel_names = self._kernel_names(candidate_path)
            kernel_patterns = tuple(
                candidate_kernel_pattern(name) for name in kernel_names
            )
            attempt, work, env = self._prepare(artifact_dir, "profile", candidate_path)
            private_cases = bool(cases) and all(
                case.id.startswith("hidden_") for case in cases
            )
            settings = dict(self.profile_settings)
            raw_mandatory = settings.get(
                "mandatory_groups", ("task_time", "pipe_utilization")
            )
            raw_optional = settings.get(
                "optional_groups", ("memory", "l2_cache", "resource_conflict")
            )
            if (
                not isinstance(raw_mandatory, Sequence)
                or isinstance(raw_mandatory, (str, bytes))
                or not all(isinstance(group, str) and group for group in raw_mandatory)
                or not isinstance(raw_optional, Sequence)
                or isinstance(raw_optional, (str, bytes))
                or not all(isinstance(group, str) and group for group in raw_optional)
            ):
                raise ValueError("profile group configuration must contain non-empty names")
            mandatory_groups = tuple(raw_mandatory)
            optional_groups = tuple(raw_optional)
            requested_groups = mandatory_groups
            if private_cases and bool(settings.get("full_profile_for_final_best", True)):
                requested_groups = tuple(
                    dict.fromkeys((*mandatory_groups, *optional_groups))
                )
            settings["requested_groups"] = list(requested_groups)
            if private_cases:
                settings["redact_case_details"] = True
            payload = self._payload(
                "profile",
                task,
                cases,
                device=self.device,
                settings=settings,
            )
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            driver = work / "profile_driver.py"
            _atomic_bytes(
                driver,
                (
                    b"from ascend_kernel_lab.worker.stage_entry import main\n"
                    b"raise SystemExit(main([]))\n"
                ),
            )
            raw_root = attempt / "profile_raw"
            argv = self.msprof_runner.build_argv(
                output_root=raw_root,
                python_executable=self.python_executable,
                script=driver,
                kernel_name=kernel_names[0],
            )
            with self._lock():
                run = self.runner.run(
                    argv,
                    cwd=work,
                    payload=encoded,
                    env=clean_environment(extra=env),
                    timeout_seconds=self.profile_timeout_seconds,
                )
            stdout = attempt / "stdout.log"
            stderr = attempt / "stderr.log"
            _atomic_bytes(stdout, b"" if private_cases else run.stdout)
            _atomic_bytes(stderr, b"" if private_cases else run.stderr)
            if run.timed_out:
                artifacts = (
                    {} if private_cases else self._publish_artifacts(attempt, work)
                )
                return StageResult.infrastructure_error(
                    EvaluationStage.PROFILE,
                    message="msprof stage exceeded its timeout",
                    error_type="ProfileTimeout",
                    retryable=True,
                    timed_out=True,
                    started_at=started,
                    details={"profile_available": False, "returncode": run.returncode},
                    artifacts=artifacts,
                )
            if run.returncode != 0 or run.output_limit_exceeded:
                artifacts = (
                    {} if private_cases else self._publish_artifacts(attempt, work)
                )
                return StageResult(
                    stage=EvaluationStage.PROFILE,
                    status=StageStatus.UNAVAILABLE,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc),
                    details={
                        "profile_available": False,
                        "returncode": run.returncode,
                        "output_limit_exceeded": run.output_limit_exceeded,
                        "unavailable_reason": (
                            "msprof returned a non-zero exit code or exceeded its log limit"
                        ),
                    },
                    artifacts=artifacts,
                )
            try:
                isolated = self._read_result(work / "stage_result.json", "profile")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                artifacts = (
                    {} if private_cases else self._publish_artifacts(attempt, work)
                )
                return StageResult(
                    stage=EvaluationStage.PROFILE,
                    status=StageStatus.UNAVAILABLE,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc),
                    details={
                        "profile_available": False,
                        "unavailable_reason": f"profile driver did not commit a result: {exc}",
                    },
                    artifacts=artifacts,
                )
            if isolated.get("passed") is not True:
                artifacts = (
                    {} if private_cases else self._publish_artifacts(attempt, work)
                )
                return StageResult.failure(
                    EvaluationStage.PROFILE,
                    started_at=started,
                    details={
                        "profile_available": False,
                        "driver": dict(isolated.get("details", {})),
                        "returncode": run.returncode,
                        "unavailable_reason": "profile driver reported candidate failure",
                    },
                    artifacts=artifacts,
                )
            summary = MsprofParser(kernel_patterns).parse(raw_root)
            summary_dict = summary.to_dict()
            missing_groups = self._missing_profile_groups(
                summary_dict, mandatory_groups
            )
            summary_dict["mandatory_groups"] = list(mandatory_groups)
            summary_dict["requested_groups"] = list(requested_groups)
            summary_dict["missing_mandatory_groups"] = list(missing_groups)
            _atomic_json(attempt / "profile_summary.json", summary_dict)
            artifacts = (
                {} if private_cases else self._publish_artifacts(attempt, work)
            )
            details = {
                "profile_available": summary.profile_available and not missing_groups,
                "summary": summary_dict,
                "driver": dict(isolated.get("details", {})),
                "returncode": run.returncode,
                "mandatory_groups": list(mandatory_groups),
                "requested_groups": list(requested_groups),
                "missing_mandatory_groups": list(missing_groups),
            }
            if not summary.profile_available or missing_groups:
                return StageResult(
                    stage=EvaluationStage.PROFILE,
                    status=StageStatus.UNAVAILABLE,
                    started_at=started,
                    finished_at=datetime.now(timezone.utc),
                    details=details,
                    artifacts=artifacts,
                )
            return StageResult.success(
                EvaluationStage.PROFILE,
                started_at=started,
                details=details,
                artifacts=artifacts,
            )
        except DeviceLockTimeout as exc:
            return StageResult.infrastructure_error(
                EvaluationStage.PROFILE,
                message=str(exc),
                error_type="DeviceLockTimeout",
                retryable=True,
                timed_out=True,
                started_at=started,
            )
        except (OSError, ValueError, TypeError) as exc:
            return StageResult.infrastructure_error(
                EvaluationStage.PROFILE,
                message=str(exc),
                error_type=type(exc).__name__,
                retryable=isinstance(exc, OSError),
                started_at=started,
            )

    def health_check(self) -> StageResult:
        started = datetime.now(timezone.utc)
        try:
            with self._lock():
                return self.health_checker.check()
        except DeviceLockTimeout as exc:
            return StageResult.infrastructure_error(
                "HEALTH_CHECK",
                message=str(exc),
                error_type="DeviceLockTimeout",
                retryable=True,
                timed_out=True,
                started_at=started,
            )

    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]:
        """Measure B0 eagerly and opportunistically measure B1/B2.

        B0 failure is an infrastructure failure and raises, because a baseline
        snapshot without eager latency cannot be consumed safely. B1/B2 absence
        is encoded per the baseline protocol and never guessed.
        """

        started = datetime.now(timezone.utc)
        if not cases:
            raise ValueError("baseline measurement requires at least one case")
        attempt = self._attempt_dir(artifact_dir, "baseline")
        work = attempt / "work"
        work.mkdir(mode=0o700)
        cache = work / "triton-cache"
        temporary = work / "tmp"
        home = work / "home"
        for path in (cache, temporary, home):
            path.mkdir(mode=0o700)
        env = {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "TRITON_CACHE_DIR": str(cache),
            "DEVICE_ID": self.device_index,
            **self.device_environment,
        }
        payload = self._payload(
            "baseline",
            task,
            cases,
            device=self.device,
            settings=self.benchmark_settings,
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        try:
            with self._lock():
                process = self.runner.run(
                    (
                        self.python_executable,
                        "-m",
                        "ascend_kernel_lab.worker.stage_entry",
                    ),
                    cwd=work,
                    payload=encoded,
                    env=env,
                    timeout_seconds=self.benchmark_timeout_seconds,
                )
        except DeviceLockTimeout as exc:
            raise RuntimeError(f"cannot measure baselines: {exc}") from exc
        stage_result = self._process_result(
            "BASELINE", "baseline", started, process, attempt, work
        )
        if not stage_result.passed:
            message = (
                str(stage_result.error.get("message"))
                if stage_result.error is not None
                else "isolated B0 baseline measurement failed"
            )
            raise RuntimeError(message)
        details = dict(stage_result.details)
        details["artifacts"] = dict(stage_result.artifacts)
        return details
