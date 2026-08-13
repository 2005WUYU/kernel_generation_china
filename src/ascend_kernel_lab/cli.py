"""Command-line control plane for local development and Ascend deployment."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ascend_kernel_lab import __version__
from ascend_kernel_lab.acceptance import aggregate_acceptance
from ascend_kernel_lab.backend import AscendTritonBackend, FakeBackend
from ascend_kernel_lab.config import ConfigError, ExperimentConfig, load_config
from ascend_kernel_lab.diagnostics import build_doctor_report
from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.evaluation import EvaluationRequest, evaluate_candidate
from ascend_kernel_lab.export import DatasetExporter, ReportExporter
from ascend_kernel_lab.llm import FakeGateway, ModelRequest, create_model_gateway
from ascend_kernel_lab.orchestration import (
    BaselineManager,
    ExperimentController,
    baseline_identity,
    validate_baseline_snapshot,
)
from ascend_kernel_lab.probe import EnvironmentProber
from ascend_kernel_lab.storage import AtomicArtifactStore, SQLiteDatabase, SQLiteStateStore
from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.tasks.loader import TaskSpecError
from ascend_kernel_lab.tasks.runtime import validate_hidden_seed
from ascend_kernel_lab.verification import RunVerifier
from ascend_kernel_lab.worker import DeviceLock, DeviceLockTimeout

DEFAULT_CONFIG = "configs/experiment_910c_kimi_k3.yaml"
EXIT_USAGE = 2
EXIT_NOT_READY = 3
EXIT_FAILED = 4
EXIT_TASK_FAILED = 5


class CommandError(RuntimeError):
    """An expected operator-facing command failure."""

    def __init__(self, message: str, *, exit_code: int = EXIT_FAILED) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _config(args: argparse.Namespace) -> ExperimentConfig:
    path = Path(str(args.config)).expanduser()
    if not path.is_file():
        raise CommandError(f"configuration file does not exist: {path}", exit_code=EXIT_USAGE)
    return load_config(path)


def _select_tasks(config: ExperimentConfig, values: Sequence[str] | None) -> tuple[str, ...]:
    selected = tuple(values or config.tasks)
    if not selected:
        raise CommandError("at least one task is required", exit_code=EXIT_USAGE)
    unknown = sorted(set(selected) - set(config.tasks))
    if unknown:
        raise CommandError(
            f"task(s) are not enabled by the configuration: {', '.join(unknown)}",
            exit_code=EXIT_USAGE,
        )
    if len(selected) != len(set(selected)):
        raise CommandError("task selection contains duplicates", exit_code=EXIT_USAGE)
    return selected


def _override_experiment_id(
    config: ExperimentConfig, explicit: str | None, *, fake: bool
) -> ExperimentConfig:
    value = explicit or (f"{config.id}_fake" if fake else config.id)
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or len(value.encode("utf-8")) > 128
    ):
        raise CommandError("experiment ID must be one safe path segment", exit_code=EXIT_USAGE)
    return dataclasses.replace(config, id=value)


def _git_commit(root: Path) -> str:
    try:
        process = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        if process.returncode == 0 and process.stdout.strip():
            commit = process.stdout.strip()
            dirty = subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
            return f"{commit}-dirty" if dirty.stdout.strip() else commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


def _require_clean_git_release(config: ExperimentConfig) -> str:
    """Fail closed before any production measurement or candidate execution."""

    try:
        top = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=config.project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        head = subprocess.run(
            ("git", "rev-parse", "--verify", "HEAD"),
            cwd=config.project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=config.project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CommandError(
            "production execution requires a clean Git checkout",
            exit_code=EXIT_NOT_READY,
        ) from exc
    revision = head.stdout.strip().lower()
    valid_revision = len(revision) in {40, 64} and all(
        character in "0123456789abcdef" for character in revision
    )
    try:
        top_level_matches = (
            top.returncode == 0
            and Path(top.stdout.strip()).resolve() == config.project_root.resolve()
        )
    except (OSError, RuntimeError):
        top_level_matches = False
    if (
        not top_level_matches
        or head.returncode != 0
        or status.returncode != 0
        or not valid_revision
        or bool(status.stdout.strip())
    ):
        raise CommandError(
            "production execution requires the configured project root to be an "
            "exact clean Git commit; commit/revert local changes and retry",
            exit_code=EXIT_NOT_READY,
        )
    return revision


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CommandError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CommandError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CommandError(f"{label} must be a JSON object: {path}")
    return value


def _probe_root(config: ExperimentConfig, explicit: str | None = None) -> Path:
    if explicit is None:
        return config.artifact_root / "probe"
    requested = Path(explicit).expanduser()
    if requested.is_symlink():
        raise CommandError(f"probe root must not be a symlink: {requested}")
    return requested.resolve()


_HARDWARE_PROFILE_RELATIVE_PATH = Path(
    "hardware_probe/kernel_authoring_environment.reported.json"
)


def _profile_value(profile: Mapping[str, Any], *path: str) -> Any:
    value: Any = profile
    try:
        for component in path:
            value = value[component]
        return value["value"]
    except (KeyError, TypeError) as exc:
        dotted = ".".join(path) + ".value"
        raise CommandError(f"hardware profile is missing {dotted}") from exc


def _compact_hardware_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if profile.get("schema_version") != "kernel_authoring_environment_v1":
        raise CommandError("hardware profile has an unexpected schema_version")

    source_profile_sha256 = profile.get("reported_profile_sha256")
    if not isinstance(source_profile_sha256, str) or len(source_profile_sha256) != 64:
        raise CommandError("hardware profile has no reported_profile_sha256")

    compiler_runtime = profile.get("compiler_runtime")
    hardware = profile.get("hardware")
    raw_evidence = profile.get("raw_evidence")
    if not all(
        isinstance(value, Mapping)
        for value in (compiler_runtime, hardware, raw_evidence)
    ):
        raise CommandError("hardware profile is missing kernel-authoring sections")

    assert isinstance(compiler_runtime, Mapping)
    assert isinstance(hardware, Mapping)
    assert isinstance(raw_evidence, Mapping)
    cache_levels = hardware["memory"]["cache_levels"]
    l2_cache = next(
        item for item in cache_levels if item.get("level") == "L2"
    )

    return {
        "backend": compiler_runtime["backend"],
        "execution": {
            "matrix_units": _profile_value(
                hardware,
                "execution",
                "schedulable_engines",
                "matrix",
                "physical_count",
            ),
            "vector_units": _profile_value(
                hardware,
                "execution",
                "schedulable_engines",
                "vector",
                "physical_count",
            ),
            "recommended_matrix_programs": _profile_value(
                compiler_runtime,
                "dispatch",
                "recommended_matrix_or_cv_programs",
            ),
            "recommended_vector_programs": _profile_value(
                compiler_runtime,
                "dispatch",
                "recommended_vector_programs",
            ),
            "max_grid_programs": _profile_value(
                compiler_runtime,
                "dispatch",
                "max_total_grid_programs",
            ),
        },
        "memory": {
            "global_memory_bytes": _profile_value(
                hardware, "memory", "global", "capacity"
            ),
            "l2_cache_bytes": _profile_value(l2_cache, "capacity"),
            "local_memory_type": hardware["memory"]["local_scratchpad"][
                "vendor_term"
            ],
        },
        "memory_access": {
            "vector_tail_alignment_bytes": _profile_value(
                compiler_runtime,
                "memory_access_constraints",
                "vector_tail_axis_alignment",
            ),
            "matrix_vector_tail_alignment_bytes": _profile_value(
                compiler_runtime,
                "memory_access_constraints",
                "cube_vector_tail_axis_alignment",
            ),
        },
        "supported": {
            "fp16": _profile_value(
                compiler_runtime,
                "verified_dtype_paths",
                "load_store_and_vector_fp16",
            ),
            "bf16": _profile_value(
                compiler_runtime,
                "verified_dtype_paths",
                "load_store_and_vector_bfloat16",
            ),
            "fp32": _profile_value(
                compiler_runtime,
                "verified_dtype_paths",
                "load_store_and_vector_fp32",
            ),
            "matrix_dot_fp16": _profile_value(
                compiler_runtime,
                "verified_dtype_paths",
                "matrix_dot_fp16",
            ),
            **{
                name: _profile_value(
                    compiler_runtime, "verified_features", name
                )
                for name in (
                    "masked_load_store",
                    "reduction_sum",
                    "max_exp",
                    "grid_2d",
                    "multiple_kernels",
                )
            },
        },
        "timing": {
            "method": raw_evidence["capability_timing"]["timing_method"]
        },
    }


def _load_probe_snapshot(
    config: ExperimentConfig,
    explicit: str | None = None,
    *,
    require_profile: bool = False,
) -> dict[str, Any]:
    root = _probe_root(config, explicit)
    environment = _load_json_object(root / "env_manifest.json", label="environment manifest")
    capabilities = _load_json_object(root / "capability_matrix.json", label="capability matrix")
    profiler = _load_json_object(root / "profiler_capabilities.json", label="profiler manifest")
    hardware_profile_path = root.parent / _HARDWARE_PROFILE_RELATIVE_PATH
    hardware_profile = _load_json_object(
        hardware_profile_path, label="kernel-authoring hardware profile"
    )
    raw_hardware_evidence = hardware_profile.get("raw_evidence")
    if (
        isinstance(raw_hardware_evidence, Mapping)
        and raw_hardware_evidence.get("runtime_error")
    ):
        raise CommandError(
            "kernel-authoring hardware probe failed: "
            + str(raw_hardware_evidence["runtime_error"]),
            exit_code=EXIT_NOT_READY,
        )
    prompt_environment = _compact_hardware_profile(hardware_profile)
    failures = _probe_readiness_failures(
        config,
        capabilities,
        profiler,
        require_profile=require_profile,
    )
    if failures:
        raise CommandError(
            "probe capability snapshot is not ready: " + "; ".join(failures),
            exit_code=EXIT_NOT_READY,
        )
    snapshot = {
        "schema_version": "ascend_prompt_environment_v1",
        "environment": environment,
        "capabilities": capabilities,
        "profiler": profiler,
        "environment_sha256": hashlib.sha256(_canonical(environment)).hexdigest(),
        "capability_sha256": hashlib.sha256(_canonical(capabilities)).hexdigest(),
        "profiler_sha256": hashlib.sha256(_canonical(profiler)).hexdigest(),
        "prompt_environment": prompt_environment,
        "hardware_profile_evidence": {
            "schema_version": "kernel_hardware_profile_evidence_v1",
            "path_base": "probe_root_parent",
            "path": _HARDWARE_PROFILE_RELATIVE_PATH.as_posix(),
            "file_sha256": hashlib.sha256(
                hardware_profile_path.read_bytes()
            ).hexdigest(),
            "reported_profile_sha256": hardware_profile[
                "reported_profile_sha256"
            ],
            "profile": hardware_profile,
        },
    }
    snapshot["probe_snapshot_sha256"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
    return snapshot


_REQUIRED_PROBE_FEATURES = (
    "vector_add",
    "masked_load_store",
    "fp16",
    "bfloat16",
    "fp32",
    "reduction_sum",
    "max_exp",
    "dot",
    "grid_2d",
    "multiple_kernels",
)


def _probe_readiness_failures(
    config: ExperimentConfig,
    capabilities: Mapping[str, Any],
    profiler: Mapping[str, Any],
    *,
    require_profile: bool,
) -> list[str]:
    failures: list[str] = []
    if capabilities.get("schema_version") != "ascend_triton_capabilities_v1":
        failures.append("invalid Triton capability schema")
    features_value = capabilities.get("features")
    features = features_value if isinstance(features_value, Mapping) else {}
    visibility_value = capabilities.get("device_visibility")
    visibility = visibility_value if isinstance(visibility_value, Mapping) else {}
    if visibility.get("available") is not True or visibility.get("count") != 1:
        failures.append("probe subprocess did not see exactly one available NPU")
    for name in _REQUIRED_PROBE_FEATURES:
        result_value = features.get(name)
        result = result_value if isinstance(result_value, Mapping) else {}
        if not all(result.get(field) is True for field in ("compile", "run", "correct")):
            failures.append(f"Triton feature {name} did not compile, run, and verify")
    timing_value = capabilities.get("timing")
    timing = timing_value if isinstance(timing_value, Mapping) else {}
    if timing.get("verified") is not True:
        failures.append("NPU timing primitive was not verified")
    if require_profile:
        emitted = profiler.get("emitted")
        emitted_mapping = emitted if isinstance(emitted, Mapping) else {}
        missing_groups = [
            group
            for group in config.profile.mandatory_groups
            if emitted_mapping.get(group) is not True
        ]
        if (
            profiler.get("schema_version") != "ascend_profiler_capabilities_v1"
            or profiler.get("msprof_available") is not True
            or profiler.get("live_smoke_completed") is not True
        ):
            failures.append("profiler live smoke did not complete")
        if missing_groups:
            failures.append(
                "profiler did not emit mandatory groups: " + ", ".join(missing_groups)
            )
    return failures


def _baseline_root(config: ExperimentConfig) -> Path:
    return config.artifact_root / "baselines"


def _load_baselines(
    config: ExperimentConfig,
    tasks: Sequence[str],
    *,
    environment_sha256: str | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "ascend_baseline_collection_v1",
        "tasks": {},
    }
    task_values: dict[str, Any] = result["tasks"]
    missing: list[str] = []
    registry = TaskRegistry(config.task_root)
    for task_id in tasks:
        path = _baseline_root(config) / task_id / "latest.json"
        if path.is_file() and not path.is_symlink():
            snapshot = _load_json_object(path, label=f"baseline for {task_id}")
            task = registry.load(task_id)
            invalid_reasons: list[str] = []
            if snapshot.get("schema_version") != "ascend_baseline_snapshot_v1":
                invalid_reasons.append("schema_version")
            if snapshot.get("task_id") != task_id:
                invalid_reasons.append("task_id")
            if snapshot.get("task_spec_sha256") != task.digest():
                invalid_reasons.append("task_spec_sha256")
            if environment_sha256 is not None:
                measured_commit = str(snapshot.get("harness_git_commit", ""))
                identity = baseline_identity(
                    task=task,
                    environment_sha256=environment_sha256,
                    harness_git_commit=measured_commit,
                    benchmark_config=dataclasses.asdict(config.benchmark),
                )
                if not validate_baseline_snapshot(snapshot, expected_identity=identity):
                    invalid_reasons.append("identity_sha256_or_measurements")
            cases = snapshot.get("per_case")
            case_ids = (
                [str(item.get("case_id")) for item in cases if isinstance(item, Mapping)]
                if isinstance(cases, list)
                else []
            )
            expected_case_ids = [case.id for case in task.benchmark_cases]
            if case_ids != expected_case_ids:
                invalid_reasons.append("benchmark_case_ids")
            if invalid_reasons:
                raise CommandError(
                    f"baseline snapshot for {task_id} is stale or incompatible "
                    f"({', '.join(invalid_reasons)}); run `akg baseline run --task {task_id}`"
                )
            task_values[task_id] = snapshot
            result[task_id] = task_values[task_id]
        else:
            missing.append(task_id)
    if missing and not allow_missing:
        raise CommandError(
            "baseline snapshots are missing for: "
            f"{', '.join(missing)}; run `akg baseline run` first"
        )
    result["missing_tasks"] = missing
    return result


def _source_guard(config: ExperimentConfig) -> Any:
    from ascend_kernel_lab.evaluation.source_guard import SourceGuard

    return SourceGuard(
        allowed_import_roots=config.security.allowed_import_roots,
        forbidden_import_roots=config.security.forbidden_import_roots,
        forbidden_call_prefixes=config.security.forbidden_call_prefixes,
        maximum_source_bytes=config.security.maximum_source_bytes,
        required_entrypoint=config.security.required_entrypoint,
    )


def _device_lock_root() -> Path:
    lock_root = Path(
        os.environ.get("AKG_DEVICE_LOCK_ROOT", "/tmp/ascend-kernel-lab-locks")
    )
    if not lock_root.is_absolute():
        raise CommandError("AKG_DEVICE_LOCK_ROOT must be absolute", exit_code=EXIT_USAGE)
    return lock_root


def _real_backend(config: ExperimentConfig) -> AscendTritonBackend:
    return AscendTritonBackend(
        device=config.worker.device,
        source_guard=_source_guard(config),
        compile_timeout_seconds=config.timeouts.compile_seconds,
        correctness_case_timeout_seconds=config.timeouts.correctness_case_seconds,
        benchmark_timeout_seconds=config.timeouts.benchmark_seconds,
        profile_timeout_seconds=config.timeouts.profile_seconds,
        benchmark_settings=dataclasses.asdict(config.benchmark),
        profile_settings=dataclasses.asdict(config.profile),
        lock_root=_device_lock_root(),
    )


def _fake_model_response(request: ModelRequest) -> dict[str, Any]:
    round_number = int(request.metadata.get("round", 1))
    source = """import torch
import triton
import triton.language as tl

@triton.jit
def _copy_kernel(x, y, n: tl.constexpr):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n
    tl.store(y + offsets, tl.load(x + offsets, mask=mask), mask=mask)

def custom_op(x):
    output = torch.empty_like(x)
    n = x.numel()
    _copy_kernel[(triton.cdiv(n, 256),)](x, output, n)
    return output
"""
    return {
        "status": "candidate",
        "round": round_number,
        "optimization_summary": ["deterministic offline pipeline candidate"],
        "expected_effect": ["exercise every recoverable controller checkpoint"],
        "assumptions": ["FakeBackend is used; no hardware result is claimed"],
        "code": source,
    }


def _hidden_seed(*, fake: bool) -> int:
    raw = os.environ.get("AKG_HIDDEN_SEED")
    if raw is None and fake:
        return 1
    if raw is None:
        raise CommandError(
            "AKG_HIDDEN_SEED is required for the blind final gate; keep it outside the repository"
        )
    try:
        return validate_hidden_seed(raw, allow_insecure_for_testing=fake)
    except ValueError as exc:
        raise CommandError(
            "AKG_HIDDEN_SEED must be a positive 128-256 bit base-10 integer"
        ) from exc


def _cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    report = build_doctor_report(config, role=args.role)
    if args.output:
        _atomic_json(Path(args.output), report)
    _print_json(report)
    return 0 if report["ready"] or args.allow_not_ready else EXIT_NOT_READY


def _cmd_db_upgrade(args: argparse.Namespace) -> int:
    config = _config(args)
    database = SQLiteDatabase(config.db_path)
    report = {
        "schema_version": "ascend_db_upgrade_result_v1",
        "database_path": str(config.db_path),
        "user_version": database.latest_schema_version,
        "status": "ready",
    }
    database.close()
    _print_json(report)
    return 0


def _cmd_probe_all(args: argparse.Namespace) -> int:
    config = _config(args)
    root = _probe_root(config, args.output)
    try:
        with DeviceLock(
            config.worker.device,
            lock_root=_device_lock_root(),
            timeout_seconds=5.0,
        ):
            bundle = EnvironmentProber(command_timeout=args.command_timeout).write_bundle(
                root, run_feature_smokes=not args.skip_feature_smokes
            )
    except DeviceLockTimeout as exc:
        raise CommandError(
            "probe refused to overlap another process using the configured NPU",
            exit_code=EXIT_NOT_READY,
        ) from exc
    capabilities = _load_json_object(bundle.capability_path, label="capability matrix")
    profiler = _load_json_object(bundle.profiler_path, label="profiler manifest")
    hardware_profile = _load_json_object(
        bundle.hardware_profile_path,
        label="kernel-authoring hardware profile",
    )
    failures = _probe_readiness_failures(
        config,
        capabilities,
        profiler,
        require_profile=True,
    )
    raw_hardware_evidence = hardware_profile.get("raw_evidence")
    if (
        isinstance(raw_hardware_evidence, Mapping)
        and raw_hardware_evidence.get("runtime_error")
    ):
        failures.append(
            "kernel-authoring hardware properties were unavailable: "
            + str(raw_hardware_evidence["runtime_error"])
        )
    try:
        _compact_hardware_profile(hardware_profile)
    except CommandError as exc:
        failures.append(str(exc))
    report = {
        "schema_version": "ascend_probe_result_v1",
        "output_root": str(bundle.output_root),
        "environment_path": str(bundle.environment_path),
        "capability_path": str(bundle.capability_path),
        "profiler_path": str(bundle.profiler_path),
        "hardware_profile_path": str(bundle.hardware_profile_path),
        "environment_sha256": bundle.environment_sha256,
        "capability_sha256": bundle.capability_sha256,
        "profiler_sha256": bundle.profiler_sha256,
        "hardware_profile_sha256": bundle.hardware_profile_sha256,
        "feature_smokes_run": not args.skip_feature_smokes,
        "ready": not failures,
        "failures": failures,
    }
    _print_json(report)
    return 0 if not failures else EXIT_NOT_READY


def _cmd_baseline_run(args: argparse.Namespace) -> int:
    config = _config(args)
    _require_clean_git_release(config)
    selected = _select_tasks(config, args.task)
    environment = _load_probe_snapshot(config, args.probe_root)
    environment_sha = str(environment["probe_snapshot_sha256"])
    backend = _real_backend(config)
    health = backend.health_check()
    if not health.passed and not args.skip_health_check:
        _print_json(health.to_dict())
        raise CommandError("Ascend backend health check failed", exit_code=EXIT_NOT_READY)
    registry = TaskRegistry(config.task_root)
    manager = BaselineManager(
        backend=backend,
        environment_sha256=environment_sha,
        harness_git_commit=_git_commit(config.project_root),
        benchmark_config=dataclasses.asdict(config.benchmark),
    )
    artifacts = AtomicArtifactStore(config.artifact_root)
    snapshots: dict[str, Any] = {}
    for task_id in selected:
        task = registry.load(task_id)
        directory = _baseline_root(config) / task_id
        try:
            snapshot = manager.measure(task, directory / "measurements")
        except (RuntimeError, ValueError) as exc:
            snapshots[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            break
        identity = str(snapshot["identity_sha256"])
        relative_root = Path("baselines") / task_id
        artifacts.put_json((relative_root / f"{identity}.json").as_posix(), snapshot)
        artifacts.put_json((relative_root / "latest.json").as_posix(), snapshot, overwrite=True)
        snapshots[task_id] = snapshot
    task_statuses = [str(snapshot.get("status", "failed")) for snapshot in snapshots.values()]
    if "failed" in task_statuses or len(snapshots) != len(selected):
        status = "failed"
    elif "partial" in task_statuses:
        status = "partial"
    else:
        status = "complete"
    report = {
        "schema_version": "ascend_baseline_run_result_v1",
        "status": status,
        "environment_sha256": environment_sha,
        "task_count": len(snapshots),
        "tasks": snapshots,
    }
    artifacts.put_json("baselines/latest_collection.json", report, overwrite=True)
    _print_json(report)
    if status == "failed":
        return EXIT_FAILED
    if status == "partial" and args.require_complete:
        return EXIT_NOT_READY
    return 0


def _cmd_worker_run(args: argparse.Namespace) -> int:
    from ascend_kernel_lab.worker import WorkerService

    config = _config(args)
    _require_clean_git_release(config)
    store = SQLiteStateStore(config.db_path)
    backend = _real_backend(config)
    if not args.skip_health_check:
        health = backend.health_check()
        if not health.passed:
            _print_json(health.to_dict())
            raise CommandError("worker refused to start because device health failed", exit_code=EXIT_NOT_READY)
    worker_id = args.worker_id or f"{socket.gethostname()}:{os.getpid()}:{config.worker.device}"
    service = WorkerService(
        queue=store.jobs,
        backend=backend,
        registry=TaskRegistry(config.task_root),
        artifact_root=config.artifact_root,
        worker_id=worker_id,
        lease_seconds=config.worker.lease_seconds,
        heartbeat_seconds=config.worker.heartbeat_seconds,
        poll_seconds=config.worker.poll_seconds,
    )
    if args.once:
        processed = service.run_once()
        _print_json({"schema_version": "ascend_worker_once_v1", "worker_id": worker_id, "processed": processed})
        return 0
    stop = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()
        service.stop()

    previous_handlers: dict[signal.Signals, Any] = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[number] = signal.signal(number, request_stop)
    try:
        processed_count = service.serve_forever(stop_event=stop, max_jobs=args.max_jobs)
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
    _print_json(
        {
            "schema_version": "ascend_worker_exit_v1",
            "worker_id": worker_id,
            "processed_jobs": processed_count,
        }
    )
    return 0


def _queue_backend(config: ExperimentConfig) -> Any:
    from ascend_kernel_lab.broker import QueueEvaluationBackend

    store = SQLiteStateStore(config.db_path)
    return QueueEvaluationBackend(
        queue=store.jobs,
        artifact_root=config.artifact_root,
        experiment_id=config.id,
        poll_interval_seconds=config.worker.poll_seconds,
        wait_timeout_seconds={
            EvaluationStage.SOURCE_CHECK: config.timeouts.source_check_seconds + 30,
            EvaluationStage.COMPILE: config.timeouts.compile_seconds + 30,
            # The blind final gate contains 20 correctness cases.  The
            # transport wait must be longer than the worker's per-case budget,
            # otherwise a healthy hidden job is abandoned midway through.
            EvaluationStage.CORRECTNESS: config.timeouts.correctness_case_seconds
            * 20
            + 30,
            EvaluationStage.BENCHMARK: config.timeouts.benchmark_seconds + 30,
            EvaluationStage.PROFILE: config.timeouts.profile_seconds + 30,
        },
        maximum_job_attempts=config.worker.maximum_job_attempts,
    )


def _start_local_worker(config: ExperimentConfig) -> tuple[Any, threading.Thread, threading.Event]:
    from ascend_kernel_lab.worker import WorkerService

    store = SQLiteStateStore(config.db_path)
    service = WorkerService(
        queue=store.jobs,
        backend=_real_backend(config),
        registry=TaskRegistry(config.task_root),
        artifact_root=config.artifact_root,
        worker_id=f"local:{socket.gethostname()}:{os.getpid()}:{config.worker.device}",
        lease_seconds=config.worker.lease_seconds,
        heartbeat_seconds=config.worker.heartbeat_seconds,
        poll_seconds=config.worker.poll_seconds,
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=service.serve_forever,
        kwargs={"stop_event": stop},
        name="akg-local-worker",
        daemon=True,
    )
    thread.start()
    return service, thread, stop


def _experiment_components(
    config: ExperimentConfig,
    args: argparse.Namespace,
    selected: Sequence[str],
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], Callable[[], None]]:
    if args.fake:
        return (
            FakeGateway(_fake_model_response),
            FakeBackend(),
            {
                "schema_version": "ascend_fake_environment_v1",
                "notice": "No NPU or model request was executed.",
            },
            {
                task_id: {
                    "schema_version": "ascend_fake_baseline_v1",
                    "task_id": task_id,
                    "notice": "Synthetic baseline for control-plane testing only.",
                }
                for task_id in selected
            },
            lambda: None,
        )
    _require_clean_git_release(config)
    environment = _load_probe_snapshot(
        config,
        args.probe_root,
        require_profile=not args.allow_unverified_profile,
    )
    baseline = _load_baselines(
        config,
        selected,
        environment_sha256=str(environment["probe_snapshot_sha256"]),
        allow_missing=bool(args.allow_missing_baseline),
    )
    gateway = create_model_gateway(config.model)
    if args.direct:
        return gateway, _real_backend(config), environment, baseline, lambda: None
    service: Any | None = None
    thread: threading.Thread | None = None
    stop: threading.Event | None = None
    if args.with_local_worker:
        service, thread, stop = _start_local_worker(config)

    def cleanup() -> None:
        if stop is not None:
            stop.set()
        if service is not None:
            service.stop()
        if thread is not None:
            thread.join(timeout=max(5.0, config.worker.poll_seconds * 3))

    return gateway, _queue_backend(config), environment, baseline, cleanup


def _cmd_experiment_run(args: argparse.Namespace) -> int:
    config = _override_experiment_id(
        _config(args), args.experiment_id, fake=bool(args.fake)
    )
    selected = _select_tasks(config, args.task)
    if selected != config.tasks:
        config = dataclasses.replace(
            config,
            tasks=selected,
            task_concurrency=min(config.task_concurrency, len(selected)),
        )
    gateway, backend, environment, baseline, cleanup = _experiment_components(
        config, args, selected
    )
    controller = ExperimentController(
        config=config,
        store=SQLiteStateStore(config.db_path),
        artifacts=AtomicArtifactStore(config.artifact_root),
        registry=TaskRegistry(config.task_root),
        model_gateway=gateway,
        backend=backend,
        environment=environment,
        baseline=baseline,
        hidden_seed=_hidden_seed(fake=args.fake),
        profile_coverage_required=not args.allow_unverified_profile,
        allow_insecure_hidden_seed_for_testing=bool(args.fake),
    )
    try:
        summaries = controller.run(selected)
    finally:
        cleanup()
    report = {
        "schema_version": "ascend_experiment_command_result_v1",
        "mode": "fake" if args.fake else "direct" if args.direct else "durable_queue",
        "experiment_id": config.id,
        "tasks": [
            {
                "task_id": summary.task_id,
                "status": summary.status,
                "best_round": summary.best_round,
                "best_candidate_id": summary.best_candidate_id,
            }
            for summary in summaries
        ],
        "experiment_root": str(config.artifact_root / config.id),
    }
    if args.output:
        _atomic_json(Path(args.output), report)
    _print_json(report)
    return (
        0
        if all(summary.status.startswith("passed") for summary in summaries)
        else EXIT_TASK_FAILED
    )


def _cmd_experiment_status(args: argparse.Namespace) -> int:
    config = _override_experiment_id(
        _config(args), args.experiment_id, fake=False
    )
    if not config.db_path.is_file():
        raise CommandError(f"experiment database does not exist: {config.db_path}")
    try:
        connection = sqlite3.connect(
            f"file:{config.db_path}?mode=ro", uri=True, timeout=10
        )
        connection.row_factory = sqlite3.Row
        try:
            experiment = connection.execute(
                "SELECT experiment_id, state, version, created_at, updated_at FROM experiments WHERE experiment_id = ?",
                (config.id,),
            ).fetchone()
            tasks = connection.execute(
                """
                SELECT task_id, state, current_round, best_candidate_id, version, updated_at
                FROM tasks WHERE experiment_id = ? ORDER BY task_id
                """,
                (config.id,),
            ).fetchall()
            jobs = connection.execute(
                """
                SELECT status, COUNT(*) AS count FROM evaluation_jobs
                WHERE experiment_id = ? GROUP BY status ORDER BY status
                """,
                (config.id,),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise CommandError(f"cannot read experiment status: {exc}") from exc
    report = {
        "schema_version": "ascend_experiment_status_v1",
        "experiment_id": config.id,
        "experiment": dict(experiment) if experiment is not None else None,
        "tasks": [dict(row) for row in tasks],
        "jobs": {str(row["status"]): int(row["count"]) for row in jobs},
    }
    _print_json(report)
    return 0 if experiment is not None else EXIT_FAILED


def _cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    requested_candidate = Path(args.candidate).expanduser()
    if requested_candidate.is_symlink() or not requested_candidate.is_file():
        raise CommandError(
            f"candidate must be a regular non-symlink file: {requested_candidate}"
        )
    candidate = requested_candidate.resolve()
    task = TaskRegistry(config.task_root).load(args.task)
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    artifact_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.artifact_root / "manual_evaluations" / task.id / digest[:16]
    )
    if args.fake:
        baseline: Mapping[str, Any] = {}
    else:
        environment = _load_probe_snapshot(
            config,
            args.probe_root,
            require_profile=not args.skip_profile,
        )
        baseline = _load_baselines(
            config,
            (task.id,),
            environment_sha256=str(environment["probe_snapshot_sha256"]),
            allow_missing=args.allow_missing_baseline,
        )
    backend = FakeBackend() if args.fake else _real_backend(config)
    result = evaluate_candidate(
        backend,
        EvaluationRequest(
            experiment_id="manual_evaluation",
            task=task,
            round_number=1,
            candidate_id=f"manual:{task.id}:{digest[:16]}",
            candidate_path=candidate,
            artifact_dir=artifact_dir,
            baseline_snapshot=baseline.get(task.id),
            run_profile=not args.skip_profile,
            profile_coverage_required=not args.skip_profile,
        ),
    )
    value = result.to_dict()
    _atomic_json(artifact_dir / "evaluation_result.json", value)
    _print_json(value)
    return 0 if result.score.is_publicly_valid else EXIT_FAILED


def _cmd_export(args: argparse.Namespace) -> int:
    root = Path(args.experiment_root).expanduser().resolve()
    if args.export_kind == "sft":
        output = Path(args.output or root / "exports/sft.jsonl")
        exporter = DatasetExporter(root)
        count = exporter.export_sft(output, main_only=not args.all_samples)
        report = {
            "schema_version": "ascend_export_result_v1",
            "kind": "sft",
            "rows": count,
            "output": str(output.resolve()),
            "selection": "all_samples" if args.all_samples else "curated",
            "trajectory_quality_summary": exporter.quality_summary(),
        }
    elif args.export_kind == "rl":
        output = Path(args.output or root / "exports/rl.jsonl")
        count = DatasetExporter(root).export_rl(output)
        report = {"schema_version": "ascend_export_result_v1", "kind": "rl", "rows": count, "output": str(output.resolve())}
    else:
        output = Path(args.output or root / "exports/report.json")
        markdown = Path(args.markdown or root / "exports/report.md")
        experiment_report = ReportExporter(root).write(output, markdown)
        report = {
            "schema_version": "ascend_export_result_v1",
            "kind": "report",
            "output": str(output.resolve()),
            "markdown": str(markdown.resolve()),
            "trajectory_quality_summary": experiment_report[
                "trajectory_quality_summary"
            ],
        }
    _print_json(report)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.experiment_root).expanduser()
    database: Path | None
    if args.database:
        database = Path(args.database).expanduser()
    elif args.config:
        database = load_config(args.config).db_path
    else:
        inferred = root.parent / "metadata.db"
        database = inferred if inferred.is_file() else None
    report = RunVerifier(root, database_path=database).verify()
    if args.output:
        _atomic_json(Path(args.output), report)
    _print_json(report)
    return 0 if report["passed"] else EXIT_FAILED


def _cmd_acceptance(args: argparse.Namespace) -> int:
    config = _override_experiment_id(
        _config(args), args.experiment_id, fake=False
    )
    doctor = build_doctor_report(config)
    run_root = config.artifact_root / config.id
    verification = (
        RunVerifier(run_root, database_path=config.db_path).verify()
        if run_root.is_dir()
        else None
    )
    report = aggregate_acceptance(
        evidence_root=Path(args.evidence_root or config.artifact_root / "acceptance_evidence"),
        experiment_id=config.id,
        harness_git_commit=_git_commit(config.project_root),
        doctor_report=doctor,
        verification_report=verification,
    )
    report["captured_at_unix"] = time.time()
    output = Path(args.output or config.artifact_root / "acceptance_report.json")
    _atomic_json(output, report)
    _print_json(report)
    return 0 if report["passed"] else EXIT_NOT_READY


def _add_config(parser: argparse.ArgumentParser, *, default: str = DEFAULT_CONFIG) -> None:
    parser.add_argument("-c", "--config", default=default, help="experiment YAML configuration")


def _add_tasks(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        action="append",
        help="task ID to run; repeat for multiple tasks (default: every configured task)",
    )


def _add_experiment_mode(parser: argparse.ArgumentParser) -> None:
    _add_config(parser)
    _add_tasks(parser)
    parser.add_argument("--probe-root", help="probe bundle directory (default: <artifact_root>/probe)")
    parser.add_argument("--experiment-id", help="override the durable experiment ID")
    parser.add_argument("--fake", action="store_true", help="offline control-plane run; never claims hardware results")
    parser.add_argument("--direct", action="store_true", help="evaluate in the controller process topology")
    parser.add_argument("--with-local-worker", action="store_true", help="start a worker thread while retaining the durable queue")
    parser.add_argument("--allow-missing-baseline", action="store_true")
    parser.add_argument("--allow-unverified-profile", action="store_true")
    parser.add_argument("-o", "--output", help="write command summary JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="akg",
        description="Recoverable Kimi K3 → Triton-Ascend kernel generation laboratory",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="show unexpected tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate deployment prerequisites without mutation")
    _add_config(doctor)
    doctor.add_argument(
        "--role",
        choices=("all", "controller", "worker"),
        default="all",
        help="validate prerequisites for all services or one container role",
    )
    doctor.add_argument("-o", "--output")
    doctor.add_argument("--allow-not-ready", action="store_true", help="return success while retaining failed checks")
    doctor.set_defaults(handler=_cmd_doctor)

    db = commands.add_parser("db", help="manage durable SQLite metadata")
    db_commands = db.add_subparsers(dest="db_command", required=True)
    upgrade = db_commands.add_parser("upgrade", help="apply crash-safe schema migrations")
    _add_config(upgrade)
    upgrade.set_defaults(handler=_cmd_db_upgrade)

    probe = commands.add_parser("probe", help="capture immutable Ascend capability evidence")
    probe_commands = probe.add_subparsers(dest="probe_command", required=True)
    probe_all = probe_commands.add_parser("all", help="probe environment, Triton features, timing, and profiler")
    _add_config(probe_all)
    probe_all.add_argument("-o", "--output", help="output directory")
    probe_all.add_argument("--skip-feature-smokes", action="store_true")
    probe_all.add_argument("--command-timeout", type=float, default=20.0)
    probe_all.set_defaults(handler=_cmd_probe_all)

    baseline = commands.add_parser(
        "baseline",
        help="measure the PyTorch eager comparison baseline",
    )
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_run = baseline_commands.add_parser("run", help="measure and atomically snapshot baselines")
    _add_config(baseline_run)
    _add_tasks(baseline_run)
    baseline_run.add_argument("--probe-root")
    baseline_run.add_argument("--skip-health-check", action="store_true")
    baseline_run.add_argument(
        "--require-complete",
        action="store_true",
        help="return nonzero unless the eager-only baseline is complete",
    )
    baseline_run.set_defaults(handler=_cmd_baseline_run)

    worker = commands.add_parser("worker", help="run the durable single-device worker")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    worker_run = worker_commands.add_parser("run", help="claim and execute fenced evaluation jobs")
    _add_config(worker_run)
    worker_run.add_argument("--worker-id")
    worker_run.add_argument("--once", action="store_true")
    worker_run.add_argument("--max-jobs", type=int)
    worker_run.add_argument("--skip-health-check", action="store_true")
    worker_run.set_defaults(handler=_cmd_worker_run)

    experiment = commands.add_parser("experiment", help="run, resume, or inspect experiments")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    experiment_run = experiment_commands.add_parser("run", help="start or idempotently continue an experiment")
    _add_experiment_mode(experiment_run)
    experiment_run.set_defaults(handler=_cmd_experiment_run)
    experiment_resume = experiment_commands.add_parser("resume", help="resume from the last committed checkpoint")
    _add_experiment_mode(experiment_resume)
    experiment_resume.set_defaults(handler=_cmd_experiment_run)
    experiment_status = experiment_commands.add_parser("status", help="read durable state without mutation")
    _add_config(experiment_status)
    experiment_status.add_argument("--experiment-id")
    experiment_status.set_defaults(handler=_cmd_experiment_status)

    evaluate = commands.add_parser("evaluate", help="evaluate one candidate without asking a model")
    _add_config(evaluate)
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output-dir")
    evaluate.add_argument("--probe-root")
    evaluate.add_argument("--fake", action="store_true")
    evaluate.add_argument("--skip-profile", action="store_true")
    evaluate.add_argument("--allow-missing-baseline", action="store_true")
    evaluate.set_defaults(handler=_cmd_evaluate)

    export = commands.add_parser("export", help="produce deterministic offline training/report artifacts")
    export_commands = export.add_subparsers(dest="export_kind", required=True)
    sft = export_commands.add_parser("sft", help="export curated supervised trajectories")
    sft.add_argument("--experiment-root", required=True)
    sft.add_argument("-o", "--output")
    sft.add_argument(
        "--all-samples",
        action="store_true",
        help="include failed, regressed, host-bound, and other non-curated rounds with labels",
    )
    sft.set_defaults(handler=_cmd_export)
    rl = export_commands.add_parser("rl", help="export all RL transitions")
    rl.add_argument("--experiment-root", required=True)
    rl.add_argument("-o", "--output")
    rl.set_defaults(handler=_cmd_export)
    report = export_commands.add_parser("report", help="export JSON and Markdown reports")
    report.add_argument("--experiment-root", required=True)
    report.add_argument("-o", "--output")
    report.add_argument("--markdown")
    report.set_defaults(handler=_cmd_export)

    verify = commands.add_parser("verify-run", help="verify lifecycle, hashes, and leakage offline")
    verify.add_argument("--experiment-root", required=True)
    verify.add_argument("--database")
    verify.add_argument("-c", "--config")
    verify.add_argument("-o", "--output")
    verify.set_defaults(handler=_cmd_verify)

    acceptance = commands.add_parser("acceptance", help="aggregate deployment and run evidence")
    _add_config(acceptance)
    acceptance.add_argument("--experiment-id")
    acceptance.add_argument("--evidence-root")
    acceptance.add_argument("-o", "--output")
    acceptance.set_defaults(handler=_cmd_acceptance)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if not callable(handler):
        parser.error("a command is required")
    if getattr(args, "direct", False) and getattr(args, "with_local_worker", False):
        parser.error("--direct and --with-local-worker are mutually exclusive")
    if getattr(args, "fake", False) and (
        getattr(args, "direct", False) or getattr(args, "with_local_worker", False)
    ):
        parser.error("--fake cannot be combined with a hardware execution mode")
    try:
        return int(handler(args))
    except CommandError as exc:
        print(f"akg: {exc}", file=sys.stderr)
        return exc.exit_code
    except (ConfigError, TaskSpecError, ValueError) as exc:
        print(f"akg: invalid input: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("akg: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"akg: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


__all__ = ["build_parser", "main"]
