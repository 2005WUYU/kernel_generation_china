"""Isolated child-process entry point for real Ascend evaluation stages.

The parent writes only a bounded JSON payload.  This module intentionally has no
import-time dependency on torch, torch_npu, or Triton so deployment diagnostics
can import the package on a controller-only host.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from ascend_kernel_lab.evaluation.benchmark import (
    speedup_summary,
    summarize_samples,
)
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec
from ascend_kernel_lab.tasks.runtime import generate_inputs, reference, validate_output

_MAX_PAYLOAD_BYTES = 16 * 1024**2
_FORBIDDEN_TORCH_CALLS = (
    "add",
    "addmm",
    "bmm",
    "div",
    "einsum",
    "exp",
    "matmul",
    "mean",
    "mm",
    "mul",
    "rsqrt",
    "softmax",
    "sub",
    "sum",
    "true_divide",
)
_FORBIDDEN_TENSOR_CALLS = (
    "__add__",
    "__matmul__",
    "__mul__",
    "__pow__",
    "__radd__",
    "__rmatmul__",
    "__rmul__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
    "add",
    "add_",
    "addmm",
    "bmm",
    "div",
    "div_",
    "exp",
    "exp_",
    "gelu",
    "matmul",
    "mean",
    "mm",
    "mul",
    "mul_",
    "pow",
    "rsqrt",
    "rsqrt_",
    "softmax",
    "sub",
    "sub_",
    "sum",
    "true_divide",
)


def _utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json_object(raw: bytes) -> dict[str, Any]:
    if len(raw) > _MAX_PAYLOAD_BYTES:
        raise ValueError("stage payload exceeds the hard limit")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stage payload must be a JSON object")
    return value


def _read_payload(path: str | None) -> dict[str, Any]:
    if path is None:
        return _json_object(sys.stdin.buffer.read(_MAX_PAYLOAD_BYTES + 1))
    candidate = Path(path)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.is_symlink():
        raise ValueError("payload file must be a regular basename in the stage cwd")
    return _json_object(candidate.read_bytes())


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _task(value: Mapping[str, Any]) -> TaskSpec:
    required = {
        "id",
        "version",
        "name",
        "description",
        "entry_point",
        "inputs",
        "outputs",
        "semantics",
        "correctness",
        "benchmark",
        "restrictions",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"serialized task is missing keys: {sorted(missing)}")
    if value["entry_point"] != "custom_op":
        raise ValueError("serialized task has an unsupported entry point")
    sequence_fields = (value["inputs"], value["outputs"])
    mapping_fields = (
        value["semantics"],
        value["correctness"],
        value["benchmark"],
        value["restrictions"],
    )
    if any(not isinstance(item, Sequence) or isinstance(item, (str, bytes)) for item in sequence_fields):
        raise ValueError("serialized task inputs and outputs must be arrays")
    if any(not isinstance(item, Mapping) for item in mapping_fields):
        raise ValueError("serialized task policy fields must be objects")
    return TaskSpec(
        id=str(value["id"]),
        version=int(value["version"]),
        name=str(value["name"]),
        description=str(value["description"]),
        entry_point="custom_op",
        inputs=tuple(dict(item) for item in value["inputs"] if isinstance(item, Mapping)),
        outputs=tuple(dict(item) for item in value["outputs"] if isinstance(item, Mapping)),
        semantics=dict(value["semantics"]),
        correctness=dict(value["correctness"]),
        benchmark=dict(value["benchmark"]),
        restrictions=dict(value["restrictions"]),
    )


def _cases(value: object) -> tuple[CaseSpec, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("serialized cases must be an array")
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("each serialized case must be an object")
        case = CaseSpec.from_dict(raw)
        if case.id in seen:
            raise ValueError(f"duplicate serialized case ID: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("at least one case is required")
    return tuple(cases)


def _runtime() -> tuple[Any, Any, Any]:
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    triton = importlib.import_module("triton")
    if not bool(torch.npu.is_available()):
        raise RuntimeError("torch reports that no Ascend NPU is available")
    return torch, torch_npu, triton


def _candidate(path: Path) -> tuple[ModuleType, Callable[..., Any]]:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved.parent != cwd or path.is_symlink() or not resolved.is_file():
        raise ValueError("candidate must be a regular file directly inside the stage cwd")
    module_spec = importlib.util.spec_from_file_location("candidate", resolved)
    if module_spec is None or module_spec.loader is None:
        raise ImportError("cannot construct a candidate module loader")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    entry = getattr(module, "custom_op", None)
    if not callable(entry):
        raise TypeError("candidate custom_op is missing or not callable")
    return module, entry


def _sync(torch: Any) -> None:
    torch.npu.synchronize()


@contextlib.contextmanager
def _candidate_guard(torch: Any) -> Iterator[None]:
    """Defense in depth; the AST guard remains the primary Python policy."""

    saved: list[tuple[Any, str, Any]] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("candidate attempted a forbidden high-level torch operation")

    def block(owner: Any, name: str) -> None:
        try:
            original = getattr(owner, name)
            setattr(owner, name, forbidden)
        except (AttributeError, RuntimeError, TypeError):
            return
        saved.append((owner, name, original))

    for name in _FORBIDDEN_TORCH_CALLS:
        if hasattr(torch, name):
            block(torch, name)
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    if functional is not None:
        for name in ("gelu", "layer_norm", "silu", "softmax"):
            if hasattr(functional, name):
                block(functional, name)
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is not None:
        for name in _FORBIDDEN_TENSOR_CALLS:
            if hasattr(tensor_type, name):
                block(tensor_type, name)
    try:
        yield
    finally:
        for owner, name, original in reversed(saved):
            with contextlib.suppress(AttributeError, RuntimeError, TypeError):
                setattr(owner, name, original)


def _unchanged(torch: Any, current: Sequence[Any], originals: Sequence[Any]) -> bool:
    return all(
        bool(torch.equal(value, original))
        for value, original in zip(current, originals, strict=True)
    )


def _compile(
    torch: Any,
    triton: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
) -> dict[str, Any]:
    began = time.perf_counter()
    case_results: list[dict[str, Any]] = []
    for case in cases:
        generated = generate_inputs(task, case, torch, device)
        originals = tuple(value.clone() for value in generated.args)
        case_start = time.perf_counter()
        with _candidate_guard(torch):
            output = entry(*generated.args)
        _sync(torch)
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"case {case.id}: custom_op did not return a Tensor")
        if not _unchanged(torch, generated.args, originals):
            raise RuntimeError(f"case {case.id}: candidate modified an input Tensor")
        case_results.append(
            {
                "case_id": case.id,
                "dtype": case.dtype,
                "compiled": True,
                "elapsed_ms": (time.perf_counter() - case_start) * 1000.0,
            }
        )
    return {
        "passed": True,
        "compiled": True,
        "compile_time_ms": (time.perf_counter() - began) * 1000.0,
        "case_results": case_results,
        "triton_version": str(getattr(triton, "__version__", "unknown")),
    }


def _correctness(
    torch: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        generated = generate_inputs(task, case, torch, device)
        originals = tuple(value.clone() for value in generated.args)
        reference_args = tuple(value.clone() for value in generated.args)
        expected = reference(task, reference_args, torch)
        _sync(torch)
        with _candidate_guard(torch):
            actual = entry(*generated.args)
        _sync(torch)
        validation = validate_output(
            task,
            case,
            actual,
            expected,
            torch,
            inputs=generated.args,
        )
        inputs_unchanged = _unchanged(torch, generated.args, originals)
        validation["inputs_unchanged"] = inputs_unchanged
        if not inputs_unchanged:
            validation["passed"] = False
            validation["error"] = "candidate modified an input Tensor"
        results.append(validation)
    passed = all(bool(item.get("passed")) for item in results)
    absolute = [
        float(item["maximum_absolute_error"])
        for item in results
        if item.get("maximum_absolute_error") is not None
    ]
    relative = [
        float(item["maximum_relative_error"])
        for item in results
        if item.get("maximum_relative_error") is not None
    ]
    return {
        "passed": passed,
        "passed_cases": sum(bool(item.get("passed")) for item in results),
        "total_cases": len(results),
        "maximum_absolute_error": max(absolute, default=None),
        "maximum_relative_error": max(relative, default=None),
        "case_results": results,
    }


def _measure_batch(
    torch: Any,
    function: Callable[[], Any],
    repeats: int,
    *,
    observe_output: Callable[[Any], None] | None = None,
) -> float:
    if repeats < 1:
        raise ValueError("measurement repeats must be positive")
    event_type = getattr(torch.npu, "Event", None)
    if event_type is not None:
        try:
            start = event_type(enable_timing=True)
            end = event_type(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = function()
            end.record()
            _sync(torch)
            elapsed_ms = float(start.elapsed_time(end))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _sync(torch)
        else:
            if math.isfinite(elapsed_ms) and elapsed_ms > 0:
                if observe_output is not None:
                    observe_output(output)
                return elapsed_ms * 1000.0 / repeats
    _sync(torch)
    started = time.perf_counter_ns()
    for _ in range(repeats):
        output = function()
    _sync(torch)
    elapsed_us = (time.perf_counter_ns() - started) / 1000.0 / repeats
    if observe_output is not None:
        observe_output(output)
    return elapsed_us


def _measurement_session(
    torch: Any,
    candidate: Callable[[], Any],
    baseline: Callable[[], Any],
    *,
    batches: int,
    target_batch_time_ms: float,
    observe_candidate_output: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    candidate_once = max(
        _measure_batch(
            torch,
            candidate,
            1,
            observe_output=observe_candidate_output,
        ),
        1e-3,
    )
    baseline_once = max(_measure_batch(torch, baseline, 1), 1e-3)
    repeats = max(
        1,
        min(
            1_000_000,
            math.ceil(target_batch_time_ms * 1000.0 / min(candidate_once, baseline_once)),
        ),
    )
    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    for batch in range(batches):
        if batch % 2 == 0:
            baseline_samples.append(_measure_batch(torch, baseline, repeats))
            candidate_samples.append(
                _measure_batch(
                    torch,
                    candidate,
                    repeats,
                    observe_output=observe_candidate_output,
                )
            )
        else:
            candidate_samples.append(
                _measure_batch(
                    torch,
                    candidate,
                    repeats,
                    observe_output=observe_candidate_output,
                )
            )
            baseline_samples.append(_measure_batch(torch, baseline, repeats))
    return (
        summarize_samples(candidate_samples).to_dict(),
        summarize_samples(baseline_samples).to_dict(),
        repeats,
    )


def _benchmark(
    torch: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    warmup = int(settings.get("warmup", task.benchmark.get("warmup", 20)))
    batches = int(
        settings.get("measurement_batches", task.benchmark.get("measurement_batches", 7))
    )
    target_ms = float(
        settings.get("target_batch_time_ms", task.benchmark.get("target_batch_time_ms", 200.0))
    )
    maximum_cv = float(settings.get("maximum_cv", task.benchmark.get("maximum_cv", 0.05)))
    rerun = bool(settings.get("rerun_if_unstable", True))
    if warmup < 1 or batches < 1 or target_ms <= 0 or not 0 < maximum_cv <= 1:
        raise ValueError("invalid benchmark settings")

    per_case: list[dict[str, Any]] = []
    for case in cases:
        candidate_inputs = generate_inputs(task, case, torch, device).args
        baseline_inputs = generate_inputs(task, case, torch, device).args
        originals = tuple(value.clone() for value in candidate_inputs)
        expected_inputs = tuple(value.clone() for value in candidate_inputs)
        expected = reference(task, expected_inputs, torch)
        _sync(torch)

        def candidate_call(inputs: tuple[Any, ...] = candidate_inputs) -> Any:
            with _candidate_guard(torch):
                return entry(*inputs)

        def baseline_call(inputs: tuple[Any, ...] = baseline_inputs) -> Any:
            return reference(task, inputs, torch)

        def validate_timed_output(
            output: Any,
            *,
            selected_case: CaseSpec = case,
            selected_expected: Any = expected,
            selected_inputs: tuple[Any, ...] = candidate_inputs,
            selected_originals: tuple[Any, ...] = originals,
        ) -> None:
            validation = validate_output(
                task,
                selected_case,
                output,
                selected_expected,
                torch,
                inputs=selected_inputs,
            )
            if not bool(validation.get("passed")):
                raise RuntimeError(
                    f"case {selected_case.id}: timed candidate output failed validation"
                )
            if not _unchanged(torch, selected_inputs, selected_originals):
                raise RuntimeError(
                    f"case {selected_case.id}: candidate modified an input during benchmark"
                )

        warmup_output: Any = None
        for _ in range(warmup):
            warmup_output = candidate_call()
            baseline_call()
        _sync(torch)
        validate_timed_output(warmup_output)
        attempts: list[dict[str, Any]] = []
        candidate_stats: dict[str, Any]
        baseline_stats: dict[str, Any]
        repeats = 1
        maximum_observed_cv = math.inf
        attempt_count = 2 if rerun else 1
        for _attempt in range(attempt_count):
            candidate_stats, baseline_stats, repeats = _measurement_session(
                torch,
                candidate_call,
                baseline_call,
                batches=batches,
                target_batch_time_ms=target_ms,
                observe_candidate_output=validate_timed_output,
            )
            maximum_observed_cv = max(
                float(candidate_stats["cv"]), float(baseline_stats["cv"])
            )
            attempts.append(
                {
                    "candidate": candidate_stats,
                    "baseline_eager": baseline_stats,
                    "repeats_per_batch": repeats,
                }
            )
            if maximum_observed_cv <= maximum_cv:
                break
        if not _unchanged(torch, candidate_inputs, originals):
            raise RuntimeError(f"case {case.id}: candidate modified an input during benchmark")
        candidate_median = float(candidate_stats["median_us"])
        baseline_median = float(baseline_stats["median_us"])
        per_case.append(
            {
                "case_id": case.id,
                "dtype": case.dtype,
                "params": dict(case.params),
                "weight": case.weight,
                "candidate": candidate_stats,
                "baseline_eager": baseline_stats,
                "speedup_vs_eager": baseline_median / candidate_median,
                "stable": maximum_observed_cv <= maximum_cv,
                "measurement_attempts": attempts,
                "repeats_per_batch": repeats,
                "timed_output_validation": True,
            }
        )
    summary = speedup_summary(per_case)
    stable = all(bool(item["stable"]) for item in per_case)
    observed_cvs = [
        max(float(item["candidate"]["cv"]), float(item["baseline_eager"]["cv"]))
        for item in per_case
    ]
    return {
        "passed": stable,
        "status": "stable" if stable else "unstable",
        "per_case": per_case,
        "maximum_cv": max(observed_cvs, default=0.0),
        **summary,
    }


def _profile(
    torch: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    case = cases[0]
    args = generate_inputs(task, case, torch, device).args
    originals = tuple(value.clone() for value in args)
    warmup = max(1, int(settings.get("warmup", 5)))
    iterations = max(1, int(settings.get("iterations", 10)))
    for _ in range(warmup):
        with _candidate_guard(torch):
            entry(*args)
    _sync(torch)
    for _ in range(iterations):
        with _candidate_guard(torch):
            output = entry(*args)
    _sync(torch)
    if not isinstance(output, torch.Tensor):
        raise TypeError("custom_op did not return a Tensor")
    if not _unchanged(torch, args, originals):
        raise RuntimeError("candidate modified an input Tensor during profiling")
    return {
        "passed": True,
        "case_id": case.id,
        "warmup": warmup,
        "iterations": iterations,
    }


def _single_measurement_session(
    torch: Any,
    function: Callable[[], Any],
    *,
    batches: int,
    target_batch_time_ms: float,
) -> tuple[dict[str, Any], int]:
    once = max(_measure_batch(torch, function, 1), 1e-3)
    repeats = max(
        1,
        min(1_000_000, math.ceil(target_batch_time_ms * 1000.0 / once)),
    )
    samples = [_measure_batch(torch, function, repeats) for _ in range(batches)]
    return summarize_samples(samples).to_dict(), repeats


def _baselines(
    torch: Any,
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    warmup = int(settings.get("warmup", task.benchmark.get("warmup", 20)))
    batches = int(
        settings.get("measurement_batches", task.benchmark.get("measurement_batches", 7))
    )
    target_ms = float(
        settings.get("target_batch_time_ms", task.benchmark.get("target_batch_time_ms", 200.0))
    )
    if warmup < 1 or batches < 1 or target_ms <= 0:
        raise ValueError("invalid baseline benchmark settings")
    per_case: list[dict[str, Any]] = []
    compile_errors: list[str] = []
    compile_successes = 0
    compile_api = getattr(torch, "compile", None)
    for case in cases:
        eager_inputs = generate_inputs(task, case, torch, device).args

        def eager_call(inputs: tuple[Any, ...] = eager_inputs) -> Any:
            return reference(task, inputs, torch)

        for _ in range(warmup):
            eager_call()
        _sync(torch)
        eager_stats, eager_repeats = _single_measurement_session(
            torch,
            eager_call,
            batches=batches,
            target_batch_time_ms=target_ms,
        )
        compiled_us: float | None = None
        compiled_stats: dict[str, Any] | None = None
        compile_error: str | None = None
        if not callable(compile_api):
            compile_error = "torch.compile is not available in this runtime"
        else:
            compiled_inputs = generate_inputs(task, case, torch, device).args

            def compile_target(*args: Any) -> Any:
                return reference(task, args, torch)

            try:
                compiled = compile_api(compile_target)

                def compiled_call(
                    compiled_function: Any = compiled,
                    inputs: tuple[Any, ...] = compiled_inputs,
                ) -> Any:
                    return compiled_function(*inputs)

                for _ in range(warmup):
                    compiled_call()
                _sync(torch)
                compiled_stats, _compiled_repeats = _single_measurement_session(
                    torch,
                    compiled_call,
                    batches=batches,
                    target_batch_time_ms=target_ms,
                )
                compiled_us = float(compiled_stats["median_us"])
                compile_successes += 1
            except Exception as exc:  # torch.compile backend errors vary by CANN release
                compile_error = f"{type(exc).__name__}: {exc}"[:4096]
                compile_errors.append(f"{case.id}: {compile_error}")
        per_case.append(
            {
                "case_id": case.id,
                "dtype": case.dtype,
                "params": dict(case.params),
                "weight": case.weight,
                "pytorch_eager_us": float(eager_stats["median_us"]),
                "pytorch_eager": eager_stats,
                "torch_compile_us": compiled_us,
                "torch_compile": compiled_stats,
                "torch_compile_error": compile_error,
                "official_us": None,
                "official": None,
                "eager_repeats_per_batch": eager_repeats,
            }
        )
    reasons: dict[str, Any] = {
        "official": (
            "task has no trusted official_baseline implementation; "
            "the shipped task baseline is the B0 eager reference"
        )
    }
    if compile_successes != len(cases):
        reasons["torch_compile"] = (
            compile_errors
            if compile_errors
            else ["torch.compile is unavailable for every requested case"]
        )
    return {
        "passed": True,
        "per_case": per_case,
        "torch_compile_available": compile_successes == len(cases),
        "torch_compile_partial": 0 < compile_successes < len(cases),
        "official_available": False,
        "unavailable_reasons": reasons,
    }


def _redact_details(stage: str, details: Mapping[str, Any]) -> dict[str, Any]:
    """Remove case identities, shapes, seeds, and element locations from output."""

    common = {"passed": bool(details.get("passed"))}
    if stage == "compile":
        return {
            **common,
            "compiled": bool(details.get("compiled")),
            "compile_time_ms": details.get("compile_time_ms"),
            "compiled_case_count": len(details.get("case_results", ())),
            "triton_version": details.get("triton_version"),
            "case_details_redacted": True,
        }
    if stage == "correctness":
        return {
            **common,
            "passed_cases": details.get("passed_cases"),
            "total_cases": details.get("total_cases"),
            "maximum_absolute_error": details.get("maximum_absolute_error"),
            "maximum_relative_error": details.get("maximum_relative_error"),
            "case_details_redacted": True,
        }
    if stage == "benchmark":
        return {
            **common,
            "status": details.get("status"),
            "geomean_speedup_vs_eager": details.get("geomean_speedup_vs_eager"),
            "minimum_speedup_vs_eager": details.get("minimum_speedup_vs_eager"),
            "maximum_speedup_vs_eager": details.get("maximum_speedup_vs_eager"),
            "maximum_cv": details.get("maximum_cv"),
            "measured_case_count": len(details.get("per_case", ())),
            "case_details_redacted": True,
        }
    if stage == "profile":
        return {
            **common,
            "warmup": details.get("warmup"),
            "iterations": details.get("iterations"),
            "case_details_redacted": True,
        }
    return {**common, "case_details_redacted": True}


def execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(payload.get("stage", ""))
    if stage not in {"baseline", "compile", "correctness", "benchmark", "profile"}:
        raise ValueError(f"unsupported isolated stage: {stage!r}")
    task_raw = payload.get("task")
    if not isinstance(task_raw, Mapping):
        raise ValueError("stage payload task must be an object")
    task = _task(task_raw)
    cases = _cases(payload.get("cases"))
    device = str(payload.get("device", "npu:0"))
    if not device.startswith("npu:") or not device.removeprefix("npu:").isdigit():
        raise ValueError("stage device must use npu:<index> syntax")
    settings_raw = payload.get("settings", {})
    if not isinstance(settings_raw, Mapping):
        raise ValueError("stage settings must be an object")
    torch, _torch_npu, triton = _runtime()
    device_index = int(device.removeprefix("npu:"))
    torch.npu.set_device(device_index)
    if stage == "baseline":
        details = _baselines(torch, task, cases, device, settings_raw)
    else:
        _module, entry = _candidate(Path(str(payload.get("candidate", "candidate.py"))))
    if stage == "compile":
        details = _compile(torch, triton, entry, task, cases, device)
    elif stage == "correctness":
        details = _correctness(torch, entry, task, cases, device)
    elif stage == "benchmark":
        details = _benchmark(torch, entry, task, cases, device, settings_raw)
    elif stage == "profile":
        details = _profile(torch, entry, task, cases, device, settings_raw)
    if bool(settings_raw.get("redact_case_details", False)):
        details = _redact_details(stage, details)
    return {
        "schema_version": "ascend_isolated_stage_v1",
        "stage": stage,
        "started_at": str(payload.get("started_at", _utc_iso())),
        "finished_at": _utc_iso(),
        "passed": bool(details.get("passed")),
        "details": details,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--payload-file")
    options = parser.parse_args(argv)
    started = _utc_iso()
    result_path = Path("stage_result.json")
    private_cases = False
    try:
        payload = _read_payload(options.payload_file)
        settings = payload.get("settings", {})
        private_cases = isinstance(settings, Mapping) and bool(
            settings.get("redact_case_details", False)
        )
        result_name = str(payload.get("result", "stage_result.json"))
        requested = Path(result_name)
        if requested.is_absolute() or len(requested.parts) != 1 or requested.is_symlink():
            raise ValueError("result path must be a basename in the stage cwd")
        result_path = requested
        result = execute(payload)
        _atomic_json(result_path, result)
        return 0 if bool(result["passed"]) else 2
    except BaseException as exc:
        failure = {
            "schema_version": "ascend_isolated_stage_v1",
            "stage": "unknown",
            "started_at": started,
            "finished_at": _utc_iso(),
            "passed": False,
            "details": {},
            "error": {
                "type": type(exc).__name__,
                "message": (
                    "hidden evaluation stage failed; diagnostic details redacted"
                    if private_cases
                    else str(exc)[:16_384]
                ),
                "traceback": (
                    None
                    if private_cases
                    else traceback.format_exc(limit=20)[-32_768:]
                ),
            },
        }
        with contextlib.suppress(OSError):
            _atomic_json(result_path, failure)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through StageRunner
    raise SystemExit(main())
