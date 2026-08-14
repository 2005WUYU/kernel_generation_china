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
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from ascend_kernel_lab.evaluation.benchmark import (
    speedup_summary,
    summarize_latency_breakdown,
    summarize_samples,
    weighted_geometric_mean,
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


class _InfrastructureStageError(RuntimeError):
    pass


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


def _candidate(
    path: Path, *, module_name: str = "candidate"
) -> tuple[ModuleType, Callable[..., Any]]:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved.parent != cwd or path.is_symlink() or not resolved.is_file():
        raise ValueError("candidate must be a regular file directly inside the stage cwd")
    module_spec = importlib.util.spec_from_file_location(module_name, resolved)
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


def _tensor_output(torch: Any, output: Any) -> bool:
    if isinstance(output, torch.Tensor):
        return True
    return isinstance(output, (tuple, list)) and bool(output) and all(
        isinstance(item, torch.Tensor) for item in output
    )


def _compile(
    torch: Any,
    triton: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
    prepared_case: dict[str, Any] | None = None,
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
        if not _tensor_output(torch, output):
            raise TypeError(f"case {case.id}: custom_op did not return Tensor output(s)")
        if not _unchanged(torch, generated.args, originals):
            raise RuntimeError(f"case {case.id}: candidate modified an input Tensor")
        if prepared_case is not None and not prepared_case:
            prepared_case.update(
                {
                    "case": case,
                    "args": generated.args,
                    "originals": originals,
                    "actual": output,
                }
            )
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
    prepared_case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    compile_probe_reused = False
    for case in cases:
        reuse = (
            prepared_case
            if prepared_case is not None and prepared_case.get("case") == case
            else None
        )
        if reuse is None:
            generated = generate_inputs(task, case, torch, device)
            args = generated.args
            originals = tuple(value.clone() for value in args)
            actual = None
        else:
            compile_probe_reused = True
            args = tuple(reuse["args"])
            originals = tuple(reuse["originals"])
            actual = reuse["actual"]
        reference_args = tuple(value.clone() for value in args)
        expected = reference(task, reference_args, torch, case=case)
        if reuse is None:
            with _candidate_guard(torch):
                actual = entry(*args)
        _sync(torch)
        validation = validate_output(
            task,
            case,
            actual,
            expected,
            torch,
            inputs=args,
        )
        inputs_unchanged = _unchanged(torch, args, originals)
        validation["inputs_unchanged"] = inputs_unchanged
        if not inputs_unchanged:
            validation["passed"] = False
            validation["error"] = "candidate modified an input Tensor"
        results.append(validation)
        if not bool(validation.get("passed")):
            break
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
        "compile_probe_reused": compile_probe_reused,
    }


def _measure_batch(
    torch: Any,
    function: Callable[[], Any],
    repeats: int,
    *,
    observe_output: Callable[[Any], None] | None = None,
    timing_sink: list[dict[str, Any]] | None = None,
) -> float:
    if repeats < 1:
        raise ValueError("measurement repeats must be positive")
    event_type = getattr(torch.npu, "Event", None)
    if event_type is not None:
        try:
            start = event_type(enable_timing=True)
            end = event_type(enable_timing=True)
            wall_started = time.perf_counter_ns()
            start.record()
            for _ in range(repeats):
                output = function()
            end.record()
            _sync(torch)
            wall_elapsed_us = (time.perf_counter_ns() - wall_started) / 1000.0 / repeats
            elapsed_ms = float(start.elapsed_time(end))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _sync(torch)
        else:
            if math.isfinite(elapsed_ms) and elapsed_ms > 0:
                if observe_output is not None:
                    observe_output(output)
                device_latency_us = elapsed_ms * 1000.0 / repeats
                if timing_sink is not None:
                    timing_sink.append(
                        {
                            "device_latency_us": device_latency_us,
                            "end_to_end_latency_us": max(
                                wall_elapsed_us, device_latency_us
                            ),
                        }
                    )
                return device_latency_us
    _sync(torch)
    started = time.perf_counter_ns()
    for _ in range(repeats):
        output = function()
    _sync(torch)
    elapsed_us = (time.perf_counter_ns() - started) / 1000.0 / repeats
    if observe_output is not None:
        observe_output(output)
    if timing_sink is not None:
        timing_sink.append(
            {
                "device_latency_us": None,
                "end_to_end_latency_us": elapsed_us,
            }
        )
    return elapsed_us


def _measurement_session(
    torch: Any,
    candidate: Callable[[], Any],
    baseline: Callable[[], Any],
    *,
    batches: int,
    target_batch_time_ms: float,
    candidate_context: Callable[[], contextlib.AbstractContextManager[None]],
    observe_candidate_output: Callable[[Any], None] | None = None,
    incumbent: Callable[[], Any] | None = None,
    incumbent_context: Callable[[], contextlib.AbstractContextManager[None]] | None = None,
    observe_incumbent_output: Callable[[Any], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, int]:
    def measure_candidate(
        repeats: int, *, timing_sink: list[dict[str, Any]] | None = None
    ) -> float:
        last_output: Any = None

        def tracked_candidate() -> Any:
            nonlocal last_output
            last_output = candidate()
            return last_output

        with candidate_context():
            latency = _measure_batch(
                torch,
                tracked_candidate,
                repeats,
                timing_sink=timing_sink,
            )
        if observe_candidate_output is not None:
            observe_candidate_output(last_output)
        return latency

    candidate_once = max(measure_candidate(1), 1e-3)
    baseline_once = max(_measure_batch(torch, baseline, 1), 1e-3)
    incumbent_once = None
    if incumbent is not None:
        assert incumbent_context is not None
        with incumbent_context():
            incumbent_once = max(_measure_batch(torch, incumbent, 1), 1e-3)
    repeats = max(
        1,
        min(
            1_000_000,
            math.ceil(
                target_batch_time_ms
                * 1000.0
                / min(
                    candidate_once,
                    baseline_once,
                    incumbent_once if incumbent_once is not None else math.inf,
                )
            ),
        ),
    )
    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    incumbent_samples: list[float] = []
    candidate_batch_timings: list[dict[str, Any]] = []
    baseline_batch_timings: list[dict[str, Any]] = []
    incumbent_batch_timings: list[dict[str, Any]] = []

    def measure_incumbent(repeats: int) -> float:
        assert incumbent is not None and incumbent_context is not None
        last_output: Any = None

        def tracked_incumbent() -> Any:
            nonlocal last_output
            last_output = incumbent()
            return last_output

        with incumbent_context():
            latency = _measure_batch(
                torch,
                tracked_incumbent,
                repeats,
                timing_sink=incumbent_batch_timings,
            )
        if observe_incumbent_output is not None:
            observe_incumbent_output(last_output)
        return latency

    for batch in range(batches):
        if batch % 2 == 0:
            if incumbent is not None:
                incumbent_samples.append(measure_incumbent(repeats))
            baseline_samples.append(
                _measure_batch(
                    torch,
                    baseline,
                    repeats,
                    timing_sink=baseline_batch_timings,
                )
            )
            candidate_samples.append(
                measure_candidate(repeats, timing_sink=candidate_batch_timings)
            )
        else:
            candidate_samples.append(
                measure_candidate(repeats, timing_sink=candidate_batch_timings)
            )
            baseline_samples.append(
                _measure_batch(
                    torch,
                    baseline,
                    repeats,
                    timing_sink=baseline_batch_timings,
                )
            )
            if incumbent is not None:
                incumbent_samples.append(measure_incumbent(repeats))
    candidate_statistics = summarize_samples(candidate_samples).to_dict()
    candidate_statistics["latency_breakdown"] = summarize_latency_breakdown(
        candidate_batch_timings
    )
    baseline_statistics = summarize_samples(baseline_samples).to_dict()
    baseline_statistics["latency_breakdown"] = summarize_latency_breakdown(
        baseline_batch_timings
    )
    incumbent_statistics = None
    if incumbent_samples:
        incumbent_statistics = summarize_samples(incumbent_samples).to_dict()
        incumbent_statistics["latency_breakdown"] = summarize_latency_breakdown(
            incumbent_batch_timings
        )
    return candidate_statistics, baseline_statistics, incumbent_statistics, repeats


def _benchmark(
    torch: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    cases: Sequence[CaseSpec],
    device: str,
    settings: Mapping[str, Any],
    incumbent_entry: Callable[..., Any] | None = None,
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
        incumbent_inputs = (
            generate_inputs(task, case, torch, device).args
            if incumbent_entry is not None
            else None
        )
        originals = tuple(value.clone() for value in candidate_inputs)
        expected_inputs = tuple(value.clone() for value in candidate_inputs)
        expected = reference(task, expected_inputs, torch, case=case)
        incumbent_originals = (
            tuple(value.clone() for value in incumbent_inputs)
            if incumbent_inputs is not None
            else None
        )
        incumbent_expected = (
            reference(
                task,
                tuple(value.clone() for value in incumbent_inputs),
                torch,
                case=case,
            )
            if incumbent_inputs is not None
            else None
        )
        _sync(torch)

        def candidate_call(inputs: tuple[Any, ...] = candidate_inputs) -> Any:
            return entry(*inputs)

        def baseline_call(inputs: tuple[Any, ...] = baseline_inputs) -> Any:
            return reference(task, inputs, torch, case=case)

        def incumbent_call(inputs: tuple[Any, ...] = incumbent_inputs or ()) -> Any:
            assert incumbent_entry is not None
            return incumbent_entry(*inputs)

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
        incumbent_warmup_output: Any = None
        with _candidate_guard(torch):
            for _ in range(warmup):
                warmup_output = candidate_call()
            if incumbent_entry is not None:
                for _ in range(warmup):
                    incumbent_warmup_output = incumbent_call()
        for _ in range(warmup):
            baseline_call()
        _sync(torch)
        validate_timed_output(warmup_output)
        if incumbent_inputs is not None and incumbent_originals is not None:
            incumbent_validation = validate_output(
                task,
                case,
                incumbent_warmup_output,
                incumbent_expected,
                torch,
                inputs=incumbent_inputs,
            )
            if not bool(incumbent_validation.get("passed")):
                raise RuntimeError(
                    f"case {case.id}: incumbent BEST output failed validation"
                )
            if not _unchanged(torch, incumbent_inputs, incumbent_originals):
                raise RuntimeError(
                    f"case {case.id}: incumbent BEST modified an input during benchmark"
                )
        attempts: list[dict[str, Any]] = []
        candidate_stats: dict[str, Any]
        baseline_stats: dict[str, Any]
        incumbent_stats: dict[str, Any] | None
        repeats = 1
        maximum_observed_cv = math.inf
        attempt_count = 2 if rerun else 1
        for _attempt in range(attempt_count):
            candidate_stats, baseline_stats, incumbent_stats, repeats = _measurement_session(
                torch,
                candidate_call,
                baseline_call,
                batches=batches,
                target_batch_time_ms=target_ms,
                candidate_context=lambda: _candidate_guard(torch),
                observe_candidate_output=validate_timed_output,
                incumbent=(incumbent_call if incumbent_entry is not None else None),
                incumbent_context=(
                    (lambda: _candidate_guard(torch))
                    if incumbent_entry is not None
                    else None
                ),
            )
            maximum_observed_cv = max(
                float(candidate_stats["cv"]), float(baseline_stats["cv"])
            )
            attempts.append(
                {
                    "candidate": candidate_stats,
                    "baseline_eager": baseline_stats,
                    "incumbent_best": incumbent_stats,
                    "repeats_per_batch": repeats,
                }
            )
            if maximum_observed_cv <= maximum_cv:
                break
        if not _unchanged(torch, candidate_inputs, originals):
            raise RuntimeError(f"case {case.id}: candidate modified an input during benchmark")
        candidate_median = float(candidate_stats["median_us"])
        baseline_median = float(baseline_stats["median_us"])
        incumbent_median = (
            float(incumbent_stats["median_us"])
            if incumbent_stats is not None
            else None
        )
        per_case.append(
            {
                "case_id": case.id,
                "dtype": case.dtype,
                "params": dict(case.params),
                "weight": case.weight,
                "candidate": candidate_stats,
                "baseline_eager": baseline_stats,
                "speedup_vs_eager": baseline_median / candidate_median,
                "incumbent_best": incumbent_stats,
                "speedup_vs_best": (
                    incumbent_median / candidate_median
                    if incumbent_median is not None
                    else None
                ),
                "candidate_vs_best_latency_delta_us": (
                    candidate_median - incumbent_median
                    if incumbent_median is not None
                    else None
                ),
                "candidate_vs_best_latency_change_fraction": (
                    candidate_median / incumbent_median - 1.0
                    if incumbent_median not in {None, 0.0}
                    else None
                ),
                "stable": maximum_observed_cv <= maximum_cv,
                "measurement_attempts": attempts,
                "repeats_per_batch": repeats,
                "timed_output_validation": True,
                "device_latency_us": candidate_stats["latency_breakdown"][
                    "device_latency_us"
                ],
                "end_to_end_latency_us": candidate_stats["latency_breakdown"][
                    "end_to_end_latency_us"
                ],
                "host_overhead_us": candidate_stats["latency_breakdown"][
                    "host_overhead_us"
                ],
            }
        )
    summary = speedup_summary(per_case)
    stable = all(bool(item["stable"]) for item in per_case)
    observed_cvs = [
        max(
            float(item["candidate"]["cv"]),
            float(item["baseline_eager"]["cv"]),
        )
        for item in per_case
    ]
    paired = [item for item in per_case if item.get("speedup_vs_best") is not None]
    paired_summary = None
    if paired:
        speeds = [float(item["speedup_vs_best"]) for item in paired]
        weights = [float(item.get("weight", 1.0)) for item in paired]
        paired_cvs = [
            max(
                float(item["candidate"]["cv"]),
                float(item["incumbent_best"]["cv"]),
            )
            for item in paired
        ]
        paired_summary = {
            "measurement": "same_process_interleaved_incumbent_candidate",
            "geomean_speedup_vs_best": weighted_geometric_mean(speeds, weights),
            "minimum_speedup_vs_best": min(speeds),
            "maximum_speedup_vs_best": max(speeds),
            "maximum_cv": max(paired_cvs, default=0.0),
            "per_case": [
                {
                    "case_id": item["case_id"],
                    "weight": item["weight"],
                    "candidate_median_us": item["candidate"]["median_us"],
                    "best_median_us": item["incumbent_best"]["median_us"],
                    "speedup_vs_best": item["speedup_vs_best"],
                    "latency_delta_us": item[
                        "candidate_vs_best_latency_delta_us"
                    ],
                    "latency_change_fraction": item[
                        "candidate_vs_best_latency_change_fraction"
                    ],
                    "candidate_cv": item["candidate"]["cv"],
                    "best_cv": item["incumbent_best"]["cv"],
                }
                for item in paired
            ],
        }
    return {
        "passed": True,
        "measurement_config": {
            "benchmark_mode": str(settings.get("benchmark_mode", "final")),
            "warmup": warmup,
            "measurement_batches": batches,
            "target_batch_time_ms": target_ms,
            "maximum_cv": maximum_cv,
            "rerun_if_unstable": rerun,
        },
        "measurement_stable": stable,
        "per_case": per_case,
        "maximum_cv": max(observed_cvs, default=0.0),
        "paired_best_comparison": paired_summary,
        **summary,
    }


def _candidate_stage(
    name: str,
    function: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    started = _utc_iso()
    began = time.perf_counter()
    try:
        details = dict(function())
    except BaseException as exc:
        infrastructure = isinstance(exc, _InfrastructureStageError)
        failure_type = (
            type(exc.__cause__).__name__
            if infrastructure and exc.__cause__ is not None
            else type(exc).__name__
        )
        return {
            "stage": name.upper(),
            "status": "fail",
            "passed": False,
            "started_at": started,
            "finished_at": _utc_iso(),
            "duration_seconds": time.perf_counter() - began,
            "details": {
                "failure_origin": (
                    "infrastructure" if infrastructure else "candidate"
                ),
                "failure_type": failure_type,
            },
            "failure_origin": (
                "infrastructure" if infrastructure else "candidate"
            ),
            "failure_type": failure_type,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:16_384],
                "traceback": traceback.format_exc(limit=20)[-32_768:],
            },
            "retryable": False,
        }
    passed = bool(details.get("passed"))
    result = {
        "stage": name.upper(),
        "status": "pass" if passed else "fail",
        "passed": passed,
        "started_at": started,
        "finished_at": _utc_iso(),
        "duration_seconds": time.perf_counter() - began,
        "details": details,
        "error": None,
        "retryable": False,
        **details,
    }
    if not passed:
        result.setdefault("failure_origin", "candidate")
        result.setdefault("failure_type", f"{name}_failed")
    return result


def _candidate_evaluation(
    torch: Any,
    triton: Any,
    entry: Callable[..., Any],
    task: TaskSpec,
    correctness_cases: Sequence[CaseSpec],
    benchmark_cases: Sequence[CaseSpec],
    device: str,
    settings: Mapping[str, Any],
    incumbent_entry: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    prepared_case: dict[str, Any] = {}
    compile_result = _candidate_stage(
        "compile",
        lambda: _compile(
            torch,
            triton,
            entry,
            task,
            correctness_cases[:1],
            device,
            prepared_case,
        ),
    )
    if not bool(compile_result["passed"]):
        return {
            "passed": False,
            "outcome": "compile_failed",
            "failure_origin": compile_result["failure_origin"],
            "failure_type": compile_result["failure_type"],
            "compile": compile_result,
            "correctness": None,
            "benchmark": None,
        }

    correctness_result = _candidate_stage(
        "correctness",
        lambda: _correctness(
            torch,
            entry,
            task,
            correctness_cases,
            device,
            prepared_case,
        ),
    )
    prepared_case.clear()
    if not bool(correctness_result["passed"]):
        return {
            "passed": False,
            "outcome": "correctness_failed",
            "failure_origin": correctness_result["failure_origin"],
            "failure_type": correctness_result["failure_type"],
            "compile": compile_result,
            "correctness": correctness_result,
            "benchmark": None,
        }

    benchmark_result = _candidate_stage(
        "benchmark",
        lambda: _benchmark(
            torch,
            entry,
            task,
            benchmark_cases,
            device,
            settings,
            incumbent_entry,
        ),
    )
    result = {
        "passed": bool(benchmark_result["passed"]),
        "outcome": (
            "correct" if bool(benchmark_result["passed"]) else "benchmark_failed"
        ),
        "compile": compile_result,
        "correctness": correctness_result,
        "benchmark": benchmark_result,
    }
    if not bool(benchmark_result["passed"]):
        result["failure_origin"] = benchmark_result["failure_origin"]
        result["failure_type"] = benchmark_result["failure_type"]
    return result


def _profile(
    torch: Any,
    torch_npu: Any,
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
    stream = torch_npu.npu.current_stream()
    range_id = torch_npu.npu.mstx.range_start("akg_candidate", stream)
    try:
        for _ in range(iterations):
            with _candidate_guard(torch):
                output = entry(*args)
    finally:
        torch_npu.npu.mstx.range_end(range_id)
    _sync(torch)
    if not _tensor_output(torch, output):
        raise TypeError("custom_op did not return Tensor output(s)")
    if not _unchanged(torch, args, originals):
        raise RuntimeError("candidate modified an input Tensor during profiling")
    return {
        "passed": True,
        "case_id": case.id,
        "warmup": warmup,
        "iterations": iterations,
        "attribution_range": "akg_candidate",
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
    comparison_baseline = str(settings.get("comparison_baseline", "pytorch_eager"))
    if comparison_baseline != "pytorch_eager":
        raise ValueError("baseline comparison must be pytorch_eager")
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
    for case in cases:
        eager_inputs = generate_inputs(task, case, torch, device).args

        def eager_call(inputs: tuple[Any, ...] = eager_inputs) -> Any:
            return reference(task, inputs, torch, case=case)

        for _ in range(warmup):
            eager_call()
        _sync(torch)
        eager_stats, eager_repeats = _single_measurement_session(
            torch,
            eager_call,
            batches=batches,
            target_batch_time_ms=target_ms,
        )
        per_case.append(
            {
                "case_id": case.id,
                "dtype": case.dtype,
                "params": dict(case.params),
                "weight": case.weight,
                "pytorch_eager_us": float(eager_stats["median_us"]),
                "pytorch_eager": eager_stats,
                "eager_repeats_per_batch": eager_repeats,
            }
        )
    return {
        "passed": True,
        "status": "complete",
        "mode": "pytorch_eager_only",
        "comparison_baseline": comparison_baseline,
        "compared_baselines": [comparison_baseline],
        "not_measured_baselines": ["torch_compile", "official"],
        "per_case": per_case,
    }


def _redact_details(stage: str, details: Mapping[str, Any]) -> dict[str, Any]:
    """Remove case identities, shapes, seeds, and element locations from output."""

    common = {"passed": bool(details.get("passed"))}
    failure = {
        "failure_origin": details.get("failure_origin"),
        "failure_type": details.get("failure_type"),
    }
    if stage == "compile":
        return {
            **common,
            **failure,
            "compiled": bool(details.get("compiled")),
            "compile_time_ms": details.get("compile_time_ms"),
            "compiled_case_count": len(details.get("case_results", ())),
            "triton_version": details.get("triton_version"),
            "case_details_redacted": True,
        }
    if stage == "correctness":
        return {
            **common,
            **failure,
            "passed_cases": details.get("passed_cases"),
            "total_cases": details.get("total_cases"),
            "maximum_absolute_error": details.get("maximum_absolute_error"),
            "maximum_relative_error": details.get("maximum_relative_error"),
            "case_details_redacted": True,
        }
    if stage == "benchmark":
        return {
            **common,
            **failure,
            "measurement_stable": details.get("measurement_stable"),
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
            **failure,
            "warmup": details.get("warmup"),
            "iterations": details.get("iterations"),
            "case_details_redacted": True,
        }
    return {**common, **failure, "case_details_redacted": True}


def execute(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(payload.get("stage", ""))
    if stage not in {
        "baseline",
        "compile",
        "correctness",
        "benchmark",
        "candidate_evaluation",
        "profile",
    }:
        raise ValueError(f"unsupported isolated stage: {stage!r}")
    task_raw = payload.get("task")
    if not isinstance(task_raw, Mapping):
        raise ValueError("stage payload task must be an object")
    task = _task(task_raw)
    cases = _cases(payload.get("cases"))
    task = replace(task, public_cases=cases)
    correctness_cases = tuple(case for case in cases if case.kind == "correctness")
    benchmark_cases = tuple(case for case in cases if case.kind == "benchmark")
    device = str(payload.get("device", "npu:0"))
    if not device.startswith("npu:") or not device.removeprefix("npu:").isdigit():
        raise ValueError("stage device must use npu:<index> syntax")
    settings_raw = payload.get("settings", {})
    if not isinstance(settings_raw, Mapping):
        raise ValueError("stage settings must be an object")
    try:
        torch, torch_npu, triton = _runtime()
        device_index = int(device.removeprefix("npu:"))
        torch.npu.set_device(device_index)
    except BaseException as exc:
        raise _InfrastructureStageError(str(exc)) from exc
    if stage == "baseline":
        details = _baselines(torch, task, cases, device, settings_raw)
    else:
        _module, entry = _candidate(Path(str(payload.get("candidate", "candidate.py"))))
    incumbent_entry = None
    incumbent_path = Path("incumbent.py")
    if stage == "candidate_evaluation" and incumbent_path.is_file():
        _incumbent_module, incumbent_entry = _candidate(
            incumbent_path, module_name="incumbent"
        )
    if stage == "compile":
        details = _compile(torch, triton, entry, task, cases, device)
    elif stage == "correctness":
        details = _correctness(torch, entry, task, cases, device)
    elif stage == "benchmark":
        details = _benchmark(torch, entry, task, cases, device, settings_raw)
    elif stage == "candidate_evaluation":
        details = _candidate_evaluation(
            torch,
            triton,
            entry,
            task,
            correctness_cases,
            benchmark_cases,
            device,
            settings_raw,
            incumbent_entry,
        )
    elif stage == "profile":
        details = _profile(
            torch, torch_npu, entry, task, cases, device, settings_raw
        )
    if bool(settings_raw.get("redact_case_details", False)):
        details = _redact_details(stage, details)
    return {
        "schema_version": "ascend_isolated_stage_v1",
        "stage": stage,
        "started_at": str(payload.get("started_at", _utc_iso())),
        "finished_at": _utc_iso(),
        "passed": bool(details.get("passed")),
        "failure_origin": details.get("failure_origin"),
        "failure_type": details.get("failure_type"),
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
        infrastructure = isinstance(exc, _InfrastructureStageError)
        failure_type = (
            type(exc.__cause__).__name__
            if infrastructure and exc.__cause__ is not None
            else type(exc).__name__
        )
        failure = {
            "schema_version": "ascend_isolated_stage_v1",
            "stage": "unknown",
            "started_at": started,
            "finished_at": _utc_iso(),
            "passed": False,
            "details": {
                "failure_origin": (
                    "infrastructure" if infrastructure else "candidate"
                ),
                "failure_type": failure_type,
            },
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
