from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ascend_kernel_lab.evaluation.benchmark import weighted_geometric_mean
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec


class BaselineBackend(Protocol):
    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]: ...


def baseline_identity(
    *,
    task: TaskSpec,
    environment_sha256: str,
    harness_git_commit: str,
    benchmark_config: Mapping[str, Any],
) -> str:
    value = {
        "task_spec_sha256": task.digest(),
        "environment_sha256": environment_sha256,
        "harness_git_commit": harness_git_commit,
        "benchmark_config": dict(benchmark_config),
        "protocol": "ascend_baseline_v1",
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_baseline_snapshot(
    snapshot: Mapping[str, Any], *, expected_identity: str
) -> bool:
    if snapshot.get("identity_sha256") != expected_identity:
        return False
    cases = snapshot.get("per_case")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return False
    for case in cases:
        if not isinstance(case, Mapping):
            return False
        eager = case.get("pytorch_eager_us")
        if eager is None:
            return False
        try:
            number = float(eager)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(number) or number <= 0:
            return False
    return True


def prompt_baseline_projection(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project an eager snapshot without raw samples or measurement attempts."""

    comparison_baseline = str(
        snapshot.get("comparison_baseline", "pytorch_eager")
    )
    raw_cases = snapshot.get("per_case")
    if not isinstance(raw_cases, Sequence) or isinstance(
        raw_cases, (str, bytes)
    ):
        return {"comparison_baseline": comparison_baseline}

    projected_cases: list[dict[str, Any]] = []
    latencies: list[float] = []
    weights: list[float] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        latency_value = raw_case.get("pytorch_eager_us")
        if latency_value is None:
            continue
        latency = float(latency_value)
        case: dict[str, Any] = {
            "case_id": str(raw_case.get("case_id", "")),
            "median_us": latency,
        }
        for name in ("dtype", "shape", "shapes"):
            if name in raw_case:
                case[name] = raw_case[name]
        params = raw_case.get("params")
        if isinstance(params, Mapping):
            case["params"] = dict(params)
        weight = float(raw_case.get("weight", 1.0))
        case["weight"] = weight
        projected_cases.append(case)
        latencies.append(latency)
        weights.append(weight)

    if not projected_cases:
        return {"comparison_baseline": comparison_baseline}
    geomean = snapshot.get("pytorch_eager_geomean_us")
    if geomean is None:
        geomean = weighted_geometric_mean(latencies, weights)
    return {
        "comparison_baseline": comparison_baseline,
        "definition": {
            "implementation": "trusted_pytorch_eager_reference",
            "measurement": "same_device_npu_event_median_per_public_benchmark_case",
            "latency_unit": "microseconds",
            "speedup_formula": "pytorch_eager_median_us / candidate_median_us",
        },
        "summary": {
            "case_count": len(projected_cases),
            "weighted_geomean_us": float(geomean),
        },
        "per_case": projected_cases,
    }


@dataclass(frozen=True)
class BaselineManager:
    backend: BaselineBackend
    environment_sha256: str
    harness_git_commit: str
    benchmark_config: Mapping[str, Any]

    def measure(self, task: TaskSpec, artifact_dir: Path) -> dict[str, Any]:
        identity = baseline_identity(
            task=task,
            environment_sha256=self.environment_sha256,
            harness_git_commit=self.harness_git_commit,
            benchmark_config=self.benchmark_config,
        )
        measured = dict(
            self.backend.measure_baselines(task, task.benchmark_cases, artifact_dir)
        )
        cases = measured.get("per_case", [])
        if not isinstance(cases, list):
            raise ValueError("baseline backend must return a per_case list")
        eager_values: list[float] = []
        weights: list[float] = []
        for case in cases:
            if not isinstance(case, Mapping) or case.get("pytorch_eager_us") is None:
                raise ValueError("B0 PyTorch eager latency is mandatory for every case")
            eager = float(case["pytorch_eager_us"])
            if not math.isfinite(eager) or eager <= 0:
                raise ValueError("baseline latencies must be finite and positive")
            eager_values.append(eager)
            weights.append(float(case.get("weight", 1.0)))
        comparison_baseline = str(
            measured.get("comparison_baseline", "pytorch_eager")
        )
        if comparison_baseline != "pytorch_eager":
            raise ValueError("baseline backend must compare against PyTorch eager")
        return {
            "schema_version": "ascend_baseline_snapshot_v1",
            "status": "complete",
            "mode": "pytorch_eager_only",
            "comparison_baseline": comparison_baseline,
            "compared_baselines": [comparison_baseline],
            "not_measured_baselines": ["torch_compile", "official"],
            "identity_sha256": identity,
            "task_id": task.id,
            "task_spec_sha256": task.digest(),
            "environment_sha256": self.environment_sha256,
            "harness_git_commit": self.harness_git_commit,
            "per_case": cases,
            "pytorch_eager_geomean_us": (
                weighted_geometric_mean(eager_values, weights)
                if eager_values
                else None
            ),
        }
