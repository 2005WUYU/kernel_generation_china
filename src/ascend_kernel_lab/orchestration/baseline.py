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
        return {
            "schema_version": "ascend_baseline_snapshot_v1",
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
            "torch_compile_available": measured.get("torch_compile_available", False),
            "official_available": measured.get("official_available", False),
            "unavailable_reasons": measured.get("unavailable_reasons", {}),
        }
