"""Deterministic backend for controller, queue, and crash-recovery tests."""

from __future__ import annotations

import copy
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec

from .base import Backend, StageResult


class FakeBackend(Backend):
    """A scriptable backend whose default path is a complete successful run."""

    def __init__(
        self,
        scripted: Mapping[EvaluationStage | str, Sequence[StageResult]] | None = None,
    ) -> None:
        self._scripted: dict[str, deque[StageResult]] = defaultdict(deque)
        for stage, results in (scripted or {}).items():
            key = stage.value if isinstance(stage, EvaluationStage) else str(stage)
            self._scripted[key].extend(results)
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _key(stage: EvaluationStage | str) -> str:
        return stage.value if isinstance(stage, EvaluationStage) else str(stage)

    def queue_result(self, result: StageResult) -> None:
        self._scripted[self._key(result.stage)].append(result)

    def _result(
        self,
        stage: EvaluationStage | str,
        *,
        candidate_path: Path | None = None,
        task: TaskSpec | None = None,
        cases: Sequence[CaseSpec] = (),
        artifact_dir: Path | None = None,
        default_details: Mapping[str, Any] | None = None,
    ) -> StageResult:
        self.calls.append(
            {
                "stage": self._key(stage),
                "candidate_path": str(candidate_path) if candidate_path else None,
                "task_id": task.id if task else None,
                "case_ids": [case.id for case in cases],
                "artifact_dir": str(artifact_dir) if artifact_dir else None,
            }
        )
        queue = self._scripted[self._key(stage)]
        if queue:
            return queue.popleft()
        return StageResult.success(stage, details=copy.deepcopy(dict(default_details or {})))

    def source_check(self, candidate_path: Path, task: TaskSpec) -> StageResult:
        return self._result(
            EvaluationStage.SOURCE_CHECK,
            candidate_path=candidate_path,
            task=task,
            default_details={"passed": True, "syntax_ok": True},
        )

    def compile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._result(
            EvaluationStage.COMPILE,
            candidate_path=candidate_path,
            task=task,
            cases=cases,
            artifact_dir=artifact_dir,
            default_details={"compiled": True, "compile_time_ms": 1.0},
        )

    def check_correctness(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._result(
            EvaluationStage.CORRECTNESS,
            candidate_path=candidate_path,
            task=task,
            cases=cases,
            artifact_dir=artifact_dir,
            default_details={
                "passed": True,
                "passed_cases": len(cases),
                "total_cases": len(cases),
                "case_results": [
                    {"case_id": case.id, "passed": True} for case in cases
                ],
            },
        )

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
    ) -> StageResult:
        del baseline_snapshot
        per_case = [
            {
                "case_id": case.id,
                "weight": case.weight,
                "candidate": {"median_us": 10.0, "cv": 0.01},
                "baseline_eager": {"median_us": 12.0, "cv": 0.01},
                "speedup_vs_eager": 1.2,
            }
            for case in cases
        ]
        return self._result(
            EvaluationStage.BENCHMARK,
            candidate_path=candidate_path,
            task=task,
            cases=cases,
            artifact_dir=artifact_dir,
            default_details={
                "status": "stable",
                "per_case": per_case,
                "geomean_speedup_vs_eager": 1.2 if per_case else None,
                "minimum_speedup_vs_eager": 1.2 if per_case else None,
            },
        )

    def profile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        return self._result(
            EvaluationStage.PROFILE,
            candidate_path=candidate_path,
            task=task,
            cases=cases,
            artifact_dir=artifact_dir,
            default_details={
                "profile_available": True,
                "kernel_count": 1,
                "candidate_kernel_coverage": 1.0,
            },
        )

    def health_check(self) -> StageResult:
        return self._result(
            "HEALTH_CHECK",
            default_details={"healthy": True, "device_available": True},
        )

    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "stage": "BASELINE",
                "candidate_path": None,
                "task_id": task.id,
                "case_ids": [case.id for case in cases],
                "artifact_dir": str(artifact_dir),
            }
        )
        return {
            "per_case": [
                {
                    "case_id": case.id,
                    "weight": case.weight,
                    "pytorch_eager_us": 12.0,
                    "torch_compile_us": 10.0,
                    "official_us": None,
                }
                for case in cases
            ],
            "torch_compile_available": True,
            "official_available": False,
            "unavailable_reasons": {
                "official": "fake backend has no official operator baseline"
            },
        }
