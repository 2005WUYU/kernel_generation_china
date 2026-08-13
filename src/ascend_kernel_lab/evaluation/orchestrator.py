from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ascend_kernel_lab.domain import CandidateScore, compute_reward
from ascend_kernel_lab.tasks import CaseSpec, TaskSpec


class StageLike(Protocol):
    @property
    def passed(self) -> bool: ...

    def to_dict(self) -> dict[str, Any]: ...


class EvaluationBackend(Protocol):
    def source_check(self, candidate_path: Path, task: TaskSpec) -> StageLike: ...

    def compile(self, candidate_path: Path, task: TaskSpec, cases: Sequence[CaseSpec], artifact_dir: Path) -> StageLike: ...

    def check_correctness(self, candidate_path: Path, task: TaskSpec, cases: Sequence[CaseSpec], artifact_dir: Path) -> StageLike: ...

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, Any] | None = None,
    ) -> StageLike: ...

    def profile(self, candidate_path: Path, task: TaskSpec, cases: Sequence[CaseSpec], artifact_dir: Path) -> StageLike: ...


@dataclass(frozen=True)
class EvaluationRequest:
    experiment_id: str
    task: TaskSpec
    round_number: int
    candidate_id: str
    candidate_path: Path
    artifact_dir: Path
    correctness_cases: tuple[CaseSpec, ...] | None = None
    benchmark_cases: tuple[CaseSpec, ...] | None = None
    profile_cases: tuple[CaseSpec, ...] | None = None
    baseline_snapshot: Mapping[str, Any] | None = None
    run_profile: bool = True
    profile_coverage_required: bool = True
    minimum_kernel_coverage: float = 0.90
    hidden: bool = False

    def __post_init__(self) -> None:
        if self.round_number < 1:
            raise ValueError("round_number must be positive")
        if not 0 <= self.minimum_kernel_coverage <= 1:
            raise ValueError("minimum_kernel_coverage must be between zero and one")


@dataclass(frozen=True)
class EvaluationResult:
    schema_version: str
    experiment_id: str
    task_id: str
    round_number: int
    candidate_id: str
    overall_status: str
    source: Mapping[str, Any]
    compile: Mapping[str, Any] | None
    correctness: Mapping[str, Any] | None
    benchmark: Mapping[str, Any] | None
    profile: Mapping[str, Any] | None
    anti_bypass: Mapping[str, Any]
    reward_vector: Mapping[str, Any]
    score: CandidateScore

    @property
    def is_valid(self) -> bool:
        return self.score.is_publicly_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "round": self.round_number,
            "candidate_id": self.candidate_id,
            "overall_status": self.overall_status,
            "source": dict(self.source),
            "compile": dict(self.compile) if self.compile is not None else None,
            "correctness": dict(self.correctness) if self.correctness is not None else None,
            "benchmark": dict(self.benchmark) if self.benchmark is not None else None,
            "profile": dict(self.profile) if self.profile is not None else None,
            "anti_bypass": dict(self.anti_bypass),
            "reward_vector": dict(self.reward_vector),
            "score": asdict(self.score),
        }


def _stage_dict(stage: StageLike) -> dict[str, Any]:
    value = stage.to_dict()
    details = value.get("details")
    if isinstance(details, Mapping):
        # Keep envelope fields and surface normalized metrics for consumers.
        return {**value, **details}
    return value


def _failure_result(
    request: EvaluationRequest,
    *,
    overall_status: str,
    source: Mapping[str, Any],
    compile_result: Mapping[str, Any] | None = None,
    correctness: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    score = CandidateScore(
        candidate_id=request.candidate_id,
        round_number=request.round_number,
        compile_passed=compile_result is not None and _passed(compile_result),
        correctness_passed=correctness is not None and _passed(correctness),
        anti_bypass_passed=False,
        hidden_correctness_passed=False if request.hidden else None,
    )
    reward = compute_reward(score)
    return EvaluationResult(
        schema_version="ascend_evaluation_result_v1",
        experiment_id=request.experiment_id,
        task_id=request.task.id,
        round_number=request.round_number,
        candidate_id=request.candidate_id,
        overall_status=overall_status,
        source=source,
        compile=compile_result,
        correctness=correctness,
        benchmark=None,
        profile=None,
        anti_bypass={"passed": False, "status": "not_evaluated", "reason": overall_status},
        reward_vector=asdict(reward),
        score=score,
    )

def _passed(result: Mapping[str, Any]) -> bool:
    if "passed" in result:
        return bool(result["passed"])
    status = str(result.get("status", "")).upper()
    return status == "PASS"


def _number(mapping: Mapping[str, Any] | None, *keys: str) -> float | None:
    if mapping is None:
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _profile_summary(profile: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if profile is None:
        return {}
    value = profile.get("summary")
    return value if isinstance(value, Mapping) else profile


def evaluate_candidate(backend: EvaluationBackend, request: EvaluationRequest) -> EvaluationResult:
    """Run compile, correctness, benchmark, then quick profile in order."""
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    source_stage = backend.source_check(request.candidate_path, request.task)
    source = _stage_dict(source_stage)
    if not source_stage.passed:
        return _failure_result(request, overall_status="source_failed", source=source)

    correctness_cases = request.correctness_cases or request.task.correctness_cases
    compile_stage = backend.compile(request.candidate_path, request.task, correctness_cases, request.artifact_dir)
    compile_result = _stage_dict(compile_stage)
    if not compile_stage.passed:
        return _failure_result(
            request, overall_status="compile_failed", source=source, compile_result=compile_result,
        )

    correctness_stage = backend.check_correctness(
        request.candidate_path, request.task, correctness_cases, request.artifact_dir,
    )
    correctness = _stage_dict(correctness_stage)
    if not correctness_stage.passed:
        return _failure_result(
            request,
            overall_status="correctness_failed",
            source=source,
            compile_result=compile_result,
            correctness=correctness,
        )

    benchmark_cases = request.benchmark_cases or request.task.benchmark_cases
    benchmark_stage = backend.benchmark(
        request.candidate_path,
        request.task,
        benchmark_cases,
        request.artifact_dir,
        request.baseline_snapshot,
    )
    benchmark = _stage_dict(benchmark_stage)
    benchmark_passed = benchmark_stage.passed

    profile: dict[str, Any] | None = None
    if request.run_profile and benchmark_passed:
        profile_stage = backend.profile(
            request.candidate_path,
            request.task,
            request.profile_cases or request.task.profile_cases,
            request.artifact_dir,
        )
        profile = _stage_dict(profile_stage)

    profile_summary = _profile_summary(profile)
    coverage = _number(profile_summary, "candidate_kernel_coverage")
    kernel_count = int(profile_summary.get("kernel_count", 0) or 0)
    if not benchmark_passed:
        anti_bypass = {
            "passed": False,
            "status": "not_evaluated",
            "reason": "benchmark_failed",
            "coverage": None,
        }
    elif not request.run_profile:
        anti_bypass = {"passed": not request.profile_coverage_required, "status": "not_run", "coverage": None}
    elif profile is None or not _passed(profile):
        anti_bypass = {"passed": not request.profile_coverage_required, "status": "not_verifiable", "coverage": coverage}
    else:
        anti_bypass = {
            "passed": kernel_count >= 1 and coverage is not None and coverage >= request.minimum_kernel_coverage,
            "status": "verified" if coverage is not None else "not_verifiable",
            "candidate_kernel_coverage": coverage,
            "minimum_required_coverage": request.minimum_kernel_coverage,
            "kernel_count": kernel_count,
        }

    geomean = (
        _number(benchmark, "geomean_speedup_vs_eager", "speedup_geomean")
        if benchmark_passed
        else None
    )
    minimum = (
        _number(benchmark, "minimum_speedup_vs_eager", "minimum_speedup")
        if benchmark_passed
        else None
    )
    stability = _number(benchmark, "maximum_cv", "stability_cv")
    score = CandidateScore(
        candidate_id=request.candidate_id,
        round_number=request.round_number,
        compile_passed=True,
        correctness_passed=True,
        anti_bypass_passed=bool(anti_bypass["passed"]),
        hidden_correctness_passed=True if request.hidden else None,
        minimum_speedup=minimum,
        geomean_speedup=geomean,
        candidate_kernel_coverage=coverage,
        stability_cv=stability,
    )
    reward = compute_reward(score)
    if not benchmark_passed:
        overall = "benchmark_failed"
    elif score.is_publicly_valid:
        overall = "correct"
    else:
        overall = "anti_bypass_failed"
    if (
        benchmark_passed
        and profile is not None
        and not _passed(profile)
        and not request.profile_coverage_required
    ):
        overall = "profile_unavailable"
    return EvaluationResult(
        schema_version="ascend_evaluation_result_v1",
        experiment_id=request.experiment_id,
        task_id=request.task.id,
        round_number=request.round_number,
        candidate_id=request.candidate_id,
        overall_status=overall,
        source=source,
        compile=compile_result,
        correctness=correctness,
        benchmark=benchmark,
        profile=profile,
        anti_bypass=anti_bypass,
        reward_vector=asdict(reward),
        score=score,
    )
