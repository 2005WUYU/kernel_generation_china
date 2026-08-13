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
        benchmark_settings: Mapping[str, Any] | None = None,
    ) -> StageLike: ...

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
    benchmark_settings: Mapping[str, Any] | None = None
    incumbent_path: Path | None = None
    combine_candidate_stages: bool = False
    run_profile: bool = True
    search_profile_policy: bool = False
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
    candidate: Mapping[str, Any]
    performance: Mapping[str, Any]
    infrastructure: Mapping[str, Any]
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
            "candidate": dict(self.candidate),
            "performance": dict(self.performance),
            "attribution": dict(self.anti_bypass),
            "infrastructure": dict(self.infrastructure),
            # Compatibility alias for already-committed v1 consumers.
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


def _combined_stage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result = dict(value)
    details = result.get("details")
    if isinstance(details, Mapping):
        result.update(details)
    return result


def _failure_result(
    request: EvaluationRequest,
    *,
    failure_stage: str,
    source: Mapping[str, Any],
    compile_result: Mapping[str, Any] | None = None,
    correctness: Mapping[str, Any] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    stage_result = {
        "source": source,
        "compile": compile_result,
        "correctness": correctness,
        "benchmark": benchmark,
    }.get(failure_stage)
    infrastructure_failure = isinstance(stage_result, Mapping) and (
        _infrastructure_failure(stage_result)
        or stage_result.get("failure_origin") == "infrastructure"
    )
    overall_status = "INFRA_RETRY" if infrastructure_failure else "INVALID_CANDIDATE"
    reason = _stage_reason(stage_result, failure_stage)
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
        benchmark=benchmark,
        profile=None,
        anti_bypass={
            "passed": False,
            "status": "UNAVAILABLE",
            "advisory": True,
            "reason": overall_status,
        },
        candidate=_candidate_projection(
            source=source,
            compile_result=compile_result,
            correctness=correctness,
            status="UNEVALUATED" if infrastructure_failure else "INVALID",
            reason=reason,
        ),
        performance={"status": "NOT_MEASURED", "vs_baseline": None, "metrics": {}},
        infrastructure={
            "status": "RETRY" if infrastructure_failure else "OK",
            "stage": failure_stage if infrastructure_failure else None,
            "reason": reason if infrastructure_failure else None,
        },
        reward_vector=asdict(reward),
        score=score,
    )

def _passed(result: Mapping[str, Any]) -> bool:
    if "passed" in result:
        return bool(result["passed"])
    status = str(result.get("status", "")).upper()
    return status == "PASS"


def _infrastructure_failure(result: Mapping[str, Any]) -> bool:
    return str(result.get("status", "")).lower() in {
        "error",
        "timeout",
        "unavailable",
    }


def _stage_reason(result: Mapping[str, Any] | None, fallback: str) -> str:
    if not isinstance(result, Mapping):
        return fallback
    error = result.get("error")
    if isinstance(error, Mapping) and error.get("message"):
        return str(error["message"])
    return str(result.get("failure_type") or result.get("status") or fallback)


def _candidate_projection(
    *,
    source: Mapping[str, Any],
    compile_result: Mapping[str, Any] | None,
    correctness: Mapping[str, Any] | None,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "source": "PASS" if _passed(source) else "FAIL",
        "compile": (
            "PASS" if compile_result is not None and _passed(compile_result)
            else "FAIL" if compile_result is not None else "NOT_RUN"
        ),
        "correctness": (
            "PASS" if correctness is not None and _passed(correctness)
            else "FAIL" if correctness is not None else "NOT_RUN"
        ),
        "status": status,
        "reason": reason,
    }


def _performance_projection(
    benchmark: Mapping[str, Any] | None,
    *,
    measured: bool,
) -> dict[str, Any]:
    geomean = _number(benchmark, "geomean_speedup_vs_eager", "speedup_geomean")
    minimum = _number(benchmark, "minimum_speedup_vs_eager", "minimum_speedup")
    stability = _number(benchmark, "maximum_cv", "stability_cv")
    if not measured or geomean is None or minimum is None:
        return {"status": "NOT_MEASURED", "vs_baseline": None, "metrics": {}}
    tolerance = max(0.01, stability or 0.0)
    vs_baseline = (
        "FASTER" if geomean > 1.0 + tolerance
        else "SLOWER" if geomean < 1.0 - tolerance
        else "TIE"
    )
    return {
        "status": "MEASURED",
        "vs_baseline": vs_baseline,
        "metrics": {
            "geomean_speedup_vs_eager": geomean,
            "minimum_speedup_vs_eager": minimum,
            "maximum_cv": stability,
            "measurement_stable": (
                benchmark.get("measurement_stable")
                if isinstance(benchmark, Mapping) else None
            ),
        },
    }


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


def _profile_attribution(
    profile: Mapping[str, Any] | None,
    *,
    minimum_kernel_coverage: float,
) -> dict[str, Any]:
    summary = _profile_summary(profile)
    coverage = _number(summary, "candidate_kernel_coverage")
    kernel_count = int(summary.get("kernel_count", 0) or 0)
    verified = (
        profile is not None
        and _passed(profile)
        and kernel_count >= 1
        and coverage is not None
        and coverage >= minimum_kernel_coverage
    )
    partial = not verified and (
        kernel_count >= 1 or (coverage is not None and coverage > 0.0)
    )
    status = "VERIFIED" if verified else "PARTIAL" if partial else "UNAVAILABLE"
    reason = None
    if profile is None:
        reason = "profile_not_run"
    elif status == "PARTIAL":
        reason = "incomplete_kernel_attribution"
    elif _passed(profile) and kernel_count == 0:
        reason = "kernel_name_not_matched"
    elif status == "UNAVAILABLE":
        reason = profile.get("unavailable_reason") or profile.get("status")
    return {
        # ``passed`` is retained only for readers of the old anti_bypass field.
        "passed": verified,
        "status": status,
        "advisory": True,
        "candidate_kernel_coverage": coverage,
        "minimum_verified_coverage": minimum_kernel_coverage,
        "kernel_count": kernel_count,
        "reason": reason,
    }


def _search_profile_selected(
    benchmark: Mapping[str, Any] | None,
    *,
    incumbent_path: Path | None,
) -> bool:
    if incumbent_path is None:
        return True
    if not isinstance(benchmark, Mapping):
        return False
    paired = benchmark.get("paired_best_comparison")
    if not isinstance(paired, Mapping):
        return False
    geomean = _number(paired, "geomean_speedup_vs_best")
    minimum = _number(paired, "minimum_speedup_vs_best")
    maximum_cv = _number(paired, "maximum_cv") or 0.0
    if geomean is None or minimum is None:
        return False
    tolerance = max(0.01, maximum_cv)
    return geomean - 1.0 > tolerance and minimum - 1.0 >= -tolerance


def evaluate_candidate(backend: EvaluationBackend, request: EvaluationRequest) -> EvaluationResult:
    """Run compile, correctness, benchmark, then quick profile in order."""
    request.artifact_dir.mkdir(parents=True, exist_ok=True)
    source_stage = backend.source_check(request.candidate_path, request.task)
    source = _stage_dict(source_stage)
    if not source_stage.passed:
        return _failure_result(
            request,
            failure_stage="source",
            source=source,
        )

    correctness_cases = request.correctness_cases or request.task.correctness_cases
    benchmark_cases = request.benchmark_cases or request.task.benchmark_cases
    compile_result: dict[str, Any] | None
    correctness: dict[str, Any] | None
    benchmark: dict[str, Any] | None
    if request.hidden or not request.combine_candidate_stages:
        compile_stage = backend.compile(
            request.candidate_path,
            request.task,
            correctness_cases,
            request.artifact_dir,
        )
        compile_result = _stage_dict(compile_stage)
        if not compile_stage.passed:
            return _failure_result(
                request,
                failure_stage="compile",
                source=source,
                compile_result=compile_result,
            )
        correctness_stage = backend.check_correctness(
            request.candidate_path,
            request.task,
            correctness_cases,
            request.artifact_dir,
        )
        correctness = _stage_dict(correctness_stage)
        if not correctness_stage.passed:
            return _failure_result(
                request,
                failure_stage="correctness",
                source=source,
                compile_result=compile_result,
                correctness=correctness,
            )
        benchmark_stage = backend.benchmark(
            request.candidate_path,
            request.task,
            benchmark_cases,
            request.artifact_dir,
            request.baseline_snapshot,
            request.benchmark_settings,
        )
        benchmark = _stage_dict(benchmark_stage)
        benchmark_passed = benchmark_stage.passed
    else:
        candidate_stage = backend.candidate_evaluation(
            request.candidate_path,
            request.task,
            correctness_cases,
            benchmark_cases,
            request.artifact_dir,
            request.baseline_snapshot,
            request.benchmark_settings,
            request.incumbent_path,
        )
        candidate_result = _stage_dict(candidate_stage)
        compile_result = _combined_stage(candidate_result.get("compile"))
        correctness = _combined_stage(candidate_result.get("correctness"))
        benchmark = _combined_stage(candidate_result.get("benchmark"))
        outcome = str(candidate_result.get("outcome", "compile_failed"))
        if compile_result is None:
            compile_result = candidate_result
        if (
            candidate_result.get("failure_origin") == "infrastructure"
            or _infrastructure_failure(candidate_result)
        ):
            failure_stage = (
                "correctness" if correctness is not None and not _passed(correctness)
                else "compile" if not _passed(compile_result)
                else "benchmark"
            )
            return _failure_result(
                request,
                failure_stage=failure_stage,
                source=source,
                compile_result=compile_result,
                correctness=correctness,
                benchmark=benchmark,
            )
        if outcome == "compile_failed":
            return _failure_result(
                request,
                failure_stage="compile",
                source=source,
                compile_result=compile_result,
            )
        if outcome == "correctness_failed":
            return _failure_result(
                request,
                failure_stage="correctness",
                source=source,
                compile_result=compile_result,
                correctness=correctness,
            )
        benchmark_passed = benchmark is not None and _passed(benchmark)

    profile: dict[str, Any] | None = None
    if (
        request.run_profile
        and benchmark_passed
        and (
            not request.search_profile_policy
            or _search_profile_selected(
                benchmark, incumbent_path=request.incumbent_path
            )
        )
    ):
        profile_stage = backend.profile(
            request.candidate_path,
            request.task,
            request.profile_cases or request.task.profile_cases,
            request.artifact_dir,
        )
        profile = _stage_dict(profile_stage)

    attribution = _profile_attribution(
        profile,
        minimum_kernel_coverage=request.minimum_kernel_coverage,
    )
    if profile is not None:
        profile = {**profile, "attribution": dict(attribution)}
    coverage = _number(attribution, "candidate_kernel_coverage")

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
        anti_bypass_passed=bool(attribution["passed"]),
        hidden_correctness_passed=True if request.hidden else None,
        minimum_speedup=minimum,
        geomean_speedup=geomean,
        candidate_kernel_coverage=coverage,
        stability_cv=stability,
    )
    reward = compute_reward(score)
    performance = _performance_projection(benchmark, measured=benchmark_passed)
    if not benchmark_passed:
        infrastructure_failure = bool(
            benchmark is not None
            and (
                _infrastructure_failure(benchmark)
                or benchmark.get("failure_origin") == "infrastructure"
            )
        )
        overall = "INFRA_RETRY" if infrastructure_failure else "INVALID_CANDIDATE"
        candidate_status = "UNEVALUATED" if infrastructure_failure else "INVALID"
        candidate_reason = _stage_reason(benchmark, "benchmark_failed")
    elif performance["status"] != "MEASURED":
        overall = "INFRA_RETRY"
        infrastructure_failure = True
        candidate_status = "VALID"
        candidate_reason = "benchmark_metrics_missing"
    else:
        overall = "VALID"
        infrastructure_failure = False
        candidate_status = "VALID"
        candidate_reason = None
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
        anti_bypass=attribution,
        candidate=_candidate_projection(
            source=source,
            compile_result=compile_result,
            correctness=correctness,
            status=candidate_status,
            reason=candidate_reason,
        ),
        performance=performance,
        infrastructure={
            "status": "RETRY" if infrastructure_failure else "OK",
            "stage": "benchmark" if infrastructure_failure else None,
            "reason": candidate_reason if infrastructure_failure else None,
        },
        reward_vector=asdict(reward),
        score=score,
    )
