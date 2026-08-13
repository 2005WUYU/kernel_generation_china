"""Numerically stable aggregation, reward calculation, and candidate ranking."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import BenchmarkSample, CandidateScore

PERFORMANCE_TIE_FLOOR = 0.01


@dataclass(frozen=True)
class PublicCandidateComparison:
    """Noise-aware comparison against the incumbent public BEST.

    ``candidate_selection_key`` remains the deterministic final ranking policy
    for legacy runs.  This comparison is the online optimization policy: it
    changes BEST only when an observed improvement is larger than benchmark
    noise and does not hide a similarly certain worst-case regression.
    """

    decision: str
    tolerance_fraction: float
    geomean_change_fraction: float | None
    minimum_change_fraction: float | None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "decision": self.decision,
            "tolerance_fraction": self.tolerance_fraction,
            "geomean_change_fraction": self.geomean_change_fraction,
            "minimum_change_fraction": self.minimum_change_fraction,
            "geomean_change_percent": (
                self.geomean_change_fraction * 100.0
                if self.geomean_change_fraction is not None
                else None
            ),
            "minimum_change_percent": (
                self.minimum_change_fraction * 100.0
                if self.minimum_change_fraction is not None
                else None
            ),
        }


def compare_public_candidate(
    candidate: CandidateScore,
    incumbent: CandidateScore | None,
    *,
    tie_floor: float = PERFORMANCE_TIE_FLOOR,
) -> PublicCandidateComparison:
    """Classify a measured public candidate as INITIAL/NEW_BEST/TIE/REGRESSION."""

    if not math.isfinite(tie_floor) or tie_floor < 0:
        raise ValueError("tie_floor must be finite and non-negative")
    if not candidate.is_publicly_valid:
        return PublicCandidateComparison(
            "INVALID", tie_floor, None, None
        )
    if incumbent is None:
        return PublicCandidateComparison(
            "INITIAL_BEST", tie_floor, None, None
        )
    if not incumbent.is_publicly_valid:
        raise ValueError("incumbent BEST must be publicly valid")

    assert candidate.geomean_speedup is not None
    assert candidate.minimum_speedup is not None
    assert incumbent.geomean_speedup is not None
    assert incumbent.minimum_speedup is not None
    tolerance = max(
        tie_floor,
        candidate.stability_cv or 0.0,
        incumbent.stability_cv or 0.0,
    )
    geomean_delta = candidate.geomean_speedup / incumbent.geomean_speedup - 1.0
    minimum_delta = candidate.minimum_speedup / incumbent.minimum_speedup - 1.0
    if geomean_delta > tolerance and minimum_delta >= -tolerance:
        decision = "NEW_BEST"
    elif geomean_delta < -tolerance or minimum_delta < -tolerance:
        decision = "REGRESSION"
    else:
        decision = "TIE"
    return PublicCandidateComparison(
        decision, tolerance, geomean_delta, minimum_delta
    )


def weighted_geometric_mean(samples: Iterable[BenchmarkSample]) -> float:
    """Compute a weighted geometric mean in log space.

    The input is materialized once so generators work correctly and an empty
    benchmark set fails loudly instead of yielding a misleading neutral value.
    """

    values = tuple(samples)
    if not values:
        raise ValueError("at least one benchmark sample is required")
    total_weight = math.fsum(sample.weight for sample in values)
    weighted_log_sum = math.fsum(
        sample.weight * math.log(sample.speedup) for sample in values
    )
    return math.exp(weighted_log_sum / total_weight)


def aggregate_candidate_score(
    *,
    candidate_id: str,
    round_number: int,
    samples: Iterable[BenchmarkSample],
    compile_passed: bool = True,
    correctness_passed: bool = True,
    anti_bypass_passed: bool = True,
    hidden_correctness_passed: bool | None = None,
    candidate_kernel_coverage: float | None = None,
) -> CandidateScore:
    """Build a candidate score from per-case benchmark observations."""

    observations = tuple(samples)
    if not observations:
        raise ValueError("at least one benchmark sample is required")
    cvs = tuple(
        sample.coefficient_of_variation
        for sample in observations
        if sample.coefficient_of_variation is not None
    )
    # Worst-case CV is intentionally conservative: one unstable shape should
    # not disappear inside an average.
    stability_cv = max(cvs) if cvs else None
    return CandidateScore(
        candidate_id=candidate_id,
        round_number=round_number,
        compile_passed=compile_passed,
        correctness_passed=correctness_passed,
        anti_bypass_passed=anti_bypass_passed,
        hidden_correctness_passed=hidden_correctness_passed,
        minimum_speedup=min(sample.speedup for sample in observations),
        geomean_speedup=weighted_geometric_mean(observations),
        candidate_kernel_coverage=candidate_kernel_coverage,
        stability_cv=stability_cv,
    )


def _finite_or(value: float | None, fallback: float) -> float:
    return value if value is not None and math.isfinite(value) else fallback


def candidate_selection_key(score: CandidateScore) -> tuple[float, ...]:
    """Return the documented lexicographic ranking key.

    Ordering is hidden correctness, worst-shape speedup, geometric-mean
    speedup, stability, then the earlier round for deterministic ties.
    Profiler attribution is advisory and does not affect selection.
    """

    publicly_valid = float(score.is_publicly_valid)
    hidden_rank = (
        2.0
        if score.hidden_correctness_passed is True
        else 1.0
        if score.hidden_correctness_passed is None
        else 0.0
    )
    return (
        publicly_valid,
        hidden_rank if publicly_valid else 0.0,
        _finite_or(score.minimum_speedup, float("-inf")),
        _finite_or(score.geomean_speedup, float("-inf")),
        -_finite_or(score.stability_cv, float("inf")),
        -float(score.round_number),
    )


def select_best_candidate(
    candidates: Iterable[CandidateScore],
    *,
    require_hidden_correctness: bool = False,
) -> CandidateScore | None:
    """Select the best eligible historical candidate, never merely the latest."""

    eligible = [
        candidate
        for candidate in candidates
        if (
            candidate.is_finally_valid
            if require_hidden_correctness
            else candidate.is_publicly_valid
        )
    ]
    if not eligible:
        return None
    return max(eligible, key=candidate_selection_key)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


@dataclass(frozen=True)
class RewardBreakdown:
    """A scalar reward plus its independently reusable components."""

    reward: float
    compile_reward: float
    correctness_reward: float
    anti_bypass_reward: float
    speedup_component: float
    coverage_component: float
    stability_penalty: float


def compute_reward(
    score: CandidateScore,
    *,
    maximum_stable_cv: float = 0.05,
    stability_penalty_scale: float = 1.0,
) -> RewardBreakdown:
    """Compute the initial reward while preserving an auditable breakdown.

    Compile, correctness, and missing benchmark measurements are hard zeroes.
    Profiler attribution is advisory and does not gate or modify reward.  The
    penalty is the amount by which CV exceeds the configured stable threshold,
    scaled by the caller.
    """

    if not math.isfinite(maximum_stable_cv) or maximum_stable_cv < 0:
        raise ValueError("maximum_stable_cv must be finite and non-negative")
    if not math.isfinite(stability_penalty_scale) or stability_penalty_scale < 0:
        raise ValueError("stability_penalty_scale must be finite and non-negative")

    compile_reward = float(score.compile_passed)
    correctness_reward = float(score.correctness_passed)
    anti_bypass_reward = float(score.anti_bypass_passed)
    if not score.is_publicly_valid:
        return RewardBreakdown(
            reward=0.0,
            compile_reward=compile_reward,
            correctness_reward=correctness_reward,
            anti_bypass_reward=anti_bypass_reward,
            speedup_component=0.0,
            coverage_component=0.0,
            stability_penalty=0.0,
        )

    assert score.geomean_speedup is not None
    speedup_component = _clip(math.log2(score.geomean_speedup), -1.0, 2.0)
    coverage_component = 0.0
    cv = score.stability_cv or 0.0
    stability_penalty = max(0.0, cv - maximum_stable_cv) * stability_penalty_scale
    reward = max(
        0.0,
        1.0 + speedup_component + coverage_component - stability_penalty,
    )
    return RewardBreakdown(
        reward=reward,
        compile_reward=compile_reward,
        correctness_reward=correctness_reward,
        anti_bypass_reward=anti_bypass_reward,
        speedup_component=speedup_component,
        coverage_component=coverage_component,
        stability_penalty=stability_penalty,
    )


def rank_candidates(candidates: Sequence[CandidateScore]) -> list[CandidateScore]:
    """Return a best-first deterministic ranking without mutating the input."""

    return sorted(candidates, key=candidate_selection_key, reverse=True)
