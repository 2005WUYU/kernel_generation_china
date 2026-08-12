"""Candidate validation, execution orchestration, statistics, and scoring."""

from .benchmark import BenchmarkStatistics, summarize_samples, weighted_geometric_mean
from .orchestrator import EvaluationRequest, EvaluationResult, evaluate_candidate
from .source_guard import SourceGuard, SourceGuardResult

__all__ = [
    "BenchmarkStatistics",
    "EvaluationRequest",
    "EvaluationResult",
    "SourceGuard",
    "SourceGuardResult",
    "evaluate_candidate",
    "summarize_samples",
    "weighted_geometric_mean",
]
