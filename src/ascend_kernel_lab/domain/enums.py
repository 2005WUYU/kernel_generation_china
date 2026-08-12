"""Stable string enumerations used by both the controller and SQLite store."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """A Python 3.10-compatible ``StrEnum`` with readable string conversion."""

    def __str__(self) -> str:
        return str(self.value)


class RoundState(StrEnum):
    """Durable checkpoints for one model-generation/evaluation round."""

    ROUND_CREATED = "ROUND_CREATED"
    PROMPT_COMMITTED = "PROMPT_COMMITTED"
    MODEL_REQUEST_SENT = "MODEL_REQUEST_SENT"
    MODEL_RESPONSE_COMMITTED = "MODEL_RESPONSE_COMMITTED"
    SOURCE_VALIDATED = "SOURCE_VALIDATED"
    COMPILE_FINISHED = "COMPILE_FINISHED"
    CORRECTNESS_FINISHED = "CORRECTNESS_FINISHED"
    BENCHMARK_FINISHED = "BENCHMARK_FINISHED"
    PROFILE_FINISHED = "PROFILE_FINISHED"
    FEEDBACK_COMMITTED = "FEEDBACK_COMMITTED"
    ROUND_FINISHED = "ROUND_FINISHED"


class TaskState(StrEnum):
    """Lifecycle of a kernel task, including the final hidden evaluation."""

    TASK_CREATED = "TASK_CREATED"
    ROUNDS_RUNNING = "ROUNDS_RUNNING"
    SELECT_BEST_CANDIDATE = "SELECT_BEST_CANDIDATE"
    HIDDEN_CORRECTNESS_TEST = "HIDDEN_CORRECTNESS_TEST"
    FINAL_BENCHMARK = "FINAL_BENCHMARK"
    FINAL_FULL_PROFILE = "FINAL_FULL_PROFILE"
    TASK_FINISHED = "TASK_FINISHED"
    TASK_FAILED = "TASK_FAILED"


class ExperimentState(StrEnum):
    """Top-level experiment state."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(StrEnum):
    """Persistent evaluation queue states."""

    QUEUED = "QUEUED"
    RETRY_WAIT = "RETRY_WAIT"
    LEASED = "LEASED"
    SUCCEEDED = "SUCCEEDED"
    DEAD = "DEAD"
    CANCELLED = "CANCELLED"


class EvaluationStage(StrEnum):
    """Worker stages represented in an evaluation job payload/result."""

    SOURCE_CHECK = "SOURCE_CHECK"
    COMPILE = "COMPILE"
    CORRECTNESS = "CORRECTNESS"
    BENCHMARK = "BENCHMARK"
    PROFILE = "PROFILE"
    FINAL_HIDDEN_CORRECTNESS = "FINAL_HIDDEN_CORRECTNESS"
    FINAL_BENCHMARK = "FINAL_BENCHMARK"
    FINAL_FULL_PROFILE = "FINAL_FULL_PROFILE"
    FULL_EVALUATION = "FULL_EVALUATION"


class CandidateValidity(StrEnum):
    """Why a candidate can or cannot receive a performance score."""

    VALID = "VALID"
    SOURCE_FAILED = "SOURCE_FAILED"
    COMPILE_FAILED = "COMPILE_FAILED"
    CORRECTNESS_FAILED = "CORRECTNESS_FAILED"
    ANTI_BYPASS_FAILED = "ANTI_BYPASS_FAILED"
    UNSTABLE = "UNSTABLE"
