"""Recoverable experiment control, feedback, and baseline workflows."""

from .baseline import (
    BaselineManager,
    baseline_identity,
    prompt_baseline_projection,
    validate_baseline_snapshot,
)
from .controller import ControllerError, ExperimentController, TaskRunSummary
from .feedback import build_feedback

__all__ = [
    "BaselineManager",
    "ControllerError",
    "ExperimentController",
    "TaskRunSummary",
    "baseline_identity",
    "build_feedback",
    "prompt_baseline_projection",
    "validate_baseline_snapshot",
]
