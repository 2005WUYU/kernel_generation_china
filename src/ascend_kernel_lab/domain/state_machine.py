"""Strict, side-effect-free lifecycle validation.

Persistence code uses these state machines before issuing compare-and-swap
updates. Keeping the transition graph in the domain layer prevents a restarted
controller from skipping durable checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Generic, TypeVar

from .enums import RoundState, TaskState
from .errors import InvalidTransitionError

StateT = TypeVar("StateT", RoundState, TaskState)


class StrictStateMachine(Generic[StateT]):
    """Validate transitions against an immutable directed graph."""

    def __init__(
        self,
        transitions: Mapping[StateT, Iterable[StateT]],
        terminal_states: Iterable[StateT],
    ) -> None:
        self._transitions: dict[StateT, frozenset[StateT]] = {
            source: frozenset(destinations)
            for source, destinations in transitions.items()
        }
        self._terminal_states: frozenset[StateT] = frozenset(terminal_states)

    def allowed_transitions(self, current: StateT) -> frozenset[StateT]:
        """Return the exact next states permitted from ``current``."""

        allowed: frozenset[StateT] | None = self._transitions.get(current)
        return allowed if allowed is not None else frozenset()

    def is_terminal(self, state: StateT) -> bool:
        return state in self._terminal_states

    def can_transition(self, current: StateT, target: StateT) -> bool:
        return target in self.allowed_transitions(current)

    def transition(self, current: StateT, target: StateT) -> StateT:
        """Return ``target`` or fail without mutating any state."""

        if not self.can_transition(current, target):
            allowed = ", ".join(
                sorted(state.value for state in self.allowed_transitions(current))
            )
            suffix = f"; allowed: {allowed}" if allowed else "; state is terminal"
            raise InvalidTransitionError(
                f"invalid transition {current.value} -> {target.value}{suffix}"
            )
        return target

    def validate_path(self, states: Iterable[StateT]) -> None:
        """Validate all adjacent edges in a proposed state history."""

        iterator = iter(states)
        try:
            previous = next(iterator)
        except StopIteration:
            return
        for current in iterator:
            self.transition(previous, current)
            previous = current


ROUND_TRANSITIONS: Mapping[RoundState, frozenset[RoundState]] = {
    RoundState.ROUND_CREATED: frozenset({RoundState.PROMPT_COMMITTED}),
    RoundState.PROMPT_COMMITTED: frozenset({RoundState.MODEL_REQUEST_SENT}),
    RoundState.MODEL_REQUEST_SENT: frozenset(
        {RoundState.MODEL_RESPONSE_COMMITTED}
    ),
    RoundState.MODEL_RESPONSE_COMMITTED: frozenset({RoundState.SOURCE_VALIDATED}),
    RoundState.SOURCE_VALIDATED: frozenset(
        {RoundState.COMPILE_FINISHED, RoundState.FEEDBACK_COMMITTED}
    ),
    RoundState.COMPILE_FINISHED: frozenset(
        {RoundState.CORRECTNESS_FINISHED, RoundState.FEEDBACK_COMMITTED}
    ),
    RoundState.CORRECTNESS_FINISHED: frozenset(
        {RoundState.BENCHMARK_FINISHED, RoundState.FEEDBACK_COMMITTED}
    ),
    RoundState.BENCHMARK_FINISHED: frozenset({RoundState.PROFILE_FINISHED}),
    RoundState.PROFILE_FINISHED: frozenset({RoundState.FEEDBACK_COMMITTED}),
    RoundState.FEEDBACK_COMMITTED: frozenset({RoundState.ROUND_FINISHED}),
    RoundState.ROUND_FINISHED: frozenset(),
}

TASK_TRANSITIONS: Mapping[TaskState, frozenset[TaskState]] = {
    TaskState.TASK_CREATED: frozenset(
        {TaskState.ROUNDS_RUNNING, TaskState.TASK_FAILED}
    ),
    TaskState.ROUNDS_RUNNING: frozenset(
        {TaskState.SELECT_BEST_CANDIDATE, TaskState.TASK_FAILED}
    ),
    TaskState.SELECT_BEST_CANDIDATE: frozenset(
        {TaskState.HIDDEN_CORRECTNESS_TEST, TaskState.TASK_FAILED}
    ),
    TaskState.HIDDEN_CORRECTNESS_TEST: frozenset(
        {TaskState.FINAL_BENCHMARK, TaskState.TASK_FAILED}
    ),
    TaskState.FINAL_BENCHMARK: frozenset(
        {
            TaskState.FINAL_FULL_PROFILE,
            TaskState.TASK_FINISHED,
            TaskState.TASK_FAILED,
        }
    ),
    TaskState.FINAL_FULL_PROFILE: frozenset(
        {TaskState.TASK_FINISHED, TaskState.TASK_FAILED}
    ),
    TaskState.TASK_FINISHED: frozenset(),
    TaskState.TASK_FAILED: frozenset(),
}


class RoundStateMachine(StrictStateMachine[RoundState]):
    def __init__(self) -> None:
        super().__init__(ROUND_TRANSITIONS, {RoundState.ROUND_FINISHED})


class TaskStateMachine(StrictStateMachine[TaskState]):
    def __init__(self) -> None:
        super().__init__(
            TASK_TRANSITIONS,
            {TaskState.TASK_FINISHED, TaskState.TASK_FAILED},
        )


round_state_machine = RoundStateMachine()
task_state_machine = TaskStateMachine()
