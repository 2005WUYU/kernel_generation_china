from __future__ import annotations

import math
import unittest

from ascend_kernel_lab.domain import (
    BenchmarkSample,
    CandidateScore,
    InvalidTransitionError,
    RoundState,
    TaskState,
    aggregate_candidate_score,
    compare_public_candidate,
    compute_reward,
    round_state_machine,
    select_best_candidate,
    task_state_machine,
    weighted_geometric_mean,
)


class RoundStateMachineTests(unittest.TestCase):
    def test_complete_success_path(self) -> None:
        states = [
            RoundState.ROUND_CREATED,
            RoundState.PROMPT_COMMITTED,
            RoundState.MODEL_REQUEST_SENT,
            RoundState.MODEL_RESPONSE_COMMITTED,
            RoundState.SOURCE_VALIDATED,
            RoundState.COMPILE_FINISHED,
            RoundState.CORRECTNESS_FINISHED,
            RoundState.BENCHMARK_FINISHED,
            RoundState.PROFILE_FINISHED,
            RoundState.FEEDBACK_COMMITTED,
            RoundState.ROUND_FINISHED,
        ]
        round_state_machine.validate_path(states)
        self.assertTrue(round_state_machine.is_terminal(states[-1]))

    def test_compile_and_correctness_failure_paths(self) -> None:
        for failed_at in (
            RoundState.SOURCE_VALIDATED,
            RoundState.COMPILE_FINISHED,
            RoundState.CORRECTNESS_FINISHED,
        ):
            with self.subTest(failed_at=failed_at):
                self.assertTrue(
                    round_state_machine.can_transition(
                        failed_at, RoundState.FEEDBACK_COMMITTED
                    )
                )

    def test_skipped_checkpoint_and_terminal_reentry_are_rejected(self) -> None:
        with self.assertRaises(InvalidTransitionError):
            round_state_machine.transition(
                RoundState.ROUND_CREATED, RoundState.MODEL_REQUEST_SENT
            )
        with self.assertRaises(InvalidTransitionError):
            round_state_machine.transition(
                RoundState.ROUND_FINISHED, RoundState.ROUND_CREATED
            )
        with self.assertRaises(InvalidTransitionError):
            round_state_machine.transition(
                RoundState.PROMPT_COMMITTED, RoundState.PROMPT_COMMITTED
            )

    def test_task_final_evaluation_path(self) -> None:
        task_state_machine.validate_path(
            [
                TaskState.TASK_CREATED,
                TaskState.ROUNDS_RUNNING,
                TaskState.SELECT_BEST_CANDIDATE,
                TaskState.HIDDEN_CORRECTNESS_TEST,
                TaskState.FINAL_BENCHMARK,
                TaskState.FINAL_FULL_PROFILE,
                TaskState.TASK_FINISHED,
            ]
        )
        self.assertTrue(task_state_machine.is_terminal(TaskState.TASK_FAILED))


class ScoringTests(unittest.TestCase):
    def test_weighted_geometric_mean_uses_log_space(self) -> None:
        result = weighted_geometric_mean(
            [
                BenchmarkSample("small", 1.0, weight=1.0),
                BenchmarkSample("large", 4.0, weight=3.0),
            ]
        )
        self.assertAlmostEqual(result, math.sqrt(8.0))

    def test_aggregate_uses_worst_shape_and_worst_cv(self) -> None:
        score = aggregate_candidate_score(
            candidate_id="candidate-1",
            round_number=1,
            samples=[
                BenchmarkSample("a", 1.5, coefficient_of_variation=0.01),
                BenchmarkSample("b", 0.9, coefficient_of_variation=0.04),
            ],
            candidate_kernel_coverage=0.98,
        )
        self.assertEqual(score.minimum_speedup, 0.9)
        self.assertEqual(score.stability_cv, 0.04)
        self.assertAlmostEqual(score.geomean_speedup or 0.0, math.sqrt(1.35))

    @staticmethod
    def _score(
        candidate_id: str,
        round_number: int,
        *,
        minimum: float,
        geomean: float,
        hidden: bool | None = None,
        cv: float = 0.01,
        anti_bypass: bool = True,
    ) -> CandidateScore:
        return CandidateScore(
            candidate_id=candidate_id,
            round_number=round_number,
            compile_passed=True,
            correctness_passed=True,
            anti_bypass_passed=anti_bypass,
            hidden_correctness_passed=hidden,
            minimum_speedup=minimum,
            geomean_speedup=geomean,
            candidate_kernel_coverage=0.99,
            stability_cv=cv,
        )

    def test_selection_prioritizes_minimum_before_geomean(self) -> None:
        high_average = self._score(
            "average", 1, minimum=0.8, geomean=2.0
        )
        robust = self._score("robust", 2, minimum=1.1, geomean=1.2)
        self.assertEqual(
            select_best_candidate([high_average, robust]), robust
        )

    def test_hidden_correctness_is_first_and_can_be_required(self) -> None:
        hidden_pass = self._score(
            "hidden", 1, minimum=0.9, geomean=1.0, hidden=True
        )
        untested = self._score(
            "public", 2, minimum=2.0, geomean=3.0, hidden=None
        )
        hidden_fail = self._score(
            "failed", 3, minimum=4.0, geomean=5.0, hidden=False
        )
        self.assertEqual(
            select_best_candidate([untested, hidden_fail, hidden_pass]),
            hidden_pass,
        )
        self.assertEqual(
            select_best_candidate(
                [untested, hidden_fail, hidden_pass],
                require_hidden_correctness=True,
            ),
            hidden_pass,
        )

    def test_invalid_candidate_never_receives_reward(self) -> None:
        bypass = self._score(
            "bypass",
            1,
            minimum=10.0,
            geomean=10.0,
            anti_bypass=False,
        )
        reward = compute_reward(bypass)
        self.assertEqual(reward.reward, 0.0)
        self.assertIsNone(select_best_candidate([bypass]))

    def test_reward_breakdown_matches_initial_formula(self) -> None:
        score = self._score("valid", 1, minimum=1.5, geomean=2.0, cv=0.06)
        reward = compute_reward(
            score, maximum_stable_cv=0.05, stability_penalty_scale=2.0
        )
        self.assertAlmostEqual(reward.speedup_component, 1.0)
        self.assertAlmostEqual(reward.coverage_component, 0.198)
        self.assertAlmostEqual(reward.stability_penalty, 0.02)
        self.assertAlmostEqual(reward.reward, 2.178)

    def test_invalid_numeric_metrics_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkSample("bad", float("nan"))
        with self.assertRaises(ValueError):
            weighted_geometric_mean([])

    def test_online_best_comparison_is_noise_aware(self) -> None:
        incumbent = self._score(
            "best", 1, minimum=1.0, geomean=1.0, cv=0.02
        )
        first = compare_public_candidate(incumbent, None)
        self.assertEqual(first.decision, "INITIAL_BEST")
        tied = self._score(
            "tie", 2, minimum=1.005, geomean=1.015, cv=0.02
        )
        self.assertEqual(
            compare_public_candidate(tied, incumbent).decision, "TIE"
        )
        better = self._score(
            "better", 3, minimum=1.01, geomean=1.04, cv=0.01
        )
        self.assertEqual(
            compare_public_candidate(better, incumbent).decision, "NEW_BEST"
        )
        regressed = self._score(
            "regressed", 4, minimum=0.95, geomean=0.99, cv=0.01
        )
        self.assertEqual(
            compare_public_candidate(regressed, incumbent).decision,
            "REGRESSION",
        )


if __name__ == "__main__":
    unittest.main()
