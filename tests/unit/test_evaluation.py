from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.backend import FakeBackend, StageResult
from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.evaluation import EvaluationRequest, evaluate_candidate
from ascend_kernel_lab.tasks import TaskRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EvaluationFlowTests(unittest.TestCase):
    def test_unstable_benchmark_still_profiles_but_is_not_selectable(self) -> None:
        benchmark = StageResult.failure(
            EvaluationStage.BENCHMARK,
            details={
                "passed": False,
                "status": "unstable",
                "geomean_speedup_vs_eager": 1.5,
                "minimum_speedup_vs_eager": 1.2,
                "maximum_cv": 0.25,
            },
        )
        profile = StageResult.success(
            EvaluationStage.PROFILE,
            details={
                "profile_available": True,
                "summary": {
                    "kernel_count": 1,
                    "candidate_kernel_coverage": 1.0,
                },
            },
        )
        backend = FakeBackend(
            {
                EvaluationStage.BENCHMARK: (benchmark,),
                EvaluationStage.PROFILE: (profile,),
            }
        )
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "candidate.py"
            candidate.write_text("def custom_op(x, y):\n    return x\n", encoding="utf-8")
            result = evaluate_candidate(
                backend,
                EvaluationRequest(
                    experiment_id="exp_test",
                    task=task,
                    round_number=1,
                    candidate_id="exp_test:k01_vector_add:r01:candidate",
                    candidate_path=candidate,
                    artifact_dir=Path(temporary) / "artifacts",
                    combine_candidate_stages=True,
                ),
            )

        self.assertEqual(result.overall_status, "benchmark_failed")
        self.assertEqual(
            [call["stage"] for call in backend.calls],
            [
                EvaluationStage.SOURCE_CHECK.value,
                EvaluationStage.FULL_EVALUATION.value,
                EvaluationStage.COMPILE.value,
                EvaluationStage.CORRECTNESS.value,
                EvaluationStage.BENCHMARK.value,
            ],
        )
        self.assertIsNone(result.profile)
        self.assertFalse(result.anti_bypass["passed"])
        self.assertIsNone(result.score.candidate_kernel_coverage)
        self.assertEqual(result.score.stability_cv, 0.25)
        self.assertIsNone(result.score.minimum_speedup)
        self.assertIsNone(result.score.geomean_speedup)
        self.assertFalse(result.score.is_publicly_valid)


if __name__ == "__main__":
    unittest.main()
