from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ascend_kernel_lab.backend import FakeBackend
from ascend_kernel_lab.orchestration import (
    BaselineManager,
    prompt_baseline_projection,
)
from ascend_kernel_lab.tasks import CaseSpec, TaskRegistry, TaskSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _EagerOnlyBaselineBackend:
    def measure_baselines(
        self,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> Mapping[str, Any]:
        del task, artifact_dir
        return {
            "per_case": [
                {
                    "case_id": case.id,
                    "weight": case.weight,
                    "pytorch_eager_us": 12.0,
                }
                for case in cases
            ],
            "status": "complete",
            "mode": "pytorch_eager_only",
            "comparison_baseline": "pytorch_eager",
        }


class BaselineManagerTests(unittest.TestCase):
    def test_prompt_projection_keeps_eager_medians_and_drops_raw_measurements(self) -> None:
        projection = prompt_baseline_projection(
            {
                "comparison_baseline": "pytorch_eager",
                "pytorch_eager_geomean_us": 15.0,
                "per_case": [
                    {
                        "case_id": "b01",
                        "dtype": "float16",
                        "params": {"n": 1024},
                        "weight": 1.0,
                        "pytorch_eager_us": 15.0,
                        "pytorch_eager": {
                            "median_us": 15.0,
                            "raw_samples_us": [14.0, 15.0, 16.0],
                            "measurement_attempts": [{"attempt": 1}],
                        },
                        "eager_repeats_per_batch": 1000,
                    }
                ],
                "measurement_attempts": [{"attempt": 1}],
            }
        )

        self.assertEqual(projection["comparison_baseline"], "pytorch_eager")
        self.assertEqual(projection["summary"]["weighted_geomean_us"], 15.0)
        self.assertEqual(
            projection["per_case"],
            [
                {
                    "case_id": "b01",
                    "median_us": 15.0,
                    "dtype": "float16",
                    "params": {"n": 1024},
                    "weight": 1.0,
                }
            ],
        )
        rendered = repr(projection)
        self.assertNotIn("raw_samples", rendered)
        self.assertNotIn("measurement_attempts", rendered)
        self.assertNotIn("repeats_per_batch", rendered)

    def test_snapshot_records_eager_only_comparison_policy(self) -> None:
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")
        manager = BaselineManager(
            backend=_EagerOnlyBaselineBackend(),
            environment_sha256="environment",
            harness_git_commit="commit",
            benchmark_config={},
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = manager.measure(task, Path(temporary))

        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["mode"], "pytorch_eager_only")
        self.assertEqual(snapshot["comparison_baseline"], "pytorch_eager")
        self.assertEqual(snapshot["compared_baselines"], ["pytorch_eager"])
        self.assertEqual(
            snapshot["not_measured_baselines"], ["torch_compile", "official"]
        )

    def test_fake_backend_uses_same_eager_only_policy(self) -> None:
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")
        manager = BaselineManager(
            backend=FakeBackend(),
            environment_sha256="environment",
            harness_git_commit="commit",
            benchmark_config={},
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = manager.measure(task, Path(temporary))

        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["compared_baselines"], ["pytorch_eager"])


if __name__ == "__main__":
    unittest.main()
