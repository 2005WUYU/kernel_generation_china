from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ascend_kernel_lab.backend import FakeBackend
from ascend_kernel_lab.orchestration import BaselineManager
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
