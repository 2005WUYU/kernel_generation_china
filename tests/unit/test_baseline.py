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


class _CompileFailingBaselineBackend:
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
                    "torch_compile_us": None,
                    "torch_compile_error": "BackendCompilerFailed: test",
                    "official_us": None,
                }
                for case in cases
            ],
            "torch_compile_available": False,
            "official_available": False,
            "official_status": "not_defined",
            "unavailable_reasons": {
                "torch_compile": ["BackendCompilerFailed: test"],
                "official": "task has no trusted official baseline",
            },
        }


class BaselineManagerTests(unittest.TestCase):
    def test_compile_failures_make_snapshot_partial_and_layer_failed(self) -> None:
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")
        manager = BaselineManager(
            backend=_CompileFailingBaselineBackend(),
            environment_sha256="environment",
            harness_git_commit="commit",
            benchmark_config={},
        )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = manager.measure(task, Path(temporary))

        self.assertEqual(snapshot["status"], "partial")
        self.assertEqual(snapshot["torch_compile_status"], "failed")
        self.assertEqual(snapshot["torch_compile_successful_case_count"], 0)
        self.assertEqual(
            snapshot["torch_compile_total_case_count"],
            len(task.benchmark_cases),
        )
        self.assertEqual(snapshot["official_status"], "not_defined")

    def test_missing_official_layer_does_not_make_snapshot_partial(self) -> None:
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
        self.assertEqual(snapshot["torch_compile_status"], "complete")
        self.assertEqual(snapshot["official_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
