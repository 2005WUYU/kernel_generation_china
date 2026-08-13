from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.worker.stage_entry import (
    _baselines,
    _measure_batch,
    _measurement_session,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _NpuWithoutEvents:
    @staticmethod
    def synchronize() -> None:
        return None


class _TorchWithoutEvents:
    npu = _NpuWithoutEvents()


class _Event:
    def __init__(self, *, enable_timing: bool) -> None:
        self.enable_timing = enable_timing

    def record(self) -> None:
        return None

    def elapsed_time(self, _end: object) -> float:
        return 1.0


class _NpuWithEvents:
    Event = _Event

    @staticmethod
    def synchronize() -> None:
        return None


class _TorchWithEvents:
    npu = _NpuWithEvents()


class StageEntryTimingTests(unittest.TestCase):
    def test_baseline_measures_only_pytorch_eager(self) -> None:
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")

        class IncompatibleTorch:
            compile_calls = 0

            def compile(self, _function: Any) -> Any:
                self.compile_calls += 1

                def compiled(*_args: Any) -> Any:
                    raise RuntimeError(
                        "cannot import name 'triton_key' from "
                        "'triton.compiler.compiler'"
                    )

                return compiled

        torch = IncompatibleTorch()
        stats = {"median_us": 12.0}
        with (
            patch(
                "ascend_kernel_lab.worker.stage_entry.generate_inputs",
                return_value=SimpleNamespace(args=(object(),)),
            ),
            patch(
                "ascend_kernel_lab.worker.stage_entry.reference",
                return_value=object(),
            ),
            patch("ascend_kernel_lab.worker.stage_entry._sync"),
            patch(
                "ascend_kernel_lab.worker.stage_entry._single_measurement_session",
                return_value=(stats, 1),
            ),
        ):
            result = _baselines(
                torch,
                task,
                task.benchmark_cases,
                "npu:0",
                {
                    "warmup": 1,
                    "measurement_batches": 1,
                    "target_batch_time_ms": 1.0,
                },
            )

        self.assertEqual(torch.compile_calls, 0)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "pytorch_eager_only")
        self.assertEqual(result["compared_baselines"], ["pytorch_eager"])
        self.assertEqual(
            result["not_measured_baselines"], ["torch_compile", "official"]
        )
        self.assertNotIn("torch_compile_us", result["per_case"][0])

    def test_measurement_observes_last_materialized_output_outside_timer(self) -> None:
        calls = 0
        observed: list[Any] = []

        def function() -> int:
            nonlocal calls
            calls += 1
            return calls

        latency = _measure_batch(
            _TorchWithoutEvents(),
            function,
            4,
            observe_output=observed.append,
        )

        self.assertGreater(latency, 0)
        self.assertEqual(calls, 4)
        self.assertEqual(observed, [4])

    def test_measurement_rejects_zero_repeats(self) -> None:
        with self.assertRaisesRegex(ValueError, "repeats"):
            _measure_batch(_TorchWithoutEvents(), lambda: None, 0)

    def test_candidate_context_is_outside_timing_and_not_reentered_per_repeat(self) -> None:
        active = False
        entries = 0
        exits = 0
        candidate_calls = 0
        baseline_calls = 0
        observations = 0

        @contextmanager
        def candidate_context() -> Any:
            nonlocal active, entries, exits
            self.assertFalse(active)
            active = True
            entries += 1
            try:
                yield
            finally:
                active = False
                exits += 1

        def candidate() -> int:
            nonlocal candidate_calls
            self.assertTrue(active)
            candidate_calls += 1
            return candidate_calls

        def baseline() -> int:
            nonlocal baseline_calls
            self.assertFalse(active)
            baseline_calls += 1
            return baseline_calls

        def observe(_output: Any) -> None:
            nonlocal observations
            self.assertFalse(active)
            observations += 1

        _measurement_session(
            _TorchWithEvents(),
            candidate,
            baseline,
            batches=2,
            target_batch_time_ms=2.0,
            candidate_context=candidate_context,
            observe_candidate_output=observe,
        )

        self.assertEqual(entries, 3)
        self.assertEqual(exits, entries)
        self.assertEqual(candidate_calls, 5)
        self.assertEqual(baseline_calls, 5)
        self.assertEqual(observations, entries)


if __name__ == "__main__":
    unittest.main()
