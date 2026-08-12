from __future__ import annotations

import unittest
from typing import Any

from ascend_kernel_lab.worker.stage_entry import _measure_batch


class _NpuWithoutEvents:
    @staticmethod
    def synchronize() -> None:
        return None


class _TorchWithoutEvents:
    npu = _NpuWithoutEvents()


class StageEntryTimingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
