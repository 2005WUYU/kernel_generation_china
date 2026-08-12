from __future__ import annotations

import math
import unittest

from ascend_kernel_lab.evaluation.benchmark import (
    speedup_summary,
    summarize_samples,
    weighted_geometric_mean,
)


class BenchmarkLogicTests(unittest.TestCase):
    def test_statistics_preserve_raw_samples_and_cv(self) -> None:
        result = summarize_samples([10.0, 11.0, 9.0, 10.0, 10.0])
        self.assertEqual(result.median_us, 10.0)
        self.assertEqual(result.sample_count, 5)
        self.assertGreater(result.cv, 0)
        self.assertEqual(result.raw_samples_us, (10.0, 11.0, 9.0, 10.0, 10.0))

    def test_weighted_geomean_is_log_stable(self) -> None:
        value = weighted_geometric_mean([0.5, 2.0], [1.0, 1.0])
        self.assertAlmostEqual(value, 1.0)
        self.assertTrue(math.isfinite(weighted_geometric_mean([1e-100, 1e100])))

    def test_speedup_summary_keeps_worst_shape(self) -> None:
        result = speedup_summary([
            {"case_id": "a", "speedup_vs_eager": 2.0, "weight": 1.0},
            {"case_id": "b", "speedup_vs_eager": 0.5, "weight": 1.0},
        ])
        self.assertAlmostEqual(result["geomean_speedup_vs_eager"], 1.0)
        self.assertEqual(result["minimum_speedup_vs_eager"], 0.5)

    def test_invalid_samples_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            summarize_samples([])
        with self.assertRaises(ValueError):
            weighted_geometric_mean([1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
