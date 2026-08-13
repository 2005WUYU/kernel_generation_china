from __future__ import annotations

import math
import unittest

from ascend_kernel_lab.evaluation.benchmark import (
    bottleneck_summary,
    classify_bottleneck,
    speedup_summary,
    summarize_latency_breakdown,
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

    def test_latency_breakdown_identifies_host_dispatch(self) -> None:
        result = summarize_latency_breakdown(
            [
                {"device_latency_us": 2.0, "end_to_end_latency_us": 12.0},
                {"device_latency_us": 2.2, "end_to_end_latency_us": 11.0},
                {"device_latency_us": 1.8, "end_to_end_latency_us": 13.0},
            ]
        )

        self.assertEqual(result["device_latency_us"], 2.0)
        self.assertEqual(result["end_to_end_latency_us"], 12.0)
        self.assertEqual(result["host_overhead_us"], 10.0)
        self.assertEqual(result["bottleneck_type"], "host_dispatch")
        self.assertEqual(
            classify_bottleneck(
                device_latency_us=8.0,
                end_to_end_latency_us=10.0,
            ),
            "device_execution",
        )

    def test_bottleneck_summary_requires_clear_case_majority(self) -> None:
        result = bottleneck_summary(
            [
                {
                    "weight": 3.0,
                    "candidate": {
                        "latency_breakdown": {"bottleneck_type": "host_dispatch"}
                    },
                },
                {
                    "weight": 1.0,
                    "candidate": {
                        "latency_breakdown": {"bottleneck_type": "device_execution"}
                    },
                },
            ]
        )

        self.assertEqual(result["bottleneck_type"], "host_dispatch")
        self.assertTrue(result["host_dispatch_limited"])
        self.assertEqual(result["host_dispatch_case_weight_fraction"], 0.75)


if __name__ == "__main__":
    unittest.main()
