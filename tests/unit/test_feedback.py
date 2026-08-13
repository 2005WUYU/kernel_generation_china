from __future__ import annotations

import unittest

from ascend_kernel_lab.orchestration.feedback import build_feedback


class FeedbackTests(unittest.TestCase):
    def test_cold_sft_feedback_has_no_hard_speedup_target(self) -> None:
        feedback = build_feedback(
            task_id="k01_vector_add",
            round_number=1,
            result={
                "overall_status": "correct",
                "source": {"passed": True},
                "compile": {"passed": True},
                "correctness": {"passed": True},
                "benchmark": {
                    "passed": True,
                    "minimum_speedup_vs_eager": 0.4,
                    "geomean_speedup_vs_eager": 0.5,
                    "per_case": [],
                },
                "profile": None,
                "anti_bypass": {"passed": True},
            },
        )

        focus = feedback["next_round_requirement"]["focus"]
        self.assertTrue(any("不设硬性加速比门槛" in item for item in focus))
        self.assertFalse(any("达到 1.0" in item for item in focus))

    def test_benchmark_failure_keeps_status_measurements_and_cv_focus(self) -> None:
        benchmark = {
            "stage": "BENCHMARK",
            "status": "fail",
            "passed": False,
            "maximum_cv": 0.18,
            "per_case": [
                {
                    "case_id": "b01",
                    "candidate": {"median_us": 44.1, "cv": 0.18},
                    "baseline_eager": {"median_us": 15.6, "cv": 0.01},
                    "speedup_vs_eager": 0.35,
                    "stable": False,
                }
            ],
        }
        feedback = build_feedback(
            task_id="k01_vector_add",
            round_number=1,
            result={
                "overall_status": "benchmark_failed",
                "source": {"passed": True},
                "compile": {"passed": True},
                "correctness": {"passed": True},
                "benchmark": benchmark,
                "profile": None,
                "anti_bypass": {
                    "passed": False,
                    "status": "not_evaluated",
                    "reason": "benchmark_failed",
                },
            },
        )

        self.assertEqual(feedback["overall_status"], "benchmark_failed")
        self.assertEqual(feedback["benchmark"], benchmark)
        focus = feedback["next_round_requirement"]["focus"]
        self.assertTrue(any("不稳定" in item and "CV" in item for item in focus))
        self.assertFalse(any("超时" in item for item in focus))

    def test_host_dispatch_bottleneck_is_compacted_and_changes_focus(self) -> None:
        feedback = build_feedback(
            task_id="k01_vector_add",
            round_number=2,
            result={
                "overall_status": "correct",
                "source": {"passed": True},
                "compile": {"passed": True},
                "correctness": {"passed": True},
                "benchmark": {
                    "passed": True,
                    "minimum_speedup_vs_eager": 0.4,
                    "geomean_speedup_vs_eager": 0.5,
                    "bottleneck_type": "host_dispatch",
                    "host_dispatch_limited": True,
                    "bottleneck": {
                        "bottleneck_type": "host_dispatch",
                        "host_dispatch_limited": True,
                    },
                    "per_case": [
                        {
                            "case_id": "b01",
                            "device_latency_us": 2.0,
                            "end_to_end_latency_us": 12.0,
                            "host_overhead_us": 10.0,
                            "bottleneck_type": "host_dispatch",
                        }
                    ],
                },
                "profile": None,
                "anti_bypass": {"passed": True},
            },
        )

        latency = feedback["latency_bottleneck"]
        self.assertEqual(latency["bottleneck_type"], "host_dispatch")
        self.assertEqual(latency["cases"][0]["device_latency_us"], 2.0)
        self.assertEqual(feedback["optimization_action"], "stop_host_bound")
        self.assertTrue(feedback["stop_recommended"])
        focus = feedback["next_round_requirement"]["focus"]
        self.assertTrue(any("host_dispatch" in item for item in focus))
        self.assertFalse(any("最慢 shape" in item for item in focus))


if __name__ == "__main__":
    unittest.main()
