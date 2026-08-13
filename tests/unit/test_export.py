from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.export.datasets import DatasetExporter, ExportError
from ascend_kernel_lab.export.report import ReportExporter


class ExportTests(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        experiment = root / "exp_test"
        round_root = experiment / "tasks" / "k01_vector_add" / "round_01"
        round_root.mkdir(parents=True)
        (round_root / "prompt.json").write_text(json.dumps({"task": "public"}), encoding="utf-8")
        (round_root / "model_response.json").write_text(json.dumps({"status": "candidate", "round": 1}), encoding="utf-8")
        (round_root / "candidate.py").write_text("def custom_op(x):\n    return x\n", encoding="utf-8")
        (round_root / "evaluation_result.json").write_text(
            json.dumps(
                {
                    "overall_status": "success",
                    "score": {
                        "compile_passed": True,
                        "correctness_passed": True,
                        "anti_bypass_passed": True,
                        "geomean_speedup": 1.1,
                        "stability_cv": 0.01,
                    },
                    "benchmark": {
                        "geomean_speedup_vs_eager": 1.1,
                        "minimum_speedup_vs_eager": 1.0,
                        "geomean_speedup_vs_compile": 0.9,
                        "minimum_speedup_vs_compile": 0.8,
                        "geomean_speedup_vs_official": None,
                        "minimum_speedup_vs_official": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        (round_root / "feedback.json").write_text(json.dumps({"overall_status": "success"}), encoding="utf-8")
        (round_root / "reward.json").write_text(json.dumps({"anti_bypass": 1, "speedup_geomean": 1.1, "stability_cv": 0.01}), encoding="utf-8")
        final_root = round_root.parent
        (final_root / "final_result.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "best_round": 1,
                    "hidden_correctness_passed": True,
                }
            ),
            encoding="utf-8",
        )
        (experiment / "environment_snapshot.json").write_text(
            json.dumps({"device": "fake"}), encoding="utf-8"
        )
        return experiment

    def test_sft_rl_and_report_are_offline_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            exporter = DatasetExporter(experiment)
            self.assertEqual(exporter.export_sft(root / "sft.jsonl"), 1)
            sft = json.loads((root / "sft.jsonl").read_text())
            self.assertEqual(sft["sample_type"], "high_quality_optimization")
            self.assertEqual(
                sft["quality"]["label"], "high_quality_optimization"
            )
            self.assertEqual(exporter.export_rl(root / "rl.jsonl"), 1)
            trajectory = json.loads((root / "rl.jsonl").read_text())
            self.assertTrue(trajectory["selected_as_best"])
            self.assertEqual(trajectory["feedback"], {"overall_status": "success"})
            self.assertEqual(
                trajectory["quality"]["label"], "high_quality_optimization"
            )
            ReportExporter(experiment).write(root / "report.json", root / "report.md")
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["passed_task_count"], 1)
            self.assertEqual(report["environment"], {"device": "fake"})
            self.assertEqual(
                report["tasks"][0]["public_best"]["geomean_speedup_vs_eager"],
                1.1,
            )
            self.assertEqual(
                report["trajectory_quality_summary"]["counts"],
                {"high_quality_optimization": 1},
            )
            markdown = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("vs PyTorch eager", markdown)
            self.assertNotIn("vs official", markdown)
            self.assertIn("## Trajectory quality", markdown)

    def test_secret_scanner_blocks_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            round_root = experiment / "tasks/k01_vector_add/round_01"
            (round_root / "prompt.json").write_text(json.dumps({"Authorization": "Bearer abcdefghijklmnop"}), encoding="utf-8")
            with self.assertRaises(ExportError):
                DatasetExporter(experiment).export_sft(root / "sft.jsonl")

    def test_main_sft_requires_final_hidden_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            final = experiment / "tasks/k01_vector_add/final_result.json"
            final.write_text(
                json.dumps(
                    {
                        "status": "failed_hidden_correctness",
                        "best_round": 1,
                        "hidden_correctness_passed": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                DatasetExporter(experiment).export_sft(root / "main.jsonl"), 0
            )
            self.assertEqual(
                DatasetExporter(experiment).export_sft(
                    root / "all.jsonl", main_only=False
                ),
                1,
            )

    def test_initial_correct_candidate_is_kept_without_a_speedup_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            evaluation_path = (
                experiment
                / "tasks/k01_vector_add/round_01/evaluation_result.json"
            )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            evaluation["score"]["geomean_speedup"] = 0.9
            evaluation["benchmark"]["geomean_speedup_vs_eager"] = 0.9
            evaluation["benchmark"]["minimum_speedup_vs_eager"] = 0.8
            evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

            output = root / "initial.jsonl"
            self.assertEqual(DatasetExporter(experiment).export_sft(output), 1)
            sample = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(sample["sample_type"], "initial_correct_candidate")
            self.assertTrue(sample["quality"]["eligible_for_default_sft"])

    def test_all_samples_really_exports_unclassified_trajectory_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            first = experiment / "tasks/k01_vector_add/round_01"
            second = first.parent / "round_02"
            second.mkdir()
            for name in (
                "prompt.json",
                "model_response.json",
                "candidate.py",
                "evaluation_result.json",
                "feedback.json",
                "reward.json",
            ):
                second.joinpath(name).write_bytes(first.joinpath(name).read_bytes())

            destination = root / "all.jsonl"
            self.assertEqual(
                DatasetExporter(experiment).export_sft(
                    destination,
                    main_only=False,
                ),
                2,
            )
            rows = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(rows[1]["sample_type"], "valid_but_not_improved")
            self.assertFalse(rows[1]["quality"]["eligible_for_default_sft"])

    def test_quality_classification_curates_only_optimization_and_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            task_root = experiment / "tasks/k01_vector_add"

            def write_round(
                number: int,
                status: str,
                speedup: float | None = None,
                *,
                host_us: float | None = None,
                device_us: float | None = None,
                host_bound: bool = False,
            ) -> None:
                round_root = task_root / f"round_{number:02d}"
                round_root.mkdir(exist_ok=True)
                (round_root / "prompt.json").write_text(
                    json.dumps({"round": number}), encoding="utf-8"
                )
                (round_root / "model_response.json").write_text(
                    json.dumps({"status": "candidate", "round": number}),
                    encoding="utf-8",
                )
                (round_root / "candidate.py").write_text(
                    f"def custom_op_{number}(x):\n    return x\n", encoding="utf-8"
                )
                valid = status in {"correct", "success"}
                evaluation: dict[str, object] = {
                    "overall_status": status,
                    "score": {
                        "compile_passed": status
                        not in {"source_failed", "compile_failed"},
                        "correctness_passed": status
                        not in {
                            "source_failed",
                            "compile_failed",
                            "correctness_failed",
                        },
                        "anti_bypass_passed": valid,
                        "geomean_speedup": speedup if valid else None,
                        "minimum_speedup": speedup if valid else None,
                        "stability_cv": 0.01 if valid else None,
                    },
                    "benchmark": (
                        {
                            "geomean_speedup_vs_eager": speedup,
                            "minimum_speedup_vs_eager": speedup,
                        }
                        if valid
                        else None
                    ),
                }
                if valid and host_bound:
                    benchmark = evaluation["benchmark"]
                    assert isinstance(benchmark, dict)
                    benchmark["bottleneck"] = {
                        "bottleneck_type": "host_dispatch",
                        "host_dispatch_limited": True,
                    }
                if host_us is not None and device_us is not None:
                    evaluation["profile"] = {
                        "summary": {
                            "scheduling": {
                                "host_enqueue_us": host_us,
                                "candidate_device_execution_us": device_us,
                            }
                        }
                    }
                (round_root / "evaluation_result.json").write_text(
                    json.dumps(evaluation), encoding="utf-8"
                )
                (round_root / "feedback.json").write_text(
                    json.dumps(
                        {
                            "overall_status": status,
                            "optimization_action": (
                                "stop_host_bound"
                                if host_bound
                                else "continue_optimization"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                (round_root / "reward.json").write_text(
                    json.dumps({"speedup_geomean": speedup}), encoding="utf-8"
                )

            write_round(1, "source_failed")
            write_round(2, "compile_failed")
            write_round(3, "correctness_failed")
            write_round(4, "correct", 1.0)
            write_round(5, "correct", 0.90)
            write_round(6, "correct", 1.10)
            write_round(7, "correct", 1.08, host_bound=True)
            (task_root / "final_result.json").write_text(
                json.dumps(
                    {
                        "status": "passed_with_profile_warning",
                        "best_round": 6,
                        "hidden_correctness_passed": True,
                    }
                ),
                encoding="utf-8",
            )

            exporter = DatasetExporter(experiment)
            summary = exporter.quality_summary()
            self.assertEqual(
                summary["counts"],
                {
                    "compile_failed": 1,
                    "correctness_failed": 1,
                    "high_quality_optimization": 1,
                    "host_bound_unoptimizable": 1,
                    "performance_regression": 1,
                    "source_failed": 1,
                    "successful_repair": 1,
                },
            )
            self.assertEqual(summary["default_sft_row_count"], 2)
            self.assertEqual(exporter.export_sft(root / "curated.jsonl"), 2)
            self.assertEqual(
                exporter.export_sft(root / "all.jsonl", main_only=False), 7
            )
            self.assertEqual(exporter.export_rl(root / "rl.jsonl"), 7)
            all_rows = [
                json.loads(line) for line in (root / "all.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["quality"]["label"] for row in all_rows],
                [
                    "source_failed",
                    "compile_failed",
                    "correctness_failed",
                    "successful_repair",
                    "performance_regression",
                    "high_quality_optimization",
                    "host_bound_unoptimizable",
                ],
            )
            host_quality = all_rows[-1]["quality"]
            self.assertTrue(host_quality["metrics"]["host_dispatch_limited"])
            self.assertEqual(
                host_quality["metrics"]["optimization_action"], "stop_host_bound"
            )
            self.assertTrue(
                any("stop_host_bound" in reason for reason in host_quality["reasons"])
            )
            rl_rows = [
                json.loads(line) for line in (root / "rl.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["quality"]["label"] for row in rl_rows],
                [row["quality"]["label"] for row in all_rows],
            )

            report = ReportExporter(experiment).build()
            self.assertEqual(
                report["trajectory_quality_summary"]["default_sft_row_count"], 2
            )

    def test_report_scanner_blocks_environment_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = self._run(root)
            (experiment / "environment_snapshot.json").write_text(
                json.dumps({"api_key": "should-not-export"}), encoding="utf-8"
            )
            with self.assertRaises(ExportError):
                ReportExporter(experiment).write(root / "report.json")


if __name__ == "__main__":
    unittest.main()
