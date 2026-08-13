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
            self.assertEqual(exporter.export_rl(root / "rl.jsonl"), 1)
            trajectory = json.loads((root / "rl.jsonl").read_text())
            self.assertTrue(trajectory["selected_as_best"])
            self.assertEqual(trajectory["feedback"], {"overall_status": "success"})
            ReportExporter(experiment).write(root / "report.json", root / "report.md")
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["passed_task_count"], 1)
            self.assertEqual(report["environment"], {"device": "fake"})
            self.assertEqual(
                report["tasks"][0]["public_best"]["geomean_speedup_vs_eager"],
                1.1,
            )
            markdown = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("vs PyTorch eager", markdown)
            self.assertNotIn("vs official", markdown)

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
            self.assertEqual(rows[1]["sample_type"], "cold_start_trajectory")

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
