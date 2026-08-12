from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from ascend_kernel_lab.cli import (
    EXIT_NOT_READY,
    CommandError,
    _require_clean_git_release,
    main,
)
from ascend_kernel_lab.config import load_config
from ascend_kernel_lab.orchestration.controller import TaskRunSummary


class CliEndToEndTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        project = Path(__file__).resolve().parents[2]
        value = yaml.safe_load(
            (project / "configs/experiment_910c_kimi_k3.yaml").read_text(encoding="utf-8")
        )
        value["experiment"]["id"] = "exp_cli"
        value["experiment"]["tasks"] = ["k01_vector_add"]
        value["storage"]["database"] = f"sqlite:///{root / 'metadata.db'}"
        value["storage"]["artifact_root"] = str(root / "artifacts")
        value["storage"]["task_root"] = str(project / "task_specs")
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        return path

    @staticmethod
    def _main(argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_fake_run_resume_status_verify_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            code, output, error = self._main(["db", "upgrade", "-c", str(config)])
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output)["status"], "ready")

            code, output, error = self._main(
                ["experiment", "run", "-c", str(config), "--fake"]
            )
            self.assertEqual(code, 0, error)
            summary = json.loads(output)
            self.assertEqual(summary["experiment_id"], "exp_cli_fake")
            experiment_root = Path(summary["experiment_root"])
            events = (experiment_root / "events.jsonl").read_bytes()

            code, _output, error = self._main(
                ["experiment", "resume", "-c", str(config), "--fake"]
            )
            self.assertEqual(code, 0, error)
            self.assertEqual((experiment_root / "events.jsonl").read_bytes(), events)

            code, output, error = self._main(
                [
                    "experiment",
                    "status",
                    "-c",
                    str(config),
                    "--experiment-id",
                    "exp_cli_fake",
                ]
            )
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(output)["experiment"]["state"], "FINISHED")

            code, output, error = self._main(
                [
                    "verify-run",
                    "--experiment-root",
                    str(experiment_root),
                    "--database",
                    str(root / "metadata.db"),
                ]
            )
            self.assertEqual(code, 0, error)
            self.assertTrue(json.loads(output)["passed"])

            for kind, filename in (("sft", "sft.jsonl"), ("rl", "rl.jsonl")):
                destination = root / filename
                code, output, error = self._main(
                    [
                        "export",
                        kind,
                        "--experiment-root",
                        str(experiment_root),
                        "-o",
                        str(destination),
                    ]
                )
                self.assertEqual(code, 0, error)
                self.assertGreater(json.loads(output)["rows"], 0)
                self.assertTrue(destination.is_file())

    def test_acceptance_never_conflates_integrity_with_hardware_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            code, _output, error = self._main(
                [
                    "experiment",
                    "run",
                    "-c",
                    str(config),
                    "--fake",
                    "--experiment-id",
                    "exp_cli",
                ]
            )
            self.assertEqual(code, 0, error)
            report_path = root / "acceptance.json"
            code, output, _error = self._main(
                [
                    "acceptance",
                    "-c",
                    str(config),
                    "-o",
                    str(report_path),
                ]
            )
            self.assertEqual(code, 3)
            report = json.loads(output)
            self.assertFalse(report["passed"])
            statuses = {item["gate"]: item["status"] for item in report["gates"]}
            self.assertEqual(statuses["G3"], "pending")
            self.assertEqual(statuses["G8"], "pending")
            self.assertTrue(report_path.is_file())

    def test_explicit_unverified_profile_success_has_zero_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            summary = TaskRunSummary(
                task_id="k01_vector_add",
                status="passed_profile_unverified",
                best_round=1,
                best_candidate_id="candidate-1",
                final_result={"status": "passed_profile_unverified"},
            )

            with patch(
                "ascend_kernel_lab.cli.ExperimentController.run",
                return_value=(summary,),
            ):
                code, output, error = self._main(
                    [
                        "experiment",
                        "run",
                        "-c",
                        str(config),
                        "--fake",
                        "--allow-unverified-profile",
                    ]
                )

            self.assertEqual(code, 0, error)
            self.assertEqual(
                json.loads(output)["tasks"][0]["status"],
                "passed_profile_unverified",
            )

    def test_production_commands_reject_unversioned_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(self._config(Path(temporary)))

            with self.assertRaisesRegex(
                CommandError,
                "clean Git",
            ) as raised:
                _require_clean_git_release(config)

            self.assertEqual(raised.exception.exit_code, EXIT_NOT_READY)


if __name__ == "__main__":
    unittest.main()
