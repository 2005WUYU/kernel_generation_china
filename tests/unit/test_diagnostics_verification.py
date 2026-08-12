from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from ascend_kernel_lab.cli import build_parser
from ascend_kernel_lab.config import load_config
from ascend_kernel_lab.diagnostics import build_doctor_report
from ascend_kernel_lab.domain import (
    CandidateScore,
    EvaluationJob,
    EvaluationStage,
    ExperimentState,
    RoundState,
    TaskState,
)
from ascend_kernel_lab.storage import AtomicArtifactStore, SQLiteStateStore
from ascend_kernel_lab.verification import RunVerifier


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostic_subprocess_preserves_runtime_but_not_credentials(self) -> None:
        from ascend_kernel_lab.diagnostics import _command

        captured: dict[str, str] = {}

        def run_double(*_args: object, **kwargs: object) -> object:
            environment = kwargs.get("env")
            assert isinstance(environment, dict)
            captured.update(environment)
            return mock.Mock(returncode=0, stdout="", stderr="")

        source = {
            "PATH": "/bin",
            "ASCEND_VISIBLE_DEVICES": "0",
            "DEVICE_ID": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/workspace",
            "ANTHROPIC_AUTH_TOKEN": "must-not-pass",
            "HTTPS_PROXY": "must-not-pass",
        }
        with (
            mock.patch.dict("os.environ", source, clear=True),
            mock.patch("ascend_kernel_lab.diagnostics.shutil.which", return_value="/bin/x"),
            mock.patch("ascend_kernel_lab.diagnostics.subprocess.run", side_effect=run_double),
        ):
            _command(("x",), cwd=Path("."))

        self.assertEqual(captured["ASCEND_VISIBLE_DEVICES"], "0")
        self.assertEqual(captured["DEVICE_ID"], "0")
        self.assertNotIn("GIT_CONFIG_VALUE_0", captured)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", captured)
        self.assertNotIn("HTTPS_PROXY", captured)

    def test_doctor_is_machine_readable_on_non_npu_host(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        report = build_doctor_report(config)
        self.assertEqual(report["schema_version"], "ascend_doctor_report_v1")
        self.assertEqual(report["role"], "all")
        self.assertIn("checks", report)
        self.assertIsInstance(report["ready"], bool)
        names = {item["name"] for item in report["checks"]}
        self.assertIn("ascend_python_runtime", names)
        self.assertIn("hidden_evaluation_secret", names)

    @staticmethod
    def _checks(report: dict[str, object]) -> dict[str, dict[str, object]]:
        checks = report["checks"]
        assert isinstance(checks, list)
        return {str(item["name"]): item for item in checks if isinstance(item, dict)}

    def test_controller_skips_worker_only_ascend_checks(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with mock.patch(
            "ascend_kernel_lab.diagnostics._python_runtime_check",
            side_effect=AssertionError("controller must not inspect the Ascend runtime"),
        ):
            report = build_doctor_report(config, role="controller")

        checks = self._checks(report)
        self.assertEqual(report["role"], "controller")
        for name in ("ascend_python_runtime", "executable_npu-smi", "executable_msprof"):
            self.assertEqual(checks[name]["status"], "skipped")
            self.assertFalse(checks[name]["mandatory"])
        self.assertTrue(checks["model_cli_capabilities"]["mandatory"])
        self.assertTrue(checks["model_environment"]["mandatory"])

    def test_worker_skips_model_checks_without_reading_model_environment(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        model_names = {
            config.model.anthropic_base_url_env,
            config.model.anthropic_auth_token_env,
        }
        real_environ = os.environ

        class GuardedEnvironment(dict[str, str]):
            def get(self, key: str, default: Any = None) -> Any:
                if key in model_names:
                    raise AssertionError("worker read model environment")
                return super().get(key, default)

        guarded = GuardedEnvironment(real_environ)
        with (
            mock.patch("ascend_kernel_lab.diagnostics.os.environ", guarded),
            mock.patch(
                "ascend_kernel_lab.diagnostics._python_runtime_check",
                return_value={
                    "runtime": {
                        name: {"available": True}
                        for name in ("torch", "torch_npu", "triton", "npu")
                    }
                },
            ),
        ):
            report = build_doctor_report(config, role="worker")

        checks = self._checks(report)
        self.assertEqual(report["role"], "worker")
        for name in ("model_cli_capabilities", "model_environment"):
            self.assertEqual(checks[name]["status"], "skipped")
            self.assertFalse(checks[name]["mandatory"])
            details = checks[name]["details"]
            self.assertIsInstance(details, dict)
            assert isinstance(details, dict)
            self.assertNotIn("configured", details)
        for name in ("ascend_python_runtime", "executable_npu-smi", "executable_msprof"):
            self.assertTrue(checks[name]["mandatory"])

    def test_invalid_role_is_rejected(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with self.assertRaisesRegex(ValueError, "unsupported doctor role"):
            build_doctor_report(config, role="invalid")  # type: ignore[arg-type]

    def test_doctor_cli_role_defaults_to_all_and_accepts_worker(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["doctor"]).role, "all")
        self.assertEqual(parser.parse_args(["doctor", "--role", "worker"]).role, "worker")


class VerificationTests(unittest.TestCase):
    def _run(
        self, root: Path, *, final_status: str = "passed"
    ) -> tuple[Path, Path]:
        artifacts = AtomicArtifactStore(root / "runs")
        store = SQLiteStateStore(root / "metadata.db")
        experiment_id = "exp_verify"
        candidate_id = "candidate-1"
        experiment = {
            "schema_version": "ascend_experiment_v1",
            "experiment": {"id": experiment_id, "rounds_per_task": 1, "tasks": ["k01_vector_add"]},
        }
        files = {
            f"{experiment_id}/experiment.json": json.dumps(experiment),
            f"{experiment_id}/tasks/k01_vector_add/round_01/prompt.json": "{}",
            f"{experiment_id}/tasks/k01_vector_add/round_01/model_response.json": "{}",
            f"{experiment_id}/tasks/k01_vector_add/round_01/candidate.py": "def custom_op(x):\n    return x\n",
            f"{experiment_id}/tasks/k01_vector_add/round_01/evaluation_result.json": "{}",
            f"{experiment_id}/tasks/k01_vector_add/round_01/feedback.json": "{}",
            f"{experiment_id}/tasks/k01_vector_add/best_candidate.py": "def custom_op(x):\n    return x\n",
            f"{experiment_id}/tasks/k01_vector_add/final_result.json": json.dumps(
                {
                    "status": final_status,
                    "best_round": 1,
                    "best_candidate_id": candidate_id,
                    "hidden_correctness_passed": True,
                }
            ),
        }
        store.create_experiment(experiment_id, experiment["experiment"])
        for relative, content in files.items():
            metadata = artifacts.put_text(relative, content)
            store.register_artifact(metadata, experiment_id=experiment_id)
        store.transition_experiment(experiment_id, ExperimentState.RUNNING)
        store.create_task(
            experiment_id,
            "k01_vector_add",
            task_version=1,
            task_spec_sha256="a" * 64,
        )
        store.transition_task(
            experiment_id, "k01_vector_add", TaskState.ROUNDS_RUNNING
        )
        store.create_round(experiment_id, "k01_vector_add", 1)
        for state in (
            RoundState.PROMPT_COMMITTED,
            RoundState.MODEL_REQUEST_SENT,
            RoundState.MODEL_RESPONSE_COMMITTED,
            RoundState.SOURCE_VALIDATED,
            RoundState.COMPILE_FINISHED,
            RoundState.CORRECTNESS_FINISHED,
            RoundState.BENCHMARK_FINISHED,
            RoundState.PROFILE_FINISHED,
            RoundState.FEEDBACK_COMMITTED,
            RoundState.ROUND_FINISHED,
        ):
            store.transition_round(experiment_id, "k01_vector_add", 1, state)
        candidate_source = files[
            f"{experiment_id}/tasks/k01_vector_add/round_01/candidate.py"
        ].encode()
        store.register_candidate(
            candidate_id=candidate_id,
            experiment_id=experiment_id,
            task_id="k01_vector_add",
            round_number=1,
            source_sha256=hashlib.sha256(candidate_source).hexdigest(),
            source_artifact_path=(
                f"{experiment_id}/tasks/k01_vector_add/round_01/candidate.py"
            ),
            response_sha256="b" * 64,
        )
        score = CandidateScore(
            candidate_id=candidate_id,
            round_number=1,
            compile_passed=True,
            correctness_passed=True,
            anti_bypass_passed=True,
            hidden_correctness_passed=True,
            minimum_speedup=1.0,
            geomean_speedup=1.0,
            candidate_kernel_coverage=1.0,
            stability_cv=0.01,
        )
        store.save_candidate_score(score)
        store.select_and_set_best_candidate(experiment_id, "k01_vector_add")
        for state in (
            TaskState.SELECT_BEST_CANDIDATE,
            TaskState.HIDDEN_CORRECTNESS_TEST,
            TaskState.FINAL_BENCHMARK,
            TaskState.FINAL_FULL_PROFILE,
            TaskState.TASK_FINISHED,
        ):
            store.transition_task(experiment_id, "k01_vector_add", state)
        store.transition_experiment(experiment_id, ExperimentState.FINISHED)
        events = store.events.read(experiment_id=experiment_id, limit=10_000)
        event_text = "".join(
            json.dumps(
                {
                    "sequence": event.sequence,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "aggregate_type": event.aggregate_type,
                    "aggregate_id": event.aggregate_id,
                },
                sort_keys=True,
            )
            + "\n"
            for event in events
        )
        artifacts.put_text(f"{experiment_id}/events.jsonl", event_text)
        return artifacts.root / experiment_id, root / "metadata.db"

    def test_verifies_hashes_and_run_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, database = self._run(Path(temporary))
            report = RunVerifier(run, database_path=database).verify()
            self.assertTrue(report["passed"], report["issues"])
            self.assertGreater(report["files_verified"], 0)

    def test_detects_tampered_registered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, database = self._run(Path(temporary))
            prompt = run / "tasks/k01_vector_add/round_01/prompt.json"
            prompt.write_text('{"changed":true}', encoding="utf-8")
            report = RunVerifier(run, database_path=database).verify()
            self.assertFalse(report["passed"])
            self.assertIn("artifact_hash_mismatch", {item["code"] for item in report["issues"]})

    def test_rejects_caller_supplied_run_root_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, database = self._run(root)
            alias = root / "run-alias"
            alias.symlink_to(run, target_is_directory=True)

            report = RunVerifier(alias, database_path=database).verify()

            self.assertFalse(report["passed"])
            self.assertIn("run_root_missing", {item["code"] for item in report["issues"]})

    def test_rejects_caller_supplied_database_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, database = self._run(root)
            alias = root / "metadata-alias.db"
            alias.symlink_to(database)

            report = RunVerifier(run, database_path=alias).verify()

            self.assertFalse(report["passed"])
            self.assertIn("database_missing", {item["code"] for item in report["issues"]})

    def test_completed_run_verification_requires_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, _database = self._run(Path(temporary))

            report = RunVerifier(run).verify()

            self.assertFalse(report["passed"])
            self.assertIn(
                "database_not_supplied",
                {item["code"] for item in report["issues"]},
            )

    def test_rejects_configured_task_with_no_rounds_or_final_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "exp_empty"
            task = run / "tasks" / "k01_vector_add"
            task.mkdir(parents=True)
            (run / "experiment.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_experiment_v1",
                        "experiment": {
                            "id": "exp_empty",
                            "rounds_per_task": 2,
                            "tasks": ["k01_vector_add"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run / "events.jsonl").write_text("", encoding="utf-8")

            report = RunVerifier(run).verify()

            codes = {item["code"] for item in report["issues"]}
            self.assertIn("incomplete_round_set", codes)
            self.assertIn("final_result_missing", codes)

    def test_detects_hidden_context_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, database = self._run(Path(temporary))
            prompt = run / "tasks/k01_vector_add/round_01/prompt.json"
            prompt.write_text('{"hidden_seed":123}', encoding="utf-8")
            digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
            self.assertEqual(len(digest), 64)
            report = RunVerifier(run, database_path=database).verify()
            codes = {item["code"] for item in report["issues"]}
            self.assertIn("hidden_data_in_model_context", codes)

    def test_detects_queued_evaluation_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, database = self._run(Path(temporary))
            store = SQLiteStateStore(database)
            store.jobs.enqueue(
                EvaluationJob(
                    job_id="verification-active-job",
                    experiment_id="exp_verify",
                    task_id="k01_vector_add",
                    round_number=1,
                    candidate_id="candidate-1",
                    stage=EvaluationStage.BENCHMARK,
                )
            )

            report = RunVerifier(run, database_path=database).verify()

            codes = {item["code"] for item in report["issues"]}
            self.assertIn("active_jobs_after_finish", codes)

    def test_accepts_explicit_profile_unverified_terminal_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run, database = self._run(
                Path(temporary), final_status="passed_profile_unverified"
            )

            report = RunVerifier(run, database_path=database).verify()

            self.assertTrue(report["passed"], report["issues"])


if __name__ == "__main__":
    unittest.main()
