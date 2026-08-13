from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ascend_kernel_lab.backend.base import StageResult
from ascend_kernel_lab.backend.fake import FakeBackend
from ascend_kernel_lab.config import StorageConfig, load_config
from ascend_kernel_lab.domain import EvaluationStage, ExperimentState, RoundState
from ascend_kernel_lab.export import DatasetExporter, ReportExporter
from ascend_kernel_lab.llm import FakeGateway
from ascend_kernel_lab.orchestration.controller import ExperimentController
from ascend_kernel_lab.storage import AtomicArtifactStore, SQLiteStateStore
from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.verification import RunVerifier

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "experiment_910c_kimi_k3.yaml"


def candidate(round_number: int) -> dict[str, object]:
    return {
        "status": "candidate",
        "round": round_number,
        "optimization_summary": [f"round {round_number}"],
        "expected_effect": ["test"],
        "assumptions": [],
        "code": f"# round {round_number}\ndef custom_op(x):\n    return x\n",
    }


class ControllerIntegrationTests(unittest.TestCase):
    def _config(self, root: Path, rounds: int = 3):
        original = load_config(BASE_CONFIG)
        return replace(
            original,
            id="exp_controller_test",
            rounds_per_task=rounds,
            tasks=("k01_vector_add",),
            model=replace(original.model, provider="fake"),
            storage=StorageConfig(
                database=f"sqlite:///{root / 'metadata.db'}",
                artifact_root=str(root / "runs"),
                task_root=str(ROOT / "task_specs"),
            ),
            project_root=ROOT,
        )

    def _controller(self, root: Path, *, backend: FakeBackend, rounds: int = 3):
        config = self._config(root, rounds)
        gateway = FakeGateway(
            lambda request: candidate(int(request.metadata.get("round", 1)))
        )
        store = SQLiteStateStore(config.db_path)
        controller = ExperimentController(
            config=config,
            store=store,
            artifacts=AtomicArtifactStore(config.artifact_root),
            registry=TaskRegistry(config.task_root),
            model_gateway=gateway,
            backend=backend,
            environment={"device": {"name": "fake-910c"}},
            baseline={
                "k01_vector_add": {
                    "schema_version": "ascend_baseline_snapshot_v1",
                    "comparison_baseline": "pytorch_eager",
                    "pytorch_eager_geomean_us": 15.0,
                    "per_case": [
                        {
                            "case_id": "b01_prime_fp16",
                            "dtype": "float16",
                            "params": {"n": 1_000_003},
                            "weight": 1.0,
                            "pytorch_eager_us": 15.0,
                            "pytorch_eager": {
                                "median_us": 15.0,
                                "raw_samples_us": [14.0, 15.0, 16.0],
                            },
                            "measurement_attempts": [{"attempt": 1}],
                        }
                    ],
                }
            },
            hidden_seed=12345,
            allow_insecure_hidden_seed_for_testing=True,
        )
        return controller, gateway, store

    def test_three_round_run_selects_history_best_and_resumes_idempotently(self) -> None:
        benchmark_results = []
        for speedup in (1.1, 1.4, 1.2):
            benchmark_results.append(
                StageResult.success(
                    EvaluationStage.BENCHMARK,
                    details={
                        "status": "stable",
                        "per_case": [],
                        "geomean_speedup_vs_eager": speedup,
                        "minimum_speedup_vs_eager": speedup - 0.05,
                        "maximum_cv": 0.01,
                    },
                )
            )
        backend = FakeBackend({EvaluationStage.BENCHMARK: benchmark_results})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, gateway, store = self._controller(root, backend=backend)
            summaries = controller.run()
            self.assertEqual(summaries[0].status, "passed")
            self.assertEqual(summaries[0].best_round, 2)
            self.assertEqual(len(gateway.requests), 3)
            first_response = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_01/model_response.json"
                ).read_text(encoding="utf-8")
            )
            first_exchange = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_01/model_exchange.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIn("optimization_summary", first_response)
            self.assertNotIn("change_summary", first_response)
            self.assertIn(
                "optimization_summary", first_exchange["response"]
            )
            first_prompt = json.loads(gateway.requests[0].user_prompt)
            self.assertEqual(
                first_prompt["baseline"]["per_case"],
                [
                    {
                        "case_id": "b01_prime_fp16",
                        "dtype": "float16",
                        "median_us": 15.0,
                        "params": {"n": 1_000_003},
                        "weight": 1.0,
                    }
                ],
            )
            self.assertEqual(
                first_prompt["baseline"]["summary"]["weighted_geomean_us"],
                15.0,
            )
            self.assertNotIn("raw_samples_us", gateway.requests[0].user_prompt)
            self.assertNotIn(
                "measurement_attempts", gateway.requests[0].user_prompt
            )
            follow_up = json.loads(gateway.requests[1].user_prompt)
            self.assertNotIn("working_candidate", follow_up["round_context"])
            self.assertEqual(follow_up["best_candidate"]["round"], 1)
            self.assertEqual(
                follow_up["environment"], {"device": {"name": "fake-910c"}}
            )
            self.assertEqual(
                follow_up["baseline"],
                {"comparison_baseline": "pytorch_eager"},
            )
            self.assertNotIn("best_candidate", follow_up["round_context"])
            self.assertNotIn("last_evaluation", follow_up["round_context"])
            self.assertIn("task_contract", follow_up)
            self.assertNotIn("collection_strategy", follow_up)
            self.assertNotIn("objective", follow_up)
            self.assertEqual(len(follow_up["round_context"]["history_summary"]), 1)
            self.assertNotIn("measurement_attempts", gateway.requests[1].user_prompt)
            for round_number in range(1, 4):
                round_record = store.get_round(
                    controller.experiment_id, "k01_vector_add", round_number
                )
                assert round_record is not None
                self.assertIs(round_record.state, RoundState.ROUND_FINISHED)
            experiment = store.get_experiment(controller.experiment_id)
            assert experiment is not None
            self.assertIs(experiment.state, ExperimentState.FINISHED)
            events_path = controller.experiment_root / "events.jsonl"
            self.assertTrue(events_path.is_file())
            first_events = events_path.read_text(encoding="utf-8")

            resumed = controller.run()
            self.assertEqual(resumed[0].best_round, 2)
            self.assertEqual(len(gateway.requests), 3)
            self.assertEqual(
                [json.loads(line)["sequence"] for line in first_events.splitlines()],
                sorted(json.loads(line)["sequence"] for line in first_events.splitlines()),
            )

    def test_committed_feedback_repairs_missing_database_best_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, _gateway, store = self._controller(
                root, backend=FakeBackend(), rounds=1
            )
            original_run_final = controller._run_final
            split_brain_observed = False

            def run_final_after_pointer_loss(task):
                nonlocal split_brain_observed
                feedback_path = (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_01/feedback.json"
                )
                self.assertTrue(feedback_path.is_file())
                feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
                expected = feedback["best_after"]["candidate_id"]
                with (
                    store.database.connection() as connection,
                    store.database.transaction(connection, immediate=True),
                ):
                    connection.execute(
                        """
                        UPDATE tasks SET best_candidate_id = NULL
                        WHERE experiment_id = ? AND task_id = ?
                        """,
                        (controller.experiment_id, task.id),
                    )
                evaluation = json.loads(
                    (
                        controller.experiment_root
                        / "tasks/k01_vector_add/round_01/evaluation_result.json"
                    ).read_text(encoding="utf-8")
                )

                controller._ensure_feedback(task, 1, evaluation)

                repaired = store.get_task(controller.experiment_id, task.id)
                assert repaired is not None
                self.assertEqual(repaired.best_candidate_id, expected)
                split_brain_observed = True
                return original_run_final(task)

            with patch.object(
                controller, "_run_final", side_effect=run_final_after_pointer_loss
            ):
                summary = controller.run()[0]

            self.assertTrue(split_brain_observed)
            self.assertEqual(summary.best_round, 1)
            self.assertEqual(
                summary.best_candidate_id,
                summary.final_result["best_candidate_id"],
            )

    def test_database_best_before_feedback_artifact_resumes_same_online_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, gateway, store = self._controller(
                root, backend=FakeBackend(), rounds=1
            )
            original_commit = controller._commit_json
            failed = False

            def fail_first_feedback_commit(relative, value, **kwargs):
                nonlocal failed
                if not failed and str(relative).endswith(
                    "tasks/k01_vector_add/round_01/feedback.json"
                ):
                    failed = True
                    raise RuntimeError("injected feedback artifact crash")
                return original_commit(relative, value, **kwargs)

            with (
                patch.object(
                    controller,
                    "_commit_json",
                    side_effect=fail_first_feedback_commit,
                ),
                self.assertRaisesRegex(RuntimeError, "feedback artifact crash"),
            ):
                controller.run()

            task_after_crash = store.get_task(
                controller.experiment_id, "k01_vector_add"
            )
            assert task_after_crash is not None
            selected_before_feedback = task_after_crash.best_candidate_id
            self.assertIsNotNone(selected_before_feedback)
            self.assertFalse(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_01/feedback.json"
                ).exists()
            )
            model_calls = len(gateway.requests)

            summary = controller.run()[0]

            self.assertEqual(len(gateway.requests), model_calls)
            self.assertEqual(summary.best_candidate_id, selected_before_feedback)
            self.assertEqual(
                summary.final_result["best_candidate_id"],
                selected_before_feedback,
            )
            task_after_resume = store.get_task(
                controller.experiment_id, "k01_vector_add"
            )
            assert task_after_resume is not None
            self.assertEqual(
                task_after_resume.best_candidate_id,
                selected_before_feedback,
            )

    def test_final_replays_online_tie_instead_of_legacy_history_ranking(self) -> None:
        public_benchmarks = [
            StageResult.success(
                EvaluationStage.BENCHMARK,
                details={
                    "status": "stable",
                    "per_case": [],
                    "geomean_speedup_vs_eager": 1.0,
                    "minimum_speedup_vs_eager": 1.0,
                    "maximum_cv": 0.02,
                },
            ),
            StageResult.success(
                EvaluationStage.BENCHMARK,
                details={
                    "status": "stable",
                    "per_case": [],
                    "geomean_speedup_vs_eager": 1.015,
                    "minimum_speedup_vs_eager": 1.005,
                    "maximum_cv": 0.02,
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, _gateway, store = self._controller(
                root,
                backend=FakeBackend(
                    {EvaluationStage.BENCHMARK: public_benchmarks}
                ),
                rounds=2,
            )
            original_run_final = controller._run_final

            def run_final_without_pointer(task):
                with (
                    store.database.connection() as connection,
                    store.database.transaction(connection, immediate=True),
                ):
                    connection.execute(
                        """
                        UPDATE tasks SET best_candidate_id = NULL
                        WHERE experiment_id = ? AND task_id = ?
                        """,
                        (controller.experiment_id, task.id),
                    )
                return original_run_final(task)

            with patch.object(
                controller, "_run_final", side_effect=run_final_without_pointer
            ):
                summary = controller.run()[0]

            second_feedback = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_02/feedback.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(second_feedback["performance_decision"], "TIE")
            self.assertEqual(second_feedback["best_after"]["round"], 1)
            self.assertEqual(summary.best_round, 1)
            self.assertEqual(summary.final_result["best_round"], 1)
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertEqual(task.best_candidate_id, summary.best_candidate_id)

    def test_identical_no_change_source_reuses_public_evaluation(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root, rounds=3)
            same_code = "def custom_op(x):\n    return x\n"
            gateway = FakeGateway(
                lambda request: {
                    **candidate(int(request.metadata.get("round", 1))),
                    "status": (
                        "candidate"
                        if int(request.metadata.get("round", 1)) == 1
                        else "no_change"
                    ),
                    "code": same_code,
                }
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={
                    "schema_version": "ascend_fake_environment_v1",
                    "notice": "offline phase verification",
                },
                hidden_seed=77,
                allow_insecure_hidden_seed_for_testing=True,
            )
            controller.run()
            self.assertEqual(
                json.loads(
                    (
                        controller.experiment_root
                        / "tasks/k01_vector_add/round_02/evaluation_result.json"
                    ).read_text()
                )["reused_from_round"],
                1,
            )
            public_benchmarks = [
                call
                for call in backend.calls
                if call["stage"] == EvaluationStage.BENCHMARK.value
            ]
            # One public benchmark and one final hidden benchmark.
            self.assertEqual(len(public_benchmarks), 2)

    def test_crash_after_model_exchange_does_not_request_model_again(self) -> None:
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller, gateway, _store = self._controller(
                root, backend=backend, rounds=1
            )
            with (
                patch.object(
                    controller,
                    "_materialize_model_exchange",
                    side_effect=RuntimeError("injected projection crash"),
                ),
                self.assertRaisesRegex(RuntimeError, "projection crash"),
            ):
                controller.run()
            exchange = (
                controller.experiment_root
                / "tasks/k01_vector_add/round_01/model_exchange.json"
            )
            self.assertTrue(exchange.is_file())
            self.assertEqual(len(gateway.requests), 1)

            resumed = controller.run()
            self.assertEqual(resumed[0].status, "passed")
            self.assertEqual(len(gateway.requests), 1)

    def test_repair_seed_does_not_consume_optimization_rounds(self) -> None:
        backend = FakeBackend(
            {
                EvaluationStage.SOURCE_CHECK: [
                    StageResult.failure(
                        EvaluationStage.SOURCE_CHECK,
                        details={
                            "findings": [
                                {
                                    "code": "forbidden_call",
                                    "message": "getattr is forbidden",
                                }
                            ]
                        },
                    ),
                    StageResult.success(
                        EvaluationStage.SOURCE_CHECK,
                        details={"passed": True, "syntax_ok": True},
                    ),
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=2),
                maximum_repair_rounds=3,
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={
                    "schema_version": "ascend_fake_environment_v1",
                    "notice": "offline phase verification",
                },
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertEqual(summary.final_result["repair_rounds"], 2)
            self.assertEqual(summary.final_result["optimization_rounds"], 2)
            self.assertEqual(len(gateway.requests), 4)
            self.assertEqual(
                [request.metadata["phase"] for request in gateway.requests],
                ["repair", "repair", "optimization", "optimization"],
            )
            second = json.loads(gateway.requests[1].user_prompt)
            self.assertEqual(
                second["failed_candidate"]["raw_stage_result"]["findings"][0],
                {
                    "code": "forbidden_call",
                    "message": "getattr is forbidden",
                },
            )
            third = json.loads(gateway.requests[2].user_prompt)
            self.assertEqual(third["phase"]["index"], 1)
            self.assertNotIn("working_candidate", third["round_context"])
            self.assertEqual(third["best_candidate"]["round"], 2)
            verification = RunVerifier(
                controller.experiment_root, database_path=config.db_path
            ).verify()
            self.assertTrue(verification["passed"], verification["issues"])

    def test_correct_unprofiled_seed_is_not_described_as_best(self) -> None:
        backend = FakeBackend(
            {
                EvaluationStage.PROFILE: [
                    StageResult.failure(
                        EvaluationStage.PROFILE,
                        details={
                            "profile_available": False,
                            "error": "smoke profile unavailable",
                        },
                    )
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=1), maximum_repair_rounds=2
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            controller = ExperimentController(
                config=config,
                store=SQLiteStateStore(config.db_path),
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={},
                baseline={"k01_vector_add": {"kind": "fake"}},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            controller.run()

            follow_up = json.loads(gateway.requests[1].user_prompt)
            instruction = follow_up["feedback_state"]["instruction"]
            self.assertIn("正确但尚未形成可比较 BEST", instruction)
            self.assertIn("benchmark 和 smoke profile", instruction)
            self.assertNotIn("这是当前真实运行得到的 BEST", instruction)
            self.assertEqual(
                follow_up["round_context"]["working_candidate"]["role"],
                "latest_correct_seed",
            )

    def test_correctness_repair_prompt_preserves_raw_failed_stage_once(self) -> None:
        correctness_failure = StageResult.failure(
            EvaluationStage.CORRECTNESS,
            details={
                "passed": False,
                "passed_cases": 5,
                "total_cases": 8,
                "maximum_absolute_error": 9.0,
                "maximum_relative_error": 8.0,
                "case_results": [
                    {
                        "case_id": "metadata_failure",
                        "passed": False,
                        "error": "output metadata or finiteness check failed",
                        "shape_ok": False,
                        "dtype_ok": True,
                        "device_ok": True,
                        "layout_ok": True,
                        "output_alias_ok": True,
                        "finite_ok": False,
                        "inputs_unchanged": True,
                        "maximum_absolute_error": None,
                        "maximum_relative_error": None,
                        "unbounded_debug_payload": "must-not-enter-prompt",
                    },
                    {
                        "case_id": "numeric_failure",
                        "passed": False,
                        "shape_ok": True,
                        "dtype_ok": True,
                        "device_ok": True,
                        "layout_ok": True,
                        "output_alias_ok": True,
                        "finite_ok": True,
                        "inputs_unchanged": True,
                        "maximum_absolute_error": 0.5,
                        "maximum_relative_error": 0.25,
                        "actual_at_maximum_error": 1.5,
                        "expected_at_maximum_error": 1.0,
                        "maximum_error_flat_index": 7,
                    },
                    {
                        "case_id": "third_failure_not_forwarded",
                        "passed": False,
                        "maximum_absolute_error": 9.0,
                        "maximum_relative_error": 8.0,
                    },
                ],
            },
        )
        backend = FakeBackend(
            {EvaluationStage.CORRECTNESS: [correctness_failure]}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=1), maximum_repair_rounds=2
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            controller.run()

            follow_up = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_02/prompt.json"
                ).read_text(encoding="utf-8")
            )["user_prompt"]
            correctness = follow_up["failed_candidate"]["raw_stage_result"]
            self.assertNotIn(
                "working_candidate", follow_up["round_context"]
            )
            self.assertNotIn(
                "raw_failure_evidence", follow_up["failed_candidate"]
            )
            self.assertEqual(
                follow_up["round_context"]["previous_round"]["key_metrics"],
                {},
            )
            self.assertEqual(correctness["passed_cases"], 5)
            self.assertEqual(correctness["total_cases"], 8)
            self.assertEqual(correctness["maximum_absolute_error"], 9.0)
            self.assertEqual(correctness["maximum_relative_error"], 8.0)
            self.assertEqual(
                [case["case_id"] for case in correctness["case_results"]],
                [
                    "metadata_failure",
                    "numeric_failure",
                    "third_failure_not_forwarded",
                ],
            )
            self.assertEqual(
                correctness["case_results"][0]["error"],
                "output metadata or finiteness check failed",
            )
            self.assertFalse(correctness["case_results"][0]["shape_ok"])
            self.assertFalse(correctness["case_results"][0]["finite_ok"])
            self.assertEqual(
                correctness["case_results"][1]["maximum_error_flat_index"],
                7,
            )
            self.assertEqual(
                correctness["case_results"][1]["actual_at_maximum_error"],
                1.5,
            )
            self.assertEqual(
                correctness["case_results"][1]["expected_at_maximum_error"],
                1.0,
            )
            self.assertEqual(
                correctness["case_results"][0]["unbounded_debug_payload"],
                "must-not-enter-prompt",
            )
            failed_candidate = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_02/prompt.json"
                ).read_text(encoding="utf-8")
            )["user_prompt"]["failed_candidate"]
            self.assertEqual(
                failed_candidate["failed_stage"], "correctness"
            )
            self.assertEqual(
                failed_candidate["raw_stage_result"]["case_results"][1][
                    "actual_at_maximum_error"
                ],
                1.5,
            )

    def test_repair_exhaustion_finishes_without_optimization(self) -> None:
        backend = FakeBackend(
            {
                EvaluationStage.SOURCE_CHECK: [
                    StageResult.failure(EvaluationStage.SOURCE_CHECK)
                    for _ in range(3)
                ]
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=5), maximum_repair_rounds=3
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertEqual(summary.status, "repair_exhausted")
            self.assertEqual(summary.final_result["repair_rounds"], 3)
            self.assertEqual(summary.final_result["optimization_rounds"], 0)
            self.assertEqual(len(gateway.requests), 3)
            verification = RunVerifier(
                controller.experiment_root, database_path=config.db_path
            ).verify()
            self.assertTrue(verification["passed"], verification["issues"])

    def test_failed_optimization_slot_repairs_failed_candidate_without_consuming_slot(self) -> None:
        backend = FakeBackend(
            {
                EvaluationStage.CORRECTNESS: [
                    StageResult.success(
                        EvaluationStage.CORRECTNESS,
                        details={"passed": True, "passed_cases": 8, "total_cases": 8},
                    ),
                    StageResult.failure(
                        EvaluationStage.CORRECTNESS,
                        details={
                            "passed": False,
                            "passed_cases": 7,
                            "total_cases": 8,
                            "case_results": [
                                {
                                    "case_id": "bad",
                                    "passed": False,
                                    "maximum_absolute_error": 1.0,
                                }
                            ],
                        },
                    ),
                ],
                EvaluationStage.BENCHMARK: [
                    StageResult.success(
                        EvaluationStage.BENCHMARK,
                        details={
                            "status": "stable",
                            "per_case": [],
                            "geomean_speedup_vs_eager": 1.0,
                            "minimum_speedup_vs_eager": 1.0,
                            "maximum_cv": 0.01,
                        },
                    ),
                    StageResult.success(
                        EvaluationStage.BENCHMARK,
                        details={
                            "status": "stable",
                            "per_case": [],
                            "geomean_speedup_vs_eager": 1.1,
                            "minimum_speedup_vs_eager": 1.05,
                            "maximum_cv": 0.01,
                        },
                    ),
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=1), maximum_repair_rounds=2
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={
                    "schema_version": "ascend_fake_environment_v1",
                    "hardware_profile": {"status": "pending_probe"},
                },
                baseline={"k01_vector_add": {"kind": "fake"}},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertEqual(summary.final_result["optimization_rounds"], 1)
            self.assertEqual(len(gateway.requests), 3)
            repaired = json.loads(gateway.requests[2].user_prompt)
            self.assertEqual(
                repaired["phase"]["name"], "optimization_repair"
            )
            self.assertEqual(
                repaired["failed_candidate"]["code"],
                "# round 2\ndef custom_op(x):\n    return x\n",
            )
            self.assertEqual(
                repaired["failed_candidate"]["candidate_generation_intent"][
                    "optimization_summary"
                ],
                ["round 2"],
            )
            self.assertIn(
                "maximum_absolute_error",
                repr(repaired["failed_candidate"]["raw_stage_result"]),
            )
            self.assertEqual(summary.best_round, 3)
            verification = RunVerifier(
                controller.experiment_root,
                database_path=config.db_path,
            ).verify()
            self.assertTrue(verification["passed"], verification["issues"])

    def test_model_format_repair_regenerates_from_committed_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=1), maximum_repair_rounds=1
            )

            def response(request):
                round_number = int(request.metadata["round"])
                return (
                    "not valid json"
                    if round_number == 2
                    else candidate(round_number)
                )

            gateway = FakeGateway(response)
            controller = ExperimentController(
                config=config,
                store=SQLiteStateStore(config.db_path),
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=FakeBackend(),
                environment={"prompt_environment": {"backend": "fake"}},
                baseline={"k01_vector_add": {"kind": "fake"}},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertTrue(summary.status.startswith("passed"))
            regenerated = json.loads(gateway.requests[-1].user_prompt)
            self.assertEqual(regenerated["phase"]["name"], "optimization_repair")
            self.assertEqual(regenerated["best_candidate"]["round"], 1)
            self.assertEqual(
                regenerated["failed_candidate"][
                    "candidate_generation_intent"
                ]["model_authored"],
                False,
            )
            self.assertIn(
                "model_response_error", regenerated["failed_candidate"]
            )
            self.assertIn(
                "failed_candidate.model_response_error",
                regenerated["phase"]["directive"],
            )
            self.assertNotIn(
                "working_candidate", regenerated["round_context"]
            )

    def test_host_dispatch_seed_stops_before_optimization_model_call(self) -> None:
        host_bound = StageResult.success(
            EvaluationStage.BENCHMARK,
            details={
                "status": "stable",
                "per_case": [],
                "geomean_speedup_vs_eager": 0.9,
                "minimum_speedup_vs_eager": 0.8,
                "bottleneck_type": "host_dispatch",
                "host_dispatch_limited": True,
                "bottleneck": {
                    "bottleneck_type": "host_dispatch",
                    "host_dispatch_limited": True,
                },
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=5), maximum_repair_rounds=3
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=FakeBackend(
                    {EvaluationStage.BENCHMARK: [host_bound]}
                ),
                environment={},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertEqual(len(gateway.requests), 1)
            self.assertEqual(summary.final_result["repair_rounds"], 1)
            self.assertEqual(summary.final_result["optimization_rounds"], 0)
            self.assertEqual(
                summary.final_result["termination_reason"],
                "host_dispatch_limited",
            )

    def test_regressed_host_bound_branch_does_not_stop_best_optimization(self) -> None:
        benchmarks = [
            StageResult.success(
                EvaluationStage.BENCHMARK,
                details={
                    "status": "stable",
                    "per_case": [],
                    "geomean_speedup_vs_eager": 1.2,
                    "minimum_speedup_vs_eager": 1.1,
                    "maximum_cv": 0.01,
                },
            ),
            StageResult.success(
                EvaluationStage.BENCHMARK,
                details={
                    "status": "stable",
                    "per_case": [],
                    "geomean_speedup_vs_eager": 0.8,
                    "minimum_speedup_vs_eager": 0.7,
                    "maximum_cv": 0.01,
                    "bottleneck_type": "host_dispatch",
                    "host_dispatch_limited": True,
                    "bottleneck": {
                        "bottleneck_type": "host_dispatch",
                        "host_dispatch_limited": True,
                    },
                },
            ),
            StageResult.success(
                EvaluationStage.BENCHMARK,
                details={
                    "status": "stable",
                    "per_case": [],
                    "geomean_speedup_vs_eager": 1.3,
                    "minimum_speedup_vs_eager": 1.2,
                    "maximum_cv": 0.01,
                },
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(
                self._config(root, rounds=2), maximum_repair_rounds=1
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata["round"]))
            )
            controller = ExperimentController(
                config=config,
                store=SQLiteStateStore(config.db_path),
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=FakeBackend(
                    {EvaluationStage.BENCHMARK: benchmarks}
                ),
                environment={},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summary = controller.run()[0]

            self.assertEqual(len(gateway.requests), 3)
            self.assertEqual(summary.best_round, 3)
            regressed = json.loads(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/round_02/feedback.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(regressed["performance_decision"], "REGRESSION")

    def test_complete_ten_task_five_round_offline_pipeline(self) -> None:
        """Exercise the production state graph at its full configured cardinality."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = load_config(BASE_CONFIG)
            config = replace(
                original,
                id="exp_full_matrix",
                rounds_per_task=5,
                model=replace(original.model, provider="fake"),
                storage=StorageConfig(
                    database=f"sqlite:///{root / 'metadata.db'}",
                    artifact_root=str(root / "runs"),
                    task_root=str(ROOT / "task_specs"),
                ),
                project_root=ROOT,
            )
            gateway = FakeGateway(
                lambda request: candidate(int(request.metadata.get("round", 1)))
            )
            store = SQLiteStateStore(config.db_path)
            controller = ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=FakeBackend(),
                environment={
                    "schema_version": "ascend_fake_environment_v1",
                    "notice": "offline matrix verification",
                },
                baseline={task_id: {"kind": "fake"} for task_id in config.tasks},
                hidden_seed=12345,
                allow_insecure_hidden_seed_for_testing=True,
            )

            summaries = controller.run()

            self.assertEqual(len(summaries), 10)
            self.assertTrue(all(item.status == "passed" for item in summaries))
            self.assertEqual(len(gateway.requests), 50)
            for task_id in config.tasks:
                for round_number in range(1, 6):
                    record = store.get_round(config.id, task_id, round_number)
                    assert record is not None
                    self.assertIs(record.state, RoundState.ROUND_FINISHED)
            experiment = store.get_experiment(config.id)
            assert experiment is not None
            self.assertIs(experiment.state, ExperimentState.FINISHED)

            calls_before_resume = len(gateway.requests)
            resumed = controller.run()
            self.assertEqual(len(resumed), 10)
            self.assertEqual(len(gateway.requests), calls_before_resume)

            verification = RunVerifier(
                controller.experiment_root,
                database_path=config.db_path,
            ).verify()
            self.assertTrue(verification["passed"], verification["issues"])
            sft_path = root / "exports" / "sft.jsonl"
            rl_path = root / "exports" / "rl.jsonl"
            self.assertEqual(
                DatasetExporter(controller.experiment_root).export_sft(sft_path),
                10,
            )
            self.assertEqual(
                DatasetExporter(controller.experiment_root).export_rl(rl_path),
                50,
            )
            report = ReportExporter(controller.experiment_root).build()
            self.assertEqual(report["task_count"], 10)
            self.assertEqual(report["passed_task_count"], 10)


if __name__ == "__main__":
    unittest.main()
