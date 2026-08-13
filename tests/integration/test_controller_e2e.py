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
        "change_summary": [f"round {round_number}"],
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
            baseline={"k01_vector_add": {"kind": "fake"}},
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
            follow_up = json.loads(gateway.requests[1].user_prompt)
            self.assertIn("last_candidate_code", follow_up["round_context"])
            self.assertNotIn("environment", follow_up)
            self.assertNotIn("baseline", follow_up)
            self.assertNotIn("best_candidate", follow_up["round_context"])
            self.assertNotIn("last_evaluation", follow_up["round_context"])
            self.assertNotIn("task_contract", follow_up)
            self.assertNotIn("collection_strategy", follow_up)
            self.assertNotIn("objective", follow_up)
            self.assertNotIn("history_summary", follow_up["round_context"])
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
                environment={},
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
