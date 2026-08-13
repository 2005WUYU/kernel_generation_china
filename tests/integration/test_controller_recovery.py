from __future__ import annotations

import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from ascend_kernel_lab.backend import FakeBackend, StageResult
from ascend_kernel_lab.config import StorageConfig, load_config
from ascend_kernel_lab.domain import (
    ConcurrentUpdateError,
    EvaluationStage,
    ExperimentState,
    TaskState,
)
from ascend_kernel_lab.llm import FakeGateway
from ascend_kernel_lab.orchestration.controller import ControllerError, ExperimentController
from ascend_kernel_lab.storage import AtomicArtifactStore, SQLiteStateStore
from ascend_kernel_lab.tasks import TaskRegistry

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "experiment_910c_kimi_k3.yaml"


def _candidate(round_number: int) -> dict[str, object]:
    return {
        "status": "candidate",
        "round": round_number,
        "change_summary": ["recovery candidate"],
        "expected_effect": ["exercise persistence boundaries"],
        "assumptions": [],
        "code": "def custom_op(x):\n    return x\n",
    }


class ControllerRecoveryTests(unittest.TestCase):
    def _config(self, root: Path, *, tasks: tuple[str, ...] = ("k01_vector_add",)):
        original = load_config(BASE_CONFIG)
        return replace(
            original,
            id="exp_recovery_test",
            rounds_per_task=1,
            tasks=tasks,
            model=replace(original.model, provider="fake"),
            storage=StorageConfig(
                database=f"sqlite:///{root / 'metadata.db'}",
                artifact_root=str(root / "runs"),
                task_root=str(ROOT / "task_specs"),
            ),
            project_root=ROOT,
        )

    def _controller(
        self,
        root: Path,
        backend: FakeBackend,
        *,
        tasks: tuple[str, ...] = ("k01_vector_add",),
        profile_coverage_required: bool = True,
    ) -> tuple[ExperimentController, FakeGateway, SQLiteStateStore]:
        config = self._config(root, tasks=tasks)
        gateway = FakeGateway(
            lambda request: _candidate(int(request.metadata.get("round", 1)))
        )
        store = SQLiteStateStore(config.db_path)
        return (
            ExperimentController(
                config=config,
                store=store,
                artifacts=AtomicArtifactStore(config.artifact_root),
                registry=TaskRegistry(config.task_root),
                model_gateway=gateway,
                backend=backend,
                environment={"device": {"name": "fake-910c"}},
                baseline={"k01_vector_add": {"kind": "fake"}},
                hidden_seed=12345,
                profile_coverage_required=profile_coverage_required,
                allow_insecure_hidden_seed_for_testing=True,
            ),
            gateway,
            store,
        )

    @staticmethod
    def _stage_count(backend: FakeBackend, stage: EvaluationStage) -> int:
        return sum(call["stage"] == stage.value for call in backend.calls)

    def test_best_candidate_file_crash_resumes_without_repeating_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            controller, gateway, store = self._controller(root, backend)
            original = store.transition_task

            def fail_before_hidden(
                experiment_id: str, task_id: str, target: TaskState, **kwargs: Any
            ) -> Any:
                if target is TaskState.HIDDEN_CORRECTNESS_TEST:
                    raise RuntimeError("injected best projection crash")
                return original(experiment_id, task_id, target, **kwargs)

            with (
                patch.object(store, "transition_task", side_effect=fail_before_hidden),
                self.assertRaisesRegex(RuntimeError, "best projection crash"),
            ):
                controller.run()
            best_path = (
                controller.experiment_root
                / "tasks/k01_vector_add/best_candidate.py"
            )
            self.assertTrue(best_path.is_file())
            self.assertEqual(len(gateway.requests), 1)
            self.assertEqual(self._stage_count(backend, EvaluationStage.CORRECTNESS), 1)

            summary = controller.run()[0]
            self.assertEqual(summary.status, "passed")
            self.assertEqual(len(gateway.requests), 1)
            self.assertEqual(self._stage_count(backend, EvaluationStage.CORRECTNESS), 2)

    def test_final_evaluation_fs_first_crash_repairs_db_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            controller, gateway, store = self._controller(root, backend)
            original = store.register_artifact
            failed = False

            def fail_registration(metadata: Any, **kwargs: Any) -> str:
                nonlocal failed
                if (
                    not failed
                    and str(metadata.relative_path).endswith(
                        "final_evaluation/final_evaluation.json"
                    )
                ):
                    failed = True
                    raise RuntimeError("injected artifact registration crash")
                return original(metadata, **kwargs)

            with (
                patch.object(store, "register_artifact", side_effect=fail_registration),
                self.assertRaisesRegex(RuntimeError, "registration crash"),
            ):
                controller.run()
            final_eval = (
                controller.experiment_root
                / "tasks/k01_vector_add/final_evaluation/final_evaluation.json"
            )
            self.assertTrue(final_eval.is_file())
            calls_after_crash = len(backend.calls)

            summary = controller.run()[0]
            self.assertEqual(summary.status, "passed")
            self.assertEqual(len(backend.calls), calls_after_crash)
            self.assertEqual(len(gateway.requests), 1)
            with store.database.connection() as connection:
                registered = connection.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE relative_path = ?",
                    (
                        "exp_recovery_test/tasks/k01_vector_add/"
                        "final_evaluation/final_evaluation.json",
                    ),
                ).fetchone()[0]
            self.assertEqual(registered, 1)

    def test_resume_rejects_final_profile_policy_flip_after_crash(self) -> None:
        hidden_profile_failure = StageResult.failure(
            EvaluationStage.PROFILE,
            details={"profile_available": False},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend(
                {
                    EvaluationStage.PROFILE: [
                        StageResult.success(
                            EvaluationStage.PROFILE,
                            details={
                                "profile_available": True,
                                "kernel_count": 1,
                                "candidate_kernel_coverage": 1.0,
                            },
                        ),
                        hidden_profile_failure,
                    ]
                }
            )
            controller, _gateway, store = self._controller(root, backend)
            original = store.register_artifact
            failed = False

            def fail_registration(metadata: Any, **kwargs: Any) -> str:
                nonlocal failed
                if (
                    not failed
                    and str(metadata.relative_path).endswith(
                        "final_evaluation/final_evaluation.json"
                    )
                ):
                    failed = True
                    raise RuntimeError("injected final evaluation commit crash")
                return original(metadata, **kwargs)

            with (
                patch.object(store, "register_artifact", side_effect=fail_registration),
                self.assertRaisesRegex(RuntimeError, "commit crash"),
            ):
                controller.run()
            calls_after_crash = len(backend.calls)
            self.assertTrue(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/final_evaluation/final_evaluation.json"
                ).is_file()
            )

            resumed, _gateway, _store = self._controller(
                root,
                backend,
                profile_coverage_required=False,
            )
            with self.assertRaisesRegex(
                ConcurrentUpdateError, "already has different config"
            ):
                resumed.run()
            self.assertEqual(len(backend.calls), calls_after_crash)

    def test_final_result_landed_before_state_progress_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            controller, gateway, store = self._controller(root, backend)
            original = store.transition_task
            failed = False

            def fail_final_state(
                experiment_id: str, task_id: str, target: TaskState, **kwargs: Any
            ) -> Any:
                nonlocal failed
                if not failed and target is TaskState.FINAL_BENCHMARK:
                    failed = True
                    raise RuntimeError("injected final state crash")
                return original(experiment_id, task_id, target, **kwargs)

            with (
                patch.object(store, "transition_task", side_effect=fail_final_state),
                self.assertRaisesRegex(RuntimeError, "final state crash"),
            ):
                controller.run()
            final_result = (
                controller.experiment_root / "tasks/k01_vector_add/final_result.json"
            )
            self.assertTrue(final_result.is_file())
            calls_after_crash = len(backend.calls)

            summary = controller.run()[0]
            self.assertEqual(summary.status, "passed")
            self.assertEqual(len(backend.calls), calls_after_crash)
            self.assertEqual(len(gateway.requests), 1)
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertIs(task.state, TaskState.TASK_FINISHED)

    def test_final_result_registration_crash_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            controller, _gateway, store = self._controller(root, backend)
            original = store.register_artifact
            failed = False

            def fail_registration(metadata: Any, **kwargs: Any) -> str:
                nonlocal failed
                if not failed and str(metadata.relative_path).endswith(
                    "tasks/k01_vector_add/final_result.json"
                ):
                    failed = True
                    raise RuntimeError("injected final result registration crash")
                return original(metadata, **kwargs)

            with (
                patch.object(store, "register_artifact", side_effect=fail_registration),
                self.assertRaisesRegex(RuntimeError, "final result registration"),
            ):
                controller.run()
            self.assertTrue(
                (
                    controller.experiment_root
                    / "tasks/k01_vector_add/final_result.json"
                ).is_file()
            )
            calls_after_crash = len(backend.calls)
            self.assertEqual(controller.run()[0].status, "passed")
            self.assertEqual(len(backend.calls), calls_after_crash)

    def test_partial_task_runs_finish_experiment_only_after_all_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            tasks = ("k01_vector_add", "k02_bias_gelu")
            controller, gateway, store = self._controller(root, backend, tasks=tasks)

            first = controller.run((tasks[0],))
            self.assertEqual(first[0].status, "passed")
            experiment = store.get_experiment(controller.experiment_id)
            assert experiment is not None
            self.assertIs(experiment.state, ExperimentState.RUNNING)

            second = controller.run((tasks[1],))
            self.assertEqual(second[0].status, "passed")
            experiment = store.get_experiment(controller.experiment_id)
            assert experiment is not None
            self.assertIs(experiment.state, ExperimentState.FINISHED)
            self.assertEqual(len(gateway.requests), 2)

    def test_distinct_tasks_run_concurrently_with_stable_summary_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            tasks = ("k01_vector_add", "k02_bias_gelu")
            controller, _gateway, _store = self._controller(
                root, backend, tasks=tasks
            )
            controller.config = replace(controller.config, task_concurrency=2)
            rendezvous = threading.Barrier(2)
            active = 0
            maximum_active = 0
            active_lock = threading.Lock()

            def respond(request: Any) -> dict[str, object]:
                nonlocal active, maximum_active
                with active_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                try:
                    rendezvous.wait(timeout=5)
                    return _candidate(int(request.metadata.get("round", 1)))
                finally:
                    with active_lock:
                        active -= 1

            controller.model_gateway = FakeGateway(respond)
            summaries = controller.run()

            self.assertEqual(maximum_active, 2)
            self.assertEqual(tuple(item.task_id for item in summaries), tasks)
            self.assertTrue(all(item.status == "passed" for item in summaries))

    def test_hidden_benchmark_failure_is_terminal_failure(self) -> None:
        public = StageResult.success(
            EvaluationStage.BENCHMARK,
            details={
                "status": "stable",
                "geomean_speedup_vs_eager": 1.1,
                "minimum_speedup_vs_eager": 0.8,
            },
        )
        hidden_failure = StageResult.failure(
            EvaluationStage.BENCHMARK,
            details={"status": "unstable"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(
                {EvaluationStage.BENCHMARK: [public, hidden_failure]}
            )
            controller, _gateway, store = self._controller(Path(temporary), backend)
            summary = controller.run()[0]
            self.assertEqual(summary.status, "failed_final_benchmark")
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertIs(task.state, TaskState.TASK_FAILED)

    def test_hidden_profile_failure_is_fail_closed_by_default(self) -> None:
        hidden_failure = StageResult.failure(
            EvaluationStage.PROFILE,
            details={"profile_available": False},
        )
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(
                {
                    EvaluationStage.PROFILE: [
                        StageResult.success(
                            EvaluationStage.PROFILE,
                            details={
                                "profile_available": True,
                                "kernel_count": 1,
                                "candidate_kernel_coverage": 1.0,
                            },
                        ),
                        hidden_failure,
                    ]
                }
            )
            controller, _gateway, store = self._controller(Path(temporary), backend)
            summary = controller.run()[0]
            self.assertEqual(summary.status, "failed_final_profile")
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertIs(task.state, TaskState.TASK_FAILED)

    def test_unverified_profile_is_explicit_when_policy_allows_it(self) -> None:
        hidden_failure = StageResult.failure(
            EvaluationStage.PROFILE,
            details={"profile_available": False},
        )
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend(
                {
                    EvaluationStage.PROFILE: [
                        StageResult.success(
                            EvaluationStage.PROFILE,
                            details={
                                "profile_available": True,
                                "kernel_count": 1,
                                "candidate_kernel_coverage": 1.0,
                            },
                        ),
                        hidden_failure,
                    ]
                }
            )
            controller, _gateway, store = self._controller(
                Path(temporary),
                backend,
                profile_coverage_required=False,
            )
            summary = controller.run()[0]
            self.assertEqual(summary.status, "passed_profile_unverified")
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertIs(task.state, TaskState.TASK_FINISHED)

    def test_exhausted_model_format_repairs_commit_failure_and_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            controller, _gateway, store = self._controller(root, backend)
            invalid_gateway = FakeGateway("not valid json")
            controller.model_gateway = invalid_gateway

            summary = controller.run()[0]

            self.assertEqual(summary.status, "failed_no_valid_candidate")
            self.assertEqual(len(invalid_gateway.requests), 2)
            failure_exchange = (
                controller.experiment_root
                / "tasks/k01_vector_add/round_01/model_failure_exchange.json"
            )
            self.assertTrue(failure_exchange.is_file())
            task = store.get_task(controller.experiment_id, "k01_vector_add")
            assert task is not None
            self.assertIs(task.state, TaskState.TASK_FAILED)

            # A terminal resume verifies the durable result and performs no model call.
            self.assertEqual(controller.run()[0].status, "failed_no_valid_candidate")
            self.assertEqual(len(invalid_gateway.requests), 2)

    def test_terminal_state_without_final_result_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = FakeBackend()
            controller, _gateway, _store = self._controller(Path(temporary), backend)
            self.assertEqual(controller.run()[0].status, "passed")
            final = (
                controller.experiment_root / "tasks/k01_vector_add/final_result.json"
            )
            final.unlink()
            with self.assertRaisesRegex(ControllerError, "missing final_result"):
                controller.run()


if __name__ == "__main__":
    unittest.main()
