from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Queue
from typing import Any
from unittest.mock import patch

from ascend_kernel_lab.backend import FakeBackend, StageResult
from ascend_kernel_lab.broker import (
    JOB_PROTOCOL_VERSION,
    PATH_MODE,
    QueueBackendConfigurationError,
    QueueEvaluationBackend,
    QueueJobFailed,
    QueueProtocolError,
    QueueWaitTimeout,
)
from ascend_kernel_lab.domain import (
    CandidateScore,
    EvaluationStage,
    ExperimentState,
    JobStatus,
    LeasedEvaluationJob,
    TaskState,
)
from ascend_kernel_lab.storage import SQLiteEvaluationJobQueue, SQLiteStateStore
from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.tasks.runtime import hidden_cases_from_template
from ascend_kernel_lab.worker import WorkerService


class QueueEvaluationBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir()
        self.store = SQLiteStateStore(self.root / "metadata.db")
        self.queue = SQLiteEvaluationJobQueue(self.store.database)
        project_root = Path(__file__).resolve().parents[2]
        self.task = TaskRegistry(project_root / "task_specs").load("k01_vector_add")

        self.store.create_experiment("exp", {"rounds": 5})
        self.store.transition_experiment("exp", ExperimentState.RUNNING)
        self.store.create_task(
            "exp",
            self.task.id,
            task_version=self.task.version,
            task_spec_sha256=self.task.digest(),
        )
        self.store.transition_task("exp", self.task.id, TaskState.ROUNDS_RUNNING)
        self.store.create_round("exp", self.task.id, 1)

        self.round_dir = (
            self.artifact_root / "exp" / "tasks" / self.task.id / "round_01"
        )
        self.round_dir.mkdir(parents=True)
        self.candidate_path = self.round_dir / "candidate.py"
        self.candidate_path.write_text(
            "def custom_op(x, y):\n    return x\n", encoding="utf-8"
        )
        self.candidate_sha = hashlib.sha256(self.candidate_path.read_bytes()).hexdigest()
        self.candidate_id = f"exp:{self.task.id}:r01:{self.candidate_sha[:16]}"
        self.store.register_candidate(
            candidate_id=self.candidate_id,
            experiment_id="exp",
            task_id=self.task.id,
            round_number=1,
            source_sha256=self.candidate_sha,
            source_artifact_path=f"exp/tasks/{self.task.id}/round_01/candidate.py",
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def backend(self, **overrides: Any) -> QueueEvaluationBackend:
        options: dict[str, Any] = {
            "artifact_root": self.artifact_root,
            "experiment_id": "exp",
            "wait_timeout_seconds": 1.0,
            "poll_interval_seconds": 0.002,
        }
        options.update(overrides)
        return QueueEvaluationBackend(self.queue, **options)

    def worker_once(
        self,
        complete: Callable[[LeasedEvaluationJob], Mapping[str, Any] | None],
    ) -> tuple[threading.Thread, Queue[LeasedEvaluationJob | BaseException]]:
        observed: Queue[LeasedEvaluationJob | BaseException] = Queue()

        def worker() -> None:
            try:
                leased = None
                deadline = time.monotonic() + 1.0
                while leased is None and time.monotonic() < deadline:
                    leased = self.queue.claim("test-worker", lease_seconds=10.0)
                    if leased is None:
                        time.sleep(0.001)
                if leased is None:
                    raise AssertionError("worker did not observe an enqueued job")
                observed.put(leased)
                result = complete(leased)
                if result is not None:
                    self.queue.complete(
                        leased.job_id,
                        "test-worker",
                        leased.lease_token,
                        result,
                    )
            except BaseException as exc:  # surfaced explicitly in the test thread
                observed.put(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return thread, observed

    def assert_worker_ok(
        self,
        thread: threading.Thread,
        observed: Queue[LeasedEvaluationJob | BaseException],
    ) -> LeasedEvaluationJob:
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        item = observed.get_nowait()
        if isinstance(item, BaseException):
            raise item
        if not observed.empty():
            trailing = observed.get_nowait()
            if isinstance(trailing, BaseException):
                raise trailing
        return item

    def success_for(self, leased: LeasedEvaluationJob) -> dict[str, Any]:
        result = StageResult.success(leased.stage, details={"worker": "ok"})
        if leased.stage is EvaluationStage.SOURCE_CHECK or "case_set" in leased.payload:
            return result.to_dict()
        published = self.artifact_root / "test-published" / leased.job_id
        published.mkdir(parents=True)
        evidence = published / "stage_result.json"
        evidence.write_text('{"passed":true}\n', encoding="utf-8")
        payload = evidence.read_bytes()
        manifest_value = {
            "schema_version": "ascend_stage_artifact_manifest_v1",
            "files": [
                {
                    "relative_path": evidence.name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "type": "application/json",
                }
            ],
        }
        manifest_payload = (
            json.dumps(manifest_value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        manifest_digest = hashlib.sha256(manifest_payload).hexdigest()
        manifest = published / f"artifact_manifest.{manifest_digest}.json"
        manifest.write_bytes(manifest_payload)
        return StageResult.success(
            leased.stage,
            details={"worker": "ok"},
            artifacts={
                "attempt_dir": str(published),
                "stage_result": str(evidence),
                "artifact_manifest": str(manifest),
            },
        ).to_dict()

    def test_enqueues_complete_immutable_payload_and_reuses_result(self) -> None:
        thread, observed = self.worker_once(self.success_for)
        result = self.backend().compile(
            self.candidate_path,
            self.task,
            self.task.correctness_cases,
            self.round_dir,
        )
        leased = self.assert_worker_ok(thread, observed)

        self.assertTrue(result.passed)
        self.assertEqual(leased.stage, EvaluationStage.COMPILE)
        payload = dict(leased.payload)
        self.assertEqual(payload["protocol_version"], JOB_PROTOCOL_VERSION)
        self.assertEqual(payload["path_mode"], PATH_MODE)
        self.assertEqual(
            payload["candidate_path"],
            f"exp/tasks/{self.task.id}/round_01/candidate.py",
        )
        self.assertEqual(payload["artifact_dir"], f"exp/tasks/{self.task.id}/round_01")
        self.assertEqual(payload["candidate_sha256"], self.candidate_sha)
        self.assertEqual(payload["task_digest"], self.task.digest())
        self.assertEqual(payload["task_bundle_digest"], self.task.bundle_digest())
        self.assertEqual(payload["task_version"], self.task.version)
        self.assertEqual(payload["task"]["name"], self.task.name)
        self.assertEqual(
            payload["task"]["public_cases"],
            [case.to_dict() for case in self.task.public_cases],
        )
        self.assertEqual(
            payload["cases"],
            [case.to_dict() for case in self.task.correctness_cases],
        )
        self.assertNotIn(str(self.artifact_root), json.dumps(payload))
        self.assertEqual(leased.candidate_id, self.candidate_id)

        # Identical controller replay resolves the already committed terminal
        # result and does not create a duplicate durable job.
        replayed = self.backend().compile(
            self.candidate_path,
            self.task,
            self.task.correctness_cases,
            self.round_dir,
        )
        self.assertEqual(replayed.to_dict(), result.to_dict())
        self.assertEqual(len(self.queue.list()), 1)

    def test_benchmark_carries_detached_baseline_and_has_distinct_key(self) -> None:
        baseline = {"schema_version": "baseline_v1", "per_case": {"b01": 12.5}}
        thread, observed = self.worker_once(self.success_for)
        result = self.backend().benchmark(
            self.candidate_path,
            self.task,
            self.task.benchmark_cases,
            self.round_dir,
            baseline,
        )
        leased = self.assert_worker_ok(thread, observed)
        self.assertTrue(result.passed)
        self.assertEqual(leased.payload["baseline_snapshot"], baseline)
        self.assertEqual(leased.stage, EvaluationStage.BENCHMARK)
        self.assertTrue(leased.job_id.startswith("eval-"))
        self.assertTrue(str(leased.idempotency_key).startswith("akg-eval-v1:"))

    def test_hidden_cases_are_replaced_by_non_secret_case_set_metadata(self) -> None:
        assert self.task.root is not None
        derived = hidden_cases_from_template(self.task.root, secret_seed=98_765)
        hidden_correctness = tuple(
            case for case in derived if case.kind == "correctness"
        )
        leaked_tokens = {
            str(case.seed) for case in hidden_correctness
        } | {
            str(dimension)
            for case in hidden_correctness
            for dimension in case.params.values()
        }
        thread, observed = self.worker_once(self.success_for)
        result = self.backend().check_correctness(
            self.candidate_path,
            self.task,
            hidden_correctness,
            self.round_dir,
        )
        leased = self.assert_worker_ok(thread, observed)
        self.assertTrue(result.passed)
        payload = dict(leased.payload)
        self.assertNotIn("cases", payload)
        case_set = payload["case_set"]
        self.assertEqual(case_set["visibility"], "hidden")
        self.assertEqual(case_set["generator"], "hidden-v1")
        self.assertEqual(case_set["kind"], "correctness")
        self.assertEqual(case_set["count"], 20)
        self.assertRegex(str(case_set["suite_commitment"]), r"^[0-9a-f]{64}$")
        encoded = json.dumps(payload, sort_keys=True)
        self.assertNotIn("hidden_correctness_", encoded)
        self.assertNotIn('"dtype"', json.dumps(payload["case_set"]))
        # Public task snapshots intentionally include public params/seeds. Shape
        # values can coincide, so verify the secret-bearing section structurally
        # and the high-entropy derived seeds textually.
        for seed in {str(case.seed) for case in hidden_correctness}:
            self.assertNotIn(seed, encoded)
        self.assertTrue(leaked_tokens)

    def test_hidden_profile_uses_profile_case_set_without_case_snapshot(self) -> None:
        assert self.task.root is not None
        derived = hidden_cases_from_template(self.task.root, secret_seed=123)
        first_benchmark = tuple(
            case for case in derived if case.kind == "benchmark"
        )[:1]
        thread, observed = self.worker_once(self.success_for)
        self.backend().profile(
            self.candidate_path,
            self.task,
            first_benchmark,
            self.round_dir,
        )
        leased = self.assert_worker_ok(thread, observed)
        self.assertNotIn("cases", leased.payload)
        case_set = leased.payload["case_set"]
        self.assertEqual(case_set["visibility"], "hidden")
        self.assertEqual(case_set["generator"], "hidden-v1")
        self.assertEqual(case_set["kind"], "profile")
        self.assertEqual(case_set["count"], 1)
        self.assertRegex(str(case_set["suite_commitment"]), r"^[0-9a-f]{64}$")

    def test_unredacted_hidden_worker_result_is_rejected(self) -> None:
        assert self.task.root is not None
        hidden = tuple(
            case
            for case in hidden_cases_from_template(self.task.root, secret_seed=555)
            if case.kind == "benchmark"
        )

        def leaked(leased: LeasedEvaluationJob) -> Mapping[str, Any]:
            return StageResult.success(
                leased.stage,
                details={"per_case": [{"params": {"n": 65_537}}]},
            ).to_dict()

        thread, observed = self.worker_once(leaked)
        with self.assertRaisesRegex(QueueProtocolError, "private case metadata"):
            self.backend().benchmark(
                self.candidate_path,
                self.task,
                hidden,
                self.round_dir,
            )
        self.assert_worker_ok(thread, observed)

    def test_hidden_job_round_trip_derives_in_worker_and_returns_redacted_result(self) -> None:
        assert self.task.root is not None
        controller_cases = tuple(
            case
            for case in hidden_cases_from_template(self.task.root, secret_seed=111)
            if case.kind == "correctness"
        )
        service = WorkerService(
            self.queue,
            FakeBackend(),
            TaskRegistry(Path(__file__).resolve().parents[2] / "task_specs"),
            self.artifact_root,
            worker_id="private-worker",
            lease_seconds=0.3,
            heartbeat_seconds=0.05,
            poll_seconds=0.002,
            retry_delay_seconds=0,
            allow_insecure_hidden_seed_for_testing=True,
        )
        observed: Queue[bool | BaseException] = Queue()

        def run_worker() -> None:
            try:
                deadline = time.monotonic() + 1.0
                handled = False
                while not handled and time.monotonic() < deadline:
                    handled = service.run_once()
                    if not handled:
                        time.sleep(0.001)
                observed.put(handled)
            except BaseException as exc:
                observed.put(exc)

        with patch.dict("os.environ", {"AKG_HIDDEN_SEED": "111"}):
            worker = threading.Thread(target=run_worker, daemon=True)
            worker.start()
            result = self.backend().check_correctness(
                self.candidate_path,
                self.task,
                controller_cases,
                self.round_dir,
            )
            worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        worker_value = observed.get_nowait()
        if isinstance(worker_value, BaseException):
            raise worker_value
        self.assertTrue(worker_value)
        self.assertTrue(result.passed)
        self.assertEqual(result.artifacts, {})
        self.assertTrue(result.details["hidden_case_details_redacted"])
        stored = self.queue.list()[0]
        self.assertNotIn("cases", stored.payload)
        self.assertIn("case_set", stored.payload)
        attempts = list(
            self.round_dir.glob("worker_jobs/*/attempt_*")
        )
        self.assertEqual(attempts, [])

    def test_mixed_public_and_hidden_cases_fail_closed(self) -> None:
        assert self.task.root is not None
        hidden = hidden_cases_from_template(
            self.task.root, secret_seed=1, count_correctness=1, count_benchmark=0
        )[0]
        with self.assertRaisesRegex(ValueError, "must never share"):
            self.backend().check_correctness(
                self.candidate_path,
                self.task,
                (self.task.correctness_cases[0], hidden),
                self.round_dir,
            )
        self.assertEqual(self.queue.list(), [])

    def test_malformed_or_wrong_stage_result_is_rejected(self) -> None:
        def wrong_stage(_leased: LeasedEvaluationJob) -> Mapping[str, Any]:
            return StageResult.success(EvaluationStage.PROFILE).to_dict()

        thread, observed = self.worker_once(wrong_stage)
        with self.assertRaisesRegex(QueueProtocolError, "stage mismatch"):
            self.backend().check_correctness(
                self.candidate_path,
                self.task,
                self.task.correctness_cases,
                self.round_dir,
            )
        self.assert_worker_ok(thread, observed)

    def test_result_artifacts_must_remain_inside_shared_root(self) -> None:
        def escaped(leased: LeasedEvaluationJob) -> Mapping[str, Any]:
            return StageResult.success(
                leased.stage,
                artifacts={"escape": "/tmp"},
            ).to_dict()

        thread, observed = self.worker_once(escaped)
        with self.assertRaisesRegex(QueueProtocolError, "escapes artifact_root"):
            self.backend().source_check(self.candidate_path, self.task)
        self.assert_worker_ok(thread, observed)

    def test_passed_public_compute_stage_requires_evidence_manifest(self) -> None:
        def no_evidence(leased: LeasedEvaluationJob) -> Mapping[str, Any]:
            return StageResult.success(leased.stage).to_dict()

        thread, observed = self.worker_once(no_evidence)
        with self.assertRaisesRegex(
            QueueProtocolError,
            "no content-addressed evidence manifest",
        ):
            self.backend().compile(
                self.candidate_path,
                self.task,
                self.task.correctness_cases,
                self.round_dir,
            )
        self.assert_worker_ok(thread, observed)

    def test_wait_timeout_can_cancel_unleased_work(self) -> None:
        backend = self.backend(
            wait_timeout_seconds=0.01,
            poll_interval_seconds=0.002,
            cancel_on_timeout=True,
        )
        with self.assertRaises(QueueWaitTimeout) as raised:
            backend.source_check(self.candidate_path, self.task)
        self.assertTrue(raised.exception.cancelled)
        jobs = self.queue.list()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, JobStatus.CANCELLED)

    def test_dead_letter_is_transport_failure_not_candidate_failure(self) -> None:
        def fail_job(leased: LeasedEvaluationJob) -> None:
            self.queue.fail(
                leased.job_id,
                "test-worker",
                leased.lease_token,
                {"type": "WorkerCrashed", "message": "boom"},
                retryable=False,
            )
            return None

        thread, observed = self.worker_once(fail_job)
        with self.assertRaises(QueueJobFailed) as raised:
            self.backend().source_check(self.candidate_path, self.task)
        self.assertEqual(raised.exception.status, JobStatus.DEAD)
        self.assertEqual(raised.exception.error["type"], "WorkerCrashed")
        self.assert_worker_ok(thread, observed)

    def test_changed_candidate_is_rejected_before_enqueue(self) -> None:
        self.candidate_path.write_text(
            "def custom_op(x, y):\n    return y\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            QueueBackendConfigurationError, "immutable persisted digest"
        ):
            self.backend().source_check(self.candidate_path, self.task)
        self.assertEqual(self.queue.list(), [])

    def test_outside_and_symlink_candidate_paths_are_rejected(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text("def custom_op(x, y): return x\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes artifact_root"):
            self.backend().source_check(outside, self.task)

        link = self.round_dir / "linked.py"
        try:
            link.symlink_to(self.candidate_path)
        except OSError as exc:  # pragma: no cover - platform policy
            self.skipTest(f"symlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.backend().source_check(link, self.task)
        self.assertEqual(self.queue.list(), [])

    def test_selected_best_copy_resolves_original_round_and_candidate(self) -> None:
        self.store.save_candidate_score(
            CandidateScore(
                candidate_id=self.candidate_id,
                round_number=1,
                compile_passed=True,
                correctness_passed=True,
                anti_bypass_passed=True,
                minimum_speedup=1.1,
                geomean_speedup=1.2,
            )
        )
        selected = self.store.select_and_set_best_candidate("exp", self.task.id)
        self.assertIsNotNone(selected)
        best_path = self.artifact_root / "exp" / "tasks" / self.task.id / "best_candidate.py"
        best_path.write_bytes(self.candidate_path.read_bytes())

        thread, observed = self.worker_once(self.success_for)
        result = self.backend().source_check(best_path, self.task)
        leased = self.assert_worker_ok(thread, observed)
        self.assertTrue(result.passed)
        self.assertEqual(leased.round_number, 1)
        self.assertEqual(leased.candidate_id, self.candidate_id)


if __name__ == "__main__":
    unittest.main()
