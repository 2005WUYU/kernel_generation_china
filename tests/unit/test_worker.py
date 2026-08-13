from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

from ascend_kernel_lab.backend import FakeBackend, StageResult
from ascend_kernel_lab.domain import (
    EvaluationJob,
    EvaluationStage,
    JobStatus,
    StoredEvaluationJob,
)
from ascend_kernel_lab.protocol import harness_digest
from ascend_kernel_lab.storage import SQLiteStateStore
from ascend_kernel_lab.tasks import CaseSpec, TaskRegistry, TaskSpec
from ascend_kernel_lab.tasks.runtime import (
    hidden_cases_from_template,
    hidden_suite_commitment,
)
from ascend_kernel_lab.worker import DeviceLock, DeviceLockTimeout, WorkerService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SlowFakeBackend(FakeBackend):
    def source_check(self, candidate_path: Path, task: object) -> StageResult:
        time.sleep(0.2)
        return StageResult.success(EvaluationStage.SOURCE_CHECK, details={"passed": True})


class LeakyHiddenBackend(FakeBackend):
    """Deliberately emits private data to prove the worker boundary redacts it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, tuple[CaseSpec, ...]] = {}

    def _private_result(
        self,
        stage: EvaluationStage,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        extra: Mapping[str, object],
    ) -> StageResult:
        values = tuple(cases)
        self.seen[stage.value] = values
        leak = artifact_dir / "private-case.json"
        leak.write_text(
            json.dumps([case.to_dict() for case in values]), encoding="utf-8"
        )
        details: dict[str, object] = {
            "passed": True,
            "case_results": [
                {
                    "case_id": case.id,
                    "params": dict(case.params),
                    "seed": case.seed,
                    "dtype": case.dtype,
                }
                for case in values
            ],
            **extra,
        }
        return StageResult.success(
            stage,
            details=details,
            artifacts={"private": str(leak)},
        )

    def check_correctness(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        del candidate_path, task
        return self._private_result(
            EvaluationStage.CORRECTNESS,
            cases,
            artifact_dir,
            {"passed_cases": len(cases), "total_cases": len(cases)},
        )

    def benchmark(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
        baseline_snapshot: Mapping[str, object] | None = None,
    ) -> StageResult:
        del candidate_path, task, baseline_snapshot
        return self._private_result(
            EvaluationStage.BENCHMARK,
            cases,
            artifact_dir,
            {
                "status": "stable",
                "geomean_speedup_vs_eager": 1.2,
                "minimum_speedup_vs_eager": 1.1,
                "maximum_speedup_vs_eager": 1.3,
                "maximum_cv": 0.01,
            },
        )

    def profile(
        self,
        candidate_path: Path,
        task: TaskSpec,
        cases: Sequence[CaseSpec],
        artifact_dir: Path,
    ) -> StageResult:
        del candidate_path, task
        return self._private_result(
            EvaluationStage.PROFILE,
            cases,
            artifact_dir,
            {
                "profile_available": True,
                "summary": {
                    "profile_available": True,
                    "kernel_count": 1,
                    "candidate_kernel_coverage": 1.0,
                },
            },
        )


class WorkerServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir()
        self.store = SQLiteStateStore(self.root / "metadata.db")
        self.registry = TaskRegistry(PROJECT_ROOT / "task_specs")
        self.task = self.registry.load("k01_vector_add")
        self.candidate = self.artifact_root / "candidates" / "candidate.py"
        self.candidate.parent.mkdir()
        self.candidate.write_text("candidate source\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def _payload(self, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "protocol_version": "ascend_eval_job_v1",
            "harness_digest": harness_digest(),
            "path_mode": "artifact_root_relative",
            "candidate_path": "candidates/candidate.py",
            "artifact_dir": "jobs/job",
            "candidate_sha256": hashlib.sha256(
                self.candidate.read_bytes()
            ).hexdigest(),
            "task_digest": self.task.digest(),
            "task_bundle_digest": self.task.bundle_digest(),
            "task_version": self.task.version,
            "task": {
                "id": self.task.id,
                "version": self.task.version,
                "name": self.task.name,
                "description": self.task.description,
                "entry_point": self.task.entry_point,
                "inputs": [dict(item) for item in self.task.inputs],
                "outputs": [dict(item) for item in self.task.outputs],
                "semantics": dict(self.task.semantics),
                "correctness": dict(self.task.correctness),
                "benchmark": dict(self.task.benchmark),
                "restrictions": dict(self.task.restrictions),
                "public_cases": [case.to_dict() for case in self.task.public_cases],
            },
            "cases": [],
        }
        value.update(overrides)
        return value

    def _enqueue(
        self,
        *,
        payload: dict[str, object] | None = None,
        max_attempts: int = 1,
    ) -> None:
        self.store.jobs.enqueue(
            EvaluationJob(
                job_id="job",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.SOURCE_CHECK,
                payload=payload or self._payload(),
                max_attempts=max_attempts,
            )
        )

    def _service(self, backend: FakeBackend) -> WorkerService:
        return WorkerService(
            self.store.jobs,
            backend,
            self.registry,
            self.artifact_root,
            worker_id="worker-1",
            lease_seconds=0.3,
            heartbeat_seconds=0.05,
            poll_seconds=0.01,
            retry_delay_seconds=0,
            allow_insecure_hidden_seed_for_testing=True,
        )

    def _hidden_case_set(self, kind: str, count: int, seed: int) -> dict[str, object]:
        assert self.task.root is not None
        derived = hidden_cases_from_template(
            self.task.root,
            secret_seed=seed,
            count_correctness=count if kind == "correctness" else 0,
            count_benchmark=count if kind in {"benchmark", "profile"} else 0,
        )
        source_kind = "benchmark" if kind == "profile" else kind
        selected = tuple(case for case in derived if case.kind == source_kind)[:count]
        return {
            "visibility": "hidden",
            "generator": "hidden-v1",
            "kind": kind,
            "count": count,
            "suite_commitment": hidden_suite_commitment(
                selected,
                experiment_id="experiment",
                task_id=self.task.id,
                generator="hidden-v1",
                kind=kind,
            ),
        }

    def _stored(self, job_id: str = "job") -> StoredEvaluationJob:
        stored = self.store.jobs.get(job_id)
        assert stored is not None
        return stored

    def test_successful_stage_is_completed_and_persisted(self) -> None:
        self._enqueue()
        backend = FakeBackend()
        self.assertTrue(self._service(backend).run_once())
        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.SUCCEEDED)
        assert stored.result is not None
        self.assertTrue(stored.result["passed"])
        self.assertEqual(stored.result["stage"], EvaluationStage.SOURCE_CHECK.value)

    def test_candidate_failure_is_a_completed_evaluation_not_a_retry(self) -> None:
        self._enqueue(max_attempts=2)
        failure = StageResult.failure(
            EvaluationStage.SOURCE_CHECK,
            details={"passed": False},
        )
        backend = FakeBackend({EvaluationStage.SOURCE_CHECK: [failure]})
        self._service(backend).run_once()
        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.SUCCEEDED)
        assert stored.result is not None
        self.assertFalse(stored.result["passed"])

    def test_retryable_infrastructure_error_returns_to_retry_wait(self) -> None:
        self._enqueue(max_attempts=2)
        transient = StageResult.infrastructure_error(
            EvaluationStage.SOURCE_CHECK,
            message="device busy",
            retryable=True,
        )
        backend = FakeBackend({EvaluationStage.SOURCE_CHECK: [transient]})
        self._service(backend).run_once()
        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.RETRY_WAIT)
        assert stored.last_error is not None
        self.assertEqual(stored.last_error["type"], "BackendError")

    def test_retryable_stage_result_is_committed_after_last_attempt(self) -> None:
        self._enqueue(max_attempts=1)
        timeout = StageResult.infrastructure_error(
            EvaluationStage.SOURCE_CHECK,
            message="stage timed out",
            error_type="StageTimeout",
            retryable=True,
            timed_out=True,
        )
        backend = FakeBackend({EvaluationStage.SOURCE_CHECK: [timeout]})

        self.assertTrue(self._service(backend).run_once())

        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.SUCCEEDED)
        assert stored.result is not None
        self.assertEqual(stored.result["status"], "timeout")
        self.assertEqual(stored.result["error"]["type"], "StageTimeout")

    def test_failed_health_check_quarantines_worker_and_stops_claiming(self) -> None:
        self._enqueue(max_attempts=2)
        second_payload = self._payload(artifact_dir="jobs/second")
        self.store.jobs.enqueue(
            EvaluationJob(
                job_id="known-good",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.SOURCE_CHECK,
                payload=second_payload,
            )
        )
        transient = StageResult.infrastructure_error(
            EvaluationStage.SOURCE_CHECK,
            message="runtime failed",
            retryable=True,
        )
        unhealthy = StageResult.failure(
            "HEALTH_CHECK",
            details={"healthy": False},
        )
        backend = FakeBackend(
            {
                EvaluationStage.SOURCE_CHECK: [transient],
                "HEALTH_CHECK": [unhealthy],
            }
        )
        service = self._service(backend)
        self.assertTrue(service.run_once())
        self.assertTrue(service.quarantined)
        assert service.quarantine_reason is not None
        self.assertEqual(service.quarantine_reason["type"], "DeviceQuarantined")
        failed = self._stored()
        self.assertEqual(failed.status, JobStatus.RETRY_WAIT)
        assert failed.last_error is not None
        self.assertEqual(failed.last_error["type"], "DeviceQuarantined")
        self.assertFalse(service.run_once())
        self.assertEqual(self._stored("known-good").status, JobStatus.QUEUED)

    def test_healthy_device_after_bad_job_can_accept_known_good(self) -> None:
        self._enqueue(max_attempts=2)
        self.store.jobs.enqueue(
            EvaluationJob(
                job_id="known-good",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.SOURCE_CHECK,
                payload=self._payload(artifact_dir="jobs/known-good"),
            )
        )
        transient = StageResult.infrastructure_error(
            EvaluationStage.SOURCE_CHECK,
            message="runtime failed",
            retryable=True,
        )
        backend = FakeBackend(
            {
                EvaluationStage.SOURCE_CHECK: [transient],
                "HEALTH_CHECK": [StageResult.success("HEALTH_CHECK")],
            }
        )
        service = WorkerService(
            self.store.jobs,
            backend,
            self.registry,
            self.artifact_root,
            worker_id="worker-healthy",
            lease_seconds=0.3,
            heartbeat_seconds=0.05,
            poll_seconds=0.01,
            retry_delay_seconds=60,
        )
        self.assertTrue(service.run_once())
        self.assertFalse(service.quarantined)
        self.assertEqual(self._stored().status, JobStatus.RETRY_WAIT)
        self.assertTrue(service.run_once())
        self.assertEqual(self._stored("known-good").status, JobStatus.SUCCEEDED)

    def test_path_traversal_and_hash_mismatch_are_dead_lettered(self) -> None:
        for payload in (
            self._payload(candidate_path="../candidate.py"),
            self._payload(candidate_sha256="0" * 64),
        ):
            with self.subTest(payload=payload):
                job_id = "job" if self.store.jobs.get("job") is None else "job-2"
                self.store.jobs.enqueue(
                    EvaluationJob(
                        job_id=job_id,
                        experiment_id="experiment",
                        task_id=self.task.id,
                        round_number=1,
                        candidate_id="candidate",
                        stage=EvaluationStage.SOURCE_CHECK,
                        payload=payload,
                        max_attempts=1,
                    )
                )
                self._service(FakeBackend()).run_once()
                self.assertEqual(self._stored(job_id).status, JobStatus.DEAD)

    def test_heartbeat_keeps_a_slow_evaluation_lease_alive(self) -> None:
        self._enqueue()
        service = WorkerService(
            self.store.jobs,
            SlowFakeBackend(),
            self.registry,
            self.artifact_root,
            worker_id="slow-worker",
            lease_seconds=0.15,
            heartbeat_seconds=0.03,
            poll_seconds=0.01,
        )
        service.run_once()
        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.SUCCEEDED)
        self.assertGreaterEqual(stored.attempt_count, 1)

    def test_distinct_stages_share_parent_artifact_dir_without_collision(self) -> None:
        source_payload = self._payload()
        compile_payload = self._payload(
            cases=[self.task.correctness_cases[0].to_dict()]
        )
        jobs = (
            EvaluationJob(
                job_id="source-job",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.SOURCE_CHECK,
                payload=source_payload,
                priority=2,
            ),
            EvaluationJob(
                job_id="compile-job",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.COMPILE,
                payload=compile_payload,
                priority=1,
            ),
        )
        for job in jobs:
            self.store.jobs.enqueue(job)
        service = self._service(FakeBackend())
        self.assertTrue(service.run_once())
        self.assertTrue(service.run_once())
        self.assertEqual(self._stored("source-job").status, JobStatus.SUCCEEDED)
        self.assertEqual(self._stored("compile-job").status, JobStatus.SUCCEEDED)
        attempts = list((self.artifact_root / "jobs" / "job" / "worker_jobs").glob("*/attempt_001"))
        self.assertEqual(len(attempts), 2)
        for attempt in attempts:
            for shared_directory in (attempt, attempt.parent):
                mode = stat.S_IMODE(shared_directory.stat().st_mode)
                self.assertEqual(mode & 0o777, 0o770)
                if sys.platform != "darwin":
                    self.assertEqual(mode & stat.S_ISGID, stat.S_ISGID)

    def test_task_snapshot_tampering_is_dead_lettered(self) -> None:
        payload = self._payload()
        raw_snapshot = payload["task"]
        assert isinstance(raw_snapshot, Mapping)
        snapshot = dict(raw_snapshot)
        snapshot["description"] = "tampered"
        payload["task"] = snapshot
        self._enqueue(payload=payload)
        self._service(FakeBackend()).run_once()
        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.DEAD)
        assert stored.last_error is not None
        self.assertEqual(stored.last_error["type"], "WorkerPayloadError")

    def test_harness_release_mismatch_is_dead_lettered_before_execution(self) -> None:
        payload = self._payload(harness_digest="0" * 64)
        self._enqueue(payload=payload)
        backend = FakeBackend()

        self._service(backend).run_once()

        stored = self._stored()
        self.assertEqual(stored.status, JobStatus.DEAD)
        self.assertEqual(backend.calls, [])
        assert stored.last_error is not None
        self.assertEqual(stored.last_error["type"], "WorkerPayloadError")

    def test_hidden_suite_commitment_mismatch_fails_closed_without_leak(self) -> None:
        payload = self._payload()
        payload.pop("cases")
        payload["case_set"] = self._hidden_case_set("correctness", 1, 912345)
        self.store.jobs.enqueue(
            EvaluationJob(
                job_id="hidden-mismatch",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.CORRECTNESS,
                payload=payload,
                max_attempts=1,
            )
        )
        backend = FakeBackend()

        with patch.dict(os.environ, {"AKG_HIDDEN_SEED": "912346"}, clear=False):
            self._service(backend).run_once()

        stored = self._stored("hidden-mismatch")
        self.assertEqual(stored.status, JobStatus.DEAD)
        self.assertEqual(backend.calls, [])
        assert stored.last_error is not None
        encoded = json.dumps(dict(stored.last_error), sort_keys=True)
        self.assertEqual(stored.last_error["type"], "HiddenWorkerPayloadError")
        self.assertNotIn("912345", encoded)
        self.assertNotIn("912346", encoded)

    def test_hidden_backend_exception_is_redacted_and_attempt_is_removed(self) -> None:
        class ExplodingBackend(FakeBackend):
            def check_correctness(
                self,
                candidate_path: Path,
                task: TaskSpec,
                cases: Sequence[CaseSpec],
                artifact_dir: Path,
            ) -> StageResult:
                del candidate_path, task
                leak = artifact_dir / "leak.txt"
                leak.write_text(str(cases[0].params), encoding="utf-8")
                raise RuntimeError(f"private shape {cases[0].params}")

        payload = self._payload()
        payload.pop("cases")
        payload["case_set"] = self._hidden_case_set("correctness", 1, 912345)
        self.store.jobs.enqueue(
            EvaluationJob(
                job_id="hidden-exception",
                experiment_id="experiment",
                task_id=self.task.id,
                round_number=1,
                candidate_id="candidate",
                stage=EvaluationStage.CORRECTNESS,
                payload=payload,
                max_attempts=1,
            )
        )
        with patch.dict(os.environ, {"AKG_HIDDEN_SEED": "912345"}, clear=False):
            self._service(ExplodingBackend()).run_once()
        stored = self._stored("hidden-exception")
        self.assertEqual(stored.status, JobStatus.DEAD)
        assert stored.last_error is not None
        encoded = json.dumps(dict(stored.last_error), sort_keys=True)
        self.assertEqual(stored.last_error["type"], "HiddenInfrastructureError")
        self.assertNotIn("private shape", encoded)
        worker_jobs = self.artifact_root / "jobs" / "job" / "worker_jobs"
        self.assertEqual(list(worker_jobs.glob("*/attempt_*")), [])

    def test_hidden_correctness_benchmark_and_profile_are_derived_and_redacted(self) -> None:
        backend = LeakyHiddenBackend()
        service = self._service(backend)
        stage_cases = (
            (EvaluationStage.CORRECTNESS, "correctness", 2),
            (EvaluationStage.BENCHMARK, "benchmark", 2),
            (EvaluationStage.PROFILE, "profile", 1),
        )
        for priority, (stage, kind, count) in enumerate(stage_cases, start=1):
            payload = self._payload(artifact_dir=f"jobs/private-{kind}")
            payload.pop("cases")
            payload["case_set"] = self._hidden_case_set(kind, count, 912345)
            self.store.jobs.enqueue(
                EvaluationJob(
                    job_id=f"private-{kind}",
                    experiment_id="experiment",
                    task_id=self.task.id,
                    round_number=1,
                    candidate_id="candidate",
                    stage=stage,
                    payload=payload,
                    priority=len(stage_cases) - priority,
                )
            )

        with patch.dict(os.environ, {"AKG_HIDDEN_SEED": "912345"}, clear=False):
            for _stage, _kind, _count in stage_cases:
                self.assertTrue(service.run_once())

        expected_seen = {
            EvaluationStage.CORRECTNESS.value: ("correctness", 2),
            EvaluationStage.BENCHMARK.value: ("benchmark", 2),
            EvaluationStage.PROFILE.value: ("benchmark", 1),
        }
        for stage_name, (derived_kind, count) in expected_seen.items():
            seen = backend.seen[stage_name]
            self.assertEqual(len(seen), count)
            self.assertTrue(all(case.kind == derived_kind for case in seen))

        for _stage, kind, _count in stage_cases:
            stored = self._stored(f"private-{kind}")
            self.assertEqual(stored.status, JobStatus.SUCCEEDED)
            self.assertNotIn("cases", stored.payload)
            assert stored.result is not None
            encoded = json.dumps(dict(stored.result), sort_keys=True)
            self.assertEqual(stored.result["artifacts"], {})
            details = stored.result["details"]
            assert isinstance(details, Mapping)
            self.assertTrue(details["hidden_case_details_redacted"])
            for forbidden in (
                '"case_results"',
                '"dtype"',
                '"params"',
                '"per_case"',
                '"seed"',
                "912345",
            ):
                self.assertNotIn(forbidden, encoded)
            attempts = list(
                (
                    self.artifact_root
                    / "jobs"
                    / f"private-{kind}"
                    / "worker_jobs"
                ).glob("*/attempt_*")
            )
            self.assertEqual(attempts, [])

    def test_idle_worker_and_bounded_serve_return(self) -> None:
        service = self._service(FakeBackend())
        self.assertFalse(service.run_once())
        self.assertEqual(service.serve_forever(max_jobs=0), 0)


class DeviceLockTests(unittest.TestCase):
    def test_second_lock_times_out_until_first_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = DeviceLock("npu:0", lock_root=temporary, timeout_seconds=1)
            second = DeviceLock("npu:0", lock_root=temporary, timeout_seconds=0.05)
            with first, self.assertRaises(DeviceLockTimeout):
                second.acquire()
            with second:
                self.assertTrue(second.acquired)


if __name__ == "__main__":
    unittest.main()
