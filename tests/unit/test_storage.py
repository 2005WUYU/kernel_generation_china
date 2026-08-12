from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ascend_kernel_lab.domain import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    CandidateScore,
    ConcurrentUpdateError,
    EvaluationJob,
    EvaluationStage,
    ExperimentState,
    IdempotencyConflictError,
    InvalidTransitionError,
    JobStatus,
    LeaseLostError,
    RoundState,
    TaskState,
)
from ascend_kernel_lab.storage import (
    AtomicArtifactStore,
    SQLiteDatabase,
    SQLiteStateStore,
)

UTC = timezone.utc
BASE_TIME = datetime(2026, 8, 12, 4, 0, tzinfo=UTC)


class SQLiteDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "metadata.db"
        self.database = SQLiteDatabase(self.database_path)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_migrations_are_idempotent_and_wal_is_enabled(self) -> None:
        self.database.initialize()
        with self.database.connection() as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            versions = connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        self.assertEqual(journal_mode.lower(), "wal")
        self.assertEqual([row[0] for row in versions], [1, 2])
        self.assertEqual(foreign_keys, 1)

    def test_database_wal_and_parent_are_group_shared_not_other_accessible(self) -> None:
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            wal_path = Path(f"{self.database_path}-wal")
            shm_path = Path(f"{self.database_path}-shm")
            self.assertTrue(wal_path.is_file())
            self.assertTrue(shm_path.is_file())
            for path in (self.database_path, wal_path, shm_path):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o660)
            connection.rollback()
        self.assertEqual(
            stat.S_IMODE(self.database_path.parent.stat().st_mode),
            0o2770,
        )

    def test_existing_database_modes_are_repaired_without_truncation(self) -> None:
        legacy_root = self.database_path.parent / "legacy"
        legacy_root.mkdir(mode=0o700)
        legacy_path = legacy_root / "metadata.db"
        with sqlite3.connect(legacy_path) as connection:
            connection.execute("CREATE TABLE preserved(value TEXT NOT NULL)")
            connection.execute("INSERT INTO preserved(value) VALUES ('before')")
        os.chmod(legacy_root, 0o700)
        os.chmod(legacy_path, 0o600)

        repaired = SQLiteDatabase(legacy_path)
        try:
            with repaired.connection() as connection:
                preserved = connection.execute(
                    "SELECT value FROM preserved"
                ).fetchone()[0]
            self.assertEqual(preserved, "before")
            self.assertEqual(
                stat.S_IMODE(legacy_root.stat().st_mode),
                0o2770,
            )
            self.assertEqual(stat.S_IMODE(legacy_path.stat().st_mode), 0o660)
        finally:
            repaired.close()

    def test_database_refuses_symlink_and_non_shared_permission_contracts(self) -> None:
        unsafe = Path(self.temporary.name) / "unsafe.db"
        unsafe.symlink_to(self.database_path)
        with self.assertRaisesRegex(ValueError, "regular file"):
            SQLiteDatabase(unsafe)
        with self.assertRaisesRegex(ValueError, "02770"):
            SQLiteDatabase(
                Path(self.temporary.name) / "other.db",
                directory_mode=0o777,
            )

    def test_nested_transaction_uses_savepoint(self) -> None:
        with self.database.connection() as connection:
            with self.database.transaction(connection, immediate=True):
                connection.execute(
                    """
                    INSERT INTO experiments(
                        experiment_id, state, config_json, created_at, updated_at
                    ) VALUES ('outer', 'CREATED', '{}', 1, 1)
                    """
                )
                with (
                    self.assertRaises(RuntimeError),
                    self.database.transaction(connection),
                ):
                        connection.execute(
                            """
                            INSERT INTO experiments(
                                experiment_id, state, config_json,
                                created_at, updated_at
                            ) VALUES ('inner', 'CREATED', '{}', 1, 1)
                            """
                        )
                        raise RuntimeError("rollback savepoint")
            rows = connection.execute(
                "SELECT experiment_id FROM experiments ORDER BY experiment_id"
            ).fetchall()
        self.assertEqual([row[0] for row in rows], ["outer"])


class StateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = SQLiteStateStore(
            Path(self.temporary.name) / "metadata.db"
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def _create_running_task(self) -> None:
        self.store.create_experiment("exp", {"rounds": 5}, now=BASE_TIME)
        self.store.transition_experiment(
            "exp", target=ExperimentState.RUNNING, now=BASE_TIME
        )
        self.store.create_task("exp", "k01", now=BASE_TIME)
        self.store.transition_task(
            "exp", "k01", TaskState.ROUNDS_RUNNING, now=BASE_TIME
        )

    def test_round_state_and_event_commit_together(self) -> None:
        self._create_running_task()
        round_record = self.store.create_round("exp", "k01", 1, now=BASE_TIME)
        updated = self.store.transition_round(
            "exp",
            "k01",
            1,
            RoundState.PROMPT_COMMITTED,
            expected_version=round_record.version,
            event_payload={"artifact": "prompt.json"},
            now=BASE_TIME,
        )
        self.assertEqual(updated.state, RoundState.PROMPT_COMMITTED)
        events = self.store.events.read(
            aggregate_type="round", aggregate_id="exp/k01/1"
        )
        self.assertEqual(
            [event.event_type for event in events],
            ["ROUND_CREATED", "PROMPT_COMMITTED"],
        )
        self.assertEqual(events[-1].payload["artifact"], "prompt.json")

    def test_invalid_and_stale_transitions_do_not_append_events(self) -> None:
        self._create_running_task()
        record = self.store.create_round("exp", "k01", 1, now=BASE_TIME)
        with self.assertRaises(InvalidTransitionError):
            self.store.transition_round(
                "exp", "k01", 1, RoundState.MODEL_REQUEST_SENT
            )
        with self.assertRaises(ConcurrentUpdateError):
            self.store.transition_round(
                "exp",
                "k01",
                1,
                RoundState.PROMPT_COMMITTED,
                expected_version=record.version + 1,
            )
        events = self.store.events.read(
            aggregate_type="round", aggregate_id="exp/k01/1"
        )
        self.assertEqual(len(events), 1)

    def test_second_round_requires_previous_round_to_finish(self) -> None:
        self._create_running_task()
        self.store.create_round("exp", "k01", 1, now=BASE_TIME)
        with self.assertRaises(InvalidTransitionError):
            self.store.create_round("exp", "k01", 2, now=BASE_TIME)

    def test_best_candidate_is_history_best_not_latest(self) -> None:
        self._create_running_task()
        self.store.create_round("exp", "k01", 1, now=BASE_TIME)
        digest_a = hashlib.sha256(b"a").hexdigest()
        self.store.register_candidate(
            candidate_id="a",
            experiment_id="exp",
            task_id="k01",
            round_number=1,
            source_sha256=digest_a,
            source_artifact_path="round_01/a.py",
            now=BASE_TIME,
        )
        self.store.save_candidate_score(
            CandidateScore(
                candidate_id="a",
                round_number=1,
                compile_passed=True,
                correctness_passed=True,
                anti_bypass_passed=True,
                minimum_speedup=1.2,
                geomean_speedup=1.3,
                stability_cv=0.02,
            ),
            now=BASE_TIME,
        )
        best = self.store.select_and_set_best_candidate(
            "exp", "k01", now=BASE_TIME
        )
        self.assertIsNotNone(best)
        self.assertEqual(best.candidate_id, "a")
        self.assertEqual(self.store.get_task("exp", "k01").best_candidate_id, "a")

    def test_event_rows_are_immutable(self) -> None:
        self.store.events.append(
            event_type="TEST",
            aggregate_type="test",
            aggregate_id="one",
            payload={},
            occurred_at=BASE_TIME,
        )
        with (
            self.store.database.connection() as connection,
            self.assertRaises(sqlite3.DatabaseError),
        ):
            connection.execute("UPDATE events SET event_type = 'CHANGED'")

    def test_explicit_event_id_is_idempotent_without_reusing_timestamp(self) -> None:
        first = self.store.events.append(
            event_id="stable-id",
            event_type="TEST",
            aggregate_type="test",
            aggregate_id="one",
            payload={"result": "ok"},
        )
        second = self.store.events.append(
            event_id="stable-id",
            event_type="TEST",
            aggregate_type="test",
            aggregate_id="one",
            payload={"result": "ok"},
        )
        self.assertEqual(first, second)


class EvaluationJobQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = SQLiteStateStore(
            Path(self.temporary.name) / "metadata.db"
        )
        self.queue = self.store.jobs

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    @staticmethod
    def _job(
        job_id: str,
        *,
        priority: int = 0,
        max_attempts: int = 3,
        idempotency_key: str | None = None,
    ) -> EvaluationJob:
        return EvaluationJob(
            job_id=job_id,
            experiment_id="exp",
            task_id="k01",
            round_number=1,
            candidate_id="candidate",
            stage=EvaluationStage.FULL_EVALUATION,
            payload={"candidate_path": "candidate.py"},
            priority=priority,
            max_attempts=max_attempts,
            available_at=BASE_TIME,
            idempotency_key=idempotency_key,
        )

    def test_claim_orders_by_priority_and_is_exclusive(self) -> None:
        self.queue.enqueue(self._job("low", priority=1))
        self.queue.enqueue(self._job("high", priority=10))
        first = self.queue.claim(
            "worker-a", lease_seconds=30, now=BASE_TIME
        )
        second = self.queue.claim(
            "worker-b", lease_seconds=30, now=BASE_TIME
        )
        self.assertEqual(first.job_id, "high")
        self.assertEqual(second.job_id, "low")
        self.assertNotEqual(first.lease_token, second.lease_token)
        self.assertIsNone(
            self.queue.claim("worker-c", lease_seconds=30, now=BASE_TIME)
        )

    def test_heartbeat_and_completion_require_live_token(self) -> None:
        self.queue.enqueue(self._job("job"))
        leased = self.queue.claim(
            "worker", lease_seconds=10, now=BASE_TIME
        )
        renewed = self.queue.heartbeat(
            "job",
            "worker",
            leased.lease_token,
            lease_seconds=20,
            now=BASE_TIME + timedelta(seconds=5),
        )
        self.assertEqual(
            renewed.lease_expires_at, BASE_TIME + timedelta(seconds=25)
        )
        with self.assertRaises(LeaseLostError):
            self.queue.complete(
                "job",
                "worker",
                "stale-token",
                {"ok": True},
                now=BASE_TIME + timedelta(seconds=6),
            )
        completed = self.queue.complete(
            "job",
            "worker",
            renewed.lease_token,
            {"ok": True},
            now=BASE_TIME + timedelta(seconds=6),
        )
        self.assertEqual(completed.status, JobStatus.SUCCEEDED)
        self.assertEqual(completed.result, {"ok": True})

    def test_failure_retries_then_dead_letters(self) -> None:
        self.queue.enqueue(self._job("job", max_attempts=2))
        first = self.queue.claim("worker", lease_seconds=10, now=BASE_TIME)
        retry = self.queue.fail(
            "job",
            "worker",
            first.lease_token,
            {"type": "runtime"},
            retry_delay_seconds=5,
            now=BASE_TIME + timedelta(seconds=1),
        )
        self.assertEqual(retry.status, JobStatus.RETRY_WAIT)
        self.assertIsNone(
            self.queue.claim(
                "worker", lease_seconds=10, now=BASE_TIME + timedelta(seconds=2)
            )
        )
        second = self.queue.claim(
            "worker", lease_seconds=10, now=BASE_TIME + timedelta(seconds=6)
        )
        dead = self.queue.fail(
            "job",
            "worker",
            second.lease_token,
            {"type": "runtime"},
            now=BASE_TIME + timedelta(seconds=7),
        )
        self.assertEqual(dead.status, JobStatus.DEAD)
        self.assertEqual(dead.attempt_count, 2)

    def test_expired_lease_is_requeued_and_stale_worker_is_rejected(self) -> None:
        self.queue.enqueue(self._job("job", max_attempts=2))
        old = self.queue.claim("old", lease_seconds=5, now=BASE_TIME)
        new = self.queue.claim(
            "new", lease_seconds=5, now=BASE_TIME + timedelta(seconds=6)
        )
        self.assertIsNotNone(new)
        self.assertEqual(new.attempt_count, 2)
        with self.assertRaises(LeaseLostError):
            self.queue.complete(
                "job",
                "old",
                old.lease_token,
                {"stale": True},
                now=BASE_TIME + timedelta(seconds=7),
            )

    def test_idempotency_replay_and_conflict(self) -> None:
        job = self._job("job", idempotency_key="same-request")
        first = self.queue.enqueue(job)
        second = self.queue.enqueue(job)
        self.assertEqual(first.job_id, second.job_id)
        with self.assertRaises(IdempotencyConflictError):
            self.queue.enqueue(
                EvaluationJob(
                    job_id="different-id",
                    experiment_id="exp",
                    task_id="k01",
                    round_number=1,
                    candidate_id="candidate",
                    stage=EvaluationStage.PROFILE,
                    payload={},
                    available_at=BASE_TIME,
                    idempotency_key="same-request",
                )
            )


class AtomicArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.artifacts = AtomicArtifactStore(Path(self.temporary.name) / "runs")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_atomic_write_digest_and_verification(self) -> None:
        metadata = self.artifacts.put_text(
            "exp/tasks/k01/round_01/candidate.py", "print('ok')\n"
        )
        self.assertEqual(
            metadata.sha256, hashlib.sha256(b"print('ok')\n").hexdigest()
        )
        self.assertTrue(self.artifacts.verify(metadata))
        self.assertEqual(
            self.artifacts.read_bytes(metadata), b"print('ok')\n"
        )
        leftovers = list(
            self.artifacts.path_for(metadata.relative_path).parent.glob("*.tmp")
        )
        self.assertEqual(leftovers, [])

    def test_shared_modes_survive_restrictive_process_umask_and_atomic_replace(self) -> None:
        previous = os.umask(0o077)
        try:
            metadata = self.artifacts.put_bytes("exp/job/result.bin", b"first")
            replaced = self.artifacts.put_bytes(
                "exp/job/result.bin",
                b"second",
                overwrite=True,
            )
        finally:
            os.umask(previous)

        self.assertNotEqual(metadata.sha256, replaced.sha256)
        for directory in (
            self.artifacts.root,
            self.artifacts.root / "exp",
            self.artifacts.root / "exp" / "job",
        ):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o2770)
        target = self.artifacts.root / "exp" / "job" / "result.bin"
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o660)
        self.assertEqual(target.read_bytes(), b"second")

    def test_existing_artifact_modes_are_repaired_and_other_stays_closed(self) -> None:
        legacy_root = self.artifacts.root / "legacy"
        legacy_root.mkdir(mode=0o700)
        target = legacy_root / "existing.bin"
        target.write_bytes(b"same")
        os.chmod(legacy_root, 0o700)
        os.chmod(target, 0o600)

        repaired = AtomicArtifactStore(legacy_root)
        metadata = repaired.put_bytes("existing.bin", b"same")

        self.assertTrue(repaired.verify(metadata))
        self.assertEqual(stat.S_IMODE(repaired.root.stat().st_mode), 0o2770)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o660)

    def test_artifact_store_rejects_modes_that_expand_access_to_other(self) -> None:
        with self.assertRaisesRegex(ValueError, "0660"):
            AtomicArtifactStore(
                Path(self.temporary.name) / "bad-file-mode",
                file_mode=0o666,
            )
        with self.assertRaisesRegex(ValueError, "02770"):
            AtomicArtifactStore(
                Path(self.temporary.name) / "bad-directory-mode",
                directory_mode=0o2777,
            )

    def test_same_content_is_idempotent_and_different_content_conflicts(self) -> None:
        first = self.artifacts.put_bytes("result.json", b"one")
        second = self.artifacts.put_bytes("result.json", b"one")
        self.assertEqual(first.sha256, second.sha256)
        with self.assertRaises(ArtifactConflictError):
            self.artifacts.put_bytes("result.json", b"two")
        replaced = self.artifacts.put_bytes(
            "result.json", b"two", overwrite=True
        )
        self.assertNotEqual(first.sha256, replaced.sha256)

    def test_path_traversal_is_rejected(self) -> None:
        for path in ("../secret", "/absolute", "a/../../secret", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.artifacts.put_bytes(path, b"bad")

    def test_intermediate_symlink_is_rejected_before_external_mkdir(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        link = self.artifacts.root / "linked"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.artifacts.put_bytes("linked/new/secret.bin", b"bad")
        self.assertFalse((outside / "new").exists())

    def test_corruption_is_detected(self) -> None:
        metadata = self.artifacts.put_bytes("artifact.bin", b"original")
        self.artifacts.path_for("artifact.bin").write_bytes(b"tampered")
        with self.assertRaises(ArtifactIntegrityError):
            self.artifacts.verify(metadata)

    def test_json_is_canonical_and_content_addressed_is_stable(self) -> None:
        metadata = self.artifacts.put_json("result.json", {"z": 1, "a": 2})
        self.assertEqual(
            self.artifacts.read_bytes(metadata), b'{"a":2,"z":1}\n'
        )
        first = self.artifacts.put_content_addressed(b"payload", suffix=".bin")
        second = self.artifacts.put_content_addressed(b"payload", suffix=".bin")
        self.assertEqual(first.relative_path, second.relative_path)


if __name__ == "__main__":
    unittest.main()
