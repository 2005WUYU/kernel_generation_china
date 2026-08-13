"""Offline integrity and lifecycle verification for a completed experiment run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ascend_kernel_lab.evidence import EvidenceIntegrityError, validate_artifact_map


@dataclass(frozen=True)
class VerificationIssue:
    severity: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


_ROUND_DIRECTORY = re.compile(r"round_(\d+)")
_TRAJECTORY_PHASES = {"repair", "optimization", "optimization_repair"}


def _round_directories(task_root: Path) -> list[Path]:
    if not task_root.is_dir():
        return []
    values: list[tuple[int, Path]] = []
    for path in task_root.iterdir():
        match = _ROUND_DIRECTORY.fullmatch(path.name)
        if match is not None and path.is_dir() and not path.is_symlink():
            values.append((int(match.group(1)), path))
    return [path for _round_number, path in sorted(values)]


def _experiment_round_directories(root: Path) -> list[Path]:
    tasks_root = root / "tasks"
    if not tasks_root.is_dir() or tasks_root.is_symlink():
        return []
    rounds: list[Path] = []
    for task_root in sorted(tasks_root.iterdir()):
        if task_root.is_dir() and not task_root.is_symlink():
            rounds.extend(_round_directories(task_root))
    return rounds


def _trajectory_phase_counts(
    rounds: list[Path],
) -> dict[str, int] | None:
    counts = {phase: 0 for phase in _TRAJECTORY_PHASES}
    for round_root in rounds:
        evaluation = _json(round_root / "evaluation_result.json")
        phase = evaluation.get("trajectory_phase") if evaluation is not None else None
        if phase not in _TRAJECTORY_PHASES:
            return None
        counts[str(phase)] += 1
    return counts


def _maximum_physical_rounds(
    optimization_rounds: int, maximum_repair_rounds: int
) -> int:
    del maximum_repair_rounds
    return optimization_rounds


class RunVerifier:
    """Verify committed artifacts without importing or executing candidate source."""

    _CREDENTIAL_VALUE = re.compile(
        r'(?i)"(?:api_key|auth_token|authorization|password|secret)"\s*:\s*"(?!<redacted>)'
    )
    _TOKEN_VALUE = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}|\bsk-[a-z0-9_-]{12,}")

    def __init__(self, experiment_root: Path | str, *, database_path: Path | str | None = None) -> None:
        # Preserve the lexical final component until it has been lstat-checked.
        # Resolving here would turn a caller-supplied symlink into its target and
        # make the formal-run symlink policy impossible to enforce.
        self.root = Path(os.path.abspath(Path(experiment_root).expanduser()))
        self.database_path = (
            Path(os.path.abspath(Path(database_path).expanduser()))
            if database_path is not None
            else None
        )
        self.issues: list[VerificationIssue] = []
        self.files_verified = 0
        self.bytes_verified = 0

    def _add(self, severity: str, code: str, message: str, path: Path | None = None) -> None:
        display: str | None = None
        if path is not None:
            try:
                display = str(path.resolve().relative_to(self.root.parent))
            except ValueError:
                display = str(path)
        self.issues.append(VerificationIssue(severity, code, message, display))

    def _check_formal_file(self, path: Path, label: str) -> bool:
        if path.is_symlink() or not path.is_file():
            self._add("error", "missing_or_unsafe_artifact", f"{label} is missing or not a regular file", path)
            return False
        return True

    def _verify_registered_artifacts(self) -> None:
        if self.database_path is None:
            self._add(
                "error",
                "database_not_supplied",
                "completed-run verification requires the durable SQLite manifest",
            )
            return
        if not self.database_path.is_file() or self.database_path.is_symlink():
            self._add("error", "database_missing", "metadata database is missing or unsafe", self.database_path)
            return
        try:
            connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro", uri=True, timeout=10
            )
            connection.row_factory = sqlite3.Row
            try:
                integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
                rows = connection.execute(
                    """
                    SELECT relative_path, sha256, size_bytes
                    FROM artifacts WHERE experiment_id = ? ORDER BY relative_path
                    """,
                    (self.root.name,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            self._add("error", "database_read_failed", f"cannot inspect metadata database: {exc}")
            return
        if integrity != "ok":
            self._add("error", "database_integrity", f"SQLite quick_check returned {integrity!r}")
        if not rows:
            self._add("error", "empty_artifact_manifest", "no artifacts are registered for this experiment")
            return
        artifact_root = self.root.parent.resolve()
        for row in rows:
            relative = str(row["relative_path"])
            candidate = artifact_root.joinpath(*Path(relative).parts)
            try:
                candidate.resolve().relative_to(artifact_root)
            except ValueError:
                self._add("error", "artifact_path_escape", "registered artifact escapes its root", candidate)
                continue
            if candidate.is_symlink() or not candidate.is_file():
                self._add("error", "artifact_missing", "registered artifact is missing or unsafe", candidate)
                continue
            digest, size = _sha256(candidate)
            self.files_verified += 1
            self.bytes_verified += size
            if digest != str(row["sha256"]) or size != int(row["size_bytes"]):
                self._add("error", "artifact_hash_mismatch", "artifact hash or size differs from SQLite manifest", candidate)

    def _verify_event_projection(self) -> None:
        path = self.root / "events.jsonl"
        if not self._check_formal_file(path, "event projection"):
            return
        previous = 0
        ids: set[str] = set()
        projected: list[tuple[int, str]] = []
        try:
            for _line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event is not an object")
                sequence = int(value["sequence"])
                event_id = str(value["event_id"])
                if sequence <= previous or event_id in ids:
                    raise ValueError("event sequence or ID is not strictly unique")
                previous = sequence
                ids.add(event_id)
                projected.append((sequence, event_id))
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._add("error", "event_projection_invalid", f"invalid events.jsonl: {exc}", path)
            return
        if self.database_path is not None and self.database_path.is_file():
            try:
                connection = sqlite3.connect(
                    f"file:{self.database_path}?mode=ro", uri=True, timeout=10
                )
                try:
                    durable = [
                        (int(row[0]), str(row[1]))
                        for row in connection.execute(
                            """
                            SELECT sequence, event_id FROM events
                            WHERE experiment_id = ? ORDER BY sequence
                            """,
                            (self.root.name,),
                        ).fetchall()
                    ]
                finally:
                    connection.close()
            except sqlite3.Error as exc:
                self._add(
                    "error",
                    "event_database_read_failed",
                    f"cannot compare durable events: {exc}",
                    self.database_path,
                )
            else:
                if projected != durable:
                    self._add(
                        "error",
                        "event_projection_mismatch",
                        "events.jsonl does not exactly match durable SQLite events",
                        path,
                    )

    def _verify_database_lifecycle(self, experiment: Mapping[str, Any]) -> None:
        if self.database_path is None or not self.database_path.is_file():
            return
        try:
            connection = sqlite3.connect(
                f"file:{self.database_path}?mode=ro", uri=True, timeout=10
            )
            connection.row_factory = sqlite3.Row
            try:
                experiment_row = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (self.root.name,),
                ).fetchone()
                task_rows = connection.execute(
                    "SELECT * FROM tasks WHERE experiment_id = ? ORDER BY task_id",
                    (self.root.name,),
                ).fetchall()
                round_rows = connection.execute(
                    """
                    SELECT * FROM rounds WHERE experiment_id = ?
                    ORDER BY task_id, round_number
                    """,
                    (self.root.name,),
                ).fetchall()
                candidate_rows = connection.execute(
                    """
                    SELECT candidate.*, score.compile_passed,
                           score.correctness_passed,
                           score.hidden_correctness_passed
                    FROM candidates AS candidate
                    LEFT JOIN candidate_scores AS score USING (candidate_id)
                    WHERE candidate.experiment_id = ?
                    """,
                    (self.root.name,),
                ).fetchall()
                active_jobs = connection.execute(
                    """
                    SELECT status, COUNT(*) AS count FROM evaluation_jobs
                    WHERE experiment_id = ?
                      AND status IN ('QUEUED', 'LEASED', 'RETRY_WAIT')
                    GROUP BY status
                    """,
                    (self.root.name,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:
            self._add(
                "error",
                "lifecycle_database_read_failed",
                f"cannot inspect durable lifecycle: {exc}",
                self.database_path,
            )
            return

        manifest = experiment.get("experiment")
        if not isinstance(manifest, Mapping):
            self._add("error", "experiment_config_missing", "experiment manifest has no config")
            return
        if experiment_row is None:
            self._add("error", "experiment_state_missing", "experiment is absent from SQLite")
            return
        if str(experiment_row["state"]) != "FINISHED":
            self._add(
                "error",
                "experiment_not_finished",
                f"durable experiment state is {experiment_row['state']!r}",
            )
        try:
            durable_config = json.loads(str(experiment_row["config_json"]))
        except json.JSONDecodeError:
            durable_config = None
        if durable_config != dict(manifest):
            self._add(
                "error",
                "experiment_config_mismatch",
                "experiment.json config differs from durable SQLite config",
            )

        configured_tasks = manifest.get("tasks")
        expected_tasks = (
            [str(value) for value in configured_tasks]
            if isinstance(configured_tasks, list)
            else []
        )
        expected_rounds = manifest.get("rounds_per_task")
        rounds_per_task = (
            int(expected_rounds)
            if isinstance(expected_rounds, int) and expected_rounds > 0
            else None
        )
        maximum_repair_rounds = manifest.get("maximum_repair_rounds", 0)
        maximum_repair_rounds = (
            int(maximum_repair_rounds)
            if isinstance(maximum_repair_rounds, int)
            and maximum_repair_rounds >= 0
            else 0
        )
        tasks = {str(row["task_id"]): row for row in task_rows}
        if sorted(tasks) != sorted(expected_tasks):
            self._add(
                "error",
                "durable_task_set_mismatch",
                "SQLite task set differs from the configured task set",
            )
        rounds_by_task: dict[str, list[sqlite3.Row]] = {}
        for row in round_rows:
            rounds_by_task.setdefault(str(row["task_id"]), []).append(row)
        candidates = {str(row["candidate_id"]): row for row in candidate_rows}
        for task_id in expected_tasks:
            row = tasks.get(task_id)
            if row is None:
                continue
            state = str(row["state"])
            if state not in {"TASK_FINISHED", "TASK_FAILED"}:
                self._add(
                    "error",
                    "task_not_terminal",
                    f"task {task_id} has durable state {state!r}",
                )
            durable_rounds = rounds_by_task.get(task_id, [])
            final_path = self.root / "tasks" / task_id / "final_result.json"
            final = _json(final_path)
            actual_rounds = len(durable_rounds)
            repair_actual = (
                int(final.get("repair_rounds", 0))
                if isinstance(final, Mapping)
                else None
            )
            optimization_actual = (
                int(final.get("optimization_rounds", 0))
                if isinstance(final, Mapping)
                else None
            )
            final_status = str(final.get("status", "")) if isinstance(final, Mapping) else ""
            termination_reason = (
                str(final.get("termination_reason", ""))
                if isinstance(final, Mapping)
                else ""
            )
            fixed_protocol = maximum_repair_rounds == 0
            artifact_rounds = _round_directories(
                self.root / "tasks" / task_id
            )
            phase_counts = _trajectory_phase_counts(artifact_rounds)
            maximum_physical_rounds = (
                _maximum_physical_rounds(
                    rounds_per_task, maximum_repair_rounds
                )
                if rounds_per_task is not None
                else None
            )
            phase_count_mismatch = bool(
                not fixed_protocol
                and phase_counts is not None
                and (
                    repair_actual != phase_counts["repair"]
                    or optimization_actual != phase_counts["optimization"]
                    or phase_counts["optimization_repair"]
                    > phase_counts["optimization"]
                    * maximum_repair_rounds
                    or sum(phase_counts.values()) != actual_rounds
                )
            )
            invalid_round_count = (
                rounds_per_task is not None
                and (
                    int(row["current_round"]) != actual_rounds
                    or len(artifact_rounds) != actual_rounds
                    or (
                        fixed_protocol
                        and actual_rounds != rounds_per_task
                    )
                    or (
                        not fixed_protocol
                        and (
                            actual_rounds < 1
                            or maximum_physical_rounds is None
                            or actual_rounds > maximum_physical_rounds
                            or phase_count_mismatch
                            or repair_actual is None
                            or repair_actual < 1
                            or repair_actual > maximum_repair_rounds
                            or optimization_actual is None
                            or optimization_actual < 0
                            or optimization_actual > rounds_per_task
                            or (
                                final_status == "repair_exhausted"
                                and (
                                    repair_actual != maximum_repair_rounds
                                    or optimization_actual != 0
                                )
                            )
                            or (
                                final_status != "repair_exhausted"
                                and termination_reason
                                == "optimization_round_budget_completed"
                                and optimization_actual != rounds_per_task
                            )
                            or (
                                final_status != "repair_exhausted"
                                and optimization_actual < rounds_per_task
                                and termination_reason != "host_dispatch_limited"
                            )
                        )
                    )
                )
            )
            if invalid_round_count:
                self._add(
                    "error",
                    "durable_round_count_mismatch",
                    f"task {task_id} has an invalid dynamic repair/optimization round count",
                )
            if any(str(round_row["state"]) != "ROUND_FINISHED" for round_row in durable_rounds):
                self._add(
                    "error",
                    "round_not_finished",
                    f"task {task_id} contains a non-terminal durable round",
                )
            if final is None:
                continue
            final_passed = str(final.get("status", "")).lower().startswith("passed")
            if final_passed != (state == "TASK_FINISHED"):
                self._add(
                    "error",
                    "task_final_state_mismatch",
                    f"task {task_id} final_result disagrees with SQLite state",
                    final_path,
                )
            best_id = row["best_candidate_id"]
            if final.get("best_candidate_id") != best_id:
                self._add(
                    "error",
                    "best_candidate_id_mismatch",
                    f"task {task_id} final_result disagrees with durable best candidate",
                    final_path,
                )
            if best_id is not None:
                candidate = candidates.get(str(best_id))
                if candidate is None:
                    self._add(
                        "error",
                        "best_candidate_missing",
                        f"task {task_id} durable best candidate row is absent",
                    )
                    continue
                if final.get("best_round") != int(candidate["round_number"]):
                    self._add(
                        "error",
                        "best_candidate_round_mismatch",
                        f"task {task_id} best round disagrees with candidate row",
                        final_path,
                    )
                source = self.root.parent.joinpath(
                    *Path(str(candidate["source_artifact_path"])).parts
                )
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or _sha256(source)[0] != str(candidate["source_sha256"])
                ):
                    self._add(
                        "error",
                        "candidate_source_identity_mismatch",
                        f"task {task_id} best source differs from its durable identity",
                        source,
                    )
                if final_passed and (
                    int(candidate["compile_passed"] or 0) != 1
                    or int(candidate["correctness_passed"] or 0) != 1
                    or int(candidate["hidden_correctness_passed"] or 0) != 1
                ):
                    self._add(
                        "error",
                        "best_candidate_gate_mismatch",
                        f"task {task_id} passed without all durable candidate gates",
                    )
        if active_jobs:
            counts = {str(row["status"]): int(row["count"]) for row in active_jobs}
            self._add(
                "error",
                "active_jobs_after_finish",
                f"finished experiment still has active jobs: {counts}",
            )

    def _verify_prompt_separation(self) -> None:
        sensitive_markers = ("hidden_seed", "secret_seed", "hidden_template", "hidden_cases")
        for pattern in ("prompt.json", "feedback.json"):
            for round_root in _experiment_round_directories(self.root):
                path = round_root / pattern
                if not path.is_file() or path.is_symlink():
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    continue
                lowered = text.lower()
                marker = next((item for item in sensitive_markers if item in lowered), None)
                if marker:
                    self._add("error", "hidden_data_in_model_context", f"model-visible artifact contains {marker!r}", path)
                if self._CREDENTIAL_VALUE.search(text) or self._TOKEN_VALUE.search(text):
                    self._add("error", "credential_in_model_context", "model-visible artifact appears to contain a credential", path)

    def _verify_public_execution_evidence(self) -> None:
        """Re-hash every worker-published public profiler/IR/log evidence tree."""

        environment = _json(self.root / "environment_snapshot.json") or {}
        fake_run = str(environment.get("schema_version", "")).startswith("ascend_fake_")
        artifact_root = self.root.parent
        for round_root in _experiment_round_directories(self.root):
            evaluation_path = round_root / "evaluation_result.json"
            if not evaluation_path.is_file() or evaluation_path.is_symlink():
                continue
            evaluation = _json(evaluation_path)
            if evaluation is None:
                continue
            for stage_name in ("compile", "correctness", "benchmark", "profile"):
                stage = evaluation.get(stage_name)
                if not isinstance(stage, Mapping):
                    continue
                artifacts = stage.get("artifacts")
                artifact_map = artifacts if isinstance(artifacts, Mapping) else {}
                try:
                    summary = validate_artifact_map(
                        artifact_map,
                        artifact_root=artifact_root,
                    )
                except (EvidenceIntegrityError, OSError) as exc:
                    self._add(
                        "error",
                        "execution_evidence_integrity",
                        f"{stage_name} evidence failed verification: {exc}",
                        evaluation_path,
                    )
                    continue
                passed = bool(stage.get("passed")) or str(
                    stage.get("status", "")
                ).lower() == "pass"
                if passed and summary is None and not fake_run:
                    self._add(
                        "error",
                        "execution_evidence_missing",
                        f"passed {stage_name} stage has no content-addressed evidence manifest",
                        evaluation_path,
                    )
                elif summary is not None:
                    self.files_verified += summary.file_count + 1
                    self.bytes_verified += summary.total_bytes + summary.manifest_path.stat().st_size

    def _verify_tasks(self, experiment: dict[str, Any]) -> None:
        manifest = experiment.get("experiment")
        configured_tasks: list[str] = []
        expected_rounds: int | None = None
        maximum_repair_rounds = 0
        if isinstance(manifest, dict):
            tasks = manifest.get("tasks")
            if isinstance(tasks, list) and all(isinstance(item, str) for item in tasks):
                configured_tasks = list(tasks)
            rounds = manifest.get("rounds_per_task")
            if isinstance(rounds, int) and rounds > 0:
                expected_rounds = rounds
            repair_rounds = manifest.get("maximum_repair_rounds", 0)
            if isinstance(repair_rounds, int) and repair_rounds >= 0:
                maximum_repair_rounds = repair_rounds
        tasks_root = self.root / "tasks"
        if not tasks_root.is_dir() or tasks_root.is_symlink():
            self._add("error", "tasks_root_missing", "run has no safe tasks directory", tasks_root)
            return
        discovered = sorted(path.name for path in tasks_root.iterdir() if path.is_dir())
        unexpected = sorted(set(discovered) - set(configured_tasks)) if configured_tasks else []
        if unexpected:
            self._add("error", "unexpected_task_artifacts", f"unexpected task directories: {unexpected}", tasks_root)
        for task_id in configured_tasks or discovered:
            task_root = tasks_root / task_id
            if not task_root.is_dir() or task_root.is_symlink():
                self._add("error", "task_artifacts_missing", f"task {task_id} is missing", task_root)
                continue
            rounds = _round_directories(task_root)
            final = _json(task_root / "final_result.json")
            repair_actual = (
                int(final.get("repair_rounds", 0))
                if isinstance(final, Mapping)
                else None
            )
            optimization_actual = (
                int(final.get("optimization_rounds", 0))
                if isinstance(final, Mapping)
                else None
            )
            final_status = str(final.get("status", "")) if isinstance(final, Mapping) else ""
            termination_reason = (
                str(final.get("termination_reason", ""))
                if isinstance(final, Mapping)
                else ""
            )
            phase_counts = _trajectory_phase_counts(rounds)
            maximum_physical_rounds = (
                _maximum_physical_rounds(
                    expected_rounds, maximum_repair_rounds
                )
                if expected_rounds is not None
                else None
            )
            phase_count_mismatch = bool(
                maximum_repair_rounds > 0
                and phase_counts is not None
                and (
                    repair_actual != phase_counts["repair"]
                    or optimization_actual != phase_counts["optimization"]
                    or phase_counts["optimization_repair"]
                    > phase_counts["optimization"]
                    * maximum_repair_rounds
                    or sum(phase_counts.values()) != len(rounds)
                )
            )
            invalid_count = (
                expected_rounds is not None
                and (
                    (
                        maximum_repair_rounds == 0
                        and len(rounds) != expected_rounds
                    )
                    or (
                        maximum_repair_rounds > 0
                        and (
                            len(rounds) < 1
                            or maximum_physical_rounds is None
                            or len(rounds) > maximum_physical_rounds
                            or phase_count_mismatch
                            or repair_actual is None
                            or repair_actual < 1
                            or repair_actual > maximum_repair_rounds
                            or optimization_actual is None
                            or optimization_actual < 0
                            or optimization_actual > expected_rounds
                            or (
                                final_status == "repair_exhausted"
                                and (
                                    repair_actual != maximum_repair_rounds
                                    or optimization_actual != 0
                                )
                            )
                            or (
                                final_status != "repair_exhausted"
                                and termination_reason
                                == "optimization_round_budget_completed"
                                and optimization_actual != expected_rounds
                            )
                            or (
                                final_status != "repair_exhausted"
                                and optimization_actual < expected_rounds
                                and termination_reason != "host_dispatch_limited"
                            )
                        )
                    )
                )
            )
            if invalid_count:
                self._add(
                    "error",
                    "incomplete_round_set",
                    "task has an invalid repair/optimization round set",
                    task_root,
                )
            for round_root in rounds:
                for name in (
                    "prompt.json",
                    "model_response.json",
                    "candidate.py",
                    "evaluation_result.json",
                    "feedback.json",
                ):
                    self._check_formal_file(round_root / name, name)
            final_path = task_root / "final_result.json"
            if final_path.is_file() and not final_path.is_symlink():
                value = _json(final_path)
                if value is None:
                    self._add("error", "final_result_invalid", "final result is not a JSON object", final_path)
                elif value.get("best_round") is not None:
                    best_round = int(value["best_round"])
                    selected = task_root / f"round_{best_round:02d}" / "candidate.py"
                    best = task_root / "best_candidate.py"
                    if (
                        self._check_formal_file(selected, "selected candidate")
                        and self._check_formal_file(best, "best candidate")
                        and _sha256(selected)[0] != _sha256(best)[0]
                    ):
                        self._add("error", "best_candidate_mismatch", "best_candidate.py differs from the selected round", best)
            else:
                self._add(
                    "error",
                    "final_result_missing",
                    "completed task has no final_result.json",
                    final_path,
                )

    def _verify_filesystem_hygiene(self) -> None:
        for path in self.root.rglob("*"):
            if path.is_symlink():
                self._add("error", "symlink_in_run", "formal run trees must not contain symlinks", path)
            if path.is_file() and (path.name.endswith(".tmp") or path.name.startswith(".tmp")):
                self._add("warning", "temporary_artifact", "leftover temporary artifact", path)

    def verify(self) -> dict[str, Any]:
        self.issues.clear()
        self.files_verified = 0
        self.bytes_verified = 0
        if not self.root.is_dir() or self.root.is_symlink():
            self._add("error", "run_root_missing", "experiment root is missing or unsafe", self.root)
        else:
            experiment_path = self.root / "experiment.json"
            if self._check_formal_file(experiment_path, "experiment manifest"):
                experiment = _json(experiment_path)
                if experiment is None:
                    self._add("error", "experiment_manifest_invalid", "experiment.json is not a JSON object", experiment_path)
                else:
                    self._verify_tasks(experiment)
                    self._verify_database_lifecycle(experiment)
            self._verify_registered_artifacts()
            self._verify_event_projection()
            self._verify_prompt_separation()
            self._verify_public_execution_evidence()
            self._verify_filesystem_hygiene()
        errors = sum(issue.severity == "error" for issue in self.issues)
        warnings = sum(issue.severity == "warning" for issue in self.issues)
        return {
            "schema_version": "ascend_run_verification_v1",
            "captured_at_unix": time.time(),
            "experiment_id": self.root.name,
            "experiment_root": str(self.root),
            "database_path": str(self.database_path) if self.database_path else None,
            "passed": errors == 0,
            "error_count": errors,
            "warning_count": warnings,
            "files_verified": self.files_verified,
            "bytes_verified": self.bytes_verified,
            "issues": [issue.to_dict() for issue in self.issues],
            "process_id": os.getpid(),
        }
