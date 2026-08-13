from __future__ import annotations

import difflib
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ascend_kernel_lab.config import ExperimentConfig
from ascend_kernel_lab.domain import (
    CandidateScore,
    ExperimentState,
    RoundState,
    TaskState,
    compare_public_candidate,
    compute_reward,
)
from ascend_kernel_lab.evaluation.orchestrator import (
    EvaluationBackend,
    EvaluationRequest,
    evaluate_candidate,
)
from ascend_kernel_lab.llm import (
    ModelGateway,
    ModelRequest,
    ModelResponse,
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    PromptBuilder,
    complete_model_response,
    validate_model_response,
)
from ascend_kernel_lab.protocol import harness_digest
from ascend_kernel_lab.storage import AtomicArtifactStore, SQLiteStateStore
from ascend_kernel_lab.tasks import TaskRegistry, TaskSpec
from ascend_kernel_lab.tasks.runtime import hidden_cases_from_template, validate_hidden_seed

from .baseline import prompt_baseline_projection
from .feedback import build_feedback


class ControllerError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerError(f"cannot read committed artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"committed artifact {path} is not a JSON object")
    return value


def _git_commit(root: Path) -> str:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if process.returncode == 0:
            commit = process.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, capture_output=True,
                text=True, timeout=5, check=False,
            ).stdout.strip()
            return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


def _score_from_mapping(value: Mapping[str, Any]) -> CandidateScore:
    return CandidateScore(
        candidate_id=str(value["candidate_id"]),
        round_number=int(value.get("round_number", value.get("round", 0))),
        compile_passed=bool(value["compile_passed"]),
        correctness_passed=bool(value["correctness_passed"]),
        anti_bypass_passed=bool(value["anti_bypass_passed"]),
        hidden_correctness_passed=value.get("hidden_correctness_passed"),
        minimum_speedup=value.get("minimum_speedup"),
        geomean_speedup=value.get("geomean_speedup"),
        candidate_kernel_coverage=value.get("candidate_kernel_coverage"),
        stability_cv=value.get("stability_cv"),
    )


def _passed_stage(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(
        value.get("passed") or str(value.get("status", "")).lower() == "pass"
    )


def _finite_positive_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0 and math.isfinite(number)


def _final_gate_status(
    evaluation: Mapping[str, Any],
    *,
    profile_coverage_required: bool,
) -> tuple[str, bool]:
    correctness = evaluation.get("correctness")
    if not _passed_stage(correctness):
        return "failed_hidden_correctness", False
    benchmark = evaluation.get("benchmark")
    if not _passed_stage(benchmark):
        return "failed_final_benchmark", True
    assert isinstance(benchmark, Mapping)
    if (
        str(benchmark.get("status", "")).lower() not in {"stable", "pass"}
        or not _finite_positive_number(
            benchmark.get("geomean_speedup_vs_eager", benchmark.get("speedup_geomean"))
        )
        or not _finite_positive_number(
            benchmark.get("minimum_speedup_vs_eager", benchmark.get("minimum_speedup"))
        )
    ):
        return "failed_final_benchmark", True
    profile = evaluation.get("profile")
    anti_bypass = evaluation.get("anti_bypass")
    profile_verified = _passed_stage(profile) and _passed_stage(anti_bypass)
    if profile_coverage_required and not profile_verified:
        return "failed_final_profile", True
    if not profile_coverage_required and not profile_verified:
        return "passed_profile_unverified", True
    return "passed", True


@dataclass(frozen=True)
class TaskRunSummary:
    task_id: str
    status: str
    best_round: int | None
    best_candidate_id: str | None
    final_result: Mapping[str, Any]


class ExperimentController:
    """Owns model calls, round progression, selection, and final hidden gates."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        store: SQLiteStateStore,
        artifacts: AtomicArtifactStore,
        registry: TaskRegistry,
        model_gateway: ModelGateway,
        backend: EvaluationBackend,
        environment: Mapping[str, Any],
        baseline: Mapping[str, Any] | None = None,
        hidden_seed: int | None = None,
        prompt_builder: PromptBuilder | None = None,
        profile_coverage_required: bool = True,
        allow_insecure_hidden_seed_for_testing: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.artifacts = artifacts
        self.registry = registry
        self.model_gateway = model_gateway
        self.backend = backend
        self.environment = dict(environment)
        self.baseline = dict(baseline or {})
        self.hidden_seed = (
            validate_hidden_seed(
                hidden_seed,
                allow_insecure_for_testing=allow_insecure_hidden_seed_for_testing,
            )
            if hidden_seed is not None
            else None
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.profile_coverage_required = profile_coverage_required
        self.experiment_id = config.id
        self.harness_commit = _git_commit(config.project_root)

    @property
    def experiment_root(self) -> Path:
        return self.artifacts.root / self.experiment_id

    def _relative(self, task_id: str | None = None, round_number: int | None = None, name: str = "") -> str:
        parts = [self.experiment_id]
        if task_id is not None:
            parts.extend(("tasks", task_id))
        if round_number is not None:
            parts.append(f"round_{round_number:02d}")
        if name:
            parts.append(name)
        return "/".join(parts)

    def _commit_json(
        self,
        relative: str,
        value: Any,
        *,
        task_id: str | None = None,
        round_number: int | None = None,
        overwrite: bool = False,
    ) -> Path:
        metadata = self.artifacts.put_json(relative, value, overwrite=overwrite)
        self.store.register_artifact(
            metadata,
            experiment_id=self.experiment_id,
            task_id=task_id,
            round_number=round_number,
            extra_metadata={"harness_git_commit": self.harness_commit},
        )
        return self.artifacts.path_for(relative)

    def _commit_text(
        self,
        relative: str,
        value: str,
        *,
        task_id: str | None = None,
        round_number: int | None = None,
        media_type: str = "text/plain; charset=utf-8",
        overwrite: bool = False,
    ) -> tuple[Path, str]:
        metadata = self.artifacts.put_text(relative, value, media_type=media_type, overwrite=overwrite)
        self.store.register_artifact(
            metadata,
            experiment_id=self.experiment_id,
            task_id=task_id,
            round_number=round_number,
            extra_metadata={"harness_git_commit": self.harness_commit},
        )
        return self.artifacts.path_for(relative), metadata.sha256

    def _artifact_path(self, task_id: str | None = None, round_number: int | None = None, name: str = "") -> Path:
        return self.artifacts.path_for(self._relative(task_id, round_number, name))

    def _experiment_manifest(self) -> dict[str, Any]:
        """Freeze every result-affecting policy without persisting the secret seed."""

        manifest = self.config.to_manifest()
        suite_identity = None
        if self.hidden_seed is not None:
            suite_identity = hashlib.sha256(
                (
                    "ascend-kernel-lab:blind-suite-identity:v1\0"
                    f"{self.experiment_id}\0{self.hidden_seed}"
                ).encode()
            ).hexdigest()
        manifest["runtime_policy"] = {
            "schema_version": "ascend_runtime_policy_v1",
            "final_profile_gate": (
                "required" if self.profile_coverage_required else "advisory"
            ),
            "blind_suite_generator": "hidden-v1",
            "blind_suite_identity_sha256": suite_identity,
            "harness_git_commit": self.harness_commit,
            "harness_protocol_sha256": harness_digest(),
        }
        return manifest

    def _load_committed_json(
        self,
        relative: str,
        *,
        task_id: str | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Read, integrity-check, and repair an FS-first artifact registration."""

        path = self.artifacts.path_for(relative)
        value = _read_json(path)
        self._commit_json(
            relative,
            value,
            task_id=task_id,
            round_number=round_number,
        )
        return value

    def initialize(self) -> None:
        manifest = self._experiment_manifest()
        record = self.store.create_experiment(self.experiment_id, manifest)
        if record.state is ExperimentState.CREATED:
            self.store.transition_experiment(self.experiment_id, ExperimentState.RUNNING)
        elif record.state not in {ExperimentState.RUNNING, ExperimentState.FINISHED}:
            raise ControllerError(f"experiment is terminal: {record.state.value}")
        self._commit_json(
            self._relative(name="experiment.json"),
            {
                "schema_version": "ascend_experiment_v1",
                "experiment": manifest,
                "harness_git_commit": self.harness_commit,
            },
        )
        self._commit_json(self._relative(name="environment_snapshot.json"), self.environment)
        self._commit_json(self._relative(name="baseline_snapshot.json"), self.baseline)

    def run(self, task_ids: Sequence[str] | None = None) -> tuple[TaskRunSummary, ...]:
        self.initialize()
        selected = tuple(self.config.tasks if task_ids is None else task_ids)
        if len(set(selected)) != len(selected):
            raise ControllerError("task selection contains duplicate task IDs")
        unknown = set(selected) - set(self.config.tasks)
        if unknown:
            raise ControllerError(f"tasks are not enabled by the experiment config: {sorted(unknown)}")
        tasks = tuple(self.registry.load(task_id) for task_id in selected)
        if len(tasks) <= 1 or self.config.task_concurrency == 1:
            summaries = tuple(self.run_task(task) for task in tasks)
        else:
            # Tasks are independent durable state machines.  Run distinct tasks
            # concurrently so the shared queue can keep multiple single-NPU
            # workers busy.  ``run_task`` itself deliberately keeps each task's
            # five rounds strictly sequential because round N+1 consumes the
            # committed feedback from round N.
            with ThreadPoolExecutor(
                max_workers=min(self.config.task_concurrency, len(tasks)),
                thread_name_prefix="akg-task",
            ) as executor:
                summaries = tuple(executor.map(self.run_task, tasks))
        experiment = self.store.get_experiment(self.experiment_id)
        all_tasks_terminal = all(
            (record := self.store.get_task(self.experiment_id, task_id)) is not None
            and record.state in {TaskState.TASK_FINISHED, TaskState.TASK_FAILED}
            for task_id in self.config.tasks
        )
        if (
            experiment is not None
            and experiment.state is ExperimentState.RUNNING
            and all_tasks_terminal
        ):
            self.store.transition_experiment(self.experiment_id, ExperimentState.FINISHED)
        self.materialize_events()
        return summaries

    def run_task(self, task: TaskSpec) -> TaskRunSummary:
        task_record = self.store.create_task(
            self.experiment_id,
            task.id,
            task_version=task.version,
            task_spec_sha256=task.digest(),
        )
        task_snapshot = {**task.public_prompt_view(), "task_spec_sha256": task.digest()}
        self._commit_json(
            self._relative(task.id, name="task_snapshot.json"),
            task_snapshot,
            task_id=task.id,
        )
        if task_record.state is TaskState.TASK_CREATED:
            task_record = self.store.transition_task(self.experiment_id, task.id, TaskState.ROUNDS_RUNNING)
        if task_record.state in {TaskState.TASK_FINISHED, TaskState.TASK_FAILED}:
            return self._run_final(task)

        # ``maximum_repair_rounds == 0`` preserves the original fixed-round
        # protocol for existing configurations.  A positive repair budget
        # changes the semantics deliberately: repair rounds continue only
        # until the first source/compile/correctness-passing seed, then five
        # (or the configured count of) optimization rounds start at index 1.
        # Round numbers remain consecutive, so crash recovery and queue
        # idempotency do not need placeholder/skipped rounds.
        if self.config.maximum_repair_rounds == 0:
            for round_number in range(1, self.config.rounds_per_task + 1):
                self._run_round(
                    task,
                    round_number,
                    phase="optimization",
                    phase_index=round_number,
                )
            return self._run_final(task)

        round_number = 1
        repair_count = 0
        seed_round: int | None = None
        while repair_count < self.config.maximum_repair_rounds:
            self._run_round(
                task,
                round_number,
                phase="repair",
                phase_index=repair_count + 1,
            )
            repair_count += 1
            evaluation = self._round_evaluation(task, round_number)
            if self._repair_seed_ready(evaluation):
                seed_round = round_number
                break
            round_number += 1

        if seed_round is None:
            return self._finish_repair_exhausted(task, repair_count)

        optimization_count = 0
        round_number = seed_round + 1
        while optimization_count < self.config.rounds_per_task:
            # A completed repair seed or optimization round may prove that the
            # immutable launch path dominates end-to-end latency.  In that
            # case another Kernel-body model call would add no useful signal.
            if self._optimization_stop_recommended(
                task, round_number - 1
            ):
                break
            self._run_round(
                task,
                round_number,
                phase="optimization",
                phase_index=optimization_count + 1,
                repair_attempt=0,
            )
            optimization_count += 1
            evaluation = self._round_evaluation(task, round_number)
            repair_attempt = 0
            while (
                not self._repair_seed_ready(evaluation)
                and repair_attempt < self.config.maximum_repair_rounds
            ):
                repair_attempt += 1
                round_number += 1
                self._run_round(
                    task,
                    round_number,
                    phase="optimization_repair",
                    phase_index=optimization_count,
                    repair_attempt=repair_attempt,
                )
                evaluation = self._round_evaluation(task, round_number)
            round_number += 1
        return self._run_final(task)

    def _trajectory_counts(self, task: TaskSpec) -> tuple[int, int]:
        repair = 0
        optimization = 0
        task_record = self.store.get_task(self.experiment_id, task.id)
        maximum = task_record.current_round if task_record is not None else 0
        for round_number in range(1, maximum + 1):
            evaluation_path = self._artifact_path(
                task.id, round_number, "evaluation_result.json"
            )
            if not evaluation_path.is_file():
                continue
            phase = _read_json(evaluation_path).get("trajectory_phase")
            if phase == "repair":
                repair += 1
            elif phase == "optimization":
                optimization += 1
        return repair, optimization

    def _trajectory_termination_reason(self, task: TaskSpec) -> str:
        _repair_rounds, optimization_rounds = self._trajectory_counts(task)
        if optimization_rounds >= self.config.rounds_per_task:
            return "optimization_round_budget_completed"
        task_record = self.store.get_task(self.experiment_id, task.id)
        if task_record is None or task_record.current_round < 1:
            return "optimization_round_budget_completed"
        if self._optimization_stop_recommended(task, task_record.current_round):
            return "host_dispatch_limited"
        return "optimization_round_budget_completed"

    def _round_evaluation(
        self, task: TaskSpec, round_number: int
    ) -> dict[str, Any]:
        path = self._artifact_path(task.id, round_number, "evaluation_result.json")
        if not path.is_file():
            raise ControllerError(
                f"finished round {round_number} is missing evaluation_result.json"
            )
        return _read_json(path)

    @staticmethod
    def _repair_seed_ready(evaluation: Mapping[str, Any]) -> bool:
        return all(
            _passed_stage(evaluation.get(stage))
            for stage in ("source", "compile", "correctness")
        )

    def _optimization_stop_recommended(
        self, task: TaskSpec, round_number: int
    ) -> bool:
        evaluation = self._round_evaluation(task, round_number)
        feedback_path = self._artifact_path(task.id, round_number, "feedback.json")
        feedback = _read_json(feedback_path) if feedback_path.is_file() else {}
        if feedback.get("stop_recommended") is not True:
            return False
        # A host-bound observation may terminate optimization only when it is
        # the committed BEST.  A slower branch can be host-bound while the
        # incumbent still has useful device-side optimization headroom.
        best_after = feedback.get("best_after")
        if (
            not isinstance(best_after, Mapping)
            or best_after.get("round") != round_number
            or feedback.get("performance_decision")
            not in {"INITIAL_BEST", "NEW_BEST"}
        ):
            return False
        benchmark = evaluation.get("benchmark")
        if not isinstance(benchmark, Mapping):
            return False
        score = evaluation.get("score")
        if not isinstance(score, Mapping) or not all(
            score.get(key) is True
            for key in (
                "compile_passed",
                "correctness_passed",
                "anti_bypass_passed",
            )
        ):
            return False
        if (
            score.get("geomean_speedup") is None
            or score.get("minimum_speedup") is None
            or not _passed_stage(benchmark)
            or str(benchmark.get("status", "")).lower() not in {"pass", "stable"}
        ):
            return False
        bottleneck = benchmark.get("bottleneck")
        return bool(
            benchmark.get("host_dispatch_limited") is True
            or (
                isinstance(bottleneck, Mapping)
                and bottleneck.get("host_dispatch_limited") is True
            )
        )

    def _finish_repair_exhausted(
        self, task: TaskSpec, repair_count: int
    ) -> TaskRunSummary:
        final = {
            "schema_version": "ascend_final_result_v1",
            "experiment_id": self.experiment_id,
            "task_id": task.id,
            "status": "repair_exhausted",
            "best_round": None,
            "best_candidate_id": None,
            "repair_rounds": repair_count,
            "optimization_rounds": 0,
            "termination_reason": "repair_exhausted",
            "reason": "no source/compile/correctness-passing seed within repair budget",
        }
        self._commit_json(
            self._relative(task.id, name="final_result.json"),
            final,
            task_id=task.id,
        )
        task_record = self.store.get_task(self.experiment_id, task.id)
        if task_record is not None and task_record.state is TaskState.ROUNDS_RUNNING:
            self.store.transition_task(
                self.experiment_id, task.id, TaskState.TASK_FAILED
            )
        return TaskRunSummary(task.id, str(final["status"]), None, None, final)

    @staticmethod
    def _follow_up_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
        """Keep only fast, actionable evidence from the previous evaluation."""

        def stage_status(name: str) -> Any:
            value = evaluation.get(name)
            if not isinstance(value, Mapping):
                return None
            return {
                "status": value.get("status"),
                "passed": value.get("passed"),
            }

        correctness = evaluation.get("correctness")
        correctness_summary: dict[str, Any] | None = None
        if isinstance(correctness, Mapping):
            failed_cases: list[dict[str, Any]] = []
            case_results = correctness.get("case_results")
            diagnostic_fields = (
                "case_id",
                "error",
                "shape_ok",
                "dtype_ok",
                "device_ok",
                "layout_ok",
                "output_alias_ok",
                "finite_ok",
                "inputs_unchanged",
                "maximum_absolute_error",
                "maximum_relative_error",
                "actual_at_maximum_error",
                "expected_at_maximum_error",
                "maximum_error_flat_index",
            )
            if isinstance(case_results, Sequence) and not isinstance(
                case_results, (str, bytes)
            ):
                for case in case_results:
                    if not isinstance(case, Mapping) or bool(case.get("passed")):
                        continue
                    failed_cases.append(
                        {
                            field: case[field]
                            for field in diagnostic_fields
                            if field in case
                        }
                    )
                    if len(failed_cases) == 2:
                        break
            correctness_summary = {
                **(stage_status("correctness") or {}),
                "passed_cases": correctness.get("passed_cases"),
                "total_cases": correctness.get("total_cases"),
                "maximum_absolute_error": correctness.get(
                    "maximum_absolute_error"
                ),
                "maximum_relative_error": correctness.get(
                    "maximum_relative_error"
                ),
                "failed_cases": failed_cases,
            }

        benchmark = evaluation.get("benchmark")
        benchmark_summary: dict[str, Any] | None = None
        if isinstance(benchmark, Mapping):
            cases: list[dict[str, Any]] = []
            per_case = benchmark.get("per_case")
            if isinstance(per_case, Sequence) and not isinstance(per_case, (str, bytes)):
                for value in per_case:
                    if not isinstance(value, Mapping):
                        continue
                    candidate = value.get("candidate")
                    eager = value.get("baseline_eager")
                    cases.append({
                        "case_id": value.get("case_id"),
                        "candidate_us": (
                            candidate.get("median_us") if isinstance(candidate, Mapping) else None
                        ),
                        "pytorch_eager_us": (
                            eager.get("median_us") if isinstance(eager, Mapping) else None
                        ),
                        "speedup_vs_pytorch_eager": value.get("speedup_vs_eager"),
                        "stable": value.get("stable"),
                        "device_latency_us": value.get("device_latency_us"),
                        "end_to_end_latency_us": value.get(
                            "end_to_end_latency_us"
                        ),
                        "host_overhead_us": value.get("host_overhead_us"),
                        "bottleneck_type": value.get("bottleneck_type"),
                    })
            benchmark_summary = {
                **(stage_status("benchmark") or {}),
                "geomean_speedup_vs_pytorch_eager": benchmark.get(
                    "geomean_speedup_vs_eager"
                ),
                "minimum_speedup_vs_pytorch_eager": benchmark.get(
                    "minimum_speedup_vs_eager"
                ),
                "maximum_cv": benchmark.get("maximum_cv"),
                "bottleneck": benchmark.get("bottleneck"),
                "bottleneck_type": benchmark.get("bottleneck_type"),
                "host_dispatch_limited": benchmark.get(
                    "host_dispatch_limited"
                ),
                "cases": cases,
            }

        profile = evaluation.get("profile")
        profile_summary: dict[str, Any] | None = None
        if isinstance(profile, Mapping):
            summary_value = profile.get("summary")
            summary = summary_value if isinstance(summary_value, Mapping) else profile
            scheduling_value = summary.get("scheduling")
            scheduling = (
                scheduling_value if isinstance(scheduling_value, Mapping) else {}
            )
            pipeline_value = summary.get("pipeline")
            pipeline = pipeline_value if isinstance(pipeline_value, Mapping) else {}
            memory_value = summary.get("memory")
            memory = memory_value if isinstance(memory_value, Mapping) else {}
            profile_summary = {
                "status": profile.get("status"),
                "passed": profile.get("passed"),
                "mode": summary.get("profile_mode", profile.get("profile_mode")),
                "kernel_count": summary.get("kernel_count"),
                "candidate_kernel_coverage": summary.get("candidate_kernel_coverage"),
                "candidate_device_execution_us": scheduling.get(
                    "candidate_device_execution_us"
                ),
                "total_device_execution_us": scheduling.get(
                    "total_device_execution_us"
                ),
                "host_overhead_us": scheduling.get(
                    "host_overhead_us", scheduling.get("host_enqueue_us")
                ),
                "pipeline_utilization": {
                    str(key): value
                    for key, value in list(pipeline.items())[:8]
                },
                "memory": {
                    str(key): value for key, value in list(memory.items())[:4]
                },
                "observations": list(summary.get("observations", []))[:3]
                if isinstance(summary.get("observations"), Sequence)
                and not isinstance(summary.get("observations"), (str, bytes))
                else [],
            }

        return {
            "overall_status": evaluation.get("overall_status"),
            "source": stage_status("source"),
            "compile": stage_status("compile"),
            "correctness": correctness_summary,
            "benchmark_vs_pytorch_eager": benchmark_summary,
            "quick_profile": profile_summary,
        }

    @staticmethod
    def _follow_up_failure_reasons(
        evaluation: Mapping[str, Any], feedback: Mapping[str, Any]
    ) -> list[str]:
        overall = str(evaluation.get("overall_status", "unknown"))
        if overall in {"correct", "success"}:
            return []
        reasons = [overall]
        for stage_name in ("source", "compile", "correctness", "benchmark", "profile"):
            stage = evaluation.get(stage_name)
            if not isinstance(stage, Mapping):
                continue
            error = stage.get("error")
            if isinstance(error, Mapping) and error.get("message"):
                reasons.append(f"{stage_name}: {error['message']}")
            if stage_name == "source":
                findings = stage.get("findings")
                details = stage.get("details")
                if findings is None and isinstance(details, Mapping):
                    findings = details.get("findings")
                if isinstance(findings, Sequence) and not isinstance(
                    findings, (str, bytes)
                ):
                    for finding in findings[:8]:
                        if isinstance(finding, Mapping):
                            code = finding.get("code", "source_check")
                            message = finding.get("message", "rejected")
                            reasons.append(f"source[{code}]: {message}")
        if overall == "correctness_failed":
            correctness_metrics = ExperimentController._follow_up_metrics(
                evaluation
            ).get("correctness")
            if isinstance(correctness_metrics, Mapping):
                failed_cases = correctness_metrics.get("failed_cases")
                if isinstance(failed_cases, Sequence) and not isinstance(
                    failed_cases, (str, bytes)
                ):
                    summaries: list[str] = []
                    for case in failed_cases:
                        if not isinstance(case, Mapping):
                            continue
                        checks = [
                            name
                            for name in (
                                "shape_ok",
                                "dtype_ok",
                                "device_ok",
                                "layout_ok",
                                "output_alias_ok",
                                "finite_ok",
                                "inputs_unchanged",
                            )
                            if case.get(name) is False
                        ]
                        values = [f"case={case.get('case_id')}"]
                        if case.get("error") is not None:
                            values.append(f"error={case['error']}")
                        if checks:
                            values.append(f"failed_checks={','.join(checks)}")
                        for name in (
                            "maximum_absolute_error",
                            "maximum_relative_error",
                            "actual_at_maximum_error",
                            "expected_at_maximum_error",
                            "maximum_error_flat_index",
                        ):
                            if case.get(name) is not None:
                                values.append(f"{name}={case[name]}")
                        summaries.append("; ".join(values))
                    if summaries:
                        reasons.append(
                            "correctness diagnostics: " + " | ".join(summaries)
                        )
        focus = feedback.get("next_round_requirement")
        if len(reasons) == 1 and isinstance(focus, Mapping):
            suggestions = focus.get("focus")
            if isinstance(suggestions, Sequence) and not isinstance(suggestions, (str, bytes)):
                reasons.extend(str(item) for item in suggestions[:1])
        return reasons

    @staticmethod
    def _follow_up_suggestions(feedback: Mapping[str, Any]) -> list[str]:
        requirement = feedback.get("next_round_requirement")
        if not isinstance(requirement, Mapping):
            return ["保持正确性并依据 PyTorch eager 对比信息尝试下一次修改"]
        focus = requirement.get("focus")
        if not isinstance(focus, Sequence) or isinstance(focus, (str, bytes)):
            return ["保持正确性并依据 PyTorch eager 对比信息尝试下一次修改"]
        result = []
        for item in focus:
            text = str(item)
            result.append(text)
        return result

    def _best_context(
        self, task: TaskSpec, *, before_round: int | None = None
    ) -> dict[str, Any] | None:
        task_record = self.store.get_task(self.experiment_id, task.id)
        best = (
            self.store.get_candidate_score(task_record.best_candidate_id)
            if task_record is not None and task_record.best_candidate_id is not None
            else None
        )
        if best is not None and before_round is not None and best.round_number >= before_round:
            best = None
        if best is None:
            scores = sorted(
                (
                    score
                    for score in self.store.list_candidate_scores(
                        self.experiment_id, task.id
                    )
                    if before_round is None or score.round_number < before_round
                ),
                key=lambda score: score.round_number,
            )
            online_best: CandidateScore | None = None
            for score in scores:
                decision = compare_public_candidate(score, online_best)
                if decision.decision in {"INITIAL_BEST", "NEW_BEST"}:
                    online_best = score
            best = online_best
        if best is None:
            return None
        code_path = self._artifact_path(task.id, best.round_number, "candidate.py")
        evaluation_path = self._artifact_path(task.id, best.round_number, "evaluation_result.json")
        response_path = self._artifact_path(
            task.id, best.round_number, "model_response.json"
        )
        response = _read_json(response_path) if response_path.is_file() else {}
        return {
            "round": best.round_number,
            "candidate_id": best.candidate_id,
            "code": code_path.read_text(encoding="utf-8"),
            "geomean_speedup": best.geomean_speedup,
            "minimum_speedup": best.minimum_speedup,
            "candidate_kernel_coverage": best.candidate_kernel_coverage,
            "stability_cv": best.stability_cv,
            "score": asdict(best),
            "optimization_summary": list(
                response.get(
                    "optimization_summary", response.get("change_summary", [])
                )
            ),
            "evaluation": _read_json(evaluation_path) if evaluation_path.is_file() else {},
        }

    def _committed_online_best(
        self, task: TaskSpec
    ) -> tuple[CandidateScore | None, int]:
        """Replay committed online BEST decisions in trajectory order.

        ``feedback.json.best_after`` is the durable result of the online,
        noise-aware comparison for a round.  Older feedback artifacts do not
        have that field, so they are replayed with the same comparison policy.
        Candidate scores without committed feedback are deliberately ignored:
        they may belong to an evaluation-to-feedback crash window.
        """

        scores = sorted(
            self.store.list_candidate_scores(self.experiment_id, task.id),
            key=lambda score: (score.round_number, score.candidate_id),
        )
        scores_by_id = {score.candidate_id: score for score in scores}
        online_best: CandidateScore | None = None
        last_committed_round = 0
        for score in scores:
            feedback_path = self._artifact_path(
                task.id, score.round_number, "feedback.json"
            )
            if not feedback_path.is_file():
                continue
            feedback = _read_json(feedback_path)
            last_committed_round = max(last_committed_round, score.round_number)
            best_after = feedback.get("best_after")
            if isinstance(best_after, Mapping):
                candidate_id = best_after.get("candidate_id")
                selected = (
                    scores_by_id.get(str(candidate_id))
                    if candidate_id is not None
                    else None
                )
                if (
                    selected is None
                    or not selected.is_publicly_valid
                    or selected.round_number > score.round_number
                ):
                    raise ControllerError(
                        "committed feedback references an invalid online BEST"
                    )
                online_best = selected
                continue

            decision = str(feedback.get("performance_decision", ""))
            if decision in {"INITIAL_BEST", "NEW_BEST"}:
                if not score.is_publicly_valid:
                    raise ControllerError(
                        "committed feedback selects an ineligible online BEST"
                    )
                online_best = score
            elif decision in {"TIE", "REGRESSION", "INVALID"}:
                continue
            else:
                comparison = compare_public_candidate(score, online_best)
                if comparison.decision in {"INITIAL_BEST", "NEW_BEST"}:
                    online_best = score
        return online_best, last_committed_round

    def _reconcile_committed_online_best(
        self,
        task: TaskSpec,
        *,
        allow_inflight_candidate: bool,
    ) -> CandidateScore | None:
        """Make the DB pointer agree with the committed feedback trajectory."""

        expected, last_committed_round = self._committed_online_best(task)
        task_record = self.store.get_task(self.experiment_id, task.id)
        if task_record is None:
            raise ControllerError("cannot reconcile BEST for a missing task")
        if expected is None:
            return None
        if task_record.best_candidate_id == expected.candidate_id:
            return expected
        current = (
            self.store.get_candidate_score(task_record.best_candidate_id)
            if task_record.best_candidate_id is not None
            else None
        )
        if (
            allow_inflight_candidate
            and current is not None
            and current.round_number > last_committed_round
        ):
            return current
        self.store.set_best_candidate(
            self.experiment_id,
            task.id,
            expected.candidate_id,
            reason="RECONCILE_COMMITTED_FEEDBACK",
        )
        return expected

    def _prompt_candidate_context(
        self,
        task: TaskSpec,
        *,
        before_round: int,
        phase: str,
    ) -> dict[str, Any]:
        previous_evaluation_path = self._artifact_path(
            task.id, before_round - 1, "evaluation_result.json"
        )
        previous = (
            _read_json(previous_evaluation_path)
            if previous_evaluation_path.is_file()
            else {}
        )
        if phase in {"repair", "optimization_repair"} and previous.get(
            "overall_status"
        ) in {
            "source_failed",
            "compile_failed",
            "correctness_failed",
        }:
            candidate_path = self._artifact_path(
                task.id, before_round - 1, "candidate.py"
            )
            if not candidate_path.is_file():
                raise ControllerError(
                    f"round {before_round - 1} failed candidate is missing"
                )
            return {
                "round": before_round - 1,
                "role": "failed_candidate_under_repair",
                "code": candidate_path.read_text(encoding="utf-8"),
            }
        best = self._best_context(task, before_round=before_round)
        if best is not None:
            return {
                "round": best["round"],
                "role": "best_public_candidate",
                "code": best["code"],
            }
        scores = sorted(
            (
                score
                for score in self.store.list_candidate_scores(
                    self.experiment_id, task.id
                )
                if score.round_number < before_round
                and score.compile_passed
                and score.correctness_passed
            ),
            key=lambda score: score.round_number,
            reverse=True,
        )
        candidate_round = scores[0].round_number if scores else before_round - 1
        candidate_path = self._artifact_path(
            task.id, candidate_round, "candidate.py"
        )
        if not candidate_path.is_file():
            raise ControllerError(
                f"round {candidate_round} candidate is missing while building follow-up"
            )
        return {
            "round": candidate_round,
            "role": (
                "latest_correct_seed" if scores else "latest_repair_candidate"
            ),
            "code": candidate_path.read_text(encoding="utf-8"),
        }

    def _history_scorecard(
        self, task: TaskSpec, *, before_round: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for previous_round in range(1, before_round):
            path = self._artifact_path(
                task.id, previous_round, "evaluation_result.json"
            )
            if not path.is_file():
                continue
            evaluation = _read_json(path)
            benchmark = evaluation.get("benchmark")
            benchmark = benchmark if isinstance(benchmark, Mapping) else {}
            rows.append(
                {
                    "round": previous_round,
                    "phase": evaluation.get("trajectory_phase", "unknown"),
                    "phase_index": evaluation.get("phase_index"),
                    "status": evaluation.get("overall_status"),
                    "source": _passed_stage(evaluation.get("source")),
                    "compile": _passed_stage(evaluation.get("compile")),
                    "correctness": _passed_stage(
                        evaluation.get("correctness")
                    ),
                    "geomean_speedup_vs_pytorch_eager": benchmark.get(
                        "geomean_speedup_vs_eager"
                    ),
                    "minimum_speedup_vs_pytorch_eager": benchmark.get(
                        "minimum_speedup_vs_eager"
                    ),
                    "bottleneck_type": benchmark.get("bottleneck_type"),
                }
            )
        # The durable artifacts retain every branch.  The model only needs a
        # short recent scorecard; allowing this list to grow across proposal
        # repair chains would recreate the prompt-bloat problem this feedback
        # protocol is intended to remove.
        return rows[-8:]

    def _candidate_intent(
        self, task: TaskSpec, round_number: int
    ) -> dict[str, Any]:
        response_path = self._artifact_path(
            task.id, round_number, "model_response.json"
        )
        if not response_path.is_file():
            return {
                "source": "model_response.optimization_summary",
                "optimization_summary": [],
                "expected_effect": [],
                "model_authored": False,
            }
        response = _read_json(response_path)
        model_failure_path = self._artifact_path(
            task.id, round_number, "model_failure.json"
        )
        if model_failure_path.is_file():
            return {
                "source": "synthetic_model_failure_sentinel",
                "optimization_summary": [],
                "expected_effect": [],
                "model_authored": False,
            }
        return {
            "source": (
                "model_response.optimization_summary"
                if "optimization_summary" in response
                else "model_response.change_summary"
            ),
            "optimization_summary": list(
                response.get(
                    "optimization_summary", response.get("change_summary", [])
                )
            ),
            "expected_effect": list(response.get("expected_effect", [])),
            "model_authored": True,
        }

    @staticmethod
    def _code_diff(before: str, after: str) -> list[str]:
        return list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile="BEST",
                tofile="candidate",
                lineterm="",
                n=3,
            )
        )

    def _successful_best_history(
        self, task: TaskSpec, *, before_round: int
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for round_number in range(1, before_round):
            feedback_path = self._artifact_path(
                task.id, round_number, "feedback.json"
            )
            if not feedback_path.is_file():
                continue
            feedback = _read_json(feedback_path)
            if feedback.get("performance_decision") not in {
                "INITIAL_BEST",
                "NEW_BEST",
            }:
                continue
            rows.append(
                {
                    "round": round_number,
                    "decision": feedback.get("performance_decision"),
                    "optimization_summary": feedback.get(
                        "candidate_generation_intent", {}
                    ).get("optimization_summary", []),
                    "geomean_speedup_vs_pytorch_eager": (
                        feedback.get("benchmark", {}).get(
                            "geomean_speedup_vs_eager"
                        )
                        if isinstance(feedback.get("benchmark"), Mapping)
                        else None
                    ),
                }
            )
        return rows[-5:]

    def _build_prompt(
        self,
        task: TaskSpec,
        round_number: int,
        *,
        phase: str,
        phase_index: int,
        repair_attempt: int = 0,
    ) -> ModelRequest:
        maximum_rounds = (
            self.config.maximum_repair_rounds
            + self.config.rounds_per_task
            * (self.config.maximum_repair_rounds + 1)
        )
        task_baseline = self.baseline.get(task.id)
        baseline_snapshot = (
            task_baseline if isinstance(task_baseline, Mapping) else self.baseline
        )
        common: dict[str, Any] = {
            "task": task.public_prompt_view(),
            "environment": self.environment,
            "baseline": prompt_baseline_projection(baseline_snapshot),
            "maximum_rounds": maximum_rounds,
            "phase": phase,
            "phase_index": phase_index,
            "optimization_rounds": self.config.rounds_per_task,
            "maximum_repair_rounds": self.config.maximum_repair_rounds,
            "model": self.config.model.model,
            "timeout_seconds": self.config.model.request_timeout_seconds,
        }
        if round_number == 1:
            return self.prompt_builder.build_first_round(**common)
        previous_evaluation_path = self._artifact_path(task.id, round_number - 1, "evaluation_result.json")
        previous_feedback_path = self._artifact_path(task.id, round_number - 1, "feedback.json")
        previous_evaluation = _read_json(previous_evaluation_path) if previous_evaluation_path.is_file() else {
            "overall_status": "model_failed",
        }
        previous_feedback = (
            _read_json(previous_feedback_path) if previous_feedback_path.is_file() else {}
        )
        candidate = self._prompt_candidate_context(
            task,
            before_round=round_number,
            phase=phase,
        )
        best = self._best_context(task, before_round=round_number)
        previous_code_path = self._artifact_path(
            task.id, round_number - 1, "candidate.py"
        )
        previous_code = (
            previous_code_path.read_text(encoding="utf-8")
            if previous_code_path.is_file()
            else ""
        )
        previous_intent = self._candidate_intent(task, round_number - 1)
        previous_overall = str(
            previous_evaluation.get("overall_status", "unknown")
        )
        feedback_state: dict[str, Any] = {
            "mode": previous_feedback.get(
                "next_prompt_mode", "REPAIR_FAILED_CANDIDATE"
            ),
            "performance_decision": previous_feedback.get(
                "performance_decision", "INVALID"
            ),
            "instruction": (
                "刚才的模型响应没有通过结构化响应校验; 不要沿用内部失败占位源码。"
                "请从当前任务或 BEST 重新生成完整 Kernel 和模型原生 optimization_summary。"
                if previous_overall == "model_failed"
                else (
                    "刚才的候选未通过 compile/correctness; 必须从该失败候选继续修复, "
                    "保留其模型原生 optimization_summary, 并只依据原始错误修改。"
                    if candidate["role"] == "failed_candidate_under_repair"
                    else (
                        "刚才的尝试造成确定性性能倒退; BEST 保持不变。"
                        "从 BEST 完整实现重新出发, 不要沿倒退候选继续优化。"
                        if previous_feedback.get("performance_decision")
                        == "REGRESSION"
                        else (
                            "候选与 BEST 的差异落在测量噪声范围, BEST 保持不变; "
                            "从 BEST 继续尝试新的优化方案。"
                            if previous_feedback.get("performance_decision")
                            == "TIE"
                            else (
                                "这是当前真实运行得到的 BEST; 从 BEST 继续优化。"
                                if best is not None
                                else (
                                    "这是正确但尚未形成可比较 BEST 的工作候选; "
                                    "请从它继续并取得 benchmark 和 smoke profile。"
                                )
                            )
                        )
                    )
                )
            ),
            "successful_best_history": self._successful_best_history(
                task, before_round=round_number
            ),
            "optimization_index": (
                phase_index
                if phase in {"optimization", "optimization_repair"}
                else None
            ),
            "repair_attempt": repair_attempt,
        }
        failed_candidate: dict[str, Any] | None = None
        if previous_overall == "model_failed":
            failure_path = self._artifact_path(
                task.id, round_number - 1, "model_failure.json"
            )
            failed_candidate = {
                "round": round_number - 1,
                "model_response_error": (
                    _read_json(failure_path) if failure_path.is_file() else None
                ),
                "candidate_generation_intent": {
                    "source": "synthetic_model_failure_sentinel",
                    "optimization_summary": [],
                    "expected_effect": [],
                    "model_authored": False,
                },
            }
        elif candidate["role"] == "failed_candidate_under_repair":
            failed_stage = next(
                (
                    name
                    for name in (
                        "source",
                        "compile",
                        "correctness",
                    )
                    if isinstance(previous_evaluation.get(name), Mapping)
                    and not _passed_stage(previous_evaluation.get(name))
                ),
                None,
            )
            failed_candidate = {
                "round": round_number - 1,
                "code": previous_code,
                "candidate_generation_intent": previous_intent,
                "failed_stage": failed_stage,
                "raw_stage_result": (
                    previous_evaluation.get(failed_stage)
                    if failed_stage is not None
                    else None
                ),
            }
        elif best is not None and previous_code and int(best["round"]) != round_number - 1:
            failed_candidate = {
                "round": round_number - 1,
                "candidate_generation_intent": previous_intent,
                "system_computed_code_diff_from_best": self._code_diff(
                    str(best["code"]), previous_code
                ),
                "candidate_metrics": self._follow_up_metrics(
                    previous_evaluation
                ),
                "system_computed_metric_and_profile_delta": previous_feedback.get(
                    "comparison_with_best"
                ),
            }
        repair_prompt = candidate["role"] == "failed_candidate_under_repair"
        prompt_best = best if best is not None and not repair_prompt else None
        use_working_candidate = prompt_best is None and failed_candidate is None
        compact_metrics = (
            self._follow_up_metrics(previous_evaluation)
            if use_working_candidate
            else {}
        )
        compact_failures = (
            self._follow_up_failure_reasons(
                previous_evaluation, previous_feedback
            )
            if use_working_candidate
            else []
        )
        if use_working_candidate:
            feedback_state["working_candidate_generation_intent"] = (
                previous_intent
            )
        return self.prompt_builder.build_follow_up(
            round_number=round_number,
            maximum_rounds=maximum_rounds,
            last_candidate_code=(
                str(candidate["code"]) if use_working_candidate else ""
            ),
            key_metrics=compact_metrics,
            failure_reasons=compact_failures,
            next_round_suggestions=self._follow_up_suggestions(previous_feedback),
            task_contract=task.public_prompt_view(),
            environment=self.environment,
            baseline={
                "comparison_baseline": (
                    self.config.benchmark.comparison_baseline
                )
            },
            candidate_round=int(candidate["round"]),
            candidate_role=str(candidate["role"]),
            history_summary=self._history_scorecard(
                task, before_round=round_number
            ),
            feedback_state=feedback_state,
            best_candidate=(
                {
                    "round": prompt_best["round"],
                    "candidate_id": prompt_best["candidate_id"],
                    "code": prompt_best["code"],
                    "metrics": self._follow_up_metrics(
                        prompt_best["evaluation"]
                    ),
                    "candidate_generation_intent": {
                        "source": "model_response.optimization_summary",
                        "optimization_summary": prompt_best[
                            "optimization_summary"
                        ],
                    },
                }
                if prompt_best is not None
                else None
            ),
            failed_candidate=failed_candidate,
            phase=phase,
            phase_index=phase_index,
            optimization_rounds=self.config.rounds_per_task,
            maximum_repair_rounds=self.config.maximum_repair_rounds,
            model=self.config.model.model,
            timeout_seconds=self.config.model.request_timeout_seconds,
        )

    def _load_model_response(self, task: TaskSpec, round_number: int) -> ModelResponse:
        return validate_model_response(
            self._load_committed_json(
                self._relative(task.id, round_number, "model_response.json"),
                task_id=task.id,
                round_number=round_number,
            ),
            expected_round=round_number,
            allow_legacy=True,
        )

    @staticmethod
    def _model_attempt(completion: Any) -> dict[str, Any]:
        return {
            "finish_reason": completion.finish_reason,
            "request_id": completion.request_id,
            "usage": dict(completion.usage),
            "raw_response": dict(completion.raw_response),
        }

    def _materialize_model_exchange(
        self,
        task: TaskSpec,
        round_number: int,
        exchange: Mapping[str, Any],
        *,
        allow_legacy: bool = False,
    ) -> ModelResponse:
        attempts = exchange.get("attempts")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)) or not attempts:
            raise ControllerError("committed model exchange has no attempts")
        for attempt_number, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, Mapping) or not isinstance(
                attempt.get("raw_response"), Mapping
            ):
                raise ControllerError("committed model exchange has an invalid attempt")
            self._commit_json(
                self._relative(
                    task.id,
                    round_number,
                    f"raw_response_attempt_{attempt_number:02d}.json",
                ),
                dict(attempt["raw_response"]),
                task_id=task.id,
                round_number=round_number,
            )
        final_attempt = attempts[-1]
        assert isinstance(final_attempt, Mapping)
        final_raw = final_attempt.get("raw_response")
        assert isinstance(final_raw, Mapping)
        self._commit_json(
            self._relative(task.id, round_number, "raw_response.json"),
            dict(final_raw),
            task_id=task.id,
            round_number=round_number,
        )
        response_value = exchange.get("response")
        if not isinstance(response_value, Mapping):
            raise ControllerError("committed model exchange has no structured response")
        response = validate_model_response(
            response_value,
            expected_round=round_number,
            allow_legacy=allow_legacy,
        )
        self._commit_json(
            self._relative(task.id, round_number, "model_response.json"),
            response.to_dict(),
            task_id=task.id,
            round_number=round_number,
        )
        return response

    def _materialize_model_failure(
        self, task: TaskSpec, round_number: int, exchange: Mapping[str, Any]
    ) -> ModelResponse:
        error_value = exchange.get("error")
        if not isinstance(error_value, Mapping):
            raise ControllerError("committed model failure exchange has no error")
        attempts = exchange.get("attempts", [])
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise ControllerError("committed model failure attempts are invalid")
        for attempt_number, attempt in enumerate(attempts, 1):
            if not isinstance(attempt, Mapping) or not isinstance(
                attempt.get("raw_response"), Mapping
            ):
                raise ControllerError("committed model failure has an invalid attempt")
            raw = dict(attempt["raw_response"])
            self._commit_json(
                self._relative(
                    task.id,
                    round_number,
                    f"raw_response_attempt_{attempt_number:02d}.json",
                ),
                raw,
                task_id=task.id,
                round_number=round_number,
            )
            if attempt_number == len(attempts):
                self._commit_json(
                    self._relative(task.id, round_number, "raw_response.json"),
                    raw,
                    task_id=task.id,
                    round_number=round_number,
                )
        self._commit_json(
            self._relative(task.id, round_number, "model_failure.json"),
            {
                "schema_version": "ascend_model_failure_v1",
                "error_type": str(error_value.get("error_type", "ModelResponseError")),
                "message": str(error_value.get("message", "invalid structured response")),
                "round": round_number,
            },
            task_id=task.id,
            round_number=round_number,
        )
        # This internal failure sentinel is deliberately not represented as a
        # model-authored optimization_summary.  ``change_summary`` is reserved
        # here solely for the legacy-compatible sentinel/read path; live model
        # completions are always validated against the current strict schema.
        synthetic_response = {
            "status": "candidate",
            "round": round_number,
            "change_summary": [],
            "expected_effect": [],
            "assumptions": [
                "Model output failed structured-response validation."
            ],
            "code": (
                "# Invalid model response; this round intentionally "
                "fails source validation.\n"
            ),
        }
        response = validate_model_response(
            synthetic_response,
            expected_round=round_number,
            allow_legacy=True,
        )
        self._commit_json(
            self._relative(task.id, round_number, "model_response.json"),
            synthetic_response,
            task_id=task.id,
            round_number=round_number,
        )
        return response

    def _call_and_commit_model(self, task: TaskSpec, round_number: int, request: ModelRequest) -> ModelResponse:
        validated = complete_model_response(
            self.model_gateway,
            request,
            expected_round=round_number,
            maximum_format_repair_retries=self.config.model.maximum_format_repair_retries,
        )
        response = validated.response
        exchange = {
            "schema_version": "ascend_model_exchange_v1",
            "round": round_number,
            "repair_attempts": validated.repair_attempts,
            "attempts": [
                self._model_attempt(completion)
                for completion in validated.completions
            ],
            "response": response.to_dict(),
        }
        self._commit_json(
            self._relative(task.id, round_number, "model_exchange.json"),
            exchange,
            task_id=task.id,
            round_number=round_number,
        )
        return self._materialize_model_exchange(
            task, round_number, exchange
        )

    def _run_round(
        self,
        task: TaskSpec,
        round_number: int,
        *,
        phase: str,
        phase_index: int,
        repair_attempt: int = 0,
    ) -> None:
        record = self.store.get_round(self.experiment_id, task.id, round_number)
        if record is None:
            record = self.store.create_round(self.experiment_id, task.id, round_number)
        if record.state is RoundState.ROUND_CREATED:
            request = self._build_prompt(
                task,
                round_number,
                phase=phase,
                phase_index=phase_index,
                repair_attempt=repair_attempt,
            )
            self._commit_json(
                self._relative(task.id, round_number, "prompt.json"),
                {
                    "system_prompt": request.system_prompt,
                    "user_prompt": json.loads(request.user_prompt),
                    "json_schema": dict(request.json_schema),
                    "metadata": dict(request.metadata),
                },
                task_id=task.id,
                round_number=round_number,
            )
            record = self.store.transition_round(
                self.experiment_id, task.id, round_number, RoundState.PROMPT_COMMITTED,
                expected_version=record.version,
            )
        prompt_artifact = self._load_committed_json(
            self._relative(task.id, round_number, "prompt.json"),
            task_id=task.id,
            round_number=round_number,
        )
        request = ModelRequest(
            system_prompt=str(prompt_artifact["system_prompt"]),
            user_prompt=json.dumps(prompt_artifact["user_prompt"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            json_schema=prompt_artifact["json_schema"],
            model=self.config.model.model,
            timeout_seconds=self.config.model.request_timeout_seconds,
            metadata=prompt_artifact.get("metadata", {}),
        )
        if record.state is RoundState.PROMPT_COMMITTED:
            record = self.store.transition_round(
                self.experiment_id, task.id, round_number, RoundState.MODEL_REQUEST_SENT,
                expected_version=record.version,
            )
        response_path = self._artifact_path(task.id, round_number, "model_response.json")
        if record.state is RoundState.MODEL_REQUEST_SENT:
            if response_path.is_file():
                response = self._load_model_response(task, round_number)
            elif self._artifact_path(
                task.id, round_number, "model_exchange.json"
            ).is_file():
                response = self._materialize_model_exchange(
                    task,
                    round_number,
                    _read_json(
                        self._artifact_path(
                            task.id, round_number, "model_exchange.json"
                        )
                    ),
                    allow_legacy=True,
                )
            elif self._artifact_path(
                task.id, round_number, "model_failure_exchange.json"
            ).is_file():
                response = self._materialize_model_failure(
                    task,
                    round_number,
                    _read_json(
                        self._artifact_path(
                            task.id,
                            round_number,
                            "model_failure_exchange.json",
                        )
                    ),
                )
            else:
                try:
                    response = self._call_and_commit_model(task, round_number, request)
                except ModelResponseError as exc:
                    # Malformed provider output consumes the round only after
                    # bounded repair has failed. Transport errors still bubble
                    # up and retry this same round on resume.
                    failure_exchange = {
                        "schema_version": "ascend_model_failure_exchange_v1",
                        "round": round_number,
                        "error": {
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                        "attempts": [
                            self._model_attempt(completion)
                            for completion in (
                                exc.attempts
                                if isinstance(exc, ModelResponseAttemptsExhausted)
                                else ()
                            )
                        ],
                    }
                    self._commit_json(
                        self._relative(
                            task.id,
                            round_number,
                            "model_failure_exchange.json",
                        ),
                        failure_exchange,
                        task_id=task.id,
                        round_number=round_number,
                    )
                    response = self._materialize_model_failure(
                        task,
                        round_number,
                        failure_exchange,
                    )
            record = self.store.transition_round(
                self.experiment_id,
                task.id,
                round_number,
                RoundState.MODEL_RESPONSE_COMMITTED,
                expected_version=record.version,
                event_payload={"status": response.status},
            )
        response = self._load_model_response(task, round_number)
        candidate_path = self._artifact_path(task.id, round_number, "candidate.py")
        candidate_path, source_sha = self._commit_text(
            self._relative(task.id, round_number, "candidate.py"),
            response.code,
            task_id=task.id,
            round_number=round_number,
            media_type="text/x-python; charset=utf-8",
        )
        candidate_id = f"{self.experiment_id}:{task.id}:r{round_number:02d}:{source_sha[:16]}"
        response_sha = hashlib.sha256(response_path.read_bytes()).hexdigest()
        self.store.register_candidate(
            candidate_id=candidate_id,
            experiment_id=self.experiment_id,
            task_id=task.id,
            round_number=round_number,
            source_sha256=source_sha,
            source_artifact_path=self._relative(task.id, round_number, "candidate.py"),
            response_sha256=response_sha,
        )
        evaluation_path = self._artifact_path(task.id, round_number, "evaluation_result.json")
        model_failure_path = self._artifact_path(
            task.id, round_number, "model_failure.json"
        )
        if evaluation_path.is_file():
            evaluation_mapping = self._load_committed_json(
                self._relative(task.id, round_number, "evaluation_result.json"),
                task_id=task.id,
                round_number=round_number,
            )
            score = _score_from_mapping(evaluation_mapping["score"])
            committed_phase = evaluation_mapping.get("trajectory_phase")
            committed_phase_index = evaluation_mapping.get("phase_index")
            committed_repair_attempt = evaluation_mapping.get("repair_attempt")
            if committed_phase not in {None, phase} or committed_phase_index not in {
                None,
                phase_index,
            } or committed_repair_attempt not in {None, repair_attempt}:
                raise ControllerError(
                    "committed evaluation trajectory phase disagrees with replay"
                )
            if model_failure_path.is_file() and score.is_publicly_valid:
                raise ControllerError(
                    "model-failure evaluation cannot contain an eligible score"
                )
        else:
            if model_failure_path.is_file():
                score = CandidateScore(
                    candidate_id=candidate_id,
                    round_number=round_number,
                    compile_passed=False,
                    correctness_passed=False,
                    anti_bypass_passed=False,
                )
                evaluation_mapping = {
                    "schema_version": "ascend_evaluation_result_v1",
                    "experiment_id": self.experiment_id,
                    "task_id": task.id,
                    "round": round_number,
                    "candidate_id": candidate_id,
                    "overall_status": "model_failed",
                    "source": {
                        "stage": "SOURCE_CHECK",
                        "passed": False,
                        "status": "model_failed",
                    },
                    "compile": None,
                    "correctness": None,
                    "benchmark": None,
                    "profile": None,
                    "anti_bypass": {
                        "passed": False,
                        "status": "not_evaluated",
                        "reason": "model_failed",
                    },
                    "reward_vector": asdict(compute_reward(score)),
                    "score": asdict(score),
                }
            else:
                reused = self._reuse_evaluation(
                    task=task,
                    round_number=round_number,
                    source_sha=source_sha,
                    candidate_id=candidate_id,
                )
                if reused is not None:
                    evaluation_mapping, score = reused
                else:
                    evaluation = evaluate_candidate(
                        self.backend,
                        EvaluationRequest(
                            experiment_id=self.experiment_id,
                            task=task,
                            round_number=round_number,
                            candidate_id=candidate_id,
                            candidate_path=candidate_path,
                            artifact_dir=candidate_path.parent,
                            baseline_snapshot=(
                                self.baseline.get(task.id)
                                if isinstance(self.baseline.get(task.id), Mapping)
                                else self.baseline
                            ),
                            run_profile=self.config.profile.run_after_correctness,
                            profile_coverage_required=self.profile_coverage_required,
                        ),
                    )
                    evaluation_mapping = evaluation.to_dict()
                    score = evaluation.score
            evaluation_mapping["trajectory_phase"] = phase
            evaluation_mapping["phase_index"] = phase_index
            evaluation_mapping["optimization_index"] = (
                phase_index
                if phase in {"optimization", "optimization_repair"}
                else None
            )
            evaluation_mapping["repair_attempt"] = repair_attempt
            self._commit_json(
                self._relative(task.id, round_number, "evaluation_result.json"),
                evaluation_mapping,
                task_id=task.id,
                round_number=round_number,
            )
        self._validate_evaluation_identity(
            evaluation_mapping,
            task=task,
            round_number=round_number,
            candidate_id=candidate_id,
        )
        self.store.save_candidate_score(score)
        self._advance_round_evaluation(record, task, round_number, evaluation_mapping)

    def _validate_evaluation_identity(
        self,
        value: Mapping[str, Any],
        *,
        task: TaskSpec,
        round_number: int,
        candidate_id: str,
    ) -> None:
        score = value.get("score")
        if not isinstance(score, Mapping):
            raise ControllerError("evaluation result has no score object")
        if (
            value.get("schema_version") != "ascend_evaluation_result_v1"
            or value.get("experiment_id") != self.experiment_id
            or value.get("task_id") != task.id
            or int(value.get("round", -1)) != round_number
            or value.get("candidate_id") != candidate_id
            or score.get("candidate_id") != candidate_id
            or int(score.get("round_number", score.get("round", -1)))
            != round_number
        ):
            raise ControllerError(
                "evaluation result identity does not match its durable round/candidate"
            )

    def _reuse_evaluation(
        self,
        *,
        task: TaskSpec,
        round_number: int,
        source_sha: str,
        candidate_id: str,
    ) -> tuple[dict[str, Any], CandidateScore] | None:
        """Reuse a prior immutable result for byte-identical full source."""
        for previous_round in range(1, round_number):
            source = self._artifact_path(task.id, previous_round, "candidate.py")
            evaluation = self._artifact_path(
                task.id, previous_round, "evaluation_result.json"
            )
            if not source.is_file() or not evaluation.is_file():
                continue
            if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
                continue
            value = json.loads(json.dumps(_read_json(evaluation)))
            value["round"] = round_number
            value["candidate_id"] = candidate_id
            score_mapping = dict(value["score"])
            score_mapping["candidate_id"] = candidate_id
            score_mapping["round_number"] = round_number
            score_mapping["hidden_correctness_passed"] = None
            value["score"] = score_mapping
            value["reused_from_round"] = previous_round
            return value, _score_from_mapping(score_mapping)
        return None

    def _advance_round_evaluation(
        self,
        initial_record: Any,
        task: TaskSpec,
        round_number: int,
        evaluation: Mapping[str, Any],
    ) -> None:
        record = self.store.get_round(self.experiment_id, task.id, round_number) or initial_record
        ordered = [
            RoundState.SOURCE_VALIDATED,
            RoundState.COMPILE_FINISHED,
            RoundState.CORRECTNESS_FINISHED,
            RoundState.BENCHMARK_FINISHED,
            RoundState.PROFILE_FINISHED,
        ]
        overall = str(evaluation.get("overall_status"))
        last = {
            "model_failed": RoundState.SOURCE_VALIDATED,
            "source_failed": RoundState.SOURCE_VALIDATED,
            "compile_failed": RoundState.COMPILE_FINISHED,
            "correctness_failed": RoundState.CORRECTNESS_FINISHED,
        }.get(overall, RoundState.PROFILE_FINISHED)
        if record.state not in {RoundState.FEEDBACK_COMMITTED, RoundState.ROUND_FINISHED}:
            current_index = -1 if record.state is RoundState.MODEL_RESPONSE_COMMITTED else ordered.index(record.state)
            target_index = ordered.index(last)
            if current_index > target_index:
                raise ControllerError(
                    "round state is ahead of the committed evaluation outcome: "
                    f"state={record.state.value}, overall_status={overall}"
                )
            for state in ordered[current_index + 1 : target_index + 1]:
                record = self.store.transition_round(
                    self.experiment_id,
                    task.id,
                    round_number,
                    state,
                    expected_version=record.version,
                    event_payload={"overall_status": overall},
                )
            if last is RoundState.BENCHMARK_FINISHED:  # defensive; current policies end at profile
                record = self.store.transition_round(
                    self.experiment_id, task.id, round_number, RoundState.PROFILE_FINISHED,
                    expected_version=record.version, event_payload={"status": "skipped"},
                )
            self._ensure_feedback(task, round_number, evaluation)
            record = self.store.transition_round(
                self.experiment_id,
                task.id,
                round_number,
                RoundState.FEEDBACK_COMMITTED,
                expected_version=record.version,
                event_payload={"overall_status": overall},
            )
        else:
            self._ensure_feedback(task, round_number, evaluation)
        if record.state is RoundState.FEEDBACK_COMMITTED:
            self.store.transition_round(
                self.experiment_id,
                task.id,
                round_number,
                RoundState.ROUND_FINISHED,
                expected_version=record.version,
            )

    def _ensure_feedback(
        self,
        task: TaskSpec,
        round_number: int,
        evaluation: Mapping[str, Any],
    ) -> dict[str, Any]:
        relative = self._relative(task.id, round_number, "feedback.json")
        path = self.artifacts.path_for(relative)
        if path.is_file():
            feedback = self._load_committed_json(
                relative,
                task_id=task.id,
                round_number=round_number,
            )
            self._reconcile_committed_online_best(
                task, allow_inflight_candidate=True
            )
            return feedback
        best_before = self._best_context(task, before_round=round_number)
        feedback = build_feedback(
            task_id=task.id,
            round_number=round_number,
            result=evaluation,
            best=best_before,
            consecutive_non_improvements=self._consecutive_non_improvements(
                task, round_number
            ),
            candidate_intent=self._candidate_intent(task, round_number),
        )
        score_value = evaluation.get("score")
        candidate_score = (
            _score_from_mapping(score_value)
            if isinstance(score_value, Mapping)
            else None
        )
        incumbent = (
            self.store.get_candidate_score(str(best_before["candidate_id"]))
            if best_before is not None
            else None
        )
        decision = (
            compare_public_candidate(candidate_score, incumbent)
            if candidate_score is not None
            else None
        )
        if candidate_score is not None and decision is not None and decision.decision in {
            "INITIAL_BEST",
            "NEW_BEST",
        }:
            self.store.set_best_candidate(
                self.experiment_id,
                task.id,
                candidate_score.candidate_id,
                reason=decision.decision,
            )
        task_record = self.store.get_task(self.experiment_id, task.id)
        best_after_score = (
            self.store.get_candidate_score(task_record.best_candidate_id)
            if task_record is not None and task_record.best_candidate_id is not None
            else None
        )
        feedback["best_before"] = (
            {
                "round": best_before["round"],
                "candidate_id": best_before["candidate_id"],
            }
            if best_before is not None
            else None
        )
        feedback["best_after"] = (
            {
                "candidate_id": task_record.best_candidate_id,
                "round": best_after_score.round_number,
            }
            if task_record is not None
            and task_record.best_candidate_id is not None
            and best_after_score is not None
            else None
        )
        self._commit_json(
            relative,
            feedback,
            task_id=task.id,
            round_number=round_number,
        )
        return feedback

    def _consecutive_non_improvements(self, task: TaskSpec, current_round: int) -> int:
        scores = sorted(
            (score for score in self.store.list_candidate_scores(self.experiment_id, task.id) if score.round_number <= current_round),
            key=lambda score: score.round_number,
        )
        count = 0
        best = float("-inf")
        for score in scores:
            value = score.geomean_speedup
            if value is not None and value > best * 1.003:
                best = value
                count = 0
            else:
                count += 1
        return count

    def _run_final(self, task: TaskSpec) -> TaskRunSummary:
        task_record = self.store.get_task(self.experiment_id, task.id)
        assert task_record is not None
        final_relative = self._relative(task.id, name="final_result.json")
        final_path = self.artifacts.path_for(final_relative)
        if task_record.state in {TaskState.TASK_FINISHED, TaskState.TASK_FAILED}:
            if not final_path.is_file():
                raise ControllerError(
                    "terminal task is missing final_result.json; terminal state "
                    "cannot be rolled back safely"
                )
            final = self._load_committed_json(final_relative, task_id=task.id)
            expected_terminal = (
                TaskState.TASK_FINISHED
                if str(final.get("status", "")).startswith("passed")
                else TaskState.TASK_FAILED
            )
            if task_record.state is not expected_terminal:
                raise ControllerError(
                    "terminal task state disagrees with final_result.json"
                )
            return TaskRunSummary(
                task.id,
                str(final.get("status")),
                final.get("best_round"),
                final.get("best_candidate_id"),
                final,
            )
        if task_record.state is TaskState.ROUNDS_RUNNING:
            best = self._reconcile_committed_online_best(
                task, allow_inflight_candidate=False
            )
            task_record = self.store.get_task(self.experiment_id, task.id) or task_record
        else:
            if task_record.best_candidate_id is None:
                raise ControllerError(
                    "final evaluation state has no durably selected candidate"
                )
            best = self.store.get_candidate_score(task_record.best_candidate_id)
        if best is None:
            repair_rounds, optimization_rounds = self._trajectory_counts(task)
            final = {
                "schema_version": "ascend_final_result_v1",
                "experiment_id": self.experiment_id,
                "task_id": task.id,
                "status": "failed_no_valid_candidate",
                "best_round": None,
                "best_candidate_id": None,
                "repair_rounds": repair_rounds,
                "optimization_rounds": optimization_rounds,
                "termination_reason": self._trajectory_termination_reason(task),
            }
            self._commit_json(final_relative, final, task_id=task.id)
            if task_record.state is TaskState.ROUNDS_RUNNING:
                self.store.transition_task(
                    self.experiment_id, task.id, TaskState.TASK_FAILED
                )
            return TaskRunSummary(task.id, final["status"], None, None, final)
        if task_record.state is TaskState.ROUNDS_RUNNING:
            task_record = self.store.transition_task(self.experiment_id, task.id, TaskState.SELECT_BEST_CANDIDATE)
        source = self._artifact_path(task.id, best.round_number, "candidate.py")
        if not source.is_file():
            raise ControllerError("selected candidate source artifact is missing")
        best_path, _ = self._commit_text(
            self._relative(task.id, name="best_candidate.py"),
            source.read_text(encoding="utf-8"),
            task_id=task.id,
            media_type="text/x-python; charset=utf-8",
        )
        if self.hidden_seed is None:
            raise ControllerError(
                "final hidden evaluation requires a deployment-only hidden seed; set AKG_HIDDEN_SEED"
            )
        if task_record.state is TaskState.SELECT_BEST_CANDIDATE:
            task_record = self.store.transition_task(self.experiment_id, task.id, TaskState.HIDDEN_CORRECTNESS_TEST)
        assert task.root is not None
        hidden = hidden_cases_from_template(task.root, secret_seed=self.hidden_seed)
        hidden_correctness = tuple(case for case in hidden if case.kind == "correctness")
        hidden_benchmark = tuple(case for case in hidden if case.kind == "benchmark")
        final_eval_relative = self._relative(
            task.id, name="final_evaluation/final_evaluation.json"
        )
        final_eval_path = self.artifacts.path_for(final_eval_relative)
        if final_eval_path.is_file():
            final_evaluation = self._load_committed_json(
                final_eval_relative,
                task_id=task.id,
            )
        else:
            result = evaluate_candidate(
                self.backend,
                EvaluationRequest(
                    experiment_id=self.experiment_id,
                    task=task,
                    round_number=best.round_number,
                    candidate_id=best.candidate_id,
                    candidate_path=best_path,
                    artifact_dir=final_eval_path.parent,
                    correctness_cases=hidden_correctness,
                    benchmark_cases=hidden_benchmark,
                    profile_cases=hidden_benchmark[:1],
                    baseline_snapshot=self.baseline.get(task.id) if isinstance(self.baseline.get(task.id), Mapping) else self.baseline,
                    run_profile=self.config.profile.run_for_final_best,
                    profile_coverage_required=self.profile_coverage_required,
                    hidden=True,
                ),
            )
            final_evaluation = result.to_dict()
            self._commit_json(
                final_eval_relative,
                final_evaluation,
                task_id=task.id,
            )
        if (
            final_evaluation.get("schema_version") != "ascend_evaluation_result_v1"
            or final_evaluation.get("experiment_id") != self.experiment_id
            or final_evaluation.get("task_id") != task.id
            or final_evaluation.get("candidate_id") != best.candidate_id
            or int(final_evaluation.get("round", -1)) != best.round_number
        ):
            raise ControllerError(
                "final evaluation identity does not match the selected candidate"
            )
        status, hidden_passed = _final_gate_status(
            final_evaluation,
            profile_coverage_required=self.profile_coverage_required,
        )
        self.store.save_candidate_score(
            replace(best, hidden_correctness_passed=hidden_passed)
        )
        final_benchmark_value = final_evaluation.get("benchmark")
        benchmark: Mapping[str, Any] = (
            final_benchmark_value
            if isinstance(final_benchmark_value, Mapping)
            else {}
        )
        final = {
            "schema_version": "ascend_final_result_v1",
            "experiment_id": self.experiment_id,
            "task_id": task.id,
            "status": status,
            "best_round": best.round_number,
            "best_candidate_id": best.candidate_id,
            "hidden_correctness_passed": hidden_passed,
            "speedup_geomean": benchmark.get("geomean_speedup_vs_eager"),
            "minimum_speedup": benchmark.get("minimum_speedup_vs_eager"),
            "final_evaluation": final_evaluation,
        }
        repair_rounds, optimization_rounds = self._trajectory_counts(task)
        final.update(
            {
                "repair_rounds": repair_rounds,
                "optimization_rounds": optimization_rounds,
                "termination_reason": self._trajectory_termination_reason(task),
            }
        )
        self._commit_json(final_relative, final, task_id=task.id)
        if status == "failed_hidden_correctness":
            if task_record.state is TaskState.HIDDEN_CORRECTNESS_TEST:
                task_record = self.store.transition_task(
                    self.experiment_id, task.id, TaskState.TASK_FAILED
                )
        else:
            if task_record.state is TaskState.HIDDEN_CORRECTNESS_TEST:
                task_record = self.store.transition_task(
                    self.experiment_id, task.id, TaskState.FINAL_BENCHMARK
                )
            if status == "failed_final_benchmark":
                if task_record.state is TaskState.FINAL_BENCHMARK:
                    task_record = self.store.transition_task(
                        self.experiment_id, task.id, TaskState.TASK_FAILED
                    )
            else:
                if task_record.state is TaskState.FINAL_BENCHMARK:
                    task_record = self.store.transition_task(
                        self.experiment_id, task.id, TaskState.FINAL_FULL_PROFILE
                    )
                terminal = (
                    TaskState.TASK_FINISHED
                    if status.startswith("passed")
                    else TaskState.TASK_FAILED
                )
                if task_record.state is TaskState.FINAL_FULL_PROFILE:
                    task_record = self.store.transition_task(
                        self.experiment_id, task.id, terminal
                    )
        return TaskRunSummary(task.id, status, best.round_number, best.candidate_id, final)

    def materialize_events(self) -> Path:
        values: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = self.store.events.read(after_sequence=cursor, limit=10_000, experiment_id=self.experiment_id)
            if not page:
                break
            values.extend({
                "sequence": event.sequence,
                "event_id": event.event_id,
                "event_type": event.event_type,
                "aggregate_type": event.aggregate_type,
                "aggregate_id": event.aggregate_id,
                "experiment_id": event.experiment_id,
                "task_id": event.task_id,
                "round": event.round_number,
                "payload": dict(event.payload),
                "occurred_at": event.occurred_at.isoformat(),
            } for event in page)
            cursor = page[-1].sequence
        text = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for value in values)
        metadata = self.artifacts.put_text(
            self._relative(name="events.jsonl"),
            text,
            media_type="application/x-ndjson",
            overwrite=True,
        )
        # This file is a rebuildable projection of immutable SQLite events, so
        # it is intentionally not registered as an immutable artifact row.
        return self.artifacts.path_for(metadata.relative_path)
