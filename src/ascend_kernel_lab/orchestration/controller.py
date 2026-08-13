from __future__ import annotations

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
    compute_reward,
    select_best_candidate,
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
        for round_number in range(1, self.config.rounds_per_task + 1):
            # Finished rounds are replayed read-only to verify/repair any
            # filesystem-first artifact registration left by a crash.
            self._run_round(task, round_number)
        return self._run_final(task)

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
            correctness_summary = {
                **(stage_status("correctness") or {}),
                "passed_cases": correctness.get("passed_cases"),
                "total_cases": correctness.get("total_cases"),
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
                "pipeline_utilization": dict(pipeline),
                "memory": dict(memory),
                "observations": summary.get("observations", []),
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
        scores = self.store.list_candidate_scores(self.experiment_id, task.id)
        if before_round is not None:
            scores = [score for score in scores if score.round_number < before_round]
        best = select_best_candidate(scores)
        if best is None:
            return None
        code_path = self._artifact_path(task.id, best.round_number, "candidate.py")
        evaluation_path = self._artifact_path(task.id, best.round_number, "evaluation_result.json")
        return {
            "round": best.round_number,
            "candidate_id": best.candidate_id,
            "code": code_path.read_text(encoding="utf-8"),
            "geomean_speedup": best.geomean_speedup,
            "minimum_speedup": best.minimum_speedup,
            "evaluation": _read_json(evaluation_path) if evaluation_path.is_file() else {},
        }

    def _build_prompt(self, task: TaskSpec, round_number: int) -> ModelRequest:
        common = {
            "task": task.public_prompt_view(),
            "environment": self.environment,
            "baseline": self.baseline.get(task.id, self.baseline),
            "maximum_rounds": self.config.rounds_per_task,
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
        previous_code = self._artifact_path(
            task.id, round_number - 1, "candidate.py"
        ).read_text(encoding="utf-8")
        return self.prompt_builder.build_follow_up(
            round_number=round_number,
            maximum_rounds=self.config.rounds_per_task,
            last_candidate_code=previous_code,
            key_metrics=self._follow_up_metrics(previous_evaluation),
            failure_reasons=self._follow_up_failure_reasons(
                previous_evaluation, previous_feedback
            ),
            next_round_suggestions=self._follow_up_suggestions(previous_feedback),
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
        self, task: TaskSpec, round_number: int, exchange: Mapping[str, Any]
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
            response_value, expected_round=round_number
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
        response = ModelResponse(
            status="candidate",
            round=round_number,
            change_summary=(),
            expected_effect=(),
            assumptions=("Model output failed structured-response validation.",),
            code=(
                "# Invalid model response; this round intentionally "
                "fails source validation.\n"
            ),
        )
        self._commit_json(
            self._relative(task.id, round_number, "model_response.json"),
            response.to_dict(),
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

    def _run_round(self, task: TaskSpec, round_number: int) -> None:
        record = self.store.get_round(self.experiment_id, task.id, round_number)
        if record is None:
            record = self.store.create_round(self.experiment_id, task.id, round_number)
        if record.state is RoundState.ROUND_CREATED:
            request = self._build_prompt(task, round_number)
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
            return self._load_committed_json(
                relative,
                task_id=task.id,
                round_number=round_number,
            )
        feedback = build_feedback(
            task_id=task.id,
            round_number=round_number,
            result=evaluation,
            best=self._best_context(task, before_round=round_number),
            consecutive_non_improvements=self._consecutive_non_improvements(
                task, round_number
            ),
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
            best = (
                self.store.get_candidate_score(task_record.best_candidate_id)
                if task_record.best_candidate_id is not None
                else self.store.select_and_set_best_candidate(
                    self.experiment_id, task.id
                )
            )
            task_record = self.store.get_task(self.experiment_id, task.id) or task_record
        else:
            if task_record.best_candidate_id is None:
                raise ControllerError(
                    "final evaluation state has no durably selected candidate"
                )
            best = self.store.get_candidate_score(task_record.best_candidate_id)
        if best is None:
            final = {
                "schema_version": "ascend_final_result_v1",
                "experiment_id": self.experiment_id,
                "task_id": task.id,
                "status": "failed_no_valid_candidate",
                "best_round": None,
                "best_candidate_id": None,
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
