from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)[\"']?authorization[\"']?\s*[:=]"),
    re.compile(
        r"(?i)[\"']?(api[_-]?key|auth[_-]?token|password|secret)[\"']?\s*[:=]"
    ),
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(r"\bsk-[a-zA-Z0-9_-]{12,}"),
)


class ExportError(RuntimeError):
    pass


def _read_json(path: Path, default: Any = None) -> Any:
    if path.is_symlink():
        raise ExportError(f"refusing to read symlinked artifact {path}")
    if not path.is_file():
        return default
    try:
        raw = path.read_bytes()
        if len(raw) > 32 * 1024**2:
            raise ExportError(f"JSON artifact exceeds 32 MiB: {path}")
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"invalid JSON artifact {path}: {exc}") from exc


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            assert_export_clean(payload)
            handle.write(payload + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def assert_export_clean(payload: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise ExportError(f"export blocked by possible credential matching {pattern.pattern!r}")
    lowered = payload.lower()
    if "hidden_cases" in lowered or "hidden_seed" in lowered or "secret_seed" in lowered:
        raise ExportError("export blocked because hidden evaluation details were detected")


@dataclass(frozen=True)
class RoundArtifacts:
    task_id: str
    round_id: int
    root: Path
    prompt: Mapping[str, Any]
    response: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    feedback: Mapping[str, Any]
    reward: Mapping[str, Any]
    code: str


@dataclass(frozen=True)
class RoundQuality:
    label: str
    eligible_for_default_sft: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "eligible_for_default_sft": self.eligible_for_default_sft,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ClassifiedRound:
    artifacts: RoundArtifacts
    quality: RoundQuality


_FAILURE_LABELS = {
    "model_failed": "model_failed",
    "source_failed": "source_failed",
    "compile_failed": "compile_failed",
    "correctness_failed": "correctness_failed",
    "benchmark_failed": "benchmark_failed",
    "anti_bypass_failed": "anti_bypass_failed",
    "profile_unavailable": "profile_unavailable",
}
_REPAIRABLE_FAILURES = frozenset(
    {"source_failed", "compile_failed", "correctness_failed"}
)
_HOST_BOUND_TERMS = (
    "host_bound",
    "host-bound",
    "host overhead",
    "host_overhead",
    "host dispatch",
    "host_dispatch",
    "dispatch overhead",
    "dispatch_bound",
    "dispatch-bound",
    "launch_bound",
    "launch-bound",
    "launch overhead",
    "unoptimizable",
    "not optimizable",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _overall_status(item: RoundArtifacts) -> str:
    return str(
        item.feedback.get(
            "overall_status", item.evaluation.get("overall_status", "unknown")
        )
    ).lower()


def _score(item: RoundArtifacts) -> Mapping[str, Any]:
    value = item.evaluation.get("score")
    return value if isinstance(value, Mapping) else {}


def candidate_generation_intent(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only the model's original structured candidate intent.

    No evaluation, feedback, or source-code analysis is used here.  The
    legacy field is accepted solely so older committed runs remain exportable.
    """

    source_field = "model_response.optimization_summary"
    summary = response.get("optimization_summary")
    if summary is None and "change_summary" in response:
        source_field = "model_response.change_summary"
        summary = response.get("change_summary")
    optimization_summary = (
        [str(item) for item in summary]
        if isinstance(summary, list) and all(isinstance(item, str) for item in summary)
        else []
    )
    expected = response.get("expected_effect")
    expected_effect = (
        [str(item) for item in expected]
        if isinstance(expected, list) and all(isinstance(item, str) for item in expected)
        else []
    )
    assumptions = response.get("assumptions")
    synthetic_model_failure = (
        isinstance(assumptions, list)
        and "Model output failed structured-response validation." in assumptions
    )
    return {
        "schema_version": "ascend_candidate_generation_intent_v1",
        "optimization_summary": optimization_summary,
        "expected_effect": expected_effect,
        "source_field": source_field,
        "optimization_summary_model_authored": bool(optimization_summary)
        and not synthetic_model_failure,
        "candidate_response_provenance": (
            "synthetic_model_failure_sentinel"
            if synthetic_model_failure
            else "model_response"
        ),
    }


def _training_model_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a legacy response to the schema required from new models."""

    normalized = dict(response)
    legacy_summary = normalized.get("change_summary")
    if (
        "optimization_summary" not in normalized
        and isinstance(legacy_summary, list)
        and bool(legacy_summary)
    ):
        normalized["optimization_summary"] = normalized.pop("change_summary")
    return normalized


def _speedup(item: RoundArtifacts) -> float | None:
    score = _score(item)
    value = _number(score.get("geomean_speedup"))
    if value is not None:
        return value
    benchmark = item.evaluation.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    return _number(
        benchmark.get("geomean_speedup_vs_eager", benchmark.get("speedup_geomean"))
    )


def _minimum_speedup(item: RoundArtifacts) -> float | None:
    score = _score(item)
    value = _number(score.get("minimum_speedup"))
    if value is not None:
        return value
    benchmark = item.evaluation.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return None
    return _number(
        benchmark.get("minimum_speedup_vs_eager", benchmark.get("minimum_speedup"))
    )


def _is_publicly_valid(item: RoundArtifacts) -> bool:
    score = _score(item)
    return (
        score.get("compile_passed") is True
        and score.get("correctness_passed") is True
        and _speedup(item) is not None
        and _minimum_speedup(item) is not None
    )


def _profile_summary(item: RoundArtifacts) -> Mapping[str, Any]:
    profile = item.evaluation.get("profile")
    if not isinstance(profile, Mapping):
        return {}
    summary = profile.get("summary")
    return summary if isinstance(summary, Mapping) else profile


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {
                "bottleneck",
                "bottleneck_type",
                "bound_type",
                "limitation",
                "type",
                "suggestion",
            } or isinstance(item, (Mapping, list, tuple)):
                yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _host_bound_evidence(item: RoundArtifacts) -> tuple[tuple[str, ...], dict[str, Any]]:
    summary = _profile_summary(item)
    scheduling_value = summary.get("scheduling")
    scheduling = scheduling_value if isinstance(scheduling_value, Mapping) else {}
    host_us = _number(
        scheduling.get("host_enqueue_us", scheduling.get("host_overhead_us"))
    )
    device_us = _number(
        scheduling.get(
            "candidate_device_execution_us", scheduling.get("device_execution_us")
        )
    )
    metrics: dict[str, Any] = {
        "host_enqueue_us": host_us,
        "candidate_device_execution_us": device_us,
        "host_to_device_ratio": (
            host_us / device_us
            if host_us is not None and device_us is not None and device_us > 0
            else None
        ),
    }
    reasons: list[str] = []
    if (
        host_us is not None
        and device_us is not None
        and device_us > 0
        and host_us >= 2.0 * device_us
    ):
        reasons.append("quick profile reports host enqueue time at least 2x device time")

    benchmark_value = item.evaluation.get("benchmark")
    benchmark = benchmark_value if isinstance(benchmark_value, Mapping) else {}
    bottleneck_value = benchmark.get("bottleneck")
    bottleneck = bottleneck_value if isinstance(bottleneck_value, Mapping) else {}
    bottleneck_type = bottleneck.get(
        "bottleneck_type", benchmark.get("bottleneck_type")
    )
    host_dispatch_limited = (
        bottleneck.get(
            "host_dispatch_limited", benchmark.get("host_dispatch_limited")
        )
        is True
    )
    stop_host_bound = item.feedback.get("optimization_action") == "stop_host_bound"
    metrics.update(
        {
            "bottleneck_type": bottleneck_type,
            "host_dispatch_limited": host_dispatch_limited,
            "optimization_action": item.feedback.get("optimization_action"),
        }
    )
    if host_dispatch_limited or bottleneck_type == "host_dispatch":
        reasons.append("benchmark identifies a host-dispatch bottleneck")
    if stop_host_bound:
        reasons.append("feedback recommends stop_host_bound")

    evidence_text = "\n".join(
        _strings(
            {
                "benchmark": benchmark,
                "profile": summary,
                "feedback": item.feedback,
            }
        )
    ).lower()
    matched = sorted(term for term in _HOST_BOUND_TERMS if term in evidence_text)
    if matched:
        metrics["matched_bottleneck_terms"] = matched
        reasons.append("profile or feedback explicitly identifies a host/dispatch bottleneck")
    return tuple(reasons), metrics


class DatasetExporter:
    """Build training datasets exclusively from committed, immutable artifacts."""

    def __init__(self, experiment_root: Path | str) -> None:
        self.root = Path(experiment_root).resolve()
        if not self.root.is_dir():
            raise ExportError(f"experiment root does not exist: {self.root}")

    def rounds(self) -> tuple[RoundArtifacts, ...]:
        items: list[RoundArtifacts] = []
        tasks_root = self.root / "tasks"
        if not tasks_root.is_dir():
            return ()
        for task_root in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
            if task_root.is_symlink():
                raise ExportError(f"refusing to traverse symlinked task directory {task_root}")
            round_roots = sorted(
                (
                    (int(match.group(1)), round_root)
                    for round_root in task_root.iterdir()
                    if (
                        (match := re.fullmatch(r"round_(\d+)", round_root.name))
                        is not None
                    )
                ),
                key=lambda value: value[0],
            )
            for round_id, round_root in round_roots:
                if round_root.is_symlink():
                    raise ExportError(
                        f"refusing to traverse symlinked round directory {round_root}"
                    )
                code_path = round_root / "candidate.py"
                if code_path.is_symlink():
                    raise ExportError(f"refusing to read symlinked candidate {code_path}")
                if not code_path.is_file():
                    continue
                prompt = _read_json(round_root / "prompt.json", {})
                response = _read_json(round_root / "model_response.json", None)
                if response is None:
                    response = _read_json(round_root / "raw_response.json", {})
                evaluation = _read_json(round_root / "evaluation_result.json", {})
                feedback = _read_json(round_root / "feedback.json", {})
                reward = _read_json(round_root / "reward.json", evaluation.get("reward_vector", {}))
                items.append(RoundArtifacts(
                    task_id=task_root.name,
                    round_id=round_id,
                    root=round_root,
                    prompt=prompt,
                    response=response,
                    evaluation=evaluation,
                    feedback=feedback,
                    reward=reward,
                    code=code_path.read_text(encoding="utf-8"),
                ))
        return tuple(items)

    @staticmethod
    def _classify(
        current: RoundArtifacts,
        previous: RoundArtifacts | None,
        previous_best_speedup: float | None,
    ) -> RoundQuality:
        overall = _overall_status(current)
        current_speedup = _speedup(current)
        metrics: dict[str, Any] = {
            "overall_status": overall,
            "geomean_speedup_vs_eager": current_speedup,
            "minimum_speedup_vs_eager": _minimum_speedup(current),
            "previous_best_geomean_speedup_vs_eager": previous_best_speedup,
            "change_vs_previous_best_percent": (
                (current_speedup / previous_best_speedup - 1.0) * 100.0
                if current_speedup is not None
                and previous_best_speedup is not None
                and previous_best_speedup > 0
                else None
            ),
        }
        failure_label = _FAILURE_LABELS.get(overall)
        if failure_label is not None:
            return RoundQuality(
                label=failure_label,
                eligible_for_default_sft=False,
                reasons=(f"evaluation ended with {overall}",),
                metrics=metrics,
            )
        if not _is_publicly_valid(current):
            return RoundQuality(
                label="invalid_or_incomplete",
                eligible_for_default_sft=False,
                reasons=("candidate did not produce a complete publicly valid score",),
                metrics=metrics,
            )

        previous_status = _overall_status(previous) if previous is not None else None
        if previous_status in _REPAIRABLE_FAILURES:
            metrics["repaired_from"] = previous_status
            return RoundQuality(
                label="successful_repair",
                eligible_for_default_sft=True,
                reasons=(f"publicly valid candidate repairs prior {previous_status}",),
                metrics=metrics,
            )

        host_reasons, host_metrics = _host_bound_evidence(current)
        metrics.update(host_metrics)
        if host_reasons:
            return RoundQuality(
                label="host_bound_unoptimizable",
                eligible_for_default_sft=False,
                reasons=host_reasons,
                metrics=metrics,
            )

        if current_speedup is not None and previous_best_speedup is not None:
            if current_speedup >= previous_best_speedup * 1.03:
                return RoundQuality(
                    label="high_quality_optimization",
                    eligible_for_default_sft=True,
                    reasons=("geomean speedup improved at least 3% over prior best",),
                    metrics=metrics,
                )
            if current_speedup <= previous_best_speedup * 0.97:
                return RoundQuality(
                    label="performance_regression",
                    eligible_for_default_sft=False,
                    reasons=("geomean speedup regressed at least 3% from prior best",),
                    metrics=metrics,
                )
            return RoundQuality(
                label="valid_but_not_improved",
                eligible_for_default_sft=False,
                reasons=("publicly valid result did not improve prior best by 3%",),
                metrics=metrics,
            )

        if current_speedup is not None and current_speedup >= 1.03:
            return RoundQuality(
                label="high_quality_optimization",
                eligible_for_default_sft=True,
                reasons=("first valid candidate beats PyTorch eager by at least 3%",),
                metrics=metrics,
            )
        return RoundQuality(
            label="initial_correct_candidate",
            eligible_for_default_sft=True,
            reasons=(
                "first publicly valid candidate is a useful cold-start correctness anchor",
            ),
            metrics=metrics,
        )

    def classified_rounds(self) -> tuple[ClassifiedRound, ...]:
        previous_by_task: dict[str, RoundArtifacts] = {}
        best_speedup_by_task: dict[str, float] = {}
        classified: list[ClassifiedRound] = []
        for item in self.rounds():
            previous = previous_by_task.get(item.task_id)
            previous_best = best_speedup_by_task.get(item.task_id)
            quality = self._classify(item, previous, previous_best)
            classified.append(ClassifiedRound(artifacts=item, quality=quality))
            previous_by_task[item.task_id] = item
            speedup = _speedup(item)
            if _is_publicly_valid(item) and speedup is not None:
                best_speedup_by_task[item.task_id] = max(
                    speedup, best_speedup_by_task.get(item.task_id, speedup)
                )
        return tuple(classified)

    def _passes_main_validation(
        self, item: RoundArtifacts, quality: RoundQuality
    ) -> bool:
        if not quality.eligible_for_default_sft:
            return False
        final = _read_json(
            self.root / "tasks" / item.task_id / "final_result.json", {}
        )
        if (
            not isinstance(final, Mapping)
            or not str(final.get("status", "")).lower().startswith("passed")
            or final.get("hidden_correctness_passed") is not True
        ):
            return False
        score = _score(item)
        if not (
            score.get("compile_passed") is True
            and score.get("correctness_passed") is True
        ):
            return False
        stability = _number(score.get("stability_cv"))
        return stability is None or stability <= 0.05

    def quality_summary(self) -> dict[str, Any]:
        classified = self.classified_rounds()
        counts = Counter(item.quality.label for item in classified)
        by_task: dict[str, dict[str, Any]] = {}
        for item in classified:
            task_id = item.artifacts.task_id
            task = by_task.setdefault(
                task_id,
                {
                    "round_count": 0,
                    "curated_category_round_count": 0,
                    "default_sft_row_count": 0,
                    "counts": Counter(),
                },
            )
            task["round_count"] += 1
            task["counts"][item.quality.label] += 1
            if item.quality.eligible_for_default_sft:
                task["curated_category_round_count"] += 1
            if self._passes_main_validation(item.artifacts, item.quality):
                task["default_sft_row_count"] += 1
        normalized_tasks = {
            task_id: {
                **{key: value for key, value in task.items() if key != "counts"},
                "counts": dict(sorted(task["counts"].items())),
            }
            for task_id, task in sorted(by_task.items())
        }
        return {
            "schema_version": "ascend_trajectory_quality_summary_v1",
            "round_count": len(classified),
            "curated_category_round_count": sum(
                item.quality.eligible_for_default_sft for item in classified
            ),
            "default_sft_row_count": sum(
                self._passes_main_validation(item.artifacts, item.quality)
                for item in classified
            ),
            "counts": dict(sorted(counts.items())),
            "tasks": normalized_tasks,
        }

    def export_sft(self, output: Path | str, *, main_only: bool = True) -> int:
        rows: list[dict[str, Any]] = []
        for classified in self.classified_rounds():
            item = classified.artifacts
            quality = classified.quality
            if main_only and not self._passes_main_validation(item, quality):
                continue
            final = _read_json(
                self.root / "tasks" / item.task_id / "final_result.json", {}
            )
            score = item.evaluation.get("score", {})
            intent = candidate_generation_intent(item.response)
            rows.append({
                "schema_version": "ascend_kernel_sft_v2",
                "sample_id": f"{self.root.name}:{item.task_id}:round-{item.round_id:02d}",
                "sample_type": quality.label,
                "candidate_generation_intent": intent,
                "messages": [
                    {"role": "user", "content": json.dumps(item.prompt, ensure_ascii=False, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps({**_training_model_response(item.response), "code": item.code}, ensure_ascii=False, sort_keys=True)},
                ],
                "quality": {
                    **quality.to_dict(),
                    "reward_vector": dict(item.reward),
                    "score": dict(score) if isinstance(score, Mapping) else {},
                    "final_hidden_correctness_passed": (
                        final.get("hidden_correctness_passed")
                        if isinstance(final, Mapping)
                        else None
                    ),
                    "selected_as_best": (
                        item.round_id == final.get("best_round")
                        if isinstance(final, Mapping)
                        else False
                    ),
                    "best_turn": (
                        final.get("best_round")
                        if isinstance(final, Mapping)
                        else None
                    ),
                },
            })
        return _atomic_jsonl(Path(output), rows)

    def export_rl(self, output: Path | str) -> int:
        rounds = self.classified_rounds()
        task_best: dict[str, int | None] = {}
        for classified in rounds:
            item = classified.artifacts
            final = _read_json(self.root / "tasks" / item.task_id / "final_result.json", {})
            task_best[item.task_id] = final.get("best_round")
        task_last: dict[str, int] = {}
        for classified in rounds:
            item = classified.artifacts
            task_last[item.task_id] = max(task_last.get(item.task_id, 0), item.round_id)
        rows: list[dict[str, Any]] = []
        for classified in rounds:
            item = classified.artifacts
            intent = candidate_generation_intent(item.response)
            rows.append({
                "schema_version": "ascend_kernel_rl_transition_v2",
                "episode_id": f"{self.root.name}:{item.task_id}",
                "turn": item.round_id,
                "observation": item.prompt,
                "action": {
                    "raw_model_response": item.response,
                    "candidate_code": item.code,
                    "candidate_generation_intent": intent,
                },
                "result": item.evaluation,
                "feedback": item.feedback,
                "reward_vector": item.reward,
                "quality": classified.quality.to_dict(),
                "done": item.round_id == task_last[item.task_id],
                "selected_as_best": item.round_id == task_best.get(item.task_id),
                "best_turn": task_best.get(item.task_id),
            })
        return _atomic_jsonl(Path(output), rows)
