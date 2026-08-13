from __future__ import annotations

import json
import os
import re
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
            for round_root in sorted(task_root.glob("round_[0-9][0-9]")):
                match = re.fullmatch(r"round_(\d+)", round_root.name)
                if not match:
                    continue
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
                    round_id=int(match.group(1)),
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
    def _quality_type(current: RoundArtifacts, previous: RoundArtifacts | None) -> str | None:
        overall = str(current.feedback.get("overall_status", current.evaluation.get("overall_status", "")))
        if current.round_id == 1 and overall not in {"model_failed", "source_failed"}:
            return "initial_generation"
        if previous is None:
            return None
        previous_status = str(previous.feedback.get("overall_status", previous.evaluation.get("overall_status", "")))
        if previous_status in {"compile_failed", "source_failed"} and overall not in {"compile_failed", "source_failed"}:
            return "compile_repair"
        if previous_status == "correctness_failed" and overall in {"correct", "success", "profile_unavailable"}:
            return "correctness_repair"
        previous_score = previous.evaluation.get("score", {})
        current_score = current.evaluation.get("score", {})
        previous_speedup = float(
            previous_score.get("geomean_speedup", 0.0)
            if isinstance(previous_score, Mapping)
            else 0.0
        )
        speedup = float(
            current_score.get("geomean_speedup", 0.0)
            if isinstance(current_score, Mapping)
            else 0.0
        )
        if previous_speedup > 0 and speedup >= previous_speedup * 1.03:
            return "performance_optimization"
        return None

    def export_sft(self, output: Path | str, *, main_only: bool = True) -> int:
        rounds = self.rounds()
        by_task: dict[str, RoundArtifacts] = {}
        rows: list[dict[str, Any]] = []
        for item in rounds:
            previous = by_task.get(item.task_id)
            sample_type = self._quality_type(item, previous)
            by_task[item.task_id] = item
            if sample_type is None:
                if main_only:
                    continue
                sample_type = "cold_start_trajectory"
            final = _read_json(
                self.root / "tasks" / item.task_id / "final_result.json", {}
            )
            score = item.evaluation.get("score", {})
            if main_only:
                if (
                    not isinstance(final, Mapping)
                    or str(final.get("status", "")).lower() != "passed"
                    or final.get("hidden_correctness_passed") is not True
                ):
                    continue
                score = item.evaluation.get("score", {})
                if not isinstance(score, Mapping):
                    continue
                if not (
                    score.get("compile_passed") is True
                    and score.get("correctness_passed") is True
                    and score.get("anti_bypass_passed") is True
                ):
                    continue
                stability = score.get("stability_cv")
                if stability is not None and float(stability) > 0.05:
                    continue
            rows.append({
                "schema_version": "ascend_kernel_sft_v1",
                "sample_id": f"{self.root.name}:{item.task_id}:round-{item.round_id:02d}",
                "sample_type": sample_type,
                "messages": [
                    {"role": "user", "content": json.dumps(item.prompt, ensure_ascii=False, sort_keys=True)},
                    {"role": "assistant", "content": json.dumps({**item.response, "code": item.code}, ensure_ascii=False, sort_keys=True)},
                ],
                "quality": {
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
        rounds = self.rounds()
        task_best: dict[str, int | None] = {}
        for item in rounds:
            final = _read_json(self.root / "tasks" / item.task_id / "final_result.json", {})
            task_best[item.task_id] = final.get("best_round")
        task_last: dict[str, int] = {}
        for item in rounds:
            task_last[item.task_id] = max(task_last.get(item.task_id, 0), item.round_id)
        rows = ({
            "schema_version": "ascend_kernel_rl_transition_v1",
            "episode_id": f"{self.root.name}:{item.task_id}",
            "turn": item.round_id,
            "observation": item.prompt,
            "action": {"raw_model_response": item.response, "candidate_code": item.code},
            "result": item.evaluation,
            "feedback": item.feedback,
            "reward_vector": item.reward,
            "done": item.round_id == task_last[item.task_id],
            "selected_as_best": item.round_id == task_best.get(item.task_id),
            "best_turn": task_best.get(item.task_id),
        } for item in rounds)
        return _atomic_jsonl(Path(output), rows)
