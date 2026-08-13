from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .datasets import (
    DatasetExporter,
    ExportError,
    assert_export_clean,
    candidate_generation_intent,
)


class ReportExporter:
    def __init__(self, experiment_root: Path | str) -> None:
        self.root = Path(experiment_root).resolve()

    @staticmethod
    def _read(path: Path) -> Mapping[str, Any]:
        if path.is_symlink():
            raise ExportError(f"refusing to read symlinked artifact {path}")
        if not path.is_file():
            return {}
        raw = path.read_bytes()
        if len(raw) > 32 * 1024**2:
            raise ExportError(f"JSON artifact exceeds 32 MiB: {path}")
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _round_roots(task_root: Path) -> list[Path]:
        values: list[tuple[int, Path]] = []
        for path in task_root.iterdir():
            match = re.fullmatch(r"round_(\d+)", path.name)
            if match is not None and path.is_dir() and not path.is_symlink():
                values.append((int(match.group(1)), path))
        return [path for _number, path in sorted(values)]

    def _trajectory_rounds(
        self,
        task_root: Path,
        result: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        phase_counts = {
            "repair": 0,
            "optimization": 0,
            "optimization_repair": 0,
            "unknown": 0,
        }
        rounds = self._round_roots(task_root)
        for round_root in rounds:
            evaluation = self._read(round_root / "evaluation_result.json")
            phase = evaluation.get("trajectory_phase")
            if isinstance(phase, str) and phase in {
                "repair",
                "optimization",
                "optimization_repair",
            }:
                phase_counts[phase] += 1
            else:
                phase_counts["unknown"] += 1

        optimization_budget = manifest.get("rounds_per_task")
        repair_budget = manifest.get("maximum_repair_rounds", 0)
        maximum_physical_rounds = None
        if (
            isinstance(optimization_budget, int)
            and optimization_budget > 0
            and isinstance(repair_budget, int)
            and repair_budget >= 0
        ):
            maximum_physical_rounds = repair_budget + optimization_budget * (
                repair_budget + 1
            )
        return {
            "actual_physical_rounds": len(rounds),
            "maximum_physical_rounds": maximum_physical_rounds,
            "initial_repair_rounds": phase_counts["repair"],
            "optimization_slots_attempted": phase_counts["optimization"],
            "optimization_repair_rounds": phase_counts[
                "optimization_repair"
            ],
            "unknown_phase_rounds": phase_counts["unknown"],
            "reported_initial_repair_rounds": result.get("repair_rounds"),
            "reported_optimization_slots": result.get("optimization_rounds"),
        }

    def build(self) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
        experiment = self._read(self.root / "experiment.json")
        manifest_value = experiment.get("experiment")
        manifest = (
            manifest_value if isinstance(manifest_value, Mapping) else {}
        )
        tasks_root = self.root / "tasks"
        if tasks_root.is_dir():
            for root in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
                if root.is_symlink():
                    raise ExportError(f"refusing to traverse symlinked task directory {root}")
                result = self._read(root / "final_result.json")
                best_round = result.get("best_round")
                public_evaluation: Mapping[str, Any] = {}
                if isinstance(best_round, int) and best_round > 0:
                    public_evaluation = self._read(
                        root
                        / f"round_{best_round:02d}"
                        / "evaluation_result.json"
                    )
                    model_response = self._read(
                        root / f"round_{best_round:02d}" / "model_response.json"
                    )
                else:
                    model_response = {}
                benchmark_value = public_evaluation.get("benchmark")
                benchmark = (
                    benchmark_value
                    if isinstance(benchmark_value, Mapping)
                    else {}
                )
                score_value = public_evaluation.get("score")
                score = score_value if isinstance(score_value, Mapping) else {}
                public_best = {
                    "overall_status": public_evaluation.get("overall_status"),
                    "geomean_speedup_vs_eager": benchmark.get(
                        "geomean_speedup_vs_eager"
                    ),
                    "minimum_speedup_vs_eager": benchmark.get(
                        "minimum_speedup_vs_eager"
                    ),
                    "candidate_kernel_coverage": score.get(
                        "candidate_kernel_coverage"
                    ),
                    "stability_cv": score.get("stability_cv"),
                    "candidate_generation_intent": candidate_generation_intent(
                        model_response
                    ),
                }
                tasks.append(
                    {
                        "task_id": root.name,
                        **result,
                        "public_best": public_best,
                        "trajectory_rounds": self._trajectory_rounds(
                            root, result, manifest
                        ),
                    }
                )
        passed = sum(
            str(task.get("status", "")).lower().startswith("passed")
            or str(task.get("status", "")).lower() in {"success", "finished"}
            for task in tasks
        )
        environment = self._read(self.root / "environment_snapshot.json")
        if not environment:
            environment = self._read(self.root / "env_manifest.json")
        return {
            "schema_version": "ascend_kernel_experiment_report_v2",
            "experiment_id": self.root.name,
            "task_count": len(tasks),
            "passed_task_count": passed,
            "tasks": tasks,
            "trajectory_quality_summary": DatasetExporter(
                self.root
            ).quality_summary(),
            "environment": environment,
            "baseline": self._read(self.root / "baseline_snapshot.json"),
        }

    def write(
        self, json_path: Path | str, markdown_path: Path | str | None = None
    ) -> dict[str, Any]:
        report = self.build()
        self._atomic(Path(json_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if markdown_path is not None:
            lines = [
                f"# Experiment {report['experiment_id']}",
                "",
                f"Passed tasks: {report['passed_task_count']} / {report['task_count']}",
                "",
                "| Task | Status | Best round | Trajectory | vs PyTorch eager | Coverage |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
            for task in report["tasks"]:
                public_best = task.get("public_best", {})
                trajectory = task.get("trajectory_rounds", {})
                actual = trajectory.get("actual_physical_rounds", "-")
                maximum = trajectory.get("maximum_physical_rounds")
                trajectory_text = (
                    f"{actual}/{maximum}" if maximum is not None else str(actual)
                )
                lines.append(
                    f"| {task['task_id']} | {task.get('status', 'unknown')} | "
                    f"{task.get('best_round', '-')} | "
                    f"{trajectory_text} | "
                    f"{public_best.get('geomean_speedup_vs_eager', '-')} | "
                    f"{public_best.get('candidate_kernel_coverage', '-')} |"
                )
            quality = report["trajectory_quality_summary"]
            lines.extend(
                [
                    "",
                    "## Trajectory quality",
                    "",
                    f"Default SFT rows: {quality['default_sft_row_count']} / "
                    f"{quality['round_count']}",
                    "",
                    "| Quality class | Rounds |",
                    "| --- | ---: |",
                ]
            )
            for label, count in quality["counts"].items():
                lines.append(f"| {label} | {count} |")
            self._atomic(Path(markdown_path), "\n".join(lines) + "\n")
        return report

    @staticmethod
    def _atomic(path: Path, text: str) -> None:
        assert_export_clean(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
