from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .datasets import ExportError, assert_export_clean


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

    def build(self) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = []
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
                }
                tasks.append(
                    {
                        "task_id": root.name,
                        **result,
                        "public_best": public_best,
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
            "schema_version": "ascend_kernel_experiment_report_v1",
            "experiment_id": self.root.name,
            "task_count": len(tasks),
            "passed_task_count": passed,
            "tasks": tasks,
            "environment": environment,
            "baseline": self._read(self.root / "baseline_snapshot.json"),
        }

    def write(self, json_path: Path | str, markdown_path: Path | str | None = None) -> None:
        report = self.build()
        self._atomic(Path(json_path), json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if markdown_path is not None:
            lines = [
                f"# Experiment {report['experiment_id']}",
                "",
                f"Passed tasks: {report['passed_task_count']} / {report['task_count']}",
                "",
                "| Task | Status | Best round | vs PyTorch eager | Coverage |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for task in report["tasks"]:
                public_best = task.get("public_best", {})
                lines.append(
                    f"| {task['task_id']} | {task.get('status', 'unknown')} | "
                    f"{task.get('best_round', '-')} | "
                    f"{public_best.get('geomean_speedup_vs_eager', '-')} | "
                    f"{public_best.get('candidate_kernel_coverage', '-')} |"
                )
            self._atomic(Path(markdown_path), "\n".join(lines) + "\n")

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
