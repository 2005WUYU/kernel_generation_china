from __future__ import annotations

import csv
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


_ALIASES: Mapping[str, tuple[str, ...]] = {
    "kernel_name": ("opname", "kernelname", "name", "算子名称", "kernel名称"),
    "duration_us": ("durationus", "taskdurationus", "aicoretimeus", "duration", "任务时长us"),
    "duration_ns": ("durationns", "taskdurationns"),
    "block_dim": ("blockdim", "blockdimension", "block维度"),
    "vector_ratio": (
        "vectorratio", "vectorutilization", "vector占比", "aivvecratio",
    ),
    "cube_ratio": (
        "cuberatio", "cubeutilization", "cube占比", "aiccuberatio",
    ),
    "scalar_ratio": (
        "scalarratio", "scalarutilization", "scalar占比",
        "aivscalarratio", "aicscalarratio",
    ),
    "mte2_ratio": (
        "mte2ratio", "mte2utilization", "mte2占比",
        "aivmte2ratio", "aicmte2ratio",
    ),
    "mte3_ratio": (
        "mte3ratio", "mte3utilization", "mte3占比",
        "aivmte3ratio", "aicmte3ratio",
    ),
    "gm_read_gbps": (
        "gmreadgbps", "gmreadbandwidthgbps", "aivgmtoubbwgbps",
        "aiv_gm_to_ub_bw(GB/s)", "aiv_main_mem_read_bw(GB/s)",
        "aic_main_mem_read_bw(GB/s)",
    ),
    "gm_write_gbps": (
        "gmwritegbps", "gmwritebandwidthgbps", "aivubtogmbwgbps",
        "aiv_ub_to_gm_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)",
        "aic_main_mem_write_bw(GB/s)",
    ),
    "ub_read_gbps": (
        "ubreadgbps", "ubreadbandwidthgbps", "aivubreadbwvectorgbps",
        "aiv_ub_read_bw_vector(GB/s)", "aiv_ub_read_bw_scalar(GB/s)",
    ),
    "ub_write_gbps": (
        "ubwritegbps", "ubwritebandwidthgbps", "aivubwritebwvectorgbps",
        "aiv_ub_write_bw_vector(GB/s)", "aiv_ub_write_bw_scalar(GB/s)",
    ),
    "l2_hit_rate": (
        "l2hit", "l2hitrate", "l2cachehitrate",
        "aivtotalhitrate", "aictotalhitrate",
    ),
    "host_enqueue_us": ("hostenqueueus", "hosttaskdurationus", "hostdurationus"),
}


def _find_column(headers: Sequence[str], metric: str) -> str | None:
    columns = _find_columns(headers, metric)
    return columns[0] if columns else None


def _find_columns(headers: Sequence[str], metric: str) -> tuple[str, ...]:
    normalized = {_key(header): header for header in headers}
    return tuple(dict.fromkeys(
        normalized[key]
        for alias in _ALIASES[metric]
        if (key := _key(alias)) in normalized
    ))


def _number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if not text or text.lower() in {"n/a", "na", "nan", "null", "--"}:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _ratio(value: Any) -> float | None:
    text = str(value).strip() if value is not None else ""
    number = _number(value)
    if number is None:
        return None
    result = number / 100.0 if "%" in text or number > 1.0 else number
    return result if 0 <= result <= 1 else None


@dataclass(frozen=True)
class KernelRecord:
    kernel_name: str
    duration_us: float
    block_dim: int | None = None


@dataclass(frozen=True)
class Observation:
    type: str
    confidence: float
    evidence: Mapping[str, Any]
    suggestion: str


@dataclass(frozen=True)
class ProfileSummary:
    profile_available: bool
    executed_candidate_kernels: tuple[KernelRecord, ...] = ()
    kernel_count: int = 0
    candidate_kernel_coverage: float | None = None
    pipeline: Mapping[str, float | None] = field(default_factory=dict)
    memory: Mapping[str, float | None] = field(default_factory=dict)
    scheduling: Mapping[str, float | None] = field(default_factory=dict)
    observations: tuple[Observation, ...] = ()
    # source_files are schema-usable operation tables. readable_source_files
    # also includes CSVs that could be decoded but did not expose the required
    # kernel-name and duration columns.
    source_files: tuple[str, ...] = ()
    readable_source_files: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MsprofParser:
    """Tolerant parser for version-varying msprof CSV artifacts.

    Column matching is alias based. Unknown or unsupported values remain ``None``;
    the parser deliberately does not infer hardware counters that were not emitted.
    """

    def __init__(self, candidate_patterns: Iterable[str]) -> None:
        patterns = tuple(candidate_patterns)
        if not patterns:
            raise ValueError("at least one candidate kernel pattern is required")
        self._patterns = tuple(re.compile(pattern) for pattern in patterns)

    def discover(self, root: Path | str) -> tuple[Path, ...]:
        base = Path(root).resolve()
        if not base.is_dir():
            return ()
        matches = []
        for path in base.rglob("*.csv"):
            resolved = path.resolve()
            if resolved.is_relative_to(base) and path.is_file() and not path.is_symlink():
                matches.append(path)
        return tuple(sorted(matches))

    def parse(self, root: Path | str) -> ProfileSummary:
        files = self.discover(root)
        if not files:
            return ProfileSummary(profile_available=False, unavailable_reason="no msprof CSV files found")
        readable: list[str] = []
        usable: list[str] = []
        normalized_rows: list[dict[str, Any]] = []
        split_metric_samples: dict[str, list[float]] = {
            metric: []
            for metric in _ALIASES
            if metric not in {"kernel_name", "duration_us", "duration_ns"}
        }
        ratio_metrics = {
            "vector_ratio",
            "cube_ratio",
            "scalar_ratio",
            "mte2_ratio",
            "mte3_ratio",
            "l2_hit_rate",
        }
        for path in files:
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue
                    readable.append(str(path))
                    all_columns = {
                        metric: _find_columns(tuple(reader.fieldnames), metric)
                        for metric in _ALIASES
                    }
                    name_column = _find_column(tuple(reader.fieldnames), "kernel_name")
                    duration_column = _find_column(tuple(reader.fieldnames), "duration_us")
                    duration_ns_column = _find_column(tuple(reader.fieldnames), "duration_ns")
                    operation_table = name_column is not None and (
                        duration_column is not None or duration_ns_column is not None
                    )
                    if operation_table:
                        usable.append(str(path))
                    for row in reader:
                        metrics: dict[str, float | None] = {}
                        for metric, columns in all_columns.items():
                            if metric in {"kernel_name", "duration_us", "duration_ns"}:
                                continue
                            values = []
                            for column in columns:
                                value = (
                                    _ratio(row.get(column))
                                    if metric in ratio_metrics
                                    else _number(row.get(column))
                                )
                                if value is not None:
                                    values.append(value)
                                    if not operation_table:
                                        split_metric_samples[metric].append(value)
                            metrics[metric] = (
                                sum(values) / len(values) if values else None
                            )
                        if not operation_table:
                            continue
                        assert name_column is not None
                        duration = (
                            _number(row.get(duration_column))
                            if duration_column
                            else None
                        )
                        if duration is None and duration_ns_column:
                            nanoseconds = _number(row.get(duration_ns_column))
                            duration = (
                                nanoseconds / 1000.0
                                if nanoseconds is not None
                                else None
                        )
                        if duration is None or duration < 0:
                            continue
                        normalized_rows.append(
                            {
                                "kernel_name": str(row.get(name_column, "")).strip(),
                                "duration_us": duration,
                                "metrics": metrics,
                            }
                        )
            except (UnicodeDecodeError, csv.Error, OSError):
                continue
        if not normalized_rows:
            return ProfileSummary(
                profile_available=False,
                source_files=tuple(usable),
                readable_source_files=tuple(readable),
                unavailable_reason=(
                    "readable msprof CSV files had no usable kernel name/duration rows"
                ),
            )

        records: list[KernelRecord] = []
        total_device_us = 0.0
        candidate_device_us = 0.0
        candidate_rows: list[Mapping[str, Any]] = []
        for row in normalized_rows:
            duration = float(row["duration_us"])
            total_device_us += duration
            name = str(row["kernel_name"])
            if any(pattern.search(name) for pattern in self._patterns):
                metrics = row["metrics"]
                assert isinstance(metrics, Mapping)
                block = metrics.get("block_dim")
                records.append(KernelRecord(name, duration, int(block) if block is not None else None))
                candidate_device_us += duration
                candidate_rows.append(metrics)

        coverage = candidate_device_us / total_device_us if total_device_us > 0 else None

        def average(metric: str) -> float | None:
            if not candidate_rows:
                return None
            row_values = [row.get(metric) for row in candidate_rows]
            present = [value for value in row_values if value is not None]
            present.extend(split_metric_samples.get(metric, ()))
            return (
                sum(float(value) for value in present) / len(present)
                if present
                else None
            )

        pipeline = {
            name: average(name)
            for name in (
                "vector_ratio",
                "cube_ratio",
                "scalar_ratio",
                "mte2_ratio",
                "mte3_ratio",
            )
        }
        memory = {
            name: average(name)
            for name in (
                "gm_read_gbps",
                "gm_write_gbps",
                "ub_read_gbps",
                "ub_write_gbps",
                "l2_hit_rate",
            )
        }
        scheduling = {
            "host_enqueue_us": average("host_enqueue_us"),
            "device_execution_us": candidate_device_us if records else None,
        }
        observations = self._observations(records, pipeline, coverage)
        return ProfileSummary(
            profile_available=True,
            executed_candidate_kernels=tuple(records),
            kernel_count=len(records),
            candidate_kernel_coverage=coverage,
            pipeline=pipeline,
            memory=memory,
            scheduling=scheduling,
            observations=observations,
            source_files=tuple(usable),
            readable_source_files=tuple(readable),
        )

    @staticmethod
    def _observations(
        records: Sequence[KernelRecord],
        pipeline: Mapping[str, float | None],
        coverage: float | None,
    ) -> tuple[Observation, ...]:
        observations: list[Observation] = []
        scalar = pipeline.get("scalar_ratio")
        if scalar is not None and scalar >= 0.15:
            observations.append(Observation(
                type="scalar_operations_high",
                confidence=min(0.98, 0.65 + scalar),
                evidence={"scalar_ratio": scalar},
                suggestion="Inspect index arithmetic and comparisons for scalar fallback.",
            ))
        if len(records) >= 4:
            observations.append(Observation(
                type="many_candidate_kernels",
                confidence=0.8,
                evidence={"kernel_count": len(records)},
                suggestion="Consider fusing short kernels to reduce launch and intermediate-memory overhead.",
            ))
        if coverage is not None and coverage < 0.9:
            observations.append(Observation(
                type="candidate_coverage_low",
                confidence=0.95,
                evidence={"candidate_kernel_coverage": coverage},
                suggestion="Remove high-level operator fallback; candidate Triton kernels must own the computation.",
            ))
        return tuple(observations)
