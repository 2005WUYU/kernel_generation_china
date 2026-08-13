from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("samples must be finite and non-negative")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass(frozen=True)
class BenchmarkStatistics:
    median_us: float
    p20_us: float
    p80_us: float
    mean_us: float
    standard_deviation_us: float
    cv: float
    sample_count: int
    raw_samples_us: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_samples(samples_us: Iterable[float]) -> BenchmarkStatistics:
    samples = tuple(float(value) for value in samples_us)
    if not samples:
        raise ValueError("samples must not be empty")
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("samples must be finite and positive")
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return BenchmarkStatistics(
        median_us=statistics.median(samples),
        p20_us=percentile(samples, 0.2),
        p80_us=percentile(samples, 0.8),
        mean_us=mean,
        standard_deviation_us=deviation,
        cv=deviation / mean if mean else 0.0,
        sample_count=len(samples),
        raw_samples_us=samples,
    )


def weighted_geometric_mean(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    if not values:
        raise ValueError("values must not be empty")
    weights = tuple(1.0 for _ in values) if weights is None else tuple(float(weight) for weight in weights)
    if len(weights) != len(values):
        raise ValueError("values and weights must have the same length")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean values must be finite and positive")
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("weights must be finite and positive")
    total = sum(weights)
    return math.exp(
        sum(
            weight * math.log(value)
            for value, weight in zip(values, weights, strict=True)
        )
        / total
    )


def speedup_summary(per_case: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in per_case if item.get("speedup_vs_eager") is not None]
    if not valid:
        return {"geomean_speedup_vs_eager": None, "minimum_speedup_vs_eager": None, "maximum_speedup_vs_eager": None}
    speeds = [float(item["speedup_vs_eager"]) for item in valid]
    weights = [float(item.get("weight", 1.0)) for item in valid]
    return {
        "geomean_speedup_vs_eager": weighted_geometric_mean(speeds, weights),
        "minimum_speedup_vs_eager": min(speeds),
        "maximum_speedup_vs_eager": max(speeds),
    }


def classify_bottleneck(
    *, device_latency_us: float, end_to_end_latency_us: float
) -> str:
    """Classify whether a single candidate call is host- or device-limited."""
    device = float(device_latency_us)
    end_to_end = float(end_to_end_latency_us)
    if not math.isfinite(device) or device <= 0:
        raise ValueError("device latency must be finite and positive")
    if not math.isfinite(end_to_end) or end_to_end <= 0:
        raise ValueError("end-to-end latency must be finite and positive")
    end_to_end = max(end_to_end, device)
    host_fraction = (end_to_end - device) / end_to_end
    if host_fraction >= 0.5:
        return "host_dispatch"
    if host_fraction <= 0.2:
        return "device_execution"
    return "mixed"


def summarize_latency_breakdown(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired NPU-event and synchronized wall-clock measurements."""
    end_to_end_samples = [
        float(value["end_to_end_latency_us"])
        for value in measurements
        if value.get("end_to_end_latency_us") is not None
    ]
    if not end_to_end_samples:
        raise ValueError("latency measurements must include end-to-end latency")
    end_to_end = statistics.median(end_to_end_samples)
    device_samples = [
        float(value["device_latency_us"])
        for value in measurements
        if value.get("device_latency_us") is not None
    ]
    if not device_samples:
        return {
            "device_latency_us": None,
            "end_to_end_latency_us": end_to_end,
            "host_overhead_us": None,
            "host_overhead_fraction": None,
            "bottleneck_type": "unknown",
            "sample_count": len(end_to_end_samples),
            "timing_source": "synchronized_wall_clock_only",
            "method": "calibrated_batch_latency_breakdown_v1",
        }
    device = statistics.median(device_samples)
    end_to_end = max(end_to_end, device)
    host_overhead = end_to_end - device
    return {
        "device_latency_us": device,
        "end_to_end_latency_us": end_to_end,
        "host_overhead_us": host_overhead,
        "host_overhead_fraction": host_overhead / end_to_end,
        "bottleneck_type": classify_bottleneck(
            device_latency_us=device,
            end_to_end_latency_us=end_to_end,
        ),
        "sample_count": len(end_to_end_samples),
        "timing_source": "npu_event_and_synchronized_wall_clock",
        "method": "calibrated_batch_latency_breakdown_v1",
    }


def bottleneck_summary(per_case: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case bottlenecks without averaging unrelated shapes."""
    counts = {
        "host_dispatch": 0,
        "device_execution": 0,
        "mixed": 0,
        "unknown": 0,
    }
    weighted = {name: 0.0 for name in counts}
    total_weight = 0.0
    for item in per_case:
        candidate = item.get("candidate")
        latency = (
            candidate.get("latency_breakdown")
            if isinstance(candidate, Mapping)
            else None
        )
        kind = (
            str(latency.get("bottleneck_type", "unknown"))
            if isinstance(latency, Mapping)
            else "unknown"
        )
        if kind not in counts:
            kind = "unknown"
        weight = float(item.get("weight", 1.0))
        counts[kind] += 1
        weighted[kind] += weight
        total_weight += weight
    measured_weight = total_weight - weighted["unknown"]
    if measured_weight <= 0:
        bottleneck_type = "unknown"
        host_fraction = None
    else:
        host_fraction = weighted["host_dispatch"] / measured_weight
        device_fraction = weighted["device_execution"] / measured_weight
        if host_fraction >= 0.75:
            bottleneck_type = "host_dispatch"
        elif device_fraction >= 0.75:
            bottleneck_type = "device_execution"
        else:
            bottleneck_type = "mixed"
    return {
        "bottleneck_type": bottleneck_type,
        "host_dispatch_limited": bottleneck_type == "host_dispatch",
        "host_dispatch_case_weight_fraction": host_fraction,
        "case_counts": counts,
        "method": "weighted_case_bottleneck_v1",
    }
