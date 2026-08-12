from __future__ import annotations

import statistics
import time
from typing import Any


def _cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean > 0 else None


def probe_timing(samples: int = 100) -> dict[str, Any]:
    """Select and validate the best available NPU timing primitive."""
    try:
        import importlib

        torch = importlib.import_module("torch")
        importlib.import_module("torch_npu")
    except Exception as exc:
        return {
            "timing_method": None,
            "verified": False,
            "empty_overhead_us": None,
            "coefficient_of_variation": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return {
            "timing_method": None,
            "verified": False,
            "empty_overhead_us": None,
            "coefficient_of_variation": None,
            "error": "torch.npu is unavailable",
        }
    event_values: list[float] = []
    try:
        for _ in range(samples):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            end.record()
            end.synchronize()
            event_values.append(float(start.elapsed_time(end)) * 1000.0)
        return {
            "timing_method": "torch_npu_event",
            "verified": True,
            "empty_overhead_us": statistics.median(event_values),
            "coefficient_of_variation": _cv(event_values),
            "sample_count": len(event_values),
            "error": None,
        }
    except Exception as event_error:
        host_values: list[float] = []
        try:
            for _ in range(samples):
                started = time.perf_counter_ns()
                torch.npu.synchronize()
                host_values.append((time.perf_counter_ns() - started) / 1000.0)
            return {
                "timing_method": "host_perf_counter_with_npu_synchronize",
                "verified": True,
                "empty_overhead_us": statistics.median(host_values),
                "coefficient_of_variation": _cv(host_values),
                "sample_count": len(host_values),
                "event_error": f"{type(event_error).__name__}: {event_error}",
                "error": None,
            }
        except Exception as host_error:
            return {
                "timing_method": None,
                "verified": False,
                "empty_overhead_us": None,
                "coefficient_of_variation": None,
                "event_error": f"{type(event_error).__name__}: {event_error}",
                "error": f"{type(host_error).__name__}: {host_error}",
            }
