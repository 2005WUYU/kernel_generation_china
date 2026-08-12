from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _status(result: Mapping[str, Any], key: str) -> str | None:
    value = result.get(key)
    if isinstance(value, Mapping):
        status = value.get("status")
        if status is not None:
            return str(status)
        for field in ("passed", "compiled", "profile_available"):
            if field in value:
                return "pass" if bool(value[field]) else "fail"
    return None


def _benchmark_cases(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, Mapping):
        return {}
    values = benchmark.get("per_case", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return {}
    cases: dict[str, Mapping[str, Any]] = {}
    for value in values:
        if isinstance(value, Mapping) and value.get("case_id") is not None:
            cases[str(value["case_id"])] = value
    return cases


def _overall_status(result: Mapping[str, Any], best: Mapping[str, Any] | None) -> str:
    if result.get("overall_status") == "model_failed":
        return "model_failed"
    source = result.get("source")
    if isinstance(source, Mapping) and not bool(source.get("passed")):
        return "source_failed"
    compile_result = result.get("compile")
    if not isinstance(compile_result, Mapping) or not bool(compile_result.get("compiled", compile_result.get("passed", False))):
        return "compile_failed"
    correctness = result.get("correctness")
    if not isinstance(correctness, Mapping) or not bool(correctness.get("passed")):
        return "correctness_failed"
    anti_bypass = result.get("anti_bypass")
    if isinstance(anti_bypass, Mapping) and anti_bypass.get("passed") is False:
        return "anti_bypass_failed"
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("status") in {"failed", "timeout"}:
        return "benchmark_failed"
    profile = result.get("profile")
    if isinstance(profile, Mapping) and profile.get("profile_available") is False:
        return "profile_unavailable"
    current_speedup = benchmark.get("geomean_speedup_vs_eager")
    if best is not None and current_speedup is not None:
        best_speedup = best.get("geomean_speedup_vs_eager", best.get("geomean_speedup"))
        if best_speedup is not None and float(current_speedup) < float(best_speedup):
            return "correct_but_slower_than_best"
    return "correct"


def build_feedback(
    *,
    task_id: str,
    round_number: int,
    result: Mapping[str, Any],
    best: Mapping[str, Any] | None = None,
    consecutive_non_improvements: int = 0,
) -> dict[str, Any]:
    """Normalize a full evaluation into the bounded evidence sent next round."""
    overall = _overall_status(result, best)
    benchmark = result.get("benchmark") if isinstance(result.get("benchmark"), Mapping) else None
    compile_result = result.get("compile") if isinstance(result.get("compile"), Mapping) else None
    correctness = result.get("correctness") if isinstance(result.get("correctness"), Mapping) else None
    profile = result.get("profile") if isinstance(result.get("profile"), Mapping) else None

    comparison: dict[str, Any] | None = None
    focus: list[str] = []
    if benchmark is not None and best is not None:
        current_cases = _benchmark_cases(result)
        best_cases = _benchmark_cases({"benchmark": best.get("benchmark", best)})
        improved: list[str] = []
        regressed: list[str] = []
        for case_id, current in current_cases.items():
            old = best_cases.get(case_id)
            current_speed = current.get("speedup_vs_eager")
            old_speed = old.get("speedup_vs_eager") if old else None
            if current_speed is None or old_speed is None:
                continue
            change = float(current_speed) / float(old_speed) - 1.0
            if change >= 0.01:
                improved.append(case_id)
            elif change <= -0.01:
                regressed.append(case_id)
        current_geomean = benchmark.get("geomean_speedup_vs_eager")
        best_geomean = best.get("geomean_speedup_vs_eager", best.get("geomean_speedup"))
        comparison = {
            "best_round": best.get("round", best.get("round_number")),
            "current_round": round_number,
            "geomean_change_percent": (
                (float(current_geomean) / float(best_geomean) - 1.0) * 100.0
                if current_geomean is not None and best_geomean not in (None, 0)
                else None
            ),
            "regressed_cases": regressed,
            "improved_cases": improved,
        }
        if regressed:
            focus.append(f"恢复退化用例的性能: {', '.join(regressed[:8])}")

    if overall == "model_failed":
        focus.append("重新生成一个满足 JSON Schema 的完整候选源码")
    elif overall == "source_failed":
        focus.append("修复静态安全检查列出的 import、调用或入口问题")
    elif overall == "compile_failed":
        focus.append("只依据编译阶段和源码位置修复 Triton-Ascend 编译错误")
    elif overall == "correctness_failed":
        focus.append("修复首个公开失败用例的 mask、边界、shape 或数值精度")
    elif overall == "anti_bypass_failed":
        focus.append("移除高层算子回退, 确保目标计算由候选 Triton kernel 完成")
    elif overall == "benchmark_failed":
        focus.append("保持正确性并降低资源使用或运行时间, 避免 benchmark 超时")
    elif overall == "profile_unavailable":
        focus.append("保持正确性和现有性能; profiler 当前不可用, 不要猜测硬件指标")
    else:
        if (
            benchmark is not None
            and benchmark.get("minimum_speedup_vs_eager") is not None
            and float(benchmark["minimum_speedup_vs_eager"]) < 1.0
        ):
            focus.append("优先改善最慢 shape, 使最低加速比达到 1.0 以上")
        if profile is not None:
            observations = profile.get("observations")
            if isinstance(observations, Sequence):
                for observation in observations[:3]:
                    if isinstance(observation, Mapping) and observation.get("suggestion"):
                        focus.append(str(observation["suggestion"]))
    if consecutive_non_improvements >= 2:
        focus.append("连续两轮未提升, 请采用明显不同的 tiling、grid 或融合方案")
    if not focus:
        focus.append("保持全部公开正确性, 并改善最低 shape 与加权几何平均性能")

    return {
        "evaluation_protocol": "ascend_kernel_feedback_v1",
        "task_id": task_id,
        "round": round_number,
        "overall_status": overall,
        "source": result.get("source"),
        "compile": compile_result,
        "correctness": correctness,
        "benchmark": benchmark,
        "profile": profile,
        "anti_bypass": result.get("anti_bypass"),
        "comparison_with_best": comparison,
        "next_round_requirement": {
            "must_keep_correctness": True,
            "must_return_full_code": True,
            "focus": focus,
        },
    }
