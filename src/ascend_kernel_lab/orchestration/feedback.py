from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ascend_kernel_lab.domain import (
    CandidateScore,
    PublicCandidateComparison,
    compare_public_candidate,
)


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


def _latency_measurements(benchmark: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if benchmark is None:
        return None
    cases: list[dict[str, Any]] = []
    values = benchmark.get("per_case", ())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for value in values:
            if not isinstance(value, Mapping):
                continue
            cases.append(
                {
                    "case_id": value.get("case_id"),
                    "device_latency_us": value.get("device_latency_us"),
                    "end_to_end_latency_us": value.get("end_to_end_latency_us"),
                    "host_overhead_us": value.get("host_overhead_us"),
                }
            )
    return {"cases": cases} if cases else None


def _score_from_result(result: Mapping[str, Any]) -> CandidateScore | None:
    value = result.get("score")
    if not isinstance(value, Mapping) or value.get("candidate_id") is None:
        return None
    try:
        return CandidateScore(
            candidate_id=str(value["candidate_id"]),
            round_number=int(value.get("round_number", value.get("round", 0))),
            compile_passed=bool(value.get("compile_passed")),
            correctness_passed=bool(value.get("correctness_passed")),
            anti_bypass_passed=bool(value.get("anti_bypass_passed")),
            hidden_correctness_passed=value.get("hidden_correctness_passed"),
            minimum_speedup=value.get("minimum_speedup"),
            geomean_speedup=value.get("geomean_speedup"),
            candidate_kernel_coverage=value.get("candidate_kernel_coverage"),
            stability_cv=value.get("stability_cv"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _best_score(best: Mapping[str, Any] | None) -> CandidateScore | None:
    if best is None:
        return None
    value = best.get("score")
    if isinstance(value, Mapping):
        candidate = _score_from_result({"score": value})
        if candidate is not None:
            return candidate
    try:
        return CandidateScore(
            candidate_id=str(best["candidate_id"]),
            round_number=int(best.get("round", best.get("round_number", 0))),
            compile_passed=True,
            correctness_passed=True,
            anti_bypass_passed=True,
            minimum_speedup=best.get("minimum_speedup"),
            geomean_speedup=best.get("geomean_speedup"),
            candidate_kernel_coverage=best.get("candidate_kernel_coverage"),
            stability_cv=best.get("stability_cv"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _best_benchmark(best: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = best.get("evaluation")
    if isinstance(evaluation, Mapping):
        benchmark = evaluation.get("benchmark")
        if isinstance(benchmark, Mapping):
            return benchmark
    benchmark = best.get("benchmark")
    return benchmark if isinstance(benchmark, Mapping) else best


def _profile_delta(
    candidate: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    def summary(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        nested = value.get("summary")
        return nested if isinstance(nested, Mapping) else value

    def number(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result

    current = summary(candidate)
    incumbent = summary(best)
    current_scheduling = current.get("scheduling")
    best_scheduling = incumbent.get("scheduling")
    current_scheduling = (
        current_scheduling if isinstance(current_scheduling, Mapping) else {}
    )
    best_scheduling = (
        best_scheduling if isinstance(best_scheduling, Mapping) else {}
    )
    fields = {
        "candidate_device_execution_us": (
            current_scheduling.get("candidate_device_execution_us"),
            best_scheduling.get("candidate_device_execution_us"),
        ),
        "host_overhead_us": (
            current_scheduling.get(
                "host_overhead_us", current_scheduling.get("host_enqueue_us")
            ),
            best_scheduling.get(
                "host_overhead_us", best_scheduling.get("host_enqueue_us")
            ),
        ),
    }
    changes: dict[str, Any] = {}
    for name, (current_value, best_value) in fields.items():
        current_number = number(current_value)
        best_number = number(best_value)
        changes[name] = {
            "candidate": current_number,
            "best": best_number,
            "difference": (
                current_number - best_number
                if current_number is not None and best_number is not None
                else None
            ),
            "change_percent": (
                (current_number / best_number - 1.0) * 100.0
                if current_number is not None
                and best_number is not None
                and best_number != 0
                else None
            ),
        }
    return changes


def _benchmark_delta(
    candidate: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None,
) -> dict[str, Any]:
    current = candidate if isinstance(candidate, Mapping) else {}
    paired = current.get("paired_best_comparison")
    if isinstance(paired, Mapping):
        return dict(paired)
    incumbent = best if isinstance(best, Mapping) else {}
    current_cases = _benchmark_cases({"benchmark": current})
    best_cases = _benchmark_cases({"benchmark": incumbent})
    rows: list[dict[str, Any]] = []
    for case_id in sorted(set(current_cases) & set(best_cases)):
        current_case = current_cases[case_id]
        best_case = best_cases[case_id]
        current_candidate = current_case.get("candidate")
        best_candidate = best_case.get("candidate")
        current_us = (
            current_candidate.get("median_us")
            if isinstance(current_candidate, Mapping)
            else current_case.get("candidate_us")
        )
        best_us = (
            best_candidate.get("median_us")
            if isinstance(best_candidate, Mapping)
            else best_case.get("candidate_us")
        )
        current_speedup = current_case.get("speedup_vs_eager")
        best_speedup = best_case.get("speedup_vs_eager")
        rows.append(
            {
                "case_id": case_id,
                "candidate_latency_us": current_us,
                "best_latency_us": best_us,
                "latency_change_percent": (
                    (float(current_us) / float(best_us) - 1.0) * 100.0
                    if current_us is not None
                    and best_us not in {None, 0}
                    else None
                ),
                "candidate_speedup": current_speedup,
                "best_speedup": best_speedup,
                "speedup_change_percent": (
                    (float(current_speedup) / float(best_speedup) - 1.0)
                    * 100.0
                    if current_speedup is not None
                    and best_speedup not in {None, 0}
                    else None
                ),
            }
        )
    return {"per_case": rows}


def _overall_status(result: Mapping[str, Any], best: Mapping[str, Any] | None) -> str:
    reported = result.get("overall_status")
    if reported in {
        "model_failed",
        "source_failed",
        "compile_failed",
        "correctness_failed",
        "benchmark_failed",
        "anti_bypass_failed",
    }:
        return str(reported)
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
    return "correct"


def build_feedback(
    *,
    task_id: str,
    round_number: int,
    result: Mapping[str, Any],
    best: Mapping[str, Any] | None = None,
    consecutive_non_improvements: int = 0,
    candidate_intent: Mapping[str, Any] | None = None,
    performance_comparison: PublicCandidateComparison | None = None,
) -> dict[str, Any]:
    """Normalize a full evaluation into the bounded evidence sent next round."""
    overall = _overall_status(result, best)
    benchmark = result.get("benchmark") if isinstance(result.get("benchmark"), Mapping) else None
    compile_result = result.get("compile") if isinstance(result.get("compile"), Mapping) else None
    correctness = result.get("correctness") if isinstance(result.get("correctness"), Mapping) else None
    profile = result.get("profile") if isinstance(result.get("profile"), Mapping) else None
    latency_measurements = _latency_measurements(benchmark)

    comparison: dict[str, Any] | None = None
    candidate_score = _score_from_result(result)
    incumbent_score = _best_score(best)
    decision = performance_comparison or (
        compare_public_candidate(candidate_score, incumbent_score)
        if candidate_score is not None
        else None
    )
    focus: list[str] = []
    if benchmark is not None and best is not None:
        current_cases = _benchmark_cases(result)
        best_benchmark = _best_benchmark(best)
        best_cases = _benchmark_cases({"benchmark": best_benchmark})
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
        comparison = {
            "best_round": best.get("round", best.get("round_number")),
            "current_round": round_number,
            "geomean_change_percent": (
                decision.geomean_change_fraction * 100.0
                if decision is not None
                and decision.geomean_change_fraction is not None
                else None
            ),
            "regressed_cases": regressed,
            "improved_cases": improved,
            "decision": decision.decision if decision is not None else "INVALID",
            "noise_tolerance_percent": (
                decision.tolerance_fraction * 100.0 if decision is not None else None
            ),
            "minimum_speedup_change_percent": (
                decision.minimum_change_fraction * 100.0
                if decision is not None
                and decision.minimum_change_fraction is not None
                else None
            ),
            "quick_profile_delta": _profile_delta(
                profile,
                (
                    best.get("evaluation", {}).get("profile")
                    if isinstance(best.get("evaluation"), Mapping)
                    else None
                ),
            ),
            "benchmark_delta": _benchmark_delta(
                benchmark, best_benchmark
            ),
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
        focus.append("保持正确性, 针对本轮 benchmark 的不稳定用例降低测量 CV, 再依据稳定结果优化延迟")
    elif overall == "profile_unavailable":
        focus.append("保持正确性和现有性能; profiler 当前不可用, 不要猜测硬件指标")
    else:
        if (
            benchmark is not None
            and benchmark.get("minimum_speedup_vs_eager") is not None
            and float(benchmark["minimum_speedup_vs_eager"]) < 1.0
        ):
            focus.append(
                "参考 PyTorch eager 对比优先改善最慢 shape; "
                "本次冷启动 SFT 轨迹不设硬性加速比门槛"
            )
    if consecutive_non_improvements >= 2:
        focus.append("连续两轮未提升, 请采用明显不同的 tiling、grid 或融合方案")
    if not focus:
        focus.append("保持全部公开正确性, 并改善最低 shape 与加权几何平均性能")

    performance_decision = (
        decision.decision
        if decision is not None and decision.decision != "INVALID"
        else "INVALID"
    )
    if overall == "model_failed":
        next_prompt_mode = "REGENERATE_INVALID_MODEL_RESPONSE"
    elif overall in {
        "source_failed",
        "compile_failed",
        "correctness_failed",
    }:
        next_prompt_mode = "REPAIR_FAILED_CANDIDATE"
    elif overall in {
        "benchmark_failed",
        "anti_bypass_failed",
        "profile_unavailable",
    }:
        next_prompt_mode = "RETURN_TO_BEST_AFTER_INVALID_PERFORMANCE_EVIDENCE"
    elif performance_decision in {"INITIAL_BEST", "NEW_BEST"}:
        next_prompt_mode = "CONTINUE_FROM_NEW_BEST"
    elif performance_decision == "REGRESSION":
        next_prompt_mode = "RETURN_TO_BEST_AFTER_REGRESSION"
    elif performance_decision == "TIE":
        next_prompt_mode = "CONTINUE_FROM_BEST_AFTER_TIE"
    else:
        next_prompt_mode = "CONTINUE_FROM_LAST_CORRECT_CANDIDATE"

    return {
        "evaluation_protocol": "ascend_kernel_feedback_v2",
        "task_id": task_id,
        "round": round_number,
        "overall_status": overall,
        "source": result.get("source"),
        "compile": compile_result,
        "correctness": correctness,
        "benchmark": benchmark,
        "latency_measurements": latency_measurements,
        "profile": profile,
        "anti_bypass": result.get("anti_bypass"),
        "comparison_with_best": comparison,
        "candidate_generation_intent": dict(candidate_intent or {}),
        "performance_decision": performance_decision,
        "next_prompt_mode": next_prompt_mode,
        "optimization_action": "continue_optimization",
        "stop_recommended": False,
        "next_round_requirement": {
            "must_keep_correctness": True,
            "must_return_full_code": True,
            "focus": focus,
        },
    }
