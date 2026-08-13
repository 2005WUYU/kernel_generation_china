"""Stable first-round and feedback-round prompt construction."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .types import ModelRequest

SYSTEM_PROMPT = """你负责生成可在华为昇腾 Triton-Ascend 环境运行的 Triton Kernel。

必须遵守:
1. 输出完整 Python 代码, 不输出补丁或 Markdown 代码块。
2. 入口函数固定为 custom_op。
3. 只允许使用 Triton Kernel 完成目标计算。
4. 可以使用 torch.empty、torch.empty_like 等函数分配输出。
5. 禁止调用 torch.matmul、torch.softmax、torch.sum、torch.nn.functional 等高层计算函数完成目标运算。
6. 禁止 CPU 回退、文件读取、网络访问、subprocess、ctypes、动态安装和进程启动。
7. 不得修改输入 Tensor。
8. 必须正确处理任务中所有公开 shape 和 dtype, 不得猜测或索取隐藏用例。
9. 只能依据提供的实际编译、正确性、延迟和 profiler 摘要进行修改。
10. 返回结果必须严格符合给定 JSON Schema; code 字段始终返回完整源码。
11. custom_op 的每一条返回路径都必须先启动候选 Triton Kernel; 禁止在 Kernel 启动前返回刚分配的输出, 包括 n == 0 等提前返回。
12. custom_op 不得调用自定义 Python 辅助函数; host 侧的 shape、grid 和 launch 参数选择逻辑必须直接内联在 custom_op 中。
13. 源码只能 import torch、triton 或 triton.language; host 侧只能分配输出、读取 Tensor 元数据、计算标量 launch 参数并启动候选 Kernel。
14. 不要在候选源码中实现 cache、warmup、benchmark、计时、异常捕获或任何运行时探测; 这些由评测框架负责。
15. Triton Kernel 名称必须是至少 8 个字符的具体标识符; custom_op 是普通 Python host wrapper, 不能用 @triton.jit 装饰。
16. 对每个公开用例, launch grid 的 program 总数(coreDim)必须小于等于 65535; 不要为每个元素或每行盲目启动一个 program, 应通过 BLOCK/tiling 让每个 program 处理多个元素。
17. optimization_summary 必须在生成候选时同步概括 code 中实际采用的具体修改或初始实现选择; 不得留给评测系统事后推断, expected_effect 则单独写预期影响。
"""


LEGAL_SOURCE_TEMPLATE = """import torch
import triton
import triton.language as tl

@triton.jit
def generated_candidate_kernel(input_ptr, output_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)

def custom_op(x):
    output = torch.empty_like(x)
    n = x.numel()
    block_size = 256
    grid = (triton.cdiv(n, block_size),)
    generated_candidate_kernel[grid](x, output, n, BLOCK_SIZE=block_size)
    return output
"""


SOURCE_CHECKER_CONTRACT: dict[str, Any] = {
    "purpose": "structural_template_only_adapt_arguments_and_math_to_the_task",
    "allowed_import_roots": ["torch", "triton"],
    "host_allowed": [
        "torch.empty/empty_like/empty_strided output allocation",
        "Tensor shape/stride/dtype/device/numel metadata",
        "scalar shape/grid/launch-parameter arithmetic",
        "triton.cdiv/triton.next_power_of_2/triton.Config",
        "launching a declared @triton.jit kernel",
    ],
    "source_checker_rejects": [
        "high-level torch computation or Tensor indexing/view/copy/data_ptr/item",
        "custom Python helper calls from custom_op",
        "getattr, dict.get, try/except, file/network/process/runtime probing",
        "candidate-side cache, warmup, benchmark, timing, or dynamic installation",
        "generic/short Triton kernel names or @triton.jit on custom_op",
        "returning an output before that exact output is passed to a candidate kernel launch",
        "top-level executable code or non-literal dynamic assignments",
    ],
    "required": [
        "one ordinary Python host entry point named custom_op",
        "at least one specifically named @triton.jit kernel (name length >= 8)",
        "every custom_op control-flow return launches a candidate kernel first",
        "complete standalone Python source, never a patch or Markdown fence",
    ],
    "runtime_launch_constraints": [
        "for every public case, the total launch-grid program count/coreDim must be <= 65535",
        "tile multiple elements or rows per program instead of launching one program per element when the grid can reach 65536",
    ],
    "legal_structure_template": LEGAL_SOURCE_TEMPLATE,
}


def _cold_start_sft_strategy(
    optimization_rounds: int, maximum_repair_rounds: int
) -> dict[str, Any]:
    return {
        "purpose": "cold_start_sft",
        "cold_start_sft": True,
        "repair_then_optimize": maximum_repair_rounds > 0,
        "maximum_repair_rounds": maximum_repair_rounds,
        "optimization_round_count": optimization_rounds,
        # Each performance proposal may need its own bounded repair chain.
        # The initial repair seed and those per-slot repairs use consecutive
        # durable round numbers and therefore all count toward the maximum.
        "maximum_total_rounds": maximum_repair_rounds
        + optimization_rounds * (maximum_repair_rounds + 1),
        "target_speedup": None,
        "comparison_baseline": "pytorch_eager",
        "policy": (
            f"先用最多 {maximum_repair_rounds} 轮得到 source/compile/correctness "
            f"全通过的起点, 再采集 {optimization_rounds} 轮性能优化轨迹; "
            "Repair 不计入 Optimization 轮数, 且不设硬性加速比门槛"
        ),
    }


def _phase_context(
    *,
    phase: str,
    phase_index: int,
    optimization_rounds: int,
    maximum_repair_rounds: int,
    initial_generation: bool = False,
    failure_evidence_path: str | None = None,
) -> dict[str, Any]:
    if phase not in {"repair", "optimization", "optimization_repair"}:
        raise ValueError(
            "phase must be repair, optimization, or optimization_repair"
        )
    if phase_index < 1:
        raise ValueError("phase_index must be at least one")
    if optimization_rounds < 1 or maximum_repair_rounds < 0:
        raise ValueError("invalid repair/optimization round counts")
    if phase == "repair" and initial_generation:
        directive = (
            "这是首次 Repair, 尚无 previous_round 或 failed_candidate; "
            "请直接依据 task、environment、baseline 和 source_checker_contract "
            "首次生成完整 Kernel, 优先保证 source/compile/correctness"
        )
    elif phase in {"repair", "optimization_repair"}:
        evidence_path = (
            failure_evidence_path
            or "failed_candidate.raw_stage_result"
        )
        if evidence_path == "failed_candidate.model_response_error":
            directive = (
                "上一轮模型响应格式失败, 没有可继续修复的 failed_candidate.code; "
                "请从 task 或 BEST 重新生成完整 Kernel; "
                "本轮原始格式错误位于 failed_candidate.model_response_error"
            )
        else:
            directive = (
                "从 failed_candidate.code 完整源码出发, 保留已通过的阶段; "
                f"本轮原始错误位于 {evidence_path}; "
                "必须只依据该原始证据精确修复, 不做无关性能改写"
            )
    else:
        directive = (
            "从已正确候选出发做一次可解释的性能修改; "
            "始终保持 source/compile/correctness"
        )
    return {
        "name": phase,
        "index": phase_index,
        "maximum_repair_rounds": maximum_repair_rounds,
        "optimization_rounds": optimization_rounds,
        "directive": directive,
    }

_SENSITIVE_EXACT = {
    "api_key", "auth_token", "authorization", "credential", "credentials",
    "hidden", "hidden_case", "hidden_cases", "hidden_input", "hidden_inputs",
    "password", "private_case", "private_cases", "reference_implementation",
    "reference_source", "secret", "secrets",
}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_EXACT
        or normalized.startswith("hidden_")
        or normalized.startswith("private_")
        or normalized.endswith("_api_key")
        or normalized.endswith("_auth_token")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
    )


def _public_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError(f"prompt payload is nested too deeply at {path}")
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(cast(Any, value))
    if isinstance(value, Enum):
        return _public_json(value.value, path=path, depth=depth + 1)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"prompt payload has non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"prompt payload key at {path} is not a string")
            if _is_sensitive_key(key):
                continue
            result[key] = _public_json(item, path=f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _public_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"prompt payload contains unsupported type at {path}: {type(value).__name__}")


def _prompt_environment(environment: Mapping[str, Any]) -> Mapping[str, Any]:
    compact = environment.get("prompt_environment")
    if isinstance(compact, Mapping):
        return compact
    return environment


class PromptBuilder:
    """Build canonical JSON requests while recursively removing hidden data."""

    def __init__(
        self,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        protocol_version: str = "ascend_kernel_generation_v1",
    ) -> None:
        if not system_prompt.strip() or not protocol_version.strip():
            raise ValueError("system_prompt and protocol_version must not be empty")
        self.system_prompt = system_prompt
        self.protocol_version = protocol_version

    @staticmethod
    def _render(payload: Mapping[str, Any]) -> str:
        rendered = json.dumps(
            _public_json(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if len(rendered.encode("utf-8")) > 2_000_000:
            raise ValueError("rendered prompt exceeds 2000000 UTF-8 bytes")
        return rendered

    def build_first_round(
        self,
        *,
        task: Mapping[str, Any],
        environment: Mapping[str, Any],
        baseline: Mapping[str, Any],
        maximum_rounds: int = 5,
        phase: str = "optimization",
        phase_index: int = 1,
        optimization_rounds: int | None = None,
        maximum_repair_rounds: int = 0,
        objective: Mapping[str, Any] | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelRequest:
        if maximum_rounds < 1:
            raise ValueError("maximum_rounds must be at least one")
        optimization_count = optimization_rounds or maximum_rounds
        phase_context = _phase_context(
            phase=phase,
            phase_index=phase_index,
            optimization_rounds=optimization_count,
            maximum_repair_rounds=maximum_repair_rounds,
            initial_generation=True,
        )
        payload = {
            "protocol_version": self.protocol_version,
            "collection_strategy": _cold_start_sft_strategy(
                optimization_count, maximum_repair_rounds
            ),
            "phase": phase_context,
            "task": task,
            "environment": _prompt_environment(environment),
            "baseline": baseline,
            "source_checker_contract": SOURCE_CHECKER_CONTRACT,
            "round_context": {
                "round": 1,
                "maximum_rounds": maximum_rounds,
                "current_candidate": None,
                "history_summary": [],
            },
            "objective": objective or {
                "first": "保证所有公开用例正确",
                "second": "与 PyTorch eager baseline 对比并记录可学习的修改与效果",
                "third": "Repair 成功后完成规定的 Optimization 轮次; 不要为加速比牺牲正确性",
            },
        }
        if objective is None and phase == "repair":
            payload["objective"] = {
                "first": "首次生成满足 source checker 契约的完整 Kernel",
                "second": "使首次候选通过编译和公开正确性",
                "third": "得到可作为 Optimization 起点的正确 Kernel; 本轮不要求性能提升",
            }
        elif objective is None:
            payload["objective"] = {
                "first": "保持 source/compile/correctness 全部通过",
                "second": "只依据上一轮紧凑证据优化相对 PyTorch eager 的 Kernel 性能",
                "third": "做一次明确修改并说明预期效果; 不重复已失败方案",
            }
        user_prompt = self._render(payload)
        return ModelRequest(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            timeout_seconds=timeout_seconds,
            metadata={
                "round": 1,
                "phase": phase,
                "phase_index": phase_index,
                "user_prompt_utf8_bytes": len(user_prompt.encode("utf-8")),
                "system_prompt_utf8_bytes": len(
                    self.system_prompt.encode("utf-8")
                ),
                "protocol_version": self.protocol_version,
            },
        )

    def build_follow_up(
        self,
        *,
        round_number: int,
        maximum_rounds: int,
        last_candidate_code: str,
        key_metrics: Mapping[str, Any],
        failure_reasons: Sequence[str],
        next_round_suggestions: Sequence[str],
        task_contract: Mapping[str, Any] | None = None,
        environment: Mapping[str, Any] | None = None,
        baseline: Mapping[str, Any] | None = None,
        candidate_round: int | None = None,
        candidate_role: str = "current_best",
        history_summary: Sequence[Mapping[str, Any]] = (),
        feedback_state: Mapping[str, Any] | None = None,
        best_candidate: Mapping[str, Any] | None = None,
        failed_candidate: Mapping[str, Any] | None = None,
        phase: str = "optimization",
        phase_index: int | None = None,
        optimization_rounds: int | None = None,
        maximum_repair_rounds: int = 0,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelRequest:
        if round_number < 2:
            raise ValueError("follow-up round must be at least 2")
        optimization_count = optimization_rounds or maximum_rounds
        phase_context = _phase_context(
            phase=phase,
            phase_index=phase_index or round_number,
            optimization_rounds=optimization_count,
            maximum_repair_rounds=maximum_repair_rounds,
            failure_evidence_path=(
                "failed_candidate.raw_stage_result"
                if isinstance(failed_candidate, Mapping)
                and "raw_stage_result" in failed_candidate
                else (
                    "failed_candidate.model_response_error"
                    if isinstance(failed_candidate, Mapping)
                    and "model_response_error" in failed_candidate
                    else None
                )
            ),
        )
        round_context: dict[str, Any] = {
            "round": round_number,
            "maximum_rounds": maximum_rounds,
            "previous_round": {
                "key_metrics": key_metrics,
                "failure_reasons": list(failure_reasons),
                "next_round_suggestions": list(next_round_suggestions),
            },
            "history_summary": list(history_summary),
        }
        if last_candidate_code:
            round_context["working_candidate"] = {
                "round": candidate_round,
                "role": candidate_role,
                "code": last_candidate_code,
            }
        payload = {
            "protocol_version": self.protocol_version,
            "phase": phase_context,
            "task_contract": task_contract or {},
            "environment": _prompt_environment(environment or {}),
            "baseline": baseline or {},
            "source_checker_contract": SOURCE_CHECKER_CONTRACT,
            "round_context": round_context,
            "feedback_state": feedback_state or {},
            "best_candidate": best_candidate,
            "failed_candidate": failed_candidate,
        }
        user_prompt = self._render(payload)
        return ModelRequest(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=model,
            timeout_seconds=timeout_seconds,
            metadata={
                "round": round_number,
                "phase": phase,
                "phase_index": phase_index or round_number,
                "user_prompt_utf8_bytes": len(user_prompt.encode("utf-8")),
                "system_prompt_utf8_bytes": len(
                    self.system_prompt.encode("utf-8")
                ),
                "protocol_version": self.protocol_version,
            },
        )

    build_initial = build_first_round
    build_next_round = build_follow_up
