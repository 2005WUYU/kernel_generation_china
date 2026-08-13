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
        "maximum_total_rounds": optimization_rounds + maximum_repair_rounds,
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
) -> dict[str, Any]:
    if phase not in {"repair", "optimization"}:
        raise ValueError("phase must be repair or optimization")
    if phase_index < 1:
        raise ValueError("phase_index must be at least one")
    if optimization_rounds < 1 or maximum_repair_rounds < 0:
        raise ValueError("invalid repair/optimization round counts")
    return {
        "name": phase,
        "index": phase_index,
        "maximum_repair_rounds": maximum_repair_rounds,
        "optimization_rounds": optimization_rounds,
        "directive": (
            "只修复 source/compile/correctness; 保留已正确部分, 不以性能改写掩盖错误"
            if phase == "repair"
            else "从已正确候选出发做一次可解释的性能修改; 始终保持 source/compile/correctness"
        ),
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
        )
        payload = {
            "protocol_version": self.protocol_version,
            "collection_strategy": _cold_start_sft_strategy(
                optimization_count, maximum_repair_rounds
            ),
            "phase": phase_context,
            "task": task,
            "environment": environment,
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
                "first": "修复 source checker 错误",
                "second": "修复编译和公开正确性错误",
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
        candidate_round: int | None = None,
        candidate_role: str = "current_best",
        history_summary: Sequence[Mapping[str, Any]] = (),
        phase: str = "optimization",
        phase_index: int | None = None,
        optimization_rounds: int | None = None,
        maximum_repair_rounds: int = 0,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelRequest:
        if not 2 <= round_number <= maximum_rounds:
            raise ValueError("follow-up round must be between 2 and maximum_rounds")
        optimization_count = optimization_rounds or maximum_rounds
        phase_context = _phase_context(
            phase=phase,
            phase_index=phase_index or round_number,
            optimization_rounds=optimization_count,
            maximum_repair_rounds=maximum_repair_rounds,
        )
        payload = {
            "protocol_version": self.protocol_version,
            "phase": phase_context,
            "task_contract": task_contract or {},
            "source_checker_contract": SOURCE_CHECKER_CONTRACT,
            "round_context": {
                "round": round_number,
                "maximum_rounds": maximum_rounds,
                "current_candidate": {
                    "round": candidate_round,
                    "role": candidate_role,
                    "code": last_candidate_code,
                },
                "previous_round": {
                    "key_metrics": key_metrics,
                    "failure_reasons": list(failure_reasons),
                    "next_round_suggestions": list(next_round_suggestions),
                },
                "history_summary": list(history_summary),
            },
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
