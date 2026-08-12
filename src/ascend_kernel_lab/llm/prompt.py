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
"""

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
        objective: Mapping[str, Any] | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelRequest:
        if maximum_rounds < 1:
            raise ValueError("maximum_rounds must be at least one")
        payload = {
            "protocol_version": self.protocol_version,
            "task": task,
            "environment": environment,
            "baseline": baseline,
            "round_context": {
                "round": 1,
                "maximum_rounds": maximum_rounds,
                "best_candidate": None,
                "history_summary": [],
            },
            "objective": objective or {
                "first": "保证所有公开用例正确",
                "second": "降低多 shape 的总体延迟",
                "third": "避免只针对单一 shape 优化",
            },
        }
        return ModelRequest(
            system_prompt=self.system_prompt,
            user_prompt=self._render(payload),
            model=model,
            timeout_seconds=timeout_seconds,
            metadata={"round": 1, "protocol_version": self.protocol_version},
        )

    def build_follow_up(
        self,
        *,
        task: Mapping[str, Any],
        environment: Mapping[str, Any],
        baseline: Mapping[str, Any],
        round_number: int,
        maximum_rounds: int,
        best_candidate: Mapping[str, Any] | None,
        last_candidate: Mapping[str, Any],
        last_evaluation: Mapping[str, Any],
        history_summary: Sequence[Mapping[str, Any]],
        objective: Mapping[str, Any] | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ModelRequest:
        if not 2 <= round_number <= maximum_rounds:
            raise ValueError("follow-up round must be between 2 and maximum_rounds")
        payload = {
            "protocol_version": self.protocol_version,
            "task": task,
            "environment": environment,
            "baseline": baseline,
            "round_context": {
                "round": round_number,
                "maximum_rounds": maximum_rounds,
                "best_candidate": best_candidate,
                "last_candidate": last_candidate,
                "last_evaluation": last_evaluation,
                "history_summary": history_summary,
            },
            "objective": objective or {
                "first": "保持所有公开用例正确",
                "second": "依据本轮结构化证据提高最差 shape 与几何平均加速比",
                "third": "保持稳定且避免高层算子回退",
            },
        }
        return ModelRequest(
            system_prompt=self.system_prompt,
            user_prompt=self._render(payload),
            model=model,
            timeout_seconds=timeout_seconds,
            metadata={"round": round_number, "protocol_version": self.protocol_version},
        )

    build_initial = build_first_round
    build_next_round = build_follow_up
