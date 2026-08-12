"""Strict parsing and one-shot repair of structured model responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import (
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    TruncatedResponseError,
)
from .types import ModelCompletion, ModelGateway, ModelRequest

_FIELDS = {
    "status", "round", "change_summary", "expected_effect", "assumptions", "code"
}
_TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}
_SUCCESS_REASONS = {"stop", "end_turn", "success", "completed"}


@dataclass(frozen=True, slots=True)
class ModelResponse:
    status: str
    round: int
    change_summary: tuple[str, ...]
    expected_effect: tuple[str, ...]
    assumptions: tuple[str, ...]
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "round": self.round,
            "change_summary": list(self.change_summary),
            "expected_effect": list(self.expected_effect),
            "assumptions": list(self.assumptions),
            "code": self.code,
        }


@dataclass(frozen=True, slots=True)
class ValidatedModelResponse:
    response: ModelResponse
    completion: ModelCompletion
    repair_attempts: int
    completions: tuple[ModelCompletion, ...]


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ModelResponseError(f"{path} must be an array of strings")
    if len(value) > 32:
        raise ModelResponseError(f"{path} contains too many items")
    if any("\x00" in item for item in value):
        raise ModelResponseError(f"{path} must not contain NUL")
    return tuple(value)


def validate_model_response(
    value: str | bytes | Mapping[str, Any], *, expected_round: int | None = None
) -> ModelResponse:
    """Validate JSON content exactly; unknown fields and coercions are rejected."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelResponseError("response is not UTF-8") from exc
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"response is not valid JSON: {exc.msg}") from exc
    else:
        decoded = dict(value)
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise ModelResponseError("response must be a JSON object")
    unknown = sorted(set(decoded) - _FIELDS)
    missing = sorted(_FIELDS - set(decoded))
    if unknown:
        raise ModelResponseError(f"unknown response field(s): {', '.join(unknown)}")
    if missing:
        raise ModelResponseError(f"missing response field(s): {', '.join(missing)}")
    status = decoded["status"]
    if status not in {"candidate", "no_change"}:
        raise ModelResponseError("status must be 'candidate' or 'no_change'")
    round_number = decoded["round"]
    if type(round_number) is not int or round_number < 1:
        raise ModelResponseError("round must be a positive integer")
    if expected_round is not None and round_number != expected_round:
        raise ModelResponseError(f"round must equal requested round {expected_round}")
    code = decoded["code"]
    if not isinstance(code, str) or not code.strip():
        raise ModelResponseError("code must be a non-empty string")
    if len(code.encode("utf-8")) > 262_144:
        raise ModelResponseError("code exceeds 262144 UTF-8 bytes")
    if "\x00" in code:
        raise ModelResponseError("code must not contain NUL")
    if code.lstrip().startswith("```") or code.rstrip().endswith("```"):
        raise ModelResponseError("code must be raw Python source, not a Markdown fence")
    return ModelResponse(
        status=status,
        round=round_number,
        change_summary=_string_list(decoded["change_summary"], "change_summary"),
        expected_effect=_string_list(decoded["expected_effect"], "expected_effect"),
        assumptions=_string_list(decoded["assumptions"], "assumptions"),
        code=code,
    )


def validate_completion(
    completion: ModelCompletion, *, expected_round: int | None = None
) -> ModelResponse:
    reason = completion.finish_reason.lower() if completion.finish_reason else None
    if reason in _TRUNCATED_REASONS:
        raise TruncatedResponseError(f"provider reported truncated response: {reason}")
    if reason not in _SUCCESS_REASONS:
        raise ModelResponseError(f"provider did not report normal completion: {reason}")
    return validate_model_response(completion.content, expected_round=expected_round)


def complete_model_response(
    gateway: ModelGateway,
    request: ModelRequest,
    *,
    expected_round: int | None = None,
    maximum_format_repair_retries: int = 1,
) -> ValidatedModelResponse:
    """Complete and, only for format failures, ask for a bounded JSON repair."""

    if maximum_format_repair_retries < 0:
        raise ValueError("maximum_format_repair_retries must be non-negative")
    current_request = request
    last_error: ModelResponseError | None = None
    completions: list[ModelCompletion] = []
    for repair_attempt in range(maximum_format_repair_retries + 1):
        completion = gateway.complete(current_request)
        completions.append(completion)
        try:
            response = validate_completion(completion, expected_round=expected_round)
            return ValidatedModelResponse(
                response, completion, repair_attempt, tuple(completions)
            )
        except ModelResponseError as exc:
            last_error = exc
            if repair_attempt >= maximum_format_repair_retries:
                raise ModelResponseAttemptsExhausted(
                    str(exc), tuple(completions)
                ) from exc
            repair_payload = {
                "protocol_version": "ascend_kernel_format_repair_v1",
                "instruction": (
                    "Return the same complete candidate as one JSON object matching the supplied "
                    "schema. Return no Markdown and no commentary."
                ),
                "validation_error": str(exc),
                "invalid_response": completion.content,
            }
            current_request = ModelRequest(
                system_prompt=request.system_prompt,
                user_prompt=json.dumps(repair_payload, ensure_ascii=False, sort_keys=True),
                json_schema=request.json_schema,
                model=request.model,
                timeout_seconds=request.timeout_seconds,
                metadata={**request.metadata, "format_repair_attempt": repair_attempt + 1},
            )
    raise ModelResponseAttemptsExhausted(
        str(last_error or "unreachable response validation state"), tuple(completions)
    )
