"""Strict parsing and one-shot repair of structured model responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .errors import (
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    TruncatedResponseError,
)
from .types import ModelCompletion, ModelGateway, ModelRequest

_REQUIRED_FIELDS = {"status", "round", "code"}
_OPTIONAL_LAYER_FIELDS = {"changes", "evidence", "hypotheses", "predictions"}
_RAW_OPTIONAL_FIELD = "raw_optional_content"
_LEGACY_V2_FIELDS = {
    "status",
    "round",
    "optimization_summary",
    "expected_effect",
    "assumptions",
    "code",
}
_LEGACY_V1_FIELDS = (
    _LEGACY_V2_FIELDS - {"optimization_summary"}
) | {"change_summary"}
_TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens", "token_limit"}
_SUCCESS_REASONS = {"stop", "end_turn", "success", "completed"}


@dataclass(frozen=True, slots=True)
class ModelResponse:
    status: str
    round: int
    changes: tuple[Any, ...]
    evidence: tuple[Any, ...]
    hypotheses: tuple[Any, ...]
    predictions: tuple[Any, ...]
    code: str
    raw_optional_content: Mapping[str, Any] | None = None
    legacy_intent: Mapping[str, tuple[str, ...]] | None = None
    legacy_summary_field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.legacy_intent is not None and self.legacy_summary_field is not None:
            return {
                "status": self.status,
                "round": self.round,
                self.legacy_summary_field: list(
                    self.legacy_intent.get(self.legacy_summary_field, ())
                ),
                "expected_effect": list(
                    self.legacy_intent.get("expected_effect", ())
                ),
                "assumptions": list(self.legacy_intent.get("assumptions", ())),
                "code": self.code,
            }
        result = {
            "status": self.status,
            "round": self.round,
            "changes": [deepcopy(item) for item in self.changes],
            "evidence": [deepcopy(item) for item in self.evidence],
            "hypotheses": [deepcopy(item) for item in self.hypotheses],
            "predictions": [deepcopy(item) for item in self.predictions],
            "code": self.code,
        }
        if self.raw_optional_content:
            result[_RAW_OPTIONAL_FIELD] = deepcopy(dict(self.raw_optional_content))
        return result

    @property
    def is_legacy(self) -> bool:
        return self.legacy_intent is not None


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


def _validate_current_layers(
    decoded: Mapping[str, Any],
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
]:
    def optional_list(name: str) -> tuple[Any, ...]:
        value = decoded.get(name)
        return tuple(deepcopy(value)) if isinstance(value, list) else ()

    return (
        optional_list("changes"),
        optional_list("evidence"),
        optional_list("hypotheses"),
        optional_list("predictions"),
    )


def validate_model_response(
    value: str | bytes | Mapping[str, Any],
    *,
    expected_round: int | None = None,
    allow_legacy: bool = False,
) -> ModelResponse:
    """Validate JSON content exactly.

    New model completions require only status, round, and complete code.
    Explanatory fields are optional advisory content and never gate code.
    Already-committed legacy artifacts retain their historical normalization
    when a caller explicitly enables it.
    """

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
    fields = set(decoded)
    legacy_summary_field: str | None = None
    if allow_legacy and fields == _LEGACY_V2_FIELDS:
        legacy_summary_field = "optimization_summary"
    elif allow_legacy and fields == _LEGACY_V1_FIELDS:
        legacy_summary_field = "change_summary"
    expected_fields = (
        _LEGACY_V2_FIELDS
        if legacy_summary_field == "optimization_summary"
        else _LEGACY_V1_FIELDS
        if legacy_summary_field == "change_summary"
        else _REQUIRED_FIELDS
    )
    missing = sorted(expected_fields - fields)
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
    if legacy_summary_field is not None:
        legacy_intent = {
            legacy_summary_field: _string_list(
                decoded[legacy_summary_field], legacy_summary_field
            ),
            "expected_effect": _string_list(
                decoded["expected_effect"], "expected_effect"
            ),
            "assumptions": _string_list(decoded["assumptions"], "assumptions"),
        }
        changes: tuple[Any, ...] = ()
        evidence: tuple[Any, ...] = ()
        hypotheses: tuple[Any, ...] = ()
        predictions: tuple[Any, ...] = ()
    else:
        changes, evidence, hypotheses, predictions = _validate_current_layers(decoded)
        legacy_intent = None
    raw_optional_content: dict[str, Any] = {}
    preserved = decoded.get(_RAW_OPTIONAL_FIELD)
    if isinstance(preserved, Mapping):
        raw_optional_content.update(deepcopy(dict(preserved)))
    elif _RAW_OPTIONAL_FIELD in decoded:
        raw_optional_content[_RAW_OPTIONAL_FIELD] = deepcopy(preserved)
    if legacy_summary_field is None:
        for field, item in decoded.items():
            if field in _REQUIRED_FIELDS or field == _RAW_OPTIONAL_FIELD:
                continue
            if field in _OPTIONAL_LAYER_FIELDS and isinstance(item, list):
                continue
            raw_optional_content[field] = deepcopy(item)
    return ModelResponse(
        status=status,
        round=round_number,
        changes=changes,
        evidence=evidence,
        hypotheses=hypotheses,
        predictions=predictions,
        code=code,
        raw_optional_content=raw_optional_content or None,
        legacy_intent=legacy_intent,
        legacy_summary_field=legacy_summary_field,
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
