"""Strict parsing and one-shot repair of structured model responses."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import (
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    TruncatedResponseError,
)
from .types import ModelCompletion, ModelGateway, ModelRequest

_CURRENT_FIELDS = {
    "status",
    "round",
    "changes",
    "evidence",
    "hypotheses",
    "predictions",
    "code",
}
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
_PREDICTION_REASON = re.compile(r"^hypothesis\[([0-9]+)\]$")


@dataclass(frozen=True, slots=True)
class ModelResponse:
    status: str
    round: int
    changes: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    hypotheses: tuple[Mapping[str, Any], ...]
    predictions: tuple[Mapping[str, Any], ...]
    code: str
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
        return {
            "status": self.status,
            "round": self.round,
            "changes": [dict(item) for item in self.changes],
            "evidence": [dict(item) for item in self.evidence],
            "hypotheses": [dict(item) for item in self.hypotheses],
            "predictions": [dict(item) for item in self.predictions],
            "code": self.code,
        }

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


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelResponseError(f"{path} must be a non-empty string")
    if "\x00" in value:
        raise ModelResponseError(f"{path} must not contain NUL")
    return value


def _object_list(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ModelResponseError(f"{path} must be an array of objects")
    if not value or len(value) > 32:
        raise ModelResponseError(f"{path} must contain between 1 and 32 items")
    if any(not isinstance(item, Mapping) for item in value):
        raise ModelResponseError(f"{path} must be an array of objects")
    return value


def _exact_fields(item: Mapping[str, Any], path: str, expected: set[str]) -> None:
    fields = set(item)
    if not all(isinstance(key, str) for key in fields):
        raise ModelResponseError(f"{path} keys must be strings")
    unknown = sorted(fields - expected)
    missing = sorted(expected - fields)
    if unknown:
        raise ModelResponseError(f"unknown {path} field(s): {', '.join(unknown)}")
    if missing:
        raise ModelResponseError(f"missing {path} field(s): {', '.join(missing)}")


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and "\x00" in value:
            raise ModelResponseError(f"{path} must not contain NUL")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelResponseError(f"{path} must be a finite JSON number")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {
            key: _json_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    raise ModelResponseError(f"{path} must be a JSON value")


def _validate_current_layers(
    decoded: Mapping[str, Any],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    changes: list[Mapping[str, Any]] = []
    for index, item in enumerate(_object_list(decoded["changes"], "changes")):
        path = f"changes[{index}]"
        _exact_fields(item, path, {"target", "before", "after"})
        changes.append(
            {
                "target": _non_empty_string(item["target"], f"{path}.target"),
                "before": _json_value(item["before"], f"{path}.before"),
                "after": _json_value(item["after"], f"{path}.after"),
            }
        )

    evidence: list[Mapping[str, Any]] = []
    for index, item in enumerate(_object_list(decoded["evidence"], "evidence")):
        path = f"evidence[{index}]"
        _exact_fields(item, path, {"fact", "source"})
        evidence.append(
            {
                "fact": _non_empty_string(item["fact"], f"{path}.fact"),
                "source": _non_empty_string(item["source"], f"{path}.source"),
            }
        )

    hypotheses: list[Mapping[str, Any]] = []
    for index, item in enumerate(_object_list(decoded["hypotheses"], "hypotheses")):
        path = f"hypotheses[{index}]"
        _exact_fields(item, path, {"claim", "confidence", "evidence_refs"})
        confidence = item["confidence"]
        if confidence not in {"low", "medium", "high"}:
            raise ModelResponseError(
                f"{path}.confidence must be low, medium, or high"
            )
        refs = item["evidence_refs"]
        if (
            not isinstance(refs, list)
            or not refs
            or len(refs) > 32
            or any(type(ref) is not int or ref < 0 for ref in refs)
            or len(set(refs)) != len(refs)
        ):
            raise ModelResponseError(
                f"{path}.evidence_refs must contain unique non-negative integers"
            )
        if any(ref >= len(evidence) for ref in refs):
            raise ModelResponseError(f"{path}.evidence_refs contains an invalid index")
        hypotheses.append(
            {
                "claim": _non_empty_string(item["claim"], f"{path}.claim"),
                "confidence": confidence,
                "evidence_refs": list(refs),
            }
        )

    predictions: list[Mapping[str, Any]] = []
    for index, item in enumerate(_object_list(decoded["predictions"], "predictions")):
        path = f"predictions[{index}]"
        _exact_fields(item, path, {"metric", "expected_direction", "reason"})
        direction = item["expected_direction"]
        if direction not in {"increase", "decrease", "unchanged"}:
            raise ModelResponseError(
                f"{path}.expected_direction must be increase, decrease, or unchanged"
            )
        reason = _non_empty_string(item["reason"], f"{path}.reason")
        match = _PREDICTION_REASON.fullmatch(reason)
        if match is None or int(match.group(1)) >= len(hypotheses):
            raise ModelResponseError(
                f"{path}.reason must reference an existing hypothesis[index]"
            )
        predictions.append(
            {
                "metric": _non_empty_string(item["metric"], f"{path}.metric"),
                "expected_direction": direction,
                "reason": reason,
            }
        )
    return tuple(changes), tuple(evidence), tuple(hypotheses), tuple(predictions)


def validate_model_response(
    value: str | bytes | Mapping[str, Any],
    *,
    expected_round: int | None = None,
    allow_legacy: bool = False,
) -> ModelResponse:
    """Validate JSON content exactly.

    New model completions must use the evidence-graded four-layer protocol.
    The older ``optimization_summary`` and ``change_summary`` protocols are
    accepted only when a caller explicitly identifies an already-committed
    historical artifact.
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
    if fields == _LEGACY_V2_FIELDS:
        legacy_summary_field = "optimization_summary"
    elif fields == _LEGACY_V1_FIELDS:
        legacy_summary_field = "change_summary"
    if legacy_summary_field is not None and not allow_legacy:
        raise ModelResponseError(
            "legacy response fields are allowed only for committed historical artifacts"
        )
    expected_fields = (
        _LEGACY_V2_FIELDS
        if legacy_summary_field == "optimization_summary"
        else _LEGACY_V1_FIELDS
        if legacy_summary_field == "change_summary"
        else _CURRENT_FIELDS
    )
    unknown = sorted(fields - expected_fields)
    missing = sorted(expected_fields - fields)
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
        changes: tuple[Mapping[str, Any], ...] = ()
        evidence: tuple[Mapping[str, Any], ...] = ()
        hypotheses: tuple[Mapping[str, Any], ...] = ()
        predictions: tuple[Mapping[str, Any], ...] = ()
    else:
        changes, evidence, hypotheses, predictions = _validate_current_layers(decoded)
        legacy_intent = None
    return ModelResponse(
        status=status,
        round=round_number,
        changes=changes,
        evidence=evidence,
        hypotheses=hypotheses,
        predictions=predictions,
        code=code,
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
