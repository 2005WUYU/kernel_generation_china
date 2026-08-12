"""Provider-envelope normalization kept separate for deterministic testing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .errors import ModelProtocolError
from .types import ModelCompletion


def decode_json_output(stdout: str) -> Mapping[str, Any]:
    """Decode a JSON document, tolerating CLI diagnostics before the last line."""

    text = stdout.strip()
    if not text:
        raise ModelProtocolError("provider returned empty stdout")
    candidates = [text]
    candidates.extend(line.strip() for line in reversed(text.splitlines()) if line.strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ModelProtocolError("provider stdout did not contain a JSON object")


def _lookup(value: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def _content_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, list):
        pieces: list[str] = []
        for block in value:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                elif block.get("type") in {"json", "output_json"} and "value" in block:
                    rendered = _content_text(block["value"])
                    if rendered is not None:
                        pieces.append(rendered)
        if pieces:
            return "".join(pieces)
    return None


def normalize_claude_envelope(raw: Mapping[str, Any]) -> ModelCompletion:
    """Normalize Claude CLI, AIPing proxy, and schema-output envelope variants."""

    if raw.get("is_error") is True or raw.get("type") == "error":
        message = _lookup(raw, ("error", "message"), ("message",))
        raise ModelProtocolError(f"provider returned an error envelope: {message or 'unknown error'}")

    # With --json-schema Claude CLI exposes this field; proxy versions sometimes
    # use output/result/content or the OpenAI-compatible message shape instead.
    candidates = (
        raw.get("structured_output"),
        _lookup(raw, ("result", "structured_output")),
        raw.get("output"),
        raw.get("result"),
        _lookup(raw, ("message", "content")),
        raw.get("content"),
        _lookup(raw, ("response", "content")),
    )
    content = next((text for item in candidates if (text := _content_text(item)) is not None), None)

    # A bare schema object is also a valid result from lightweight AIPing wrappers.
    if content is None and {"status", "round", "code"}.issubset(raw):
        content = _content_text(raw)
    if content is None:
        raise ModelProtocolError("provider JSON envelope contains no model content")

    finish_reason = _lookup(
        raw,
        ("finish_reason",),
        ("stop_reason",),
        ("message", "stop_reason"),
        ("response", "finish_reason"),
    )
    if finish_reason is None and raw.get("subtype") == "success":
        finish_reason = "stop"
    if finish_reason is not None and not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    request_id = _lookup(raw, ("request_id",), ("id",), ("session_id",))
    usage = _lookup(raw, ("usage",), ("result", "usage"))
    return ModelCompletion(
        content=content,
        finish_reason=finish_reason,
        raw_response=raw,
        request_id=str(request_id) if request_id is not None else None,
        usage=usage if isinstance(usage, Mapping) else {},
    )


def normalize_openai_envelope(raw: Mapping[str, Any]) -> ModelCompletion:
    if isinstance(raw.get("error"), Mapping):
        raise ModelProtocolError(
            f"provider returned an error envelope: {raw['error'].get('message', 'unknown error')}"
        )
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ModelProtocolError("OpenAI-compatible response has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ModelProtocolError("OpenAI-compatible choice has no message")
    content = _content_text(message.get("content"))
    if content is None:
        content = _content_text(message.get("parsed"))
    if content is None:
        raise ModelProtocolError("OpenAI-compatible message has no content")
    finish_reason = choice.get("finish_reason")
    usage_value = raw.get("usage")
    usage: Mapping[str, Any] = usage_value if isinstance(usage_value, Mapping) else {}
    return ModelCompletion(
        content=content,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        raw_response=raw,
        request_id=str(raw["id"]) if raw.get("id") is not None else None,
        usage=usage,
    )
