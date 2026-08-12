"""Provider-output bounds and audit-safe redaction helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import ModelProtocolError

_SENSITIVE_KEY = re.compile(
    r"(?i)^(?:api[_-]?key|auth(?:entication)?[_-]?token|authorization|bearer|"
    r"cookie|credential|password|private[_-]?key|secret|session[_-]?token|token)$"
)
_BEARER = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}")
_OPENAI_STYLE = re.compile(r"\bsk-[a-zA-Z0-9_-]{8,}")
_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|auth[_-]?token|authorization|password|secret)"
    r"(\s*[:=]\s*)([^\s,;]{4,})"
)


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def require_text_limit(value: str, maximum_bytes: int, label: str) -> None:
    if utf8_size(value) > maximum_bytes:
        raise ModelProtocolError(f"{label} exceeds {maximum_bytes} UTF-8 bytes")


def truncate_utf8(value: str, maximum_bytes: int) -> str:
    if maximum_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    suffix = b"...<truncated>"
    if maximum_bytes <= len(suffix):
        return suffix[:maximum_bytes].decode("ascii")
    keep = max(0, maximum_bytes - len(suffix))
    return (encoded[:keep] + suffix).decode("utf-8", errors="ignore")


def _is_sensitive_key(key: str) -> bool:
    if _SENSITIVE_KEY.fullmatch(key):
        return True
    normalized = key.replace("-", "_").lower()
    parts = normalized.split("_")
    return (
        "token" in parts
        or "secret" in parts
        or normalized.endswith("_credential")
        or normalized.endswith("_private_key")
        or normalized in {"set_cookie", "x_api_key"}
    )


def redact_text(
    value: str,
    *,
    secrets: Sequence[str] = (),
    maximum_bytes: int | None = None,
) -> str:
    """Remove explicit credentials and common credential-shaped fragments."""

    result = value
    for secret in secrets:
        if not secret:
            continue
        result = result.replace(secret, "<redacted>")
    result = _BEARER.sub("Bearer <redacted>", result)
    result = _OPENAI_STYLE.sub("sk-<redacted>", result)
    result = _ASSIGNMENT.sub(r"\1\2<redacted>", result)
    return truncate_utf8(result, maximum_bytes) if maximum_bytes is not None else result


def sanitize_audit_value(
    value: Any,
    *,
    secrets: Sequence[str] = (),
    maximum_bytes: int,
) -> Any:
    """Return a detached JSON value with credential-bearing fields redacted."""

    def clean(item: Any, depth: int) -> Any:
        if depth > 64:
            raise ModelProtocolError("provider response nesting exceeds 64 levels")
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            return redact_text(item, secrets=secrets)
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ModelProtocolError("provider response contains a non-string key")
                result[key] = "<redacted>" if _is_sensitive_key(key) else clean(child, depth + 1)
            return result
        if isinstance(item, list):
            return [clean(child, depth + 1) for child in item]
        raise ModelProtocolError(
            f"provider response contains unsupported {type(item).__name__} value"
        )

    cleaned = clean(value, 0)
    try:
        encoded = json.dumps(
            cleaned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelProtocolError("provider response is not strict JSON") from exc
    if len(encoded) > maximum_bytes:
        raise ModelProtocolError(
            f"sanitized provider response exceeds {maximum_bytes} UTF-8 bytes"
        )
    return cleaned


__all__ = [
    "redact_text",
    "require_text_limit",
    "sanitize_audit_value",
    "truncate_utf8",
    "utf8_size",
]
