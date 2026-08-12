"""Bounded OpenAI-compatible structured-output adapter."""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any, cast

from ascend_kernel_lab.config import ModelConfig

from .envelopes import normalize_openai_envelope
from .errors import (
    ModelAuthenticationError,
    ModelGatewayError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelTransientError,
)
from .safety import redact_text, sanitize_audit_value
from .types import ModelCompletion, ModelRequest

HTTPCall = Callable[[urllib.request.Request, float], bytes]


def _urlopen(
    request: urllib.request.Request,
    timeout: float,
    *,
    maximum_bytes: int = 4_194_304,
) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = cast(bytes, response.read(maximum_bytes + 1))
        if len(data) > maximum_bytes:
            raise ModelProtocolError(
                f"OpenAI-compatible response exceeds {maximum_bytes} bytes"
            )
        return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _parse_retry_after(value: str | None, *, now: float) -> float | None:
    """Parse either Retry-After seconds or an RFC 7231 HTTP date."""

    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            return None
        seconds = retry_at.timestamp() - now
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _safe_header_value(name: str, value: str) -> None:
    if not value or any(character in value for character in ("\r", "\n", "\x00")):
        raise ModelGatewayError(f"environment value for HTTP header {name!r} is invalid")
    if len(value.encode("utf-8")) > 8_192:
        raise ModelGatewayError(f"environment value for HTTP header {name!r} is too large")


class OpenAICompatibleGateway:
    """POST to ``/chat/completions`` using bearer auth from an env reference."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
        http_call: HTTPCall | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._source_env = dict(os.environ if environ is None else environ)
        if http_call is None:
            maximum_bytes = config.maximum_response_bytes

            def bounded_urlopen(request: urllib.request.Request, timeout: float) -> bytes:
                return _urlopen(request, timeout, maximum_bytes=maximum_bytes)

            self._http_call: HTTPCall = bounded_urlopen
        else:
            self._http_call = http_call
        self._sleep = sleeper
        self._random = random_source
        self._wall_clock = wall_clock

    def _headers(self) -> tuple[dict[str, str], tuple[str, ...]]:
        http = self._config.openai
        api_key = self._source_env.get(http.api_key_env)
        if not api_key:
            raise ModelAuthenticationError(
                f"required environment variable {http.api_key_env!r} is not set"
            )
        _safe_header_value("Authorization", api_key)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        secrets = [api_key]
        if http.organization_env:
            organization = self._source_env.get(http.organization_env)
            if organization:
                _safe_header_value("OpenAI-Organization", organization)
                headers["OpenAI-Organization"] = organization
                secrets.append(organization)
        for header, env_reference in http.extra_header_env.items():
            value = self._source_env.get(env_reference)
            if not value:
                raise ModelAuthenticationError(
                    f"required environment variable {env_reference!r} is not set"
                )
            _safe_header_value(header, value)
            headers[header] = value
            secrets.append(value)
        return headers, tuple(secrets)

    def _delay(self, retry_number: int) -> float:
        retry = self._config.retry
        base = min(
            retry.maximum_backoff_seconds,
            retry.initial_backoff_seconds * retry.multiplier ** max(0, retry_number - 1),
        )
        return max(0.0, base + base * retry.jitter_fraction * (2 * self._random() - 1))

    def _retry_delay(self, error: ModelTransientError, retry_number: int) -> float | None:
        if isinstance(error, ModelRateLimitError) and error.retry_after_seconds is not None:
            if error.retry_after_seconds > self._config.retry.maximum_retry_after_seconds:
                return None
            return error.retry_after_seconds
        return self._delay(retry_number)

    def _sanitize_transient(
        self,
        error: ModelTransientError,
        secrets: tuple[str, ...],
    ) -> ModelTransientError:
        message = redact_text(
            str(error),
            secrets=secrets,
            maximum_bytes=self._config.maximum_error_bytes,
        )
        if isinstance(error, ModelRateLimitError):
            return ModelRateLimitError(
                message,
                retry_after_seconds=error.retry_after_seconds,
            )
        if isinstance(error, ModelTimeoutError):
            return ModelTimeoutError(message)
        return ModelTransientError(message)

    def _request_wire(self, request: ModelRequest) -> bytes:
        model = request.model or self._config.model
        if not model.strip() or "\x00" in model:
            raise ModelProtocolError("model name must be non-empty and contain no NUL")
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ascend_kernel_candidate",
                    "strict": True,
                    "schema": dict(request.json_schema),
                },
            },
        }
        if self._config.reasoning_effort:
            body["reasoning_effort"] = self._config.reasoning_effort
        try:
            wire = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise ModelProtocolError("model request is not strict UTF-8 JSON") from None
        if len(wire) > self._config.maximum_request_bytes:
            raise ModelProtocolError(
                f"OpenAI-compatible request exceeds {self._config.maximum_request_bytes} bytes"
            )
        return wire

    def _decode_response(self, raw_bytes: bytes, secrets: tuple[str, ...]) -> ModelCompletion:
        if not isinstance(raw_bytes, bytes):
            raise ModelProtocolError("OpenAI-compatible transport returned a non-bytes response")
        if len(raw_bytes) > self._config.maximum_response_bytes:
            raise ModelProtocolError(
                f"OpenAI-compatible response exceeds {self._config.maximum_response_bytes} bytes"
            )
        try:
            raw: Any = json.loads(
                raw_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ModelProtocolError(
                "OpenAI-compatible response is not strict UTF-8 JSON"
            ) from None
        if not isinstance(raw, dict):
            raise ModelProtocolError("OpenAI-compatible response must be a JSON object")
        sanitized = sanitize_audit_value(
            raw,
            secrets=secrets,
            maximum_bytes=self._config.maximum_response_bytes,
        )
        if not isinstance(sanitized, dict):  # pragma: no cover - guarded by raw check
            raise ModelProtocolError("OpenAI-compatible response must be a JSON object")
        return normalize_openai_envelope(sanitized)

    def complete(self, request: ModelRequest) -> ModelCompletion:
        endpoint = self._config.openai.base_url.rstrip("/") + "/chat/completions"
        wire = self._request_wire(request)
        headers, secrets = self._headers()
        timeout = request.timeout_seconds or self._config.request_timeout_seconds
        last_error: ModelTransientError | None = None
        for attempt in range(1, self._config.api_attempts + 1):
            http_request = urllib.request.Request(
                endpoint,
                data=wire,
                headers=headers,
                method="POST",
            )
            try:
                return self._decode_response(self._http_call(http_request, timeout), secrets)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403}:
                    raise ModelAuthenticationError(
                        f"OpenAI-compatible authentication failed with HTTP status {exc.code}"
                    ) from exc
                if exc.code == 429:
                    retry_after = _parse_retry_after(
                        exc.headers.get("Retry-After") if exc.headers is not None else None,
                        now=self._wall_clock(),
                    )
                    last_error = ModelRateLimitError(
                        "OpenAI-compatible rate limited the request (HTTP 429)",
                        retry_after_seconds=retry_after,
                    )
                elif exc.code in {408, 409, 425} or 500 <= exc.code < 600:
                    last_error = ModelTransientError(
                        f"OpenAI-compatible transient HTTP status {exc.code}"
                    )
                else:
                    raise ModelGatewayError(
                        f"OpenAI-compatible HTTP status {exc.code} is not retryable"
                    ) from exc
            except TimeoutError:
                last_error = ModelTimeoutError(
                    f"model HTTP request timed out after {timeout:g} seconds"
                )
            except urllib.error.URLError as exc:
                detail = redact_text(
                    str(exc.reason),
                    secrets=secrets,
                    maximum_bytes=self._config.maximum_error_bytes,
                )
                if isinstance(exc.reason, TimeoutError):
                    last_error = ModelTimeoutError(
                        f"model HTTP request timed out after {timeout:g} seconds"
                    )
                else:
                    last_error = ModelTransientError(
                        f"model HTTP transport failed: {detail or 'unknown transport error'}"
                    )
            except ModelTransientError as exc:
                last_error = self._sanitize_transient(exc, secrets)
            if attempt == self._config.api_attempts:
                break
            delay = self._retry_delay(last_error, attempt)
            if delay is None:
                raise last_error
            self._sleep(delay)
        raise last_error or ModelGatewayError("OpenAI-compatible request failed")

    generate = complete
