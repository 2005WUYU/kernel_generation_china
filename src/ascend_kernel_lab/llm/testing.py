"""Deterministic gateways for tests, dry-runs, and offline trajectory replay."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .errors import ModelGatewayError
from .types import ModelCompletion, ModelRequest

CompletionLike = ModelCompletion | str | Mapping[str, Any] | Exception


def _completion(value: CompletionLike) -> ModelCompletion:
    if isinstance(value, Exception):
        raise value
    if isinstance(value, ModelCompletion):
        return value
    raw: dict[str, Any]
    if isinstance(value, str):
        content = value
        raw = {"content": value, "finish_reason": "stop"}
    else:
        content = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        raw = {"structured_output": dict(value), "finish_reason": "stop"}
    return ModelCompletion(content=content, finish_reason="stop", raw_response=raw)


class ReplayGateway:
    """Consume a finite, thread-safe sequence of recorded completions."""

    def __init__(self, completions: Iterable[CompletionLike]) -> None:
        self._completions = deque(completions)
        self._lock = threading.Lock()
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        with self._lock:
            self.requests.append(request)
            if not self._completions:
                raise ModelGatewayError("replay is exhausted")
            value = self._completions.popleft()
        return _completion(value)

    generate = complete

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._completions)


class FakeGateway:
    """Configurable gateway whose responder may inspect each request."""

    def __init__(
        self,
        response: CompletionLike | Callable[[ModelRequest], CompletionLike],
    ) -> None:
        self._response = response
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelCompletion:
        self.requests.append(request)
        value = self._response(request) if callable(self._response) else self._response
        return _completion(value)

    generate = complete
