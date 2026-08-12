"""Provider-neutral LLM request and completion types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ascend_kernel_lab.schemas import model_response_schema


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One stateless request. Both prompts are sent on every invocation."""

    system_prompt: str
    user_prompt: str
    json_schema: Mapping[str, Any] = field(default_factory=model_response_schema)
    model: str | None = None
    timeout_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not self.user_prompt.strip():
            raise ValueError("user_prompt must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        object.__setattr__(self, "json_schema", MappingProxyType(dict(self.json_schema)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def stdin_text(self) -> str:
        """Render a self-contained prompt without putting prompt text in argv."""

        return f"<system>\n{self.system_prompt}\n</system>\n\n<request>\n{self.user_prompt}\n</request>\n"


@dataclass(frozen=True, slots=True)
class ModelCompletion:
    """Normalized provider result, retaining the raw envelope for auditing."""

    content: str
    finish_reason: str | None
    raw_response: Mapping[str, Any]
    request_id: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        object.__setattr__(self, "raw_response", MappingProxyType(dict(self.raw_response)))
        object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))


@runtime_checkable
class ModelGateway(Protocol):
    """Minimal gateway contract used by the experiment controller."""

    def complete(self, request: ModelRequest) -> ModelCompletion:
        """Return one provider-normalized completion or raise ModelGatewayError."""
