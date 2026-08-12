"""Errors raised by model gateways and response validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import ModelCompletion


class ModelGatewayError(RuntimeError):
    """The external model process or service did not complete successfully."""


class ModelTransientError(ModelGatewayError):
    """A transport/service failure that is safe to retry with backoff."""


class ModelTimeoutError(ModelTransientError):
    """A model request exceeded its configured deadline."""


class ModelProtocolError(ModelGatewayError):
    """A successful transport returned an unusable provider envelope."""


class ModelAuthenticationError(ModelGatewayError):
    """Provider credentials were rejected; retrying cannot repair them."""


class ModelCapabilityError(ModelGatewayError):
    """The installed provider client cannot prove the required safe mode."""


class ModelRateLimitError(ModelTransientError):
    """The provider asked the caller to retry later."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class ModelResponseError(ValueError):
    """Model content did not satisfy the candidate response protocol."""


class TruncatedResponseError(ModelResponseError):
    """The provider explicitly reported that generation hit a length limit."""


class ModelResponseAttemptsExhausted(ModelResponseError):
    """Every bounded structured-output/repair attempt was invalid."""

    def __init__(self, message: str, attempts: tuple[ModelCompletion, ...]) -> None:
        self.attempts = attempts
        super().__init__(message)
