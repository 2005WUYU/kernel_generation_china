"""Provider-neutral, structured and tool-isolated model access."""

from .claude_cli import ClaudeCliGateway
from .errors import (
    ModelAuthenticationError,
    ModelCapabilityError,
    ModelGatewayError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    ModelTimeoutError,
    ModelTransientError,
    TruncatedResponseError,
)
from .factory import create_model_gateway
from .openai_http import OpenAICompatibleGateway
from .prompt import SYSTEM_PROMPT, PromptBuilder
from .response import (
    ModelResponse,
    ValidatedModelResponse,
    complete_model_response,
    validate_completion,
    validate_model_response,
)
from .testing import FakeGateway, ReplayGateway
from .types import ModelCompletion, ModelGateway, ModelRequest

AIPingClaudeCliGateway = ClaudeCliGateway

__all__ = [
    "SYSTEM_PROMPT",
    "AIPingClaudeCliGateway",
    "ClaudeCliGateway",
    "FakeGateway",
    "ModelAuthenticationError",
    "ModelCapabilityError",
    "ModelCompletion",
    "ModelGateway",
    "ModelGatewayError",
    "ModelProtocolError",
    "ModelRateLimitError",
    "ModelRequest",
    "ModelResponse",
    "ModelResponseAttemptsExhausted",
    "ModelResponseError",
    "ModelTimeoutError",
    "ModelTransientError",
    "OpenAICompatibleGateway",
    "PromptBuilder",
    "ReplayGateway",
    "TruncatedResponseError",
    "ValidatedModelResponse",
    "complete_model_response",
    "create_model_gateway",
    "validate_completion",
    "validate_model_response",
]
