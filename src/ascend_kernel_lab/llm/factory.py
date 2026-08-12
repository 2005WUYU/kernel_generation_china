"""Gateway construction from the strict model configuration."""

from __future__ import annotations

from collections.abc import Mapping

from ascend_kernel_lab.config import ModelConfig

from .claude_cli import ClaudeCliGateway
from .openai_http import OpenAICompatibleGateway
from .types import ModelGateway


def create_model_gateway(
    config: ModelConfig, *, environ: Mapping[str, str] | None = None
) -> ModelGateway:
    if config.provider == "claude_cli":
        return ClaudeCliGateway(config, environ=environ)
    if config.provider == "openai_compatible":
        return OpenAICompatibleGateway(config, environ=environ)
    raise ValueError(
        f"provider {config.provider!r} requires an explicitly supplied replay/fake gateway"
    )
