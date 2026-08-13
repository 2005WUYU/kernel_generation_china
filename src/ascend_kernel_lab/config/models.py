"""Configuration value objects.

Secrets are represented exclusively as environment-variable *names*.  The
configuration objects are therefore safe to serialize into run manifests.
Gateways resolve the referenced variable only immediately before invocation.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HTTP_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_RESERVED_HTTP_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "content-type",
    "host",
    "proxy-authorization",
    "transfer-encoding",
}
_MAXIMUM_PROVIDER_PAYLOAD_BYTES = 16 * 1024 * 1024


def _positive(name: str, value: int | float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")


def _nonempty(name: str, value: str) -> None:
    if not value or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string")


def _env_name(name: str, value: str | None) -> None:
    if value is not None and not _ENV_NAME.fullmatch(value):
        raise ValueError(f"{name} must be an environment variable name")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    maximum_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 20.0
    maximum_retry_after_seconds: float = 120.0
    multiplier: float = 2.0
    jitter_fraction: float = 0.1

    def __post_init__(self) -> None:
        if type(self.maximum_attempts) is not int:
            raise ValueError("maximum_attempts must be an integer")
        _positive("maximum_attempts", self.maximum_attempts)
        _positive("initial_backoff_seconds", self.initial_backoff_seconds)
        _positive("maximum_backoff_seconds", self.maximum_backoff_seconds)
        _positive("maximum_retry_after_seconds", self.maximum_retry_after_seconds)
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("maximum_backoff_seconds must not be less than initial_backoff_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least one")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be between zero and one")


@dataclass(frozen=True, slots=True)
class OpenAIHTTPConfig:
    base_url: str = "https://api.moonshot.cn/v1"
    api_key_env: str = "KIMI_API_KEY"
    organization_env: str | None = None
    extra_header_env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("base_url must not contain credentials, a query, or a fragment")
        _env_name("api_key_env", self.api_key_env)
        _env_name("organization_env", self.organization_env)
        normalized_headers: set[str] = set()
        for header, env_name in self.extra_header_env.items():
            _nonempty("extra header name", header)
            if not _HTTP_HEADER_NAME.fullmatch(header):
                raise ValueError(f"invalid HTTP header name: {header!r}")
            if header.lower() in _RESERVED_HTTP_HEADERS:
                raise ValueError(f"extra_header_env must not override reserved header {header!r}")
            if header.lower() in normalized_headers:
                raise ValueError("extra_header_env contains duplicate case-insensitive header names")
            normalized_headers.add(header.lower())
            _env_name(f"environment reference for {header}", env_name)
        object.__setattr__(self, "extra_header_env", dict(self.extra_header_env))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str = "claude_cli"
    model: str = "kimi-k3"
    reasoning_effort: str = "high"
    structured_output: bool = True
    request_timeout_seconds: float = 300.0
    claude_capability_timeout_seconds: float = 15.0
    maximum_request_bytes: int = 4_194_304
    maximum_response_bytes: int = 4_194_304
    maximum_error_bytes: int = 65_536
    # Deprecated compatibility alias. New configurations use retry.maximum_attempts.
    maximum_api_retries: int | None = None
    maximum_format_repair_retries: int = 1
    retry: RetryConfig = field(default_factory=RetryConfig)
    claude_executable: str = "claude"
    claude_extra_args: tuple[str, ...] = ()
    anthropic_base_url_env: str = "ANTHROPIC_BASE_URL"
    anthropic_auth_token_env: str = "ANTHROPIC_AUTH_TOKEN"
    anthropic_model_env: str | None = "ANTHROPIC_MODEL"
    openai: OpenAIHTTPConfig = field(default_factory=OpenAIHTTPConfig)

    def __post_init__(self) -> None:
        if self.provider not in {"claude_cli", "openai_compatible", "replay", "fake"}:
            raise ValueError("model.provider is not supported")
        _nonempty("model", self.model)
        _positive("request_timeout_seconds", self.request_timeout_seconds)
        _positive("claude_capability_timeout_seconds", self.claude_capability_timeout_seconds)
        for name in ("maximum_request_bytes", "maximum_response_bytes", "maximum_error_bytes"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            _positive(name, value)
            if value > _MAXIMUM_PROVIDER_PAYLOAD_BYTES:
                raise ValueError(f"{name} must not exceed 16 MiB")
        if self.maximum_api_retries is not None and (
            type(self.maximum_api_retries) is not int or self.maximum_api_retries < 0
        ):
            raise ValueError("maximum_api_retries must be a non-negative integer")
        if (
            self.maximum_api_retries is not None
            and self.maximum_api_retries + 1 != self.retry.maximum_attempts
        ):
            raise ValueError(
                "maximum_api_retries is a deprecated alias and must equal "
                "retry.maximum_attempts - 1 when both are configured"
            )
        if self.maximum_format_repair_retries < 0:
            raise ValueError("maximum_format_repair_retries must be non-negative")
        _nonempty("claude_executable", self.claude_executable)
        for item in self.claude_extra_args:
            _nonempty("claude_extra_args item", item)
        if self.claude_extra_args:
            raise ValueError(
                "claude_extra_args must be empty so one-shot/tool-free invocation remains provable"
            )
        _env_name("anthropic_base_url_env", self.anthropic_base_url_env)
        _env_name("anthropic_auth_token_env", self.anthropic_auth_token_env)
        _env_name("anthropic_model_env", self.anthropic_model_env)

    @property
    def api_attempts(self) -> int:
        """Effective attempts from the canonical retry block or its legacy alias."""

        return self.retry.maximum_attempts


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    device: str = "npu:0"
    exclusive_device: bool = True
    fresh_process_per_stage: bool = True
    heartbeat_seconds: float = 10.0
    lease_seconds: float = 120.0
    poll_seconds: float = 1.0
    maximum_job_attempts: int = 3

    def __post_init__(self) -> None:
        _nonempty("device", self.device)
        for name in ("heartbeat_seconds", "lease_seconds", "poll_seconds"):
            _positive(name, getattr(self, name))
        _positive("maximum_job_attempts", self.maximum_job_attempts)
        if self.lease_seconds <= self.heartbeat_seconds:
            raise ValueError("lease_seconds must be greater than heartbeat_seconds")


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    source_check_seconds: float = 10.0
    compile_seconds: float = 180.0
    correctness_case_seconds: float = 30.0
    benchmark_seconds: float = 300.0
    profile_seconds: float = 600.0

    def __post_init__(self) -> None:
        for value in dataclasses.fields(self):
            _positive(value.name, getattr(self, value.name))


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    comparison_baseline: str = "pytorch_eager"
    warmup: int = 20
    measurement_batches: int = 7
    target_batch_time_ms: float = 200.0
    maximum_cv: float = 0.05
    rerun_if_unstable: bool = True

    def __post_init__(self) -> None:
        if self.comparison_baseline != "pytorch_eager":
            raise ValueError("comparison_baseline must be pytorch_eager")
        _positive("warmup", self.warmup)
        _positive("measurement_batches", self.measurement_batches)
        _positive("target_batch_time_ms", self.target_batch_time_ms)
        if not 0 < self.maximum_cv <= 1:
            raise ValueError("maximum_cv must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    mode: str = "quick"
    run_after_correctness: bool = True
    run_for_final_best: bool = True
    warmup: int = 1
    iterations: int = 1
    mandatory_groups: tuple[str, ...] = ("task_time", "pipe_utilization")
    optional_groups: tuple[str, ...] = ("memory", "l2_cache", "resource_conflict")
    full_profile_for_final_best: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"quick", "full"}:
            raise ValueError("profile mode must be quick or full")
        _positive("profile warmup", self.warmup)
        _positive("profile iterations", self.iterations)
        for group in self.mandatory_groups + self.optional_groups:
            _nonempty("profile group", group)
        if len(set(self.mandatory_groups + self.optional_groups)) != len(
            self.mandatory_groups + self.optional_groups
        ):
            raise ValueError("profile groups must be unique")


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    allowed_import_roots: tuple[str, ...] = ("torch", "triton")
    forbidden_import_roots: tuple[str, ...] = (
        "ctypes", "http", "multiprocessing", "os", "pathlib", "requests",
        "shutil", "socket", "subprocess", "urllib",
    )
    forbidden_call_prefixes: tuple[str, ...] = (
        "torch.matmul", "torch.mm", "torch.bmm", "torch.softmax", "torch.sum",
        "torch.nn", "torch.nn.functional", "torch.compile", "torch.ops",
    )
    maximum_source_bytes: int = 262_144
    required_entrypoint: str = "custom_op"

    def __post_init__(self) -> None:
        _positive("maximum_source_bytes", self.maximum_source_bytes)
        _nonempty("required_entrypoint", self.required_entrypoint)


@dataclass(frozen=True, slots=True)
class StorageConfig:
    database: str = "sqlite:///runs/metadata.db"
    artifact_root: str = "runs"
    task_root: str = "task_specs"

    def __post_init__(self) -> None:
        if not self.database.startswith("sqlite:///"):
            raise ValueError("only sqlite:/// database URLs are supported")
        _nonempty("artifact_root", self.artifact_root)
        _nonempty("task_root", self.task_root)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    id: str
    rounds_per_task: int
    tasks: tuple[str, ...]
    model: ModelConfig
    worker: WorkerConfig
    timeouts: TimeoutConfig
    benchmark: BenchmarkConfig
    profile: ProfileConfig
    storage: StorageConfig
    security: SecurityConfig = field(default_factory=SecurityConfig)
    task_concurrency: int = 1
    config_path: Path = field(default=Path("<memory>"), repr=False, compare=False)
    project_root: Path = field(default_factory=Path.cwd, repr=False, compare=False)

    def __post_init__(self) -> None:
        _nonempty("experiment.id", self.id)
        _positive("rounds_per_task", self.rounds_per_task)
        _positive("task_concurrency", self.task_concurrency)
        if not self.tasks:
            raise ValueError("experiment.tasks must not be empty")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("experiment.tasks must be unique")
        for task in self.tasks:
            _nonempty("task id", task)
        object.__setattr__(self, "config_path", self.config_path.resolve())
        object.__setattr__(self, "project_root", self.project_root.resolve())

    @property
    def artifact_root(self) -> Path:
        return self._resolve_project_path(self.storage.artifact_root)

    @property
    def task_root(self) -> Path:
        return self._resolve_project_path(self.storage.task_root)

    @property
    def db_path(self) -> Path:
        raw_path = unquote(self.storage.database.removeprefix("sqlite:///"))
        return self._resolve_project_path(raw_path)

    def _resolve_project_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe representation containing no secret values."""

        value = dataclasses.asdict(self)
        value.pop("config_path", None)
        value.pop("project_root", None)
        return value
