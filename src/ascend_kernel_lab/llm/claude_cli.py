"""One-shot, capability-checked, tool-free Claude CLI adapter for AIPing."""

from __future__ import annotations

import errno
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ascend_kernel_lab.config import ModelConfig

from .envelopes import decode_json_output, normalize_claude_envelope
from .errors import (
    ModelAuthenticationError,
    ModelCapabilityError,
    ModelGatewayError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelTransientError,
)
from .safety import redact_text, require_text_limit, sanitize_audit_value
from .types import ModelCompletion, ModelRequest

ProcessRunner = Callable[
    [Sequence[str], str, Path, Mapping[str, str], float], tuple[int, str, str]
]

_SAFE_INHERITED_ENV = {
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_AUTH_FAILURE = re.compile(
    r"(?i)(?:\b(?:401|403)\b|unauthori[sz]ed|forbidden|authentication\s+failed|"
    r"invalid\s+(?:api[_ -]?key|auth(?:entication)?[_ -]?token|credential))"
)
_RATE_LIMIT = re.compile(r"(?i)(?:\b429\b|rate[_ -]?limit|too\s+many\s+requests)")
_TRANSIENT_FAILURE = re.compile(
    r"(?i)(?:\b5\d\d\b|bad\s+gateway|service\s+unavailable|gateway\s+timeout|"
    r"internal\s+server\s+error|temporar(?:y|ily)\s+unavailable|timed?\s*out|"
    r"timeout|econnreset|etimedout|connection\s+reset|network\s+error)"
)
_RETRY_AFTER = re.compile(r"(?i)retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)")
_UNSUPPORTED_CLAUDE_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "title",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
}


def _claude_json_schema(value: Any) -> Any:
    """Project a strict schema onto Claude structured output's supported subset."""

    if isinstance(value, Mapping):
        return {
            key: _claude_json_schema(item)
            for key, item in value.items()
            if key not in _UNSUPPORTED_CLAUDE_SCHEMA_KEYWORDS
        }
    if isinstance(value, list):
        return [_claude_json_schema(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ClaudeCliCapabilities:
    """Safe invocation switches proven by one installed CLI's help output."""

    print_option: str
    output_format_option: str
    tools_option: str
    json_schema_option: str | None
    no_session_persistence_option: str | None
    model_option: str | None


def _has_option(help_text: str, option: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(option)}(?![A-Za-z0-9_-])",
        help_text,
    ) is not None


def _option_block(help_text: str, *options: str) -> str:
    lines = help_text.splitlines()
    for index, line in enumerate(lines):
        if not any(_has_option(line, option) for option in options):
            continue
        block = [line]
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if stripped.startswith("-") and re.match(r"^-{1,2}[A-Za-z]", stripped):
                break
            if not stripped and len(block) > 1:
                break
            block.append(following)
        return "\n".join(block)
    return ""


def parse_claude_cli_capabilities(
    help_text: str,
    *,
    require_json_schema: bool,
) -> ClaudeCliCapabilities:
    """Accept only help text that proves non-interactive JSON and disabled tools."""

    print_option = "--print" if _has_option(help_text, "--print") else "-p"
    print_block = _option_block(help_text, "--print", "-p").lower()
    if not _has_option(help_text, print_option) or not any(
        marker in print_block
        for marker in ("non-interactive", "non interactive", "one-shot", "exit")
    ):
        raise ModelCapabilityError(
            "Claude CLI help does not prove a non-interactive --print/-p mode"
        )

    output_block = _option_block(help_text, "--output-format").lower()
    if not output_block or re.search(r"\bjson\b", output_block) is None:
        raise ModelCapabilityError(
            "Claude CLI help does not prove JSON support for --output-format"
        )

    tools_block = _option_block(help_text, "--tools").lower()
    if not tools_block:
        raise ModelCapabilityError("Claude CLI help does not advertise required --tools support")

    json_schema_option: str | None = None
    if require_json_schema:
        if not _has_option(help_text, "--json-schema"):
            raise ModelCapabilityError(
                "Claude CLI help does not advertise required --json-schema support"
            )
        json_schema_option = "--json-schema"

    return ClaudeCliCapabilities(
        print_option=print_option,
        output_format_option="--output-format",
        tools_option="--tools",
        json_schema_option=json_schema_option,
        no_session_persistence_option=(
            "--no-session-persistence"
            if _has_option(help_text, "--no-session-persistence")
            else None
        ),
        model_option="--model" if _has_option(help_text, "--model") else None,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def _subprocess_runner(
    argv: Sequence[str],
    stdin: str,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    *,
    maximum_stdout_bytes: int = 4_194_304,
    maximum_stderr_bytes: int = 65_536,
) -> tuple[int, str, str]:
    """Run with file-backed capture and terminate as soon as a bound is crossed."""

    try:
        stdin_bytes = stdin.encode("utf-8")
    except UnicodeEncodeError:
        raise ModelProtocolError("Claude CLI stdin is not valid UTF-8") from None
    with (
        tempfile.TemporaryFile(mode="w+b") as stdin_file,
        tempfile.TemporaryFile(mode="w+b") as stdout_file,
        tempfile.TemporaryFile(mode="w+b") as stderr_file,
    ):
        stdin_file.write(stdin_bytes)
        stdin_file.seek(0)
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=stderr_file,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            stdout_size = os.fstat(stdout_file.fileno()).st_size
            stderr_size = os.fstat(stderr_file.fileno()).st_size
            if stdout_size > maximum_stdout_bytes or stderr_size > maximum_stderr_bytes:
                _kill_process_group(process)
                process.wait()
                stream = "stdout" if stdout_size > maximum_stdout_bytes else "stderr"
                maximum = (
                    maximum_stdout_bytes if stream == "stdout" else maximum_stderr_bytes
                )
                raise ModelProtocolError(
                    f"Claude CLI {stream} exceeds configured {maximum}-byte limit"
                )
            if time.monotonic() >= deadline:
                _kill_process_group(process)
                process.wait()
                raise ModelTimeoutError(f"Claude CLI timed out after {timeout:g} seconds")
            time.sleep(0.01)

        stdout_size = os.fstat(stdout_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        if stdout_size > maximum_stdout_bytes or stderr_size > maximum_stderr_bytes:
            stream = "stdout" if stdout_size > maximum_stdout_bytes else "stderr"
            maximum = maximum_stdout_bytes if stream == "stdout" else maximum_stderr_bytes
            raise ModelProtocolError(
                f"Claude CLI {stream} exceeds configured {maximum}-byte limit"
            )
        stdout_file.seek(0)
        stderr_file.seek(0)
        try:
            stdout = stdout_file.read().decode("utf-8")
        except UnicodeDecodeError:
            raise ModelProtocolError("Claude CLI stdout is not valid UTF-8") from None
        stderr = stderr_file.read().decode("utf-8", errors="replace")
        return process.returncode, stdout, stderr


class ClaudeCliGateway:
    """Invoke a fresh Claude CLI process with prompt on stdin and all tools disabled."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
        runner: ProcessRunner | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._config = config
        self._source_env = dict(os.environ if environ is None else environ)
        if runner is None:
            maximum_stdout_bytes = config.maximum_response_bytes
            maximum_stderr_bytes = config.maximum_error_bytes

            def bounded_runner(
                argv: Sequence[str],
                stdin: str,
                cwd: Path,
                env: Mapping[str, str],
                timeout: float,
            ) -> tuple[int, str, str]:
                return _subprocess_runner(
                    argv,
                    stdin,
                    cwd,
                    env,
                    timeout,
                    maximum_stdout_bytes=maximum_stdout_bytes,
                    maximum_stderr_bytes=maximum_stderr_bytes,
                )

            self._runner: ProcessRunner = bounded_runner
        else:
            self._runner = runner
        self._sleep = sleeper
        self._random = random_source
        self._capabilities: ClaudeCliCapabilities | None = None
        self._capability_lock = threading.Lock()

    def _resolve_executable(self) -> str:
        executable = shutil.which(
            self._config.claude_executable,
            path=self._source_env.get("PATH"),
        )
        if executable is None:
            raise ModelCapabilityError(
                f"Claude CLI executable not found: {self._config.claude_executable}"
            )
        return executable

    @staticmethod
    def _isolated_environment(root: Path, base: Mapping[str, str]) -> dict[str, str]:
        env = dict(base)
        env.update(
            {
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CONFIG_DIR": str(root / "claude-config"),
                "HOME": str(root),
                "TMPDIR": str(root),
                "XDG_CACHE_HOME": str(root / "cache"),
                "XDG_CONFIG_HOME": str(root / "config"),
            }
        )
        return env

    def _probe_capabilities(self, executable: str) -> ClaudeCliCapabilities:
        safe_base = {
            key: value for key, value in self._source_env.items() if key in _SAFE_INHERITED_ENV
        }
        failures: list[str] = []
        with tempfile.TemporaryDirectory(prefix="ascend-kernel-claude-help-") as temp_dir:
            isolated_root = Path(temp_dir)
            env = self._isolated_environment(isolated_root, safe_base)
            for help_option in ("--help", "-h"):
                try:
                    return_code, stdout, stderr = self._runner(
                        [executable, help_option],
                        "",
                        isolated_root,
                        env,
                        self._config.claude_capability_timeout_seconds,
                    )
                    require_text_limit(
                        stdout,
                        self._config.maximum_response_bytes,
                        "Claude CLI help stdout",
                    )
                    require_text_limit(
                        stderr,
                        self._config.maximum_error_bytes,
                        "Claude CLI help stderr",
                    )
                except (ModelGatewayError, OSError) as exc:
                    failures.append(type(exc).__name__)
                    continue
                if return_code != 0:
                    failures.append(f"{help_option} exited with status {return_code}")
                    continue
                try:
                    return parse_claude_cli_capabilities(
                        f"{stdout}\n{stderr}",
                        require_json_schema=self._config.structured_output,
                    )
                except ModelCapabilityError as exc:
                    failures.append(str(exc))
        summary = "; ".join(failures[-2:]) or "no usable help output"
        raise ModelCapabilityError(f"Claude CLI safe capabilities could not be proven: {summary}")

    def _get_capabilities(self, executable: str) -> ClaudeCliCapabilities:
        with self._capability_lock:
            if self._capabilities is None:
                self._capabilities = self._probe_capabilities(executable)
            return self._capabilities

    def _environment(self, model: str) -> tuple[dict[str, str], tuple[str, ...]]:
        env = {
            key: value for key, value in self._source_env.items() if key in _SAFE_INHERITED_ENV
        }
        base_url = self._source_env.get(self._config.anthropic_base_url_env)
        if not base_url:
            raise ModelGatewayError(
                f"required environment variable {self._config.anthropic_base_url_env!r} is not set"
            )
        parts = urlsplit(base_url)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.netloc
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ModelGatewayError("configured ANTHROPIC_BASE_URL is not a safe absolute HTTP(S) URL")
        auth_token = self._source_env.get(self._config.anthropic_auth_token_env)
        if not auth_token:
            raise ModelAuthenticationError(
                f"required environment variable {self._config.anthropic_auth_token_env!r} is not set"
            )
        if any(character in auth_token for character in ("\r", "\n", "\x00")):
            raise ModelAuthenticationError("configured ANTHROPIC_AUTH_TOKEN is invalid")
        if len(auth_token.encode("utf-8")) > 8_192:
            raise ModelAuthenticationError("configured ANTHROPIC_AUTH_TOKEN is too large")
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_AUTH_TOKEN"] = auth_token
        env["ANTHROPIC_MODEL"] = model
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        return env, (auth_token,)

    def _request_payload(self, request: ModelRequest) -> tuple[str, str | None, str]:
        model = request.model or self._config.model
        if not model.strip() or any(character in model for character in ("\r", "\n", "\x00")):
            raise ModelProtocolError("model name must be non-empty and contain no control line break")
        try:
            model_size = len(model.encode("utf-8"))
        except UnicodeEncodeError:
            raise ModelProtocolError("model name is not valid UTF-8") from None
        if model_size > 512:
            raise ModelProtocolError("model name exceeds 512 UTF-8 bytes")
        stdin = request.stdin_text()
        schema: str | None = None
        if self._config.structured_output:
            try:
                # Claude Code 2.1.228 accepts a documented subset of JSON
                # Schema. The stricter limits remain enforced by
                # validate_model_response after the provider returns.
                cli_schema = _claude_json_schema(request.json_schema)
                schema = json.dumps(
                    cli_schema,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError, UnicodeError):
                raise ModelProtocolError("model JSON schema is not strict UTF-8 JSON") from None
        try:
            request_size = len(stdin.encode("utf-8")) + (
                len(schema.encode("utf-8")) if schema is not None else 0
            )
        except UnicodeEncodeError:
            raise ModelProtocolError("Claude CLI request is not valid UTF-8") from None
        if request_size > self._config.maximum_request_bytes:
            raise ModelProtocolError(
                f"Claude CLI request exceeds {self._config.maximum_request_bytes} bytes"
            )
        return stdin, schema, model

    @staticmethod
    def _argv(
        executable: str,
        capabilities: ClaudeCliCapabilities,
        *,
        schema: str | None,
        model: str,
    ) -> list[str]:
        argv = [
            executable,
            capabilities.print_option,
            capabilities.output_format_option,
            "json",
            capabilities.tools_option,
            "",
        ]
        if schema is not None:
            if capabilities.json_schema_option is None:  # pragma: no cover - parser invariant
                raise ModelCapabilityError("Claude CLI lacks required JSON schema support")
            argv.extend((capabilities.json_schema_option, schema))
        if capabilities.no_session_persistence_option is not None:
            argv.append(capabilities.no_session_persistence_option)
        if capabilities.model_option is not None:
            argv.extend((capabilities.model_option, model))
        return argv

    def _delay(self, retry_number: int) -> float:
        retry = self._config.retry
        base = min(
            retry.maximum_backoff_seconds,
            retry.initial_backoff_seconds * retry.multiplier ** max(0, retry_number - 1),
        )
        jitter = base * retry.jitter_fraction * (2 * self._random() - 1)
        return max(0.0, base + jitter)

    def _retry_delay(self, error: ModelTransientError, retry_number: int) -> float | None:
        if isinstance(error, ModelRateLimitError) and error.retry_after_seconds is not None:
            if error.retry_after_seconds > self._config.retry.maximum_retry_after_seconds:
                return None
            return error.retry_after_seconds
        return self._delay(retry_number)

    def _sanitize_transient(
        self,
        error: ModelTransientError,
        secrets: Sequence[str],
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

    def _classify_exit(
        self,
        return_code: int,
        stdout: str,
        stderr: str,
        secrets: Sequence[str],
    ) -> ModelGatewayError:
        raw_detail = stderr.strip() or stdout.strip()
        detail = redact_text(
            raw_detail,
            secrets=secrets,
            maximum_bytes=self._config.maximum_error_bytes,
        )
        classification_text = f"{stderr}\n{stdout}"
        if _AUTH_FAILURE.search(classification_text):
            return ModelAuthenticationError(
                f"Claude CLI authentication failed (status {return_code})"
            )
        if _RATE_LIMIT.search(classification_text):
            retry_after_match = _RETRY_AFTER.search(classification_text)
            retry_after = (
                float(retry_after_match.group(1)) if retry_after_match is not None else None
            )
            if retry_after is not None and not math.isfinite(retry_after):
                retry_after = None
            return ModelRateLimitError(
                f"Claude CLI was rate limited (status {return_code})",
                retry_after_seconds=retry_after,
            )
        if _TRANSIENT_FAILURE.search(classification_text):
            return ModelTransientError(
                f"Claude CLI transient failure (status {return_code}): "
                f"{detail or 'no diagnostic'}"
            )
        return ModelGatewayError(
            f"Claude CLI exited with status {return_code}: {detail or 'no diagnostic'}"
        )

    @staticmethod
    def _classify_os_error(exc: OSError) -> ModelGatewayError:
        transient_errnos = {
            errno.EAGAIN,
            errno.ECONNABORTED,
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.EINTR,
            errno.ENETDOWN,
            errno.ENETUNREACH,
            errno.ETIMEDOUT,
        }
        if isinstance(exc, TimeoutError) or exc.errno in transient_errnos:
            return ModelTransientError(f"Claude CLI transient execution failure: {exc.strerror or 'I/O error'}")
        return ModelGatewayError(f"Claude CLI could not execute: {exc.strerror or type(exc).__name__}")

    def complete(self, request: ModelRequest) -> ModelCompletion:
        executable = self._resolve_executable()
        stdin, schema, model = self._request_payload(request)
        base_env, secrets = self._environment(model)
        capabilities = self._get_capabilities(executable)
        argv = self._argv(
            executable,
            capabilities,
            schema=schema,
            model=model,
        )
        timeout = request.timeout_seconds or self._config.request_timeout_seconds
        last_error: ModelTransientError | None = None
        for attempt in range(1, self._config.api_attempts + 1):
            try:
                with tempfile.TemporaryDirectory(prefix="ascend-kernel-llm-") as temp_dir:
                    isolated_root = Path(temp_dir)
                    env = self._isolated_environment(isolated_root, base_env)
                    return_code, stdout, stderr = self._runner(
                        argv,
                        stdin,
                        isolated_root,
                        env,
                        timeout,
                    )
                require_text_limit(
                    stdout,
                    self._config.maximum_response_bytes,
                    "Claude CLI stdout",
                )
                require_text_limit(
                    stderr,
                    self._config.maximum_error_bytes,
                    "Claude CLI stderr",
                )
                if return_code != 0:
                    error = self._classify_exit(return_code, stdout, stderr, secrets)
                    raise error
                raw = decode_json_output(stdout)
                sanitized = sanitize_audit_value(
                    raw,
                    secrets=secrets,
                    maximum_bytes=self._config.maximum_response_bytes,
                )
                if not isinstance(sanitized, Mapping):  # pragma: no cover - raw is a mapping
                    raise ModelProtocolError("Claude CLI response must be a JSON object")
                return normalize_claude_envelope(sanitized)
            except OSError as exc:
                error = self._classify_os_error(exc)
                if not isinstance(error, ModelTransientError):
                    raise error from exc
                last_error = error
            except ModelTransientError as exc:
                last_error = self._sanitize_transient(exc, secrets)
            if attempt == self._config.api_attempts:
                break
            delay = self._retry_delay(last_error, attempt)
            if delay is None:
                raise last_error
            self._sleep(delay)
        raise last_error or ModelGatewayError("Claude CLI request failed")

    generate = complete
