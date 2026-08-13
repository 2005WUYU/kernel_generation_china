from __future__ import annotations

import io
import json
import unittest
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from ascend_kernel_lab.config import ModelConfig, OpenAIHTTPConfig, RetryConfig
from ascend_kernel_lab.llm import (
    SYSTEM_PROMPT,
    ClaudeCliGateway,
    ModelAuthenticationError,
    ModelCapabilityError,
    ModelGatewayError,
    ModelProtocolError,
    ModelRequest,
    ModelResponseAttemptsExhausted,
    ModelResponseError,
    OpenAICompatibleGateway,
    PromptBuilder,
    ReplayGateway,
    TruncatedResponseError,
    complete_model_response,
    validate_completion,
    validate_model_response,
)
from ascend_kernel_lab.llm.claude_cli import parse_claude_cli_capabilities
from ascend_kernel_lab.llm.envelopes import normalize_claude_envelope
from ascend_kernel_lab.llm.safety import redact_text, truncate_utf8
from ascend_kernel_lab.llm.types import ModelCompletion

CLAUDE_HELP = """Usage: claude [options]
  -p, --print                    Print response and exit (non-interactive)
  --output-format <format>       Output format: text, json, stream-json
  --json-schema <schema>         Validate output with a JSON schema
  --tools <tools...>             Use "" to disable all tools
  --no-session-persistence       Do not persist a session
  --model <model>                Model name
"""


def is_help_call(argv: Sequence[str]) -> bool:
    return len(argv) == 2 and argv[1] in {"--help", "-h"}


def http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://gateway.example/v1/chat/completions",
        code,
        "provider failure",
        headers,
        io.BytesIO(b"response body must never be logged"),
    )


def candidate(round_number: int = 1) -> dict[str, object]:
    return {
        "status": "candidate",
        "round": round_number,
        "change_summary": ["coalesced loads"],
        "expected_effect": ["lower latency"],
        "assumptions": [],
        "code": "def custom_op(x):\n    return x\n",
    }


class ResponseTests(unittest.TestCase):
    def test_strict_response_validation(self) -> None:
        response = validate_model_response(candidate(), expected_round=1)
        self.assertEqual(response.status, "candidate")
        invalid = candidate()
        invalid["extra"] = True
        with self.assertRaises(ModelResponseError):
            validate_model_response(invalid)

    def test_finish_reason_length_is_rejected(self) -> None:
        completion = ModelCompletion(json.dumps(candidate()), "length", {})
        with self.assertRaises(TruncatedResponseError):
            validate_completion(completion)

    def test_claude_structured_output_tool_use_is_a_successful_terminal_result(self) -> None:
        raw = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "stop_reason": "tool_use",
            "structured_output": candidate(),
        }

        completion = normalize_claude_envelope(raw)
        response = validate_completion(completion, expected_round=1)

        self.assertEqual(completion.finish_reason, "stop")
        self.assertEqual(completion.raw_response["stop_reason"], "tool_use")
        self.assertEqual(response.status, "candidate")

    def test_claude_tool_use_without_structured_output_remains_rejected(self) -> None:
        completion = normalize_claude_envelope(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "tool_use",
                "result": json.dumps(candidate()),
            }
        )

        with self.assertRaises(ModelResponseError):
            validate_completion(completion, expected_round=1)

    def test_structured_output_tool_use_does_not_skip_local_schema_validation(self) -> None:
        invalid = candidate(round_number=2)
        completion = normalize_claude_envelope(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "stop_reason": "tool_use",
                "structured_output": invalid,
            }
        )

        with self.assertRaises(ModelResponseError):
            validate_completion(completion, expected_round=1)

    def test_bounded_format_repair(self) -> None:
        replay = ReplayGateway(["not json", candidate()])
        result = complete_model_response(
            replay,
            ModelRequest("system", "request"),
            expected_round=1,
            maximum_format_repair_retries=1,
        )
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(len(result.completions), 2)
        self.assertEqual(len(replay.requests), 2)
        self.assertIn("format_repair", repr(replay.requests[1].metadata))

    def test_exhausted_format_attempts_retain_raw_audit_envelopes(self) -> None:
        replay = ReplayGateway(["not json", "still not json"])
        with self.assertRaises(ModelResponseAttemptsExhausted) as raised:
            complete_model_response(
                replay,
                ModelRequest("system", "request"),
                expected_round=1,
                maximum_format_repair_retries=1,
            )
        self.assertEqual(len(raised.exception.attempts), 2)
        self.assertEqual(raised.exception.attempts[0].content, "not json")


class PromptTests(unittest.TestCase):
    def test_system_prompt_states_source_guard_host_wrapper_contract(self) -> None:
        self.assertIn("每一条返回路径都必须先启动候选 Triton Kernel", SYSTEM_PROMPT)
        self.assertIn("包括 n == 0 等提前返回", SYSTEM_PROMPT)
        self.assertIn("不得调用自定义 Python 辅助函数", SYSTEM_PROMPT)

    def test_first_and_followup_prompts_remove_hidden_content_recursively(self) -> None:
        builder = PromptBuilder()
        first = builder.build_first_round(
            task={"task_id": "k01", "public_cases": [1], "hidden_cases": ["DO_NOT_LEAK"]},
            environment={"device": {}, "api_key": "DO_NOT_LEAK"},
            baseline={},
        )
        self.assertNotIn("DO_NOT_LEAK", first.user_prompt)
        payload = json.loads(first.user_prompt)
        self.assertEqual(payload["round_context"]["round"], 1)
        self.assertTrue(payload["collection_strategy"]["cold_start_sft"])
        self.assertIsNone(payload["collection_strategy"]["target_speedup"])
        followup = builder.build_follow_up(
            round_number=2,
            maximum_rounds=5,
            last_candidate_code="def custom_op(x):\n    return x\n",
            key_metrics={
                "benchmark_vs_pytorch_eager": {"geomean": 0.8},
                "hidden_correctness": "DO_NOT_LEAK",
            },
            failure_reasons=["benchmark_failed"],
            next_round_suggestions=["try a different block size"],
        )
        followup_payload = json.loads(followup.user_prompt)
        self.assertNotIn("DO_NOT_LEAK", followup.user_prompt)
        self.assertEqual(
            followup_payload["round_context"]["current_candidate"]["code"],
            "def custom_op(x):\n    return x\n",
        )
        self.assertNotIn("environment", followup_payload)
        self.assertNotIn("baseline", followup_payload)
        self.assertIn("source_checker_contract", followup_payload)
        self.assertIn("legal_structure_template", followup_payload["source_checker_contract"])
        self.assertEqual(followup_payload["task_contract"], {})
        self.assertEqual(followup_payload["round_context"]["history_summary"], [])
        self.assertNotIn("best_candidate", followup.user_prompt)
        self.assertNotIn("last_evaluation", followup.user_prompt)

    def test_repair_followup_is_bounded_and_phase_aware(self) -> None:
        request = PromptBuilder().build_follow_up(
            round_number=3,
            maximum_rounds=8,
            phase="repair",
            phase_index=3,
            optimization_rounds=5,
            maximum_repair_rounds=3,
            task_contract={"task_id": "k01", "description": "add"},
            candidate_round=2,
            candidate_role="latest_repair_candidate",
            last_candidate_code="def custom_op(x):\n    return x\n",
            key_metrics={"source": {"status": "fail"}},
            failure_reasons=["source[forbidden_call]: getattr is forbidden"],
            next_round_suggestions=["remove getattr"],
            history_summary=[
                {"round": 1, "status": "source_failed"},
                {"round": 2, "status": "source_failed"},
            ],
        )
        payload = json.loads(request.user_prompt)
        self.assertEqual(payload["phase"]["name"], "repair")
        self.assertEqual(payload["phase"]["index"], 3)
        self.assertEqual(request.metadata["phase"], "repair")
        self.assertGreater(request.metadata["user_prompt_utf8_bytes"], 0)
        self.assertGreater(request.metadata["system_prompt_utf8_bytes"], 0)
        self.assertEqual(
            payload["round_context"]["current_candidate"]["round"], 2
        )
        self.assertEqual(len(payload["round_context"]["history_summary"]), 2)
        self.assertNotIn("environment", payload)
        self.assertNotIn("baseline", payload)


class ProviderSafetyTests(unittest.TestCase):
    def test_redaction_handles_known_short_tokens_and_enforces_utf8_limit(self) -> None:
        message = "token=abc; Authorization: Bearer sk-provider-secret; key=短密钥"
        redacted = redact_text(
            message,
            secrets=("abc", "短密钥"),
            maximum_bytes=48,
        )
        self.assertNotIn("abc", redacted)
        self.assertNotIn("短密钥", redacted)
        self.assertNotIn("sk-provider-secret", redacted)
        self.assertLessEqual(len(redacted.encode("utf-8")), 48)
        self.assertLessEqual(len(truncate_utf8("你" * 20, 17).encode("utf-8")), 17)


class ClaudeCliTests(unittest.TestCase):
    def test_cli_accepts_installed_tools_option_wording(self) -> None:
        current_help = CLAUDE_HELP.replace(
            'Use "" to disable all tools',
            "Specify the available built-in tools",
        )

        capabilities = parse_claude_cli_capabilities(
            current_help,
            require_json_schema=True,
        )

        self.assertEqual(capabilities.tools_option, "--tools")

    def test_cli_is_one_shot_tool_free_stdin_and_clean_environment(self) -> None:
        captured: dict[str, object] = {}
        help_environment: dict[str, str] = {}

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            if is_help_call(argv):
                help_environment.update(env)
                return 0, CLAUDE_HELP, ""
            captured.update(argv=list(argv), stdin=stdin, cwd=cwd, env=dict(env), timeout=timeout)
            envelope = {"subtype": "success", "structured_output": candidate()}
            return 0, json.dumps(envelope), ""

        config = ModelConfig(retry=RetryConfig(maximum_attempts=1))
        environment = {
            "PATH": "/usr/bin",
            "HOME": "/tmp/test-home",
            "API_TIMEOUT_MS": "3000000",
            "ANTHROPIC_BASE_URL": "https://aiping.example",
            "ANTHROPIC_AUTH_TOKEN": "top-secret",
            "ANTHROPIC_MODEL": "kimi-k3",
            "UNRELATED_SECRET": "must-not-pass",
        }
        gateway = ClaudeCliGateway(config, environ=environment, runner=runner)
        with patch("ascend_kernel_lab.llm.claude_cli.shutil.which", return_value="/usr/bin/claude"):
            completion = gateway.complete(ModelRequest("system value", "user value"))
        argv = captured["argv"]
        self.assertIn("--print", argv)
        self.assertIn("--json-schema", argv)
        cli_schema = json.loads(argv[argv.index("--json-schema") + 1])
        self.assertNotIn("$schema", cli_schema)
        self.assertNotIn("$id", cli_schema)
        self.assertNotIn("title", cli_schema)
        self.assertFalse(cli_schema["additionalProperties"])
        self.assertNotIn("minimum", cli_schema["properties"]["round"])
        self.assertNotIn("maxItems", cli_schema["properties"]["assumptions"])
        self.assertNotIn("minLength", cli_schema["properties"]["code"])
        self.assertNotIn("maxLength", cli_schema["properties"]["code"])
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertNotIn("user value", argv)
        self.assertIn("user value", captured["stdin"])
        self.assertNotIn("UNRELATED_SECRET", captured["env"])
        self.assertEqual(captured["env"]["API_TIMEOUT_MS"], "3000000")
        self.assertEqual(captured["env"]["ANTHROPIC_BASE_URL"], "https://aiping.example")
        self.assertEqual(captured["env"]["ANTHROPIC_AUTH_TOKEN"], "top-secret")
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", help_environment)
        self.assertNotEqual(help_environment["HOME"], "/tmp/test-home")
        self.assertEqual(completion.finish_reason, "stop")

    def test_cli_supports_short_print_and_optional_flag_absence(self) -> None:
        captured: dict[str, object] = {}
        short_help = """Options:
  -p                   Print response then exit in non-interactive mode
  --output-format FMT  Use text or json output
  --json-schema VALUE  JSON schema
  --tools VALUE        An empty value disables all tools
"""

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            if is_help_call(argv):
                return 0, short_help, ""
            captured.update(argv=list(argv), env=dict(env))
            return 0, json.dumps({"structured_output": candidate()}), ""

        gateway = ClaudeCliGateway(
            ModelConfig(retry=RetryConfig(maximum_attempts=1)),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
        )
        with patch("ascend_kernel_lab.llm.claude_cli.shutil.which", return_value="/usr/bin/claude"):
            gateway.complete(ModelRequest("system", "user", model="request-model"))
        argv = captured["argv"]
        self.assertIn("-p", argv)
        self.assertNotIn("--print", argv)
        self.assertNotIn("--no-session-persistence", argv)
        self.assertNotIn("--model", argv)
        self.assertEqual(captured["env"]["ANTHROPIC_MODEL"], "request-model")

    def test_cli_retries_only_classified_transient_exit(self) -> None:
        api_count = 0
        delays: list[float] = []

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, CLAUDE_HELP, ""
            api_count += 1
            if api_count == 1:
                return 1, "", "HTTP 503 service unavailable"
            return 0, json.dumps({"structured_output": candidate(), "finish_reason": "stop"}), ""

        config = ModelConfig(
            retry=RetryConfig(
                maximum_attempts=2,
                initial_backoff_seconds=0.001,
                maximum_backoff_seconds=0.001,
                jitter_fraction=0,
            ),
        )
        gateway = ClaudeCliGateway(
            config,
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret",
            },
            runner=runner,
            sleeper=delays.append,
        )
        with patch("ascend_kernel_lab.llm.claude_cli.shutil.which", return_value="/usr/bin/claude"):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(api_count, 2)
        self.assertEqual(delays, [0.001])

    def test_cli_429_retry_after_is_honored(self) -> None:
        api_count = 0
        delays: list[float] = []

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, CLAUDE_HELP, ""
            api_count += 1
            if api_count == 1:
                return 1, "", "HTTP 429 too many requests; Retry-After: 2.5"
            return 0, json.dumps({"structured_output": candidate()}), ""

        gateway = ClaudeCliGateway(
            ModelConfig(retry=RetryConfig(maximum_attempts=2)),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
            sleeper=delays.append,
        )
        with patch("ascend_kernel_lab.llm.claude_cli.shutil.which", return_value="/usr/bin/claude"):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(api_count, 2)
        self.assertEqual(delays, [2.5])

    def test_cli_authentication_and_unknown_exit_fail_fast_with_redaction(self) -> None:
        for diagnostic, expected_error in (
            ("HTTP 401 invalid auth token secret-value", ModelAuthenticationError),
            ("compilation wrapper rejected secret-value", ModelGatewayError),
        ):
            with self.subTest(diagnostic=diagnostic):
                api_count = 0

                def runner(
                    argv: Sequence[str],
                    stdin: str,
                    cwd: Path,
                    env: Mapping[str, str],
                    timeout: float,
                    diagnostic_value: str = diagnostic,
                ) -> tuple[int, str, str]:
                    nonlocal api_count
                    if is_help_call(argv):
                        return 0, CLAUDE_HELP, ""
                    api_count += 1
                    return 1, "", diagnostic_value

                gateway = ClaudeCliGateway(
                    ModelConfig(),
                    environ={
                        "PATH": "/usr/bin",
                        "ANTHROPIC_BASE_URL": "https://aiping.example",
                        "ANTHROPIC_AUTH_TOKEN": "secret-value",
                    },
                    runner=runner,
                    sleeper=lambda _: self.fail("fail-fast error slept before retry"),
                )
                with patch(
                    "ascend_kernel_lab.llm.claude_cli.shutil.which",
                    return_value="/usr/bin/claude",
                ), self.assertRaises(expected_error) as raised:
                    gateway.complete(ModelRequest("system", "user"))
                self.assertEqual(api_count, 1)
                self.assertNotIn("secret-value", str(raised.exception))

    def test_cli_protocol_error_is_not_retried(self) -> None:
        api_count = 0

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, CLAUDE_HELP, ""
            api_count += 1
            return 0, "not-json", ""

        gateway = ClaudeCliGateway(
            ModelConfig(),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
            sleeper=lambda _: self.fail("protocol error slept before retry"),
        )
        with (
            patch(
                "ascend_kernel_lab.llm.claude_cli.shutil.which",
                return_value="/usr/bin/claude",
            ),
            self.assertRaises(ModelProtocolError),
        ):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(api_count, 1)

    def test_cli_missing_tools_option_never_sends_prompt(self) -> None:
        api_count = 0
        unsafe_help = CLAUDE_HELP.replace(
            '  --tools <tools...>             Use "" to disable all tools\n',
            "",
        )

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, unsafe_help, ""
            api_count += 1
            return 0, "{}", ""

        gateway = ClaudeCliGateway(
            ModelConfig(),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
        )
        with (
            patch(
                "ascend_kernel_lab.llm.claude_cli.shutil.which",
                return_value="/usr/bin/claude",
            ),
            self.assertRaises(ModelCapabilityError),
        ):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(api_count, 0)

    def test_cli_response_is_bounded_and_audit_envelope_is_redacted(self) -> None:
        api_count = 0

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, CLAUDE_HELP, ""
            api_count += 1
            envelope = {
                "structured_output": candidate(),
                "api_key": "secret-value",
                "nested": {"access_token": "provider-token"},
            }
            return 0, json.dumps(envelope), ""

        gateway = ClaudeCliGateway(
            ModelConfig(maximum_response_bytes=2_000),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
        )
        with patch("ascend_kernel_lab.llm.claude_cli.shutil.which", return_value="/usr/bin/claude"):
            completion = gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(completion.raw_response["api_key"], "<redacted>")
        self.assertEqual(completion.raw_response["nested"]["access_token"], "<redacted>")
        self.assertEqual(api_count, 1)

    def test_cli_oversized_output_fails_without_api_retry(self) -> None:
        api_count = 0

        def runner(
            argv: Sequence[str], stdin: str, cwd: Path, env: Mapping[str, str], timeout: float
        ) -> tuple[int, str, str]:
            nonlocal api_count
            if is_help_call(argv):
                return 0, CLAUDE_HELP, ""
            api_count += 1
            return 0, "x" * 1_001, ""

        gateway = ClaudeCliGateway(
            ModelConfig(maximum_response_bytes=1_000),
            environ={
                "PATH": "/usr/bin",
                "ANTHROPIC_BASE_URL": "https://aiping.example",
                "ANTHROPIC_AUTH_TOKEN": "secret-value",
            },
            runner=runner,
            sleeper=lambda _: self.fail("oversized output slept before retry"),
        )
        with (
            patch(
                "ascend_kernel_lab.llm.claude_cli.shutil.which",
                return_value="/usr/bin/claude",
            ),
            self.assertRaises(ModelProtocolError),
        ):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(api_count, 1)


class HTTPGatewayTests(unittest.TestCase):
    def test_openai_compatible_wire_format(self) -> None:
        captured: dict[str, object] = {}

        def call(request: urllib.request.Request, timeout: float) -> bytes:
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return json.dumps(
                {
                    "id": "req-1",
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(candidate())}}],
                    "usage": {"completion_tokens": 10},
                }
            ).encode()

        config = ModelConfig(
            provider="openai_compatible",
            retry=RetryConfig(maximum_attempts=1),
            openai=OpenAIHTTPConfig(base_url="https://gateway.example/v1", api_key_env="TEST_KEY"),
        )
        completion = OpenAICompatibleGateway(
            config, environ={"TEST_KEY": "secret"}, http_call=call
        ).complete(ModelRequest("system", "user"))
        self.assertEqual(captured["url"], "https://gateway.example/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertEqual(captured["body"]["response_format"]["type"], "json_schema")
        self.assertEqual(completion.request_id, "req-1")

    def test_http_429_honors_retry_after(self) -> None:
        calls = 0
        delays: list[float] = []

        def call(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise http_error(429, retry_after="3")
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(candidate())}}]}
            ).encode()

        config = ModelConfig(
            provider="openai_compatible",
            retry=RetryConfig(maximum_attempts=2),
            openai=OpenAIHTTPConfig(
                base_url="https://gateway.example/v1",
                api_key_env="TEST_KEY",
            ),
        )
        completion = OpenAICompatibleGateway(
            config,
            environ={"TEST_KEY": "secret-value"},
            http_call=call,
            sleeper=delays.append,
        ).complete(ModelRequest("system", "user"))
        self.assertEqual(json.loads(completion.content)["status"], "candidate")
        self.assertEqual(calls, 2)
        self.assertEqual(delays, [3.0])

    def test_http_401_fails_fast_without_reading_error_body(self) -> None:
        calls = 0

        def call(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal calls
            calls += 1
            raise http_error(401)

        gateway = OpenAICompatibleGateway(
            ModelConfig(
                provider="openai_compatible",
                openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
            ),
            environ={"TEST_KEY": "secret-value"},
            http_call=call,
            sleeper=lambda _: self.fail("authentication failure slept before retry"),
        )
        with self.assertRaises(ModelAuthenticationError) as raised:
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(calls, 1)
        self.assertNotIn("secret-value", str(raised.exception))
        self.assertNotIn("response body", str(raised.exception))

    def test_http_5xx_and_timeout_retry(self) -> None:
        for first_error in (http_error(503), TimeoutError("socket timeout")):
            with self.subTest(error=type(first_error).__name__):
                calls = 0
                delays: list[float] = []

                def call(
                    request: urllib.request.Request,
                    timeout: float,
                    error: BaseException = first_error,
                ) -> bytes:
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise error
                    return json.dumps(
                        {"choices": [{"message": {"content": json.dumps(candidate())}}]}
                    ).encode()

                gateway = OpenAICompatibleGateway(
                    ModelConfig(
                        provider="openai_compatible",
                        retry=RetryConfig(
                            maximum_attempts=2,
                            initial_backoff_seconds=0.25,
                            maximum_backoff_seconds=0.25,
                            jitter_fraction=0,
                        ),
                        openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
                    ),
                    environ={"TEST_KEY": "secret-value"},
                    http_call=call,
                    sleeper=delays.append,
                )
                gateway.complete(ModelRequest("system", "user"))
                self.assertEqual(calls, 2)
                self.assertEqual(delays, [0.25])

    def test_http_protocol_and_json_errors_are_not_retried(self) -> None:
        bad_responses = (
            b"not-json",
            json.dumps({"object": "missing choices"}).encode(),
        )
        for response in bad_responses:
            with self.subTest(response=response):
                calls = 0

                def call(
                    request: urllib.request.Request,
                    timeout: float,
                    response_value: bytes = response,
                ) -> bytes:
                    nonlocal calls
                    calls += 1
                    return response_value

                gateway = OpenAICompatibleGateway(
                    ModelConfig(
                        provider="openai_compatible",
                        openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
                    ),
                    environ={"TEST_KEY": "secret-value"},
                    http_call=call,
                    sleeper=lambda _: self.fail("protocol error slept before retry"),
                )
                with self.assertRaises(ModelProtocolError):
                    gateway.complete(ModelRequest("system", "user"))
                self.assertEqual(calls, 1)

    def test_http_response_and_request_limits_fail_fast(self) -> None:
        response_calls = 0

        def oversized_response(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal response_calls
            response_calls += 1
            return b"x" * 101

        response_gateway = OpenAICompatibleGateway(
            ModelConfig(
                provider="openai_compatible",
                maximum_response_bytes=100,
                openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
            ),
            environ={"TEST_KEY": "secret-value"},
            http_call=oversized_response,
            sleeper=lambda _: self.fail("oversized response slept before retry"),
        )
        with self.assertRaises(ModelProtocolError):
            response_gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(response_calls, 1)

        request_calls = 0

        def never_called(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal request_calls
            request_calls += 1
            return b"{}"

        request_gateway = OpenAICompatibleGateway(
            ModelConfig(
                provider="openai_compatible",
                maximum_request_bytes=100,
                openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
            ),
            environ={"TEST_KEY": "secret-value"},
            http_call=never_called,
        )
        with self.assertRaises(ModelProtocolError):
            request_gateway.complete(ModelRequest("system", "x" * 200))
        self.assertEqual(request_calls, 0)

    def test_http_audit_envelope_redacts_credentials(self) -> None:
        def call(request: urllib.request.Request, timeout: float) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(candidate())}}],
                    "api_key": "secret-value",
                    "nested": {"access_token": "provider-token"},
                }
            ).encode()

        completion = OpenAICompatibleGateway(
            ModelConfig(
                provider="openai_compatible",
                openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
            ),
            environ={"TEST_KEY": "secret-value"},
            http_call=call,
        ).complete(ModelRequest("system", "user"))
        self.assertEqual(completion.raw_response["api_key"], "<redacted>")
        self.assertEqual(completion.raw_response["nested"]["access_token"], "<redacted>")

    def test_http_excessive_retry_after_fails_without_sleeping(self) -> None:
        calls = 0

        def call(request: urllib.request.Request, timeout: float) -> bytes:
            nonlocal calls
            calls += 1
            raise http_error(429, retry_after="999")

        gateway = OpenAICompatibleGateway(
            ModelConfig(
                provider="openai_compatible",
                retry=RetryConfig(maximum_retry_after_seconds=10),
                openai=OpenAIHTTPConfig(api_key_env="TEST_KEY"),
            ),
            environ={"TEST_KEY": "secret-value"},
            http_call=call,
            sleeper=lambda _: self.fail("unsafe Retry-After value should not be slept"),
        )
        with self.assertRaises(ModelGatewayError):
            gateway.complete(ModelRequest("system", "user"))
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
