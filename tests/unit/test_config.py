from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.config import (
    ConfigError,
    ModelConfig,
    OpenAIHTTPConfig,
    RetryConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "experiment_910c_kimi_k3.yaml"


class ConfigTests(unittest.TestCase):
    def test_loads_full_config_and_resolves_paths(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config.id, "exp_910c_kimi_k3_v1")
        self.assertEqual(config.model.provider, "claude_cli")
        self.assertEqual(config.model.api_attempts, 4)
        self.assertEqual(config.model.maximum_response_bytes, 4_194_304)
        self.assertEqual(config.artifact_root, ROOT / "runs")
        self.assertEqual(config.task_root, ROOT / "task_specs")
        self.assertEqual(config.db_path, ROOT / "runs" / "metadata.db")

    def test_manifest_has_only_environment_references(self) -> None:
        config = load_config(CONFIG)
        manifest = config.to_manifest()
        self.assertEqual(manifest["model"]["anthropic_auth_token_env"], "ANTHROPIC_AUTH_TOKEN")
        self.assertNotIn("actual-token", repr(manifest))

    def test_unknown_field_is_rejected_with_path(self) -> None:
        text = CONFIG.read_text(encoding="utf-8").replace(
            "  model: kimi-k3", "  model: kimi-k3\n  surprise: false", 1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, r"configuration\.model: unknown field.*surprise"):
                load_config(path, project_root=ROOT)

    def test_yaml_type_coercion_is_rejected(self) -> None:
        text = CONFIG.read_text(encoding="utf-8").replace("  rounds_per_task: 5", "  rounds_per_task: '5'")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "expected an integer"):
                load_config(path, project_root=ROOT)

    def test_literal_secret_field_is_unknown(self) -> None:
        text = CONFIG.read_text(encoding="utf-8").replace(
            "  anthropic_auth_token_env: ANTHROPIC_AUTH_TOKEN",
            "  anthropic_auth_token_env: ANTHROPIC_AUTH_TOKEN\n  api_key: should-not-load",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(path, project_root=ROOT)

    def test_retry_maximum_attempts_is_canonical_and_legacy_alias_is_checked(self) -> None:
        self.assertEqual(
            ModelConfig(retry=RetryConfig(maximum_attempts=2)).api_attempts,
            2,
        )
        self.assertEqual(ModelConfig(maximum_api_retries=3).api_attempts, 4)
        with self.assertRaisesRegex(ValueError, "deprecated alias"):
            ModelConfig(
                maximum_api_retries=1,
                retry=RetryConfig(maximum_attempts=4, initial_backoff_seconds=2),
            )

    def test_claude_extra_args_are_rejected_to_preserve_safe_capabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be empty"):
            ModelConfig(claude_extra_args=("--verbose",))

    def test_openai_extra_headers_cannot_override_transport_headers(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved header"):
            OpenAIHTTPConfig(extra_header_env={"Authorization": "OTHER_TOKEN"})
        with self.assertRaisesRegex(ValueError, "invalid HTTP header"):
            OpenAIHTTPConfig(extra_header_env={"Bad Header": "OTHER_TOKEN"})


if __name__ == "__main__":
    unittest.main()
