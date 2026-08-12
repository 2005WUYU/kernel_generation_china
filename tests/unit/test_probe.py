from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ascend_kernel_lab.cli import (
    _REQUIRED_PROBE_FEATURES,
    CommandError,
    _device_lock_root,
    _load_probe_snapshot,
)
from ascend_kernel_lab.config import load_config
from ascend_kernel_lab.probe.environment import EnvironmentProber
from ascend_kernel_lab.probe.smoke import _run_source


class ProbeTests(unittest.TestCase):
    @staticmethod
    def _passing_features() -> dict[str, object]:
        return {
            name: {"compile": True, "run": True, "correct": True, "error": None}
            for name in _REQUIRED_PROBE_FEATURES
        }

    def test_non_npu_host_produces_honest_manifests(self) -> None:
        prober = EnvironmentProber(command_timeout=2)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = prober.write_bundle(temporary, run_feature_smokes=False)
            self.assertTrue(bundle.environment_path.is_file())
            self.assertTrue(bundle.capability_path.is_file())
            self.assertTrue(bundle.profiler_path.is_file())
            manifest = json.loads(bundle.environment_path.read_text())
            self.assertIn("npu_smi_info", manifest["commands"])
            self.assertEqual(len(bundle.environment_sha256), 64)
            profiler = json.loads(bundle.profiler_path.read_text())
            self.assertFalse(profiler["live_smoke_attempted"])
            self.assertFalse(profiler["live_smoke_completed"])
            self.assertIn("emitted", profiler)
            self.assertNotIn("probe profiler --live", profiler["note"])

    def test_smoke_source_exists_while_run_is_called(self) -> None:
        source = """
from pathlib import Path

def run():
    return Path(__file__).is_file()
"""
        self.assertTrue(_run_source(source))

    def test_probe_bundle_refuses_to_mix_with_existing_output(self) -> None:
        prober = EnvironmentProber(command_timeout=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "old-evidence.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                prober.write_bundle(root, run_feature_smokes=False)

    def test_experiment_probe_gate_requires_live_mandatory_profile_groups(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "env_manifest.json").write_text(
                json.dumps({"schema_version": "ascend_environment_v1"}),
                encoding="utf-8",
            )
            (root / "capability_matrix.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_triton_capabilities_v1",
                        "device_visibility": {"available": True, "count": 1},
                        "features": self._passing_features(),
                        "timing": {"verified": True},
                    }
                ),
                encoding="utf-8",
            )
            (root / "profiler_capabilities.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_profiler_capabilities_v1",
                        "msprof_available": True,
                        "live_smoke_completed": True,
                        "emitted": {
                            "task_time": True,
                            "pipe_utilization": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CommandError, "pipe_utilization"):
                _load_probe_snapshot(config, str(root), require_profile=True)
            snapshot = _load_probe_snapshot(
                config,
                str(root),
                require_profile=False,
            )
            self.assertEqual(snapshot["schema_version"], "ascend_prompt_environment_v1")

    def test_probe_snapshot_rejects_failed_or_missing_triton_features(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = self._passing_features()
            features["dot"] = {"compile": True, "run": True, "correct": False}
            (root / "env_manifest.json").write_text("{}", encoding="utf-8")
            (root / "capability_matrix.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_triton_capabilities_v1",
                        "device_visibility": {"available": True, "count": 1},
                        "features": features,
                        "timing": {"verified": True},
                    }
                ),
                encoding="utf-8",
            )
            (root / "profiler_capabilities.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CommandError, "feature dot"):
                _load_probe_snapshot(config, str(root), require_profile=False)

    def test_probe_subprocess_environment_preserves_single_device_and_home(self) -> None:
        source = {
            "HOME": "/tmp/akg-home",
            "ASCEND_VISIBLE_DEVICES": "0",
            "DEVICE_ID": "0",
            "ANTHROPIC_AUTH_TOKEN": "must-not-pass",
        }
        with mock.patch.dict("os.environ", source, clear=True):
            environment = EnvironmentProber._probe_env()
        self.assertEqual(environment["HOME"], "/tmp/akg-home")
        self.assertEqual(environment["ASCEND_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["DEVICE_ID"], "0")
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)

    def test_device_lock_root_must_be_absolute(self) -> None:
        with (
            mock.patch.dict("os.environ", {"AKG_DEVICE_LOCK_ROOT": "relative"}),
            self.assertRaisesRegex(CommandError, "must be absolute"),
        ):
            _device_lock_root()


if __name__ == "__main__":
    unittest.main()
