from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.cli import CommandError, _load_probe_snapshot
from ascend_kernel_lab.config import load_config
from ascend_kernel_lab.probe.environment import EnvironmentProber


class ProbeTests(unittest.TestCase):
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

    def test_experiment_probe_gate_requires_live_mandatory_profile_groups(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "env_manifest.json").write_text(
                json.dumps({"schema_version": "ascend_environment_v1"}),
                encoding="utf-8",
            )
            (root / "capability_matrix.json").write_text(
                json.dumps({"schema_version": "ascend_triton_capabilities_v1"}),
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


if __name__ == "__main__":
    unittest.main()
