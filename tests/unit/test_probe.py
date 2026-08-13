from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ascend_kernel_lab.cli import (
    _REQUIRED_PROBE_FEATURES,
    CommandError,
    _compact_hardware_profile,
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

    @staticmethod
    def _write_hardware_profile(probe_root: Path) -> Path:
        def fact(value: object) -> dict[str, object]:
            return {
                "value": value,
                "unit": None,
                "evidence_kind": "reported",
                "source": "test",
            }

        profile = {
            "schema_version": "kernel_authoring_environment_v1",
            "reported_profile_sha256": "a" * 64,
            "compiler_runtime": {
                "backend": "triton-ascend",
                "dispatch": {
                    "max_total_grid_programs": fact(65535),
                    "recommended_matrix_or_cv_programs": fact(24),
                    "recommended_vector_programs": fact(48),
                },
                "memory_access_constraints": {
                    "vector_tail_axis_alignment": fact(32),
                    "cube_vector_tail_axis_alignment": fact(512),
                },
                "verified_dtype_paths": {
                    "load_store_and_vector_fp16": fact(True),
                    "load_store_and_vector_bfloat16": fact(True),
                    "load_store_and_vector_fp32": fact(True),
                    "matrix_dot_fp16": fact(True),
                },
                "verified_features": {
                    name: fact(True)
                    for name in (
                        "masked_load_store",
                        "reduction_sum",
                        "max_exp",
                        "grid_2d",
                        "multiple_kernels",
                    )
                },
            },
            "hardware": {
                "execution": {
                    "schedulable_engines": {
                        "matrix": {"physical_count": fact(24)},
                        "vector": {"physical_count": fact(48)},
                    }
                },
                "memory": {
                    "global": {"capacity": fact(65_787_658_240)},
                    "cache_levels": [
                        {"level": "L2", "capacity": fact(201_326_592)}
                    ],
                    "local_scratchpad": {"vendor_term": "UB"},
                },
            },
            "raw_evidence": {
                "capability_timing": {"timing_method": "torch_npu_event"}
            },
        }
        path = (
            probe_root.parent
            / "hardware_probe/kernel_authoring_environment.reported.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def test_non_npu_host_produces_honest_manifests(self) -> None:
        prober = EnvironmentProber(command_timeout=2)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = prober.write_bundle(
                Path(temporary) / "probe", run_feature_smokes=False
            )
            self.assertTrue(bundle.environment_path.is_file())
            self.assertTrue(bundle.capability_path.is_file())
            self.assertTrue(bundle.profiler_path.is_file())
            self.assertTrue(bundle.hardware_profile_path.is_file())
            manifest = json.loads(bundle.environment_path.read_text())
            self.assertIn("npu_smi_info", manifest["commands"])
            self.assertEqual(len(bundle.environment_sha256), 64)
            profiler = json.loads(bundle.profiler_path.read_text())
            self.assertFalse(profiler["live_smoke_attempted"])
            self.assertFalse(profiler["live_smoke_completed"])
            self.assertIn("emitted", profiler)
            self.assertNotIn("probe profiler --live", profiler["note"])
            hardware = json.loads(bundle.hardware_profile_path.read_text())
            self.assertEqual(
                hardware["schema_version"], "kernel_authoring_environment_v1"
            )
            self.assertEqual(len(hardware["reported_profile_sha256"]), 64)
            self.assertEqual(
                bundle.hardware_profile_path,
                Path(temporary).resolve()
                / "hardware_probe/kernel_authoring_environment.reported.json",
            )

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
            root = Path(temporary) / "probe"
            root.mkdir()
            (root / "old-evidence.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                prober.write_bundle(root, run_feature_smokes=False)

    def test_new_probe_atomically_updates_shared_hardware_profile(self) -> None:
        prober = EnvironmentProber(command_timeout=2)
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = prober.write_bundle(
                parent / "probe-one", run_feature_smokes=False
            )
            first.hardware_profile_path.write_text(
                '{"stale":true}\n', encoding="utf-8"
            )

            second = prober.write_bundle(
                parent / "probe-two", run_feature_smokes=False
            )

            updated = json.loads(second.hardware_profile_path.read_text())
            self.assertEqual(
                updated["schema_version"], "kernel_authoring_environment_v1"
            )
            self.assertNotIn("stale", updated)

    def test_runtime_properties_generate_the_verified_compact_projection(self) -> None:
        properties = types.SimpleNamespace(
            name="Ascend910_9382",
            cube_core_num=24,
            vector_core_num=48,
            total_memory=65_787_658_240,
            L2_cache_size=201_326_592,
        )
        fake_npu = types.SimpleNamespace(
            current_device=lambda: 0,
            device_count=lambda: 1,
            get_device_properties=lambda _device: properties,
        )
        fake_driver = types.SimpleNamespace(
            active=types.SimpleNamespace(
                utils=types.SimpleNamespace(
                    get_device_properties=lambda _device: {
                        "num_aicore": 24,
                        "num_vectorcore": 48,
                    }
                ),
                get_current_target=lambda: types.SimpleNamespace(
                    backend="npu", arch="Ascend910_9382", warp_size=0
                ),
            )
        )
        modules = {
            "torch": types.SimpleNamespace(__version__="2.9.0", npu=fake_npu),
            "torch_npu": types.SimpleNamespace(
                __version__="2.9.0.post2",
                npu=types.SimpleNamespace(get_soc_version=lambda: 253),
            ),
            "triton": types.SimpleNamespace(__version__="3.5.1"),
            "triton.runtime": types.SimpleNamespace(driver=fake_driver),
        }
        capabilities = {
            "schema_version": "ascend_triton_capabilities_v1",
            "features": self._passing_features(),
            "timing": {"verified": True, "timing_method": "torch_npu_event"},
        }
        with mock.patch(
            "ascend_kernel_lab.probe.environment.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            profile = EnvironmentProber().collect_kernel_hardware_profile(
                {"system": {"python_executable": "/usr/bin/python3"}},
                capabilities,
            )

        self.assertEqual(len(profile["reported_profile_sha256"]), 64)
        self.assertFalse(profile["measured"]["direct_microbench_executed"])
        self.assertEqual(
            profile["compiler_runtime"]["effective_schedulable_engines"][
                "vector"
            ]["count"]["value"],
            48,
        )
        compact = _compact_hardware_profile(profile)
        self.assertEqual(compact["backend"], "triton-ascend")
        self.assertEqual(
            compact["execution"],
            {
                "matrix_units": 24,
                "vector_units": 48,
                "recommended_matrix_programs": 24,
                "recommended_vector_programs": 48,
                "max_grid_programs": 65535,
            },
        )
        self.assertEqual(compact["timing"], {"method": "torch_npu_event"})

    def test_experiment_probe_gate_requires_live_mandatory_profile_groups(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "probe"
            root.mkdir()
            hardware_path = self._write_hardware_profile(root)
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
            self.assertEqual(
                snapshot["prompt_environment"],
                {
                    "backend": "triton-ascend",
                    "execution": {
                        "matrix_units": 24,
                        "vector_units": 48,
                        "recommended_matrix_programs": 24,
                        "recommended_vector_programs": 48,
                        "max_grid_programs": 65535,
                    },
                    "memory": {
                        "global_memory_bytes": 65_787_658_240,
                        "l2_cache_bytes": 201_326_592,
                        "local_memory_type": "UB",
                    },
                    "memory_access": {
                        "vector_tail_alignment_bytes": 32,
                        "matrix_vector_tail_alignment_bytes": 512,
                    },
                    "supported": {
                        "fp16": True,
                        "bf16": True,
                        "fp32": True,
                        "matrix_dot_fp16": True,
                        "masked_load_store": True,
                        "reduction_sum": True,
                        "max_exp": True,
                        "grid_2d": True,
                        "multiple_kernels": True,
                    },
                    "timing": {"method": "torch_npu_event"},
                },
            )
            evidence = snapshot["hardware_profile_evidence"]
            self.assertEqual(
                evidence["path"],
                "hardware_probe/kernel_authoring_environment.reported.json",
            )
            self.assertEqual(
                evidence["file_sha256"],
                hashlib.sha256(hardware_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                evidence["profile"],
                json.loads(hardware_path.read_text(encoding="utf-8")),
            )
            failed_hardware = json.loads(
                hardware_path.read_text(encoding="utf-8")
            )
            failed_hardware["raw_evidence"]["runtime_error"] = (
                "ImportError: failed to map npu_utils.so"
            )
            hardware_path.write_text(
                json.dumps(failed_hardware), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                CommandError, "failed to map npu_utils.so"
            ):
                _load_probe_snapshot(
                    config, str(root), require_profile=False
                )

    def test_probe_snapshot_rejects_failed_or_missing_triton_features(self) -> None:
        config = load_config("configs/experiment_910c_kimi_k3.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "probe"
            root.mkdir()
            self._write_hardware_profile(root)
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
