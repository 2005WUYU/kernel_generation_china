from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ascend_kernel_lab.storage.permissions import (
    SHARED_FILE_MODE,
    ensure_shared_directory,
    ensure_shared_regular_file,
)

_MAX_OUTPUT = 256_000
_PROBE_RESULT_PREFIX = "AKG_TRITON_PROBE_RESULT="


def _probe_result(text: str) -> dict[str, Any] | None:
    position = text.rfind(_PROBE_RESULT_PREFIX)
    if position < 0:
        return None
    payload = text[position + len(_PROBE_RESULT_PREFIX) :].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    ensure_shared_regular_file(path, mode=SHARED_FILE_MODE)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def _atomic_text(path: Path, value: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.chmod(mode)
    os.replace(temporary, path)


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _redact(value: str) -> str:
    value = re.sub(r"(?i)(token|secret|password|api[_-]?key)\s*[:=]\s*\S+", r"\1=<redacted>", value)
    return value[:_MAX_OUTPUT]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    available: bool
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeBundle:
    output_root: Path
    environment_path: Path
    capability_path: Path
    profiler_path: Path
    hardware_profile_path: Path
    environment_sha256: str
    capability_sha256: str
    profiler_sha256: str
    hardware_profile_sha256: str


class EnvironmentProber:
    schema_version = "ascend_environment_v1"

    def __init__(self, *, python_executable: str = sys.executable, command_timeout: float = 20.0) -> None:
        self.python_executable = python_executable
        self.command_timeout = command_timeout

    def command(self, argv: Sequence[str], timeout: float | None = None) -> CommandResult:
        started = time.monotonic()
        executable = shutil.which(argv[0]) if argv else None
        if not executable:
            return CommandResult(tuple(argv), False, None, False, "", f"{argv[0]} not found", 0.0)
        try:
            process = subprocess.run(
                [executable, *argv[1:]],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout or self.command_timeout,
                check=False,
                env=self._probe_env(),
            )
            return CommandResult(
                tuple(argv), True, process.returncode, False,
                _redact(process.stdout), _redact(process.stderr),
                (time.monotonic() - started) * 1000,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                tuple(argv), True, None, True,
                _redact((exc.stdout or "") if isinstance(exc.stdout, str) else ""),
                _redact((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
                (time.monotonic() - started) * 1000,
            )

    @staticmethod
    def _probe_env() -> dict[str, str]:
        allowed = {
            "PATH", "HOME", "LD_LIBRARY_PATH", "PYTHONPATH", "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH", "ASCEND_TOOLKIT_HOME",
            "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES",
            "DEVICE_ID", "LANG", "LC_ALL", "TMPDIR", "TRITON_CACHE_DIR",
            "TRITON_DUMP_DIR",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    @staticmethod
    def _read_file(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:_MAX_OUTPUT]
        except (OSError, PermissionError):
            return None

    def collect_environment(self) -> dict[str, Any]:
        commands = {
            "uname": self.command(("uname", "-a")).to_dict(),
            "os_release": self.command(("cat", "/etc/os-release")).to_dict(),
            "lscpu": self.command(("lscpu",)).to_dict(),
            "npu_smi_info": self.command(("npu-smi", "info"), timeout=30).to_dict(),
            "npu_smi_list": self.command(("npu-smi", "info", "-l"), timeout=30).to_dict(),
            "bisheng_version": self.command(("bishengir-compile", "--version")).to_dict(),
            "msprof_help": self.command(("msprof", "--help")).to_dict(),
            "msprof_op_help": self.command(("msprof", "op", "--help")).to_dict(),
        }
        package_names = ("torch", "torch-npu", "torch_npu", "triton", "triton-ascend")
        packages: dict[str, str | None] = {}
        for name in package_names:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        install_roots = [Path("/usr/local/Ascend"), Path.home() / "Ascend"]
        cann_candidates: list[str] = []
        for root in install_roots:
            try:
                for path in root.glob("*/set_env.sh"):
                    cann_candidates.append(str(path.parent))
                for path in root.glob("*/*/set_env.sh"):
                    cann_candidates.append(str(path.parent))
            except OSError:
                continue
        return {
            "schema_version": self.schema_version,
            "captured_at_unix": time.time(),
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_executable": self.python_executable,
            },
            "ascend": {
                "driver_version_info": self._read_file(Path("/usr/local/Ascend/driver/version.info")),
                "install_info": self._read_file(Path("/etc/ascend_install.info")),
                "cann_candidates": sorted(set(cann_candidates)),
            },
            "software": packages,
            "commands": commands,
        }

    def collect_capabilities(self) -> dict[str, Any]:
        from .timing import probe_timing

        import_check = self.command((
            self.python_executable,
            "-c",
            "import json; r={};\n"
            "for n in ('torch','torch_npu','triton'):\n"
            "  try:\n   m=__import__(n); r[n]={'available':True,'version':getattr(m,'__version__',None)}\n"
            "  except Exception as e: r[n]={'available':False,'error':type(e).__name__+': '+str(e)}\n"
            "print(json.dumps(r))",
        )).to_dict()
        features = (
            "vector_add", "masked_load_store", "fp16", "bfloat16", "fp32",
            "reduction_sum", "max_exp", "dot", "grid_2d", "multiple_kernels",
        )
        results: dict[str, Any] = {}
        for feature in features:
            print(f"[probe] {feature}: running", file=sys.stderr, flush=True)
            probe = self.command((
                self.python_executable, "-m", "ascend_kernel_lab.probe.smoke", "--feature", feature,
            ), timeout=120)
            parsed = _probe_result(probe.stdout)
            results[feature] = parsed or {
                "feature": feature,
                "compile": False,
                "run": False,
                "correct": False,
                "error": (
                    probe.stderr[-4000:]
                    or probe.stdout[-4000:]
                    or f"exit code {probe.returncode}"
                ),
                "returncode": probe.returncode,
                "timed_out": probe.timed_out,
            }
            outcome = results[feature]
            passed = bool(outcome.get("correct"))
            detail = "" if passed else f": {outcome.get('error') or 'unknown error'}"
            print(
                f"[probe] {feature}: {'PASS' if passed else 'FAIL'}{detail}",
                file=sys.stderr,
                flush=True,
            )
        device_visibility = self.command(
            (
                self.python_executable,
                "-c",
                "import json, torch, torch_npu; "
                "print(json.dumps({'available': bool(torch.npu.is_available()), "
                "'count': int(torch.npu.device_count())}))",
            )
        )
        visibility_value: Any = None
        for line in reversed(device_visibility.stdout.splitlines()):
            try:
                visibility_value = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        return {
            "schema_version": "ascend_triton_capabilities_v1",
            "captured_at_unix": time.time(),
            "import_check": import_check,
            "device_visibility": visibility_value or {
                "available": False,
                "count": None,
                "error": device_visibility.stderr
                or f"exit code {device_visibility.returncode}",
            },
            "features": results,
            "ir": {
                "debug_environment": {"TRITON_DEBUG": "1", "TRITON_ALWAYS_COMPILE": "1"},
                "default_dump_directory": str(Path.home() / ".triton" / "dump"),
                "ttir_expected": "kernel.ttir.mlir",
                "ttadapter_expected": "kernel.ttadapter.mlir",
            },
            "timing": probe_timing(),
        }

    @staticmethod
    def _fact(
        value: Any,
        unit: str | None,
        evidence_kind: str,
        source: str | None,
    ) -> dict[str, Any]:
        return {
            "value": value,
            "unit": unit,
            "evidence_kind": evidence_kind,
            "source": source,
        }

    @classmethod
    def _unknown(cls, unit: str | None) -> dict[str, Any]:
        return cls._fact(None, unit, "unknown", None)

    @classmethod
    def _verified_feature(
        cls, capabilities: Mapping[str, Any], name: str
    ) -> dict[str, Any]:
        features = capabilities.get("features")
        result = features.get(name) if isinstance(features, Mapping) else None
        if not isinstance(result, Mapping):
            return cls._unknown(None)
        passed = all(result.get(field) is True for field in ("compile", "run", "correct"))
        return cls._fact(
            passed,
            None,
            "measured",
            f"capability_matrix.json.features.{name}",
        )

    def collect_kernel_hardware_profile(
        self,
        environment: Mapping[str, Any],
        capabilities: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Describe only kernel-authoring facts; run no profiler or benchmark."""

        torch: Any = None
        torch_npu: Any = None
        triton: Any = None
        properties: Any = None
        triton_properties: dict[str, Any] = {}
        target: Any = None
        device: int | None = None
        visible_device_count: int | None = None
        runtime_error: str | None = None
        try:
            torch = importlib.import_module("torch")
            torch_npu = importlib.import_module("torch_npu")
            triton = importlib.import_module("triton")
            # Triton exposes the active driver config from the runtime package.
            # In Triton-Ascend this package attribute is a DriverConfig object,
            # while importing the same-named submodule with importlib returns
            # the implementation module (which has no module-level ``active``).
            triton_runtime = importlib.import_module("triton.runtime")
            triton_driver = triton_runtime.driver
            device = int(torch.npu.current_device())
            visible_device_count = int(torch.npu.device_count())
            properties = torch.npu.get_device_properties(device)
            triton_properties = dict(
                triton_driver.active.utils.get_device_properties(device)
            )
            target = triton_driver.active.get_current_target()
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"

        def reported(
            value: Any, unit: str | None, source: str
        ) -> dict[str, Any]:
            if value is None:
                return self._unknown(unit)
            return self._fact(value, unit, "reported", source)

        physical_matrix = getattr(properties, "cube_core_num", None)
        physical_vector = getattr(properties, "vector_core_num", None)
        global_memory = getattr(properties, "total_memory", None)
        l2_cache = getattr(properties, "L2_cache_size", None)
        effective_matrix = triton_properties.get("num_aicore")
        effective_vector = triton_properties.get("num_vectorcore")
        timing = capabilities.get("timing")
        timing_mapping = timing if isinstance(timing, Mapping) else {}
        features = capabilities.get("features")
        feature_mapping = features if isinstance(features, Mapping) else {}
        system = environment.get("system")
        system_mapping = system if isinstance(system, Mapping) else {}
        software = environment.get("software")
        software_mapping = software if isinstance(software, Mapping) else {}
        ascend = environment.get("ascend")
        ascend_mapping = ascend if isinstance(ascend, Mapping) else {}
        commands = environment.get("commands")
        command_mapping = commands if isinstance(commands, Mapping) else {}
        get_soc_version = (
            getattr(getattr(torch_npu, "npu", None), "get_soc_version", None)
            if torch_npu is not None
            else None
        )
        try:
            soc_version = get_soc_version() if callable(get_soc_version) else None
        except Exception:
            soc_version = None

        verified_features = {
            name: self._verified_feature(capabilities, name)
            for name in (
                "vector_add",
                "masked_load_store",
                "fp16",
                "bfloat16",
                "fp32",
                "reduction_sum",
                "max_exp",
                "dot",
                "grid_2d",
                "multiple_kernels",
            )
        }
        profile: dict[str, Any] = {
            "schema_version": "kernel_authoring_environment_v1",
            "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "profile_scope": {
                "runtime_visible_device_index": device,
                "runtime_visible_device_count": visible_device_count,
                "physical_device_selector": os.environ.get("ASCEND_VISIBLE_DEVICES"),
            },
            "leaf_contract": {
                "evidence_kind": ["reported", "measured", "unknown"],
                "canonical_units": [
                    "count",
                    "bytes",
                    "bits",
                    "bytes_per_second",
                    "microseconds",
                    "elements",
                    "ratio_0_to_1",
                ],
            },
            "identity_metadata": {
                "vendor": "Huawei",
                "device_family": "Ascend",
                "device_name": getattr(properties, "name", None),
                "soc_version": soc_version,
            },
            "hardware": {
                "execution": {
                    "schedulable_engines": {
                        "vector": {
                            "vendor_term": "AI Vector Core",
                            "physical_count": reported(
                                physical_vector,
                                "count",
                                "torch.npu.get_device_properties.vector_core_num",
                            ),
                        },
                        "matrix": {
                            "vendor_term": "AI Core / Cube Core",
                            "physical_count": reported(
                                physical_matrix,
                                "count",
                                "torch.npu.get_device_properties.cube_core_num",
                            ),
                        },
                    },
                    "execution_group_width": self._unknown("elements"),
                    "independent_program_scheduling": self._unknown(None),
                    "program_local_barrier": self._unknown(None),
                    "cross_program_barrier": self._unknown(None),
                },
                "compute": {
                    "scalar_unit": {
                        "available": self._unknown(None),
                        "native_width_bits": self._unknown("bits"),
                    },
                    "vector_unit": {
                        "available": reported(
                            physical_vector is not None and physical_vector > 0,
                            None,
                            "torch.npu.get_device_properties.vector_core_num",
                        ),
                        "native_width_bits": self._unknown("bits"),
                        "native_dtypes": self._unknown(None),
                    },
                    "matrix_unit": {
                        "available": reported(
                            physical_matrix is not None and physical_matrix > 0,
                            None,
                            "torch.npu.get_device_properties.cube_core_num",
                        ),
                        "native_tile_shapes": self._unknown("elements"),
                        "native_input_dtypes": self._unknown(None),
                        "native_accumulator_dtypes": self._unknown(None),
                    },
                    "special_function_hardware": {
                        name: self._unknown(None)
                        for name in ("exp", "log", "rsqrt", "sin", "cos")
                    },
                },
                "memory": {
                    "global": {
                        "capacity": reported(
                            global_memory,
                            "bytes",
                            "torch.npu.get_device_properties.total_memory",
                        ),
                        "peak_bandwidth": self._unknown("bytes_per_second"),
                        "access_granularity": self._unknown("bytes"),
                    },
                    "cache_levels": [
                        {
                            "level": "L2",
                            "scope": "device",
                            "capacity": reported(
                                l2_cache,
                                "bytes",
                                "torch.npu.get_device_properties.L2_cache_size",
                            ),
                            "line_size": self._unknown("bytes"),
                        }
                    ],
                    "local_scratchpad": {
                        "vendor_term": "UB",
                        "scope": "compute_unit",
                        "capacity_per_compute_unit": self._unknown("bytes"),
                        "allocation_granularity": self._unknown("bytes"),
                        "bank_count": self._unknown("count"),
                        "bank_width": self._unknown("bytes"),
                    },
                    "register_file": {
                        "capacity_per_compute_unit": self._unknown("bytes"),
                        "register_width": self._unknown("bits"),
                        "allocation_granularity": self._unknown("count"),
                    },
                },
                "data_movement": {
                    "global_to_local": {
                        "supported": self._unknown(None),
                        "async_supported": self._unknown(None),
                        "preferred_alignment": self._unknown("bytes"),
                        "preferred_transaction": self._unknown("bytes"),
                    },
                    "local_to_global": {
                        "supported": self._unknown(None),
                        "async_supported": self._unknown(None),
                    },
                    "local_to_matrix_unit": {
                        "supported": self._unknown(None),
                        "async_supported": self._unknown(None),
                    },
                    "compute_copy_overlap": self._unknown(None),
                    "double_buffering": self._unknown(None),
                    "multicast": self._unknown(None),
                },
                "resource_limits": {
                    "max_local_scratchpad_per_program": self._unknown("bytes"),
                    "max_registers_per_program": self._unknown("count"),
                    "max_programs_per_compute_unit": self._unknown("count"),
                    "max_grid_dimensions": self._unknown("count"),
                    "max_static_allocation": self._unknown("bytes"),
                },
            },
            "compiler_runtime": {
                "backend": "triton-ascend",
                "versions": {
                    "torch": getattr(torch, "__version__", software_mapping.get("torch")),
                    "torch_npu": getattr(
                        torch_npu,
                        "__version__",
                        software_mapping.get("torch-npu")
                        or software_mapping.get("torch_npu"),
                    ),
                    "triton": getattr(
                        triton, "__version__", software_mapping.get("triton")
                    ),
                    "triton_ascend_distribution": software_mapping.get(
                        "triton-ascend"
                    ),
                    "driver_version_info": ascend_mapping.get("driver_version_info"),
                    "install_info": ascend_mapping.get("install_info"),
                },
                "target": {
                    "backend": getattr(target, "backend", None),
                    "arch": getattr(target, "arch", None),
                    "raw_compatibility_warp_size": getattr(target, "warp_size", None),
                    "raw_compatibility_warp_size_note": (
                        "compatibility field only; not an Ascend execution-group width"
                    ),
                },
                "effective_schedulable_engines": {
                    "vector": {
                        "vendor_term": "AI Vector Core",
                        "count": reported(
                            effective_vector,
                            "count",
                            "triton.runtime.driver.active.utils.get_device_properties.num_vectorcore",
                        ),
                    },
                    "matrix": {
                        "vendor_term": "AI Core / Cube Core",
                        "count": reported(
                            effective_matrix,
                            "count",
                            "triton.runtime.driver.active.utils.get_device_properties.num_aicore",
                        ),
                    },
                },
                "verified_features": verified_features,
                "verified_dtype_paths": {
                    "load_store_and_vector_fp16": verified_features["fp16"],
                    "load_store_and_vector_bfloat16": verified_features["bfloat16"],
                    "load_store_and_vector_fp32": verified_features["fp32"],
                    "matrix_dot_fp16": verified_features["dot"],
                },
                "dispatch": {
                    "max_total_grid_programs": self._fact(
                        65535,
                        "count",
                        "reported",
                        "Triton-Ascend runtime coreDim contract",
                    ),
                    "recommended_vector_programs": reported(
                        effective_vector,
                        "count",
                        "Triton driver effective vector core count",
                    ),
                    "recommended_matrix_or_cv_programs": reported(
                        effective_matrix,
                        "count",
                        "Triton driver effective matrix core count",
                    ),
                    "verified_minimum_grid_dimensions": (
                        self._fact(
                            2,
                            "count",
                            "measured",
                            "capability_matrix.json.features.grid_2d",
                        )
                        if verified_features["grid_2d"]["value"] is True
                        else self._unknown("count")
                    ),
                },
                "memory_access_constraints": {
                    "vector_tail_axis_alignment": self._fact(
                        32,
                        "bytes",
                        "reported",
                        "Triton-Ascend operator development guide",
                    ),
                    "cube_vector_tail_axis_alignment": self._fact(
                        512,
                        "bytes",
                        "reported",
                        "Triton-Ascend operator development guide",
                    ),
                },
                "automatic_transformations": {
                    "double_buffering": self._unknown(None),
                    "compute_copy_overlap": self._unknown(None),
                },
                "environment_limits": {
                    "NPU_DEVICE_LIMIT": os.environ.get("NPU_DEVICE_LIMIT")
                },
            },
            "measured": {
                "direct_microbench_executed": False,
                "launch": {
                    "minimal_device_latency": self._unknown("microseconds"),
                    "host_enqueue_per_launch": self._unknown("microseconds"),
                    "idle_synchronize": self._unknown("microseconds"),
                },
                "global_memory": {
                    "copy_effective_bandwidth": self._unknown("bytes_per_second"),
                    "add_effective_bandwidth": self._unknown("bytes_per_second"),
                },
                "vector_compute": {},
                "matrix_compute": {},
                "reduction": {},
            },
            "raw_evidence": {
                "runtime_error": runtime_error,
                "python_executable": system_mapping.get("python_executable"),
                "torch_npu_device_properties_repr": (
                    repr(properties) if properties is not None else None
                ),
                "triton_driver_properties": triton_properties,
                "capability_matrix_schema": capabilities.get("schema_version"),
                "capability_timing": dict(timing_mapping),
                "capability_features": dict(feature_mapping),
                "npu_smi_list": command_mapping.get("npu_smi_list"),
                "npu_smi_info": command_mapping.get("npu_smi_info"),
            },
        }
        profile = _json_safe(profile)
        stable_identity = {
            "identity_metadata": profile["identity_metadata"],
            "hardware": profile["hardware"],
            "compiler_runtime": profile["compiler_runtime"],
        }
        profile["reported_profile_sha256"] = _digest(stable_identity)
        return profile

    def collect_profiler_capabilities(
        self, *, live_root: Path | None = None
    ) -> dict[str, Any]:
        from ascend_kernel_lab.profiling import MsprofParser

        executable = shutil.which("msprof")
        help_result = self.command(("msprof", "op", "--help"))
        help_text = f"{help_result.stdout}\n{help_result.stderr}".lower()
        flags = {
            "kernel_name": "--kernel-name" in help_text,
            "output": "--output" in help_text,
            "task_time": any(term in help_text for term in ("task-time", "task time", "duration")),
            "pipe_utilization": any(term in help_text for term in ("pipe", "pipeline")),
            "memory": any(term in help_text for term in ("memory", "bandwidth", "gm")),
            "l2_cache": "l2" in help_text,
            "resource_conflict": "conflict" in help_text,
        }
        live: dict[str, Any] = {
            "attempted": False,
            "completed": False,
            "output_root": str(live_root) if live_root is not None else None,
            "command": None,
            "summary": None,
            "unavailable_reason": (
                "live profiler smoke was not requested"
                if live_root is None
                else "msprof executable was not found"
                if executable is None
                else None
            ),
        }
        if executable is not None and live_root is not None:
            if live_root.exists() and live_root.is_symlink():
                raise ValueError("profiler live-smoke root must not be a symlink")
            live_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            driver = live_root / "profile_smoke_driver.py"
            _atomic_text(
                driver,
                "from __future__ import annotations\n"
                "import json\n"
                "from ascend_kernel_lab.probe.smoke import run_feature\n"
                "result = run_feature('vector_add')\n"
                "print(json.dumps(result, sort_keys=True))\n"
                "raise SystemExit(0 if result.get('correct') else 1)\n",
            )
            raw_root = live_root / "raw"
            command = self.command(
                (
                    executable,
                    "op",
                    "--kernel-name=_add",
                    f"--output={raw_root}",
                    self.python_executable,
                    str(driver),
                ),
                timeout=180.0,
            )
            summary = MsprofParser((r".*_add.*",)).parse(raw_root)
            live.update(
                {
                    "attempted": True,
                    "completed": (
                        command.returncode == 0
                        and not command.timed_out
                        and summary.profile_available
                        and summary.kernel_count >= 1
                    ),
                    "command": command.to_dict(),
                    "summary": summary.to_dict(),
                    "unavailable_reason": (
                        None
                        if summary.profile_available
                        else summary.unavailable_reason
                        or "msprof live smoke did not emit usable metrics"
                    ),
                }
            )
        summary_value = live.get("summary")
        summary_mapping = (
            summary_value if isinstance(summary_value, Mapping) else {}
        )
        pipeline = summary_mapping.get("pipeline")
        memory = summary_mapping.get("memory")
        scheduling = summary_mapping.get("scheduling")
        emitted = {
            "task_time": bool(
                isinstance(scheduling, Mapping)
                and scheduling.get("device_execution_us") is not None
            ),
            "pipe_utilization": bool(
                isinstance(pipeline, Mapping)
                and any(value is not None for value in pipeline.values())
            ),
            "memory": bool(
                isinstance(memory, Mapping)
                and any(value is not None for value in memory.values())
            ),
            "l2_cache": bool(
                isinstance(memory, Mapping) and memory.get("l2_hit_rate") is not None
            ),
            "resource_conflict": False,
        }
        return {
            "schema_version": "ascend_profiler_capabilities_v1",
            "captured_at_unix": time.time(),
            "msprof_available": executable is not None,
            "msprof_path": executable,
            "op_help_returncode": help_result.returncode,
            "supported": flags,
            "live_smoke_attempted": live["attempted"],
            "live_smoke_completed": live["completed"],
            "emitted": emitted,
            "live_smoke": live,
            "note": (
                "Help flags describe advertised capabilities; emitted fields are "
                "accepted only from the live vector-add profiler smoke."
            ),
        }

    def write_bundle(self, output_root: Path | str, *, run_feature_smokes: bool = True) -> ProbeBundle:
        requested_root = Path(output_root).expanduser()
        if requested_root.exists() and requested_root.is_symlink():
            raise ValueError("probe output root must not be a symlink")
        root = requested_root.resolve()
        ensure_shared_directory(root)
        if any(root.iterdir()):
            raise ValueError("probe output root must be empty; preserve or remove the old bundle first")
        environment = self.collect_environment()
        capabilities = self.collect_capabilities() if run_feature_smokes else {
            "schema_version": "ascend_triton_capabilities_v1",
            "captured_at_unix": time.time(),
            "features": {},
            "smokes_skipped": True,
        }
        profiler = self.collect_profiler_capabilities(
            live_root=root / "profiler_live_smoke" if run_feature_smokes else None
        )
        hardware_profile = self.collect_kernel_hardware_profile(
            environment, capabilities
        )
        env_path = root / "env_manifest.json"
        capability_path = root / "capability_matrix.json"
        profiler_path = root / "profiler_capabilities.json"
        hardware_profile_path = (
            root.parent
            / "hardware_probe/kernel_authoring_environment.reported.json"
        )
        ensure_shared_directory(hardware_profile_path.parent)
        _atomic_json(env_path, environment)
        _atomic_json(capability_path, capabilities)
        _atomic_json(profiler_path, profiler)
        _atomic_json(hardware_profile_path, hardware_profile)
        return ProbeBundle(
            output_root=root,
            environment_path=env_path,
            capability_path=capability_path,
            profiler_path=profiler_path,
            hardware_profile_path=hardware_profile_path,
            environment_sha256=_digest(environment),
            capability_sha256=_digest(capabilities),
            profiler_sha256=_digest(profiler),
            hardware_profile_sha256=_digest(hardware_profile),
        )
