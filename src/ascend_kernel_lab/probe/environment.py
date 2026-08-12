from __future__ import annotations

import hashlib
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
    environment_sha256: str
    capability_sha256: str
    profiler_sha256: str


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
        env_path = root / "env_manifest.json"
        capability_path = root / "capability_matrix.json"
        profiler_path = root / "profiler_capabilities.json"
        _atomic_json(env_path, environment)
        _atomic_json(capability_path, capabilities)
        _atomic_json(profiler_path, profiler)
        return ProbeBundle(
            output_root=root,
            environment_path=env_path,
            capability_path=capability_path,
            profiler_path=profiler_path,
            environment_sha256=_digest(environment),
            capability_sha256=_digest(capabilities),
            profiler_sha256=_digest(profiler),
        )
