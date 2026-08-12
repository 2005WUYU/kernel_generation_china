"""Machine-readable deployment diagnostics and acceptance preflight checks."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ascend_kernel_lab.config import ExperimentConfig
from ascend_kernel_lab.llm.claude_cli import parse_claude_cli_capabilities
from ascend_kernel_lab.llm.errors import ModelCapabilityError
from ascend_kernel_lab.storage import MIGRATIONS
from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.tasks.runtime import validate_hidden_seed

DoctorRole = Literal["all", "controller", "worker"]


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: str
    mandatory: bool
    summary: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check(
    name: str,
    passed: bool | None,
    summary: str,
    *,
    mandatory: bool = True,
    details: Mapping[str, Any] | None = None,
) -> DiagnosticCheck:
    status = "pass" if passed is True else "fail" if passed is False else "warning"
    return DiagnosticCheck(name, status, mandatory, summary, dict(details or {}))


def _skipped(name: str, summary: str, *, role: DoctorRole) -> DiagnosticCheck:
    return DiagnosticCheck(
        name=name,
        status="skipped",
        mandatory=False,
        summary=summary,
        details={"role": role, "reason": "not required for this container role"},
    )


def _command(argv: tuple[str, ...], *, cwd: Path, timeout: float = 10.0) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"available": False, "returncode": None, "timed_out": False}
    started = time.monotonic()
    try:
        process = subprocess.run(
            (executable, *argv[1:]),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env={
                key: value
                for key in (
                    "PATH",
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "LD_LIBRARY_PATH",
                    "PYTHONPATH",
                    "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH",
                    "ASCEND_TOOLKIT_HOME",
                )
                if (value := os.environ.get(key)) is not None
            },
        )
        return {
            "available": True,
            "returncode": process.returncode,
            "timed_out": False,
            "stdout": process.stdout[-8192:],
            "stderr": process.stderr[-8192:],
            "duration_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "returncode": None,
            "timed_out": True,
            "duration_seconds": time.monotonic() - started,
        }


def _python_runtime_check(cwd: Path) -> dict[str, Any]:
    script = (
        "import importlib,json\n"
        "r={}\n"
        "for n in ('torch','torch_npu','triton'):\n"
        " try:\n"
        "  m=importlib.import_module(n);r[n]={'available':True,'version':str(getattr(m,'__version__','unknown'))}\n"
        " except Exception as e:r[n]={'available':False,'error':type(e).__name__+': '+str(e)}\n"
        "try:\n"
        " t=importlib.import_module('torch');r['npu']={'available':bool(hasattr(t,'npu') and t.npu.is_available()),'count':int(t.npu.device_count()) if hasattr(t,'npu') else 0}\n"
        "except Exception as e:r['npu']={'available':False,'error':type(e).__name__+': '+str(e)}\n"
        "print(json.dumps(r))\n"
    )
    result = _command((sys.executable, "-c", script), cwd=cwd, timeout=30.0)
    if result.get("returncode") == 0:
        try:
            result["runtime"] = json.loads(str(result.get("stdout", "")).splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            result["runtime"] = {}
    return result


def _sqlite_schema(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "user_version": None, "integrity": None}
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        finally:
            connection.close()
        return {"exists": True, "user_version": version, "integrity": integrity}
    except sqlite3.Error as exc:
        return {"exists": True, "user_version": None, "integrity": "error", "error": str(exc)}


def build_doctor_report(
    config: ExperimentConfig, *, role: DoctorRole = "all"
) -> dict[str, Any]:
    """Inspect a checkout without invoking a model or executing candidate code."""

    if role not in {"all", "controller", "worker"}:
        raise ValueError(f"unsupported doctor role: {role}")

    checks: list[DiagnosticCheck] = []
    version_ok = sys.version_info >= (3, 10)
    checks.append(
        _check(
            "python_version",
            version_ok,
            f"Python {platform.python_version()}",
            details={"executable": sys.executable, "minimum": "3.10"},
        )
    )

    registry = TaskRegistry(config.task_root)
    loaded: list[str] = []
    task_error: str | None = None
    try:
        loaded = [task.id for task in registry.load_many(config.tasks)]
    except Exception as exc:  # diagnostic boundary
        task_error = f"{type(exc).__name__}: {exc}"
    checks.append(
        _check(
            "task_specs",
            task_error is None and tuple(loaded) == config.tasks,
            "all configured task specifications validate" if task_error is None else task_error,
            details={"configured": list(config.tasks), "loaded": loaded},
        )
    )

    artifact_root = config.artifact_root
    storage_safe = not artifact_root.is_symlink()
    usage = shutil.disk_usage(artifact_root.parent if artifact_root.parent.exists() else config.project_root)
    checks.append(
        _check(
            "artifact_storage",
            storage_safe and usage.free >= 5 * 1024**3,
            "artifact root is confined and has at least 5 GiB free",
            details={
                "path": str(artifact_root),
                "is_symlink": artifact_root.is_symlink(),
                "free_bytes": usage.free,
            },
        )
    )

    git = _command(("git", "status", "--porcelain=v1"), cwd=config.project_root)
    git_ok = git.get("returncode") == 0 and not str(git.get("stdout", "")).strip()
    checks.append(
        _check(
            "git_release",
            git_ok,
            "checkout is a clean Git commit" if git_ok else "checkout is unversioned or dirty",
            details={
                "available": git.get("available"),
                "returncode": git.get("returncode"),
                "dirty_entries": len(str(git.get("stdout", "")).splitlines()),
            },
        )
    )

    schema = _sqlite_schema(config.db_path)
    latest = MIGRATIONS[-1].version if MIGRATIONS else 0
    db_ok: bool | None
    if not schema["exists"]:
        db_ok = None
        db_summary = "database does not exist yet; run `akg db upgrade`"
    else:
        db_ok = schema.get("user_version") == latest and schema.get("integrity") == "ok"
        db_summary = "database schema and quick integrity check pass"
    checks.append(
        _check(
            "database",
            db_ok,
            db_summary,
            details={**schema, "expected_user_version": latest, "path": str(config.db_path)},
        )
    )

    if role == "controller":
        checks.append(
            _skipped(
                "ascend_python_runtime",
                "Ascend Python runtime is checked in the worker container",
                role=role,
            )
        )
    else:
        runtime = _python_runtime_check(config.project_root)
        runtime_values = runtime.get("runtime", {})
        runtime_ok = (
            isinstance(runtime_values, Mapping)
            and all(
                isinstance(runtime_values.get(name), Mapping)
                and bool(runtime_values[name].get("available"))
                for name in ("torch", "torch_npu", "triton", "npu")
            )
        )
        checks.append(
            _check(
                "ascend_python_runtime",
                runtime_ok,
                "torch, torch_npu, triton-ascend, and an NPU are available",
                details=runtime_values if isinstance(runtime_values, Mapping) else {},
            )
        )

    for executable in ("npu-smi", "msprof"):
        if role == "controller":
            checks.append(
                _skipped(
                    f"executable_{executable}",
                    f"{executable} is checked in the worker container",
                    role=role,
                )
            )
        else:
            path = shutil.which(executable)
            checks.append(
                _check(
                    f"executable_{executable}",
                    path is not None,
                    f"{executable} {'found' if path else 'not found'}",
                    details={"path": path},
                )
            )

    if role == "worker":
        checks.extend(
            (
                _skipped(
                    "model_cli_capabilities",
                    "model CLI is intentionally absent from the worker container",
                    role=role,
                ),
                _skipped(
                    "model_environment",
                    "model environment is intentionally unavailable to the worker container",
                    role=role,
                ),
            )
        )
    elif config.model.provider == "claude_cli":
        help_result = _command(
            (config.model.claude_executable, "--help"), cwd=config.project_root, timeout=15.0
        )
        help_text = f"{help_result.get('stdout', '')}\n{help_result.get('stderr', '')}"
        capabilities = None
        capability_error: str | None = None
        if help_result.get("returncode") == 0:
            try:
                capabilities = parse_claude_cli_capabilities(
                    help_text,
                    require_json_schema=config.model.structured_output,
                )
            except ModelCapabilityError as exc:
                capability_error = str(exc)
        cli_ok = capabilities is not None
        flags = {
            "print": capabilities.print_option if capabilities else None,
            "output_format": (
                capabilities.output_format_option if capabilities else None
            ),
            "json_schema": (
                capabilities.json_schema_option if capabilities else None
            ),
            "tools": capabilities.tools_option if capabilities else None,
            "no_session_persistence": (
                capabilities.no_session_persistence_option if capabilities else None
            ),
            "model": capabilities.model_option if capabilities else None,
        }
        checks.append(
            _check(
                "model_cli_capabilities",
                cli_ok,
                "Claude CLI proves a safe one-shot, tool-free invocation",
                details={
                    "executable": config.model.claude_executable,
                    "capabilities": flags,
                    "error": capability_error,
                },
            )
        )
        references = {
            config.model.anthropic_base_url_env: False,
            config.model.anthropic_auth_token_env: True,
        }
        configured = {name: bool(os.environ.get(name)) for name in references}
        checks.append(
            _check(
                "model_environment",
                all(configured.values()),
                "required AIPing/Anthropic environment references are populated",
                details={"configured": configured},
            )
        )
    else:
        api_reference = config.model.openai.api_key_env
        checks.append(
            _check(
                "model_environment",
                bool(os.environ.get(api_reference)),
                "model API credential reference is populated",
                details={"configured": {api_reference: bool(os.environ.get(api_reference))}},
            )
        )

    hidden_reference = "AKG_HIDDEN_SEED"
    hidden_raw = os.environ.get(hidden_reference)
    hidden_valid = False
    if hidden_raw is not None:
        try:
            validate_hidden_seed(hidden_raw)
        except ValueError:
            pass
        else:
            hidden_valid = True
    checks.append(
        _check(
            "hidden_evaluation_secret",
            hidden_valid,
            "deployment-only hidden evaluation seed satisfies the 128-256 bit policy",
            details={
                "environment_variable": hidden_reference,
                "configured": hidden_raw is not None,
                "policy": "positive base-10 integer with 128-256 significant bits",
            },
        )
    )

    mandatory_failures = [item.name for item in checks if item.mandatory and item.status == "fail"]
    warnings = [item.name for item in checks if item.status == "warning"]
    return {
        "schema_version": "ascend_doctor_report_v1",
        "role": role,
        "captured_at_unix": time.time(),
        "experiment_id": config.id,
        "project_root": str(config.project_root),
        "ready": not mandatory_failures,
        "mandatory_failures": mandatory_failures,
        "warnings": warnings,
        "checks": [item.to_dict() for item in checks],
    }


def installed_package_available(name: str) -> bool:
    """Small public helper used by packaging/self-test commands."""

    return importlib.util.find_spec(name) is not None
