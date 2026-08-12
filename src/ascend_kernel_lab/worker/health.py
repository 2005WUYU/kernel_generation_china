"""Non-mutating Ascend runtime health checks."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from ascend_kernel_lab.backend.base import StageResult, StageStatus

from .stage_runner import StageRunner


class AscendHealthChecker:
    """Check the management CLI and a minimal Python/NPU synchronization path."""

    def __init__(
        self,
        *,
        runner: StageRunner | None = None,
        python_executable: str = sys.executable,
        device: str = "npu:0",
        timeout_seconds: float = 30.0,
        work_dir: Path | str | None = None,
    ) -> None:
        self.runner = runner or StageRunner(maximum_output_bytes=512 * 1024)
        self.python_executable = python_executable
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir or "/tmp").resolve()

    def check(self) -> StageResult:
        started = datetime.now(timezone.utc)
        npu_smi = shutil.which("npu-smi")
        if npu_smi is None:
            return StageResult(
                stage="HEALTH_CHECK",
                status=StageStatus.UNAVAILABLE,
                started_at=started,
                finished_at=datetime.now(timezone.utc),
                details={"healthy": False, "npu_smi_available": False},
                error={"type": "MissingExecutable", "message": "npu-smi was not found"},
            )
        smi = self.runner.run(
            (npu_smi, "info"),
            cwd=self.work_dir,
            timeout_seconds=self.timeout_seconds,
        )
        script = """
import importlib, json
torch = importlib.import_module("torch")
importlib.import_module("torch_npu")
triton = importlib.import_module("triton")
torch.npu.set_device(%s)
available = bool(torch.npu.is_available())
count = int(torch.npu.device_count()) if available else 0
if available:
    torch.npu.synchronize()
print("AKG_HEALTH_RESULT=" + json.dumps({"available": available, "count": count,
                                         "torch": str(torch.__version__),
                                         "triton": str(triton.__version__)}), flush=True)
""".strip() % int(self.device.removeprefix("npu:"))
        runtime = self.runner.run(
            (self.python_executable, "-I", "-c", script),
            cwd=self.work_dir,
            timeout_seconds=self.timeout_seconds,
        )
        details: dict[str, object] = {
            "healthy": False,
            "npu_smi_available": True,
            "npu_smi_returncode": smi.returncode,
            "npu_smi_usable": smi.succeeded,
            "npu_smi_stderr": smi.stderr_text()[-4096:],
        }
        if not runtime.succeeded:
            details.update(
                {
                    "runtime_returncode": runtime.returncode,
                    "runtime_stderr": runtime.stderr_text()[-4096:],
                }
            )
            return StageResult.infrastructure_error(
                "HEALTH_CHECK",
                message="Ascend Python runtime import/synchronization failed",
                error_type="RuntimeHealthError",
                retryable=True,
                timed_out=runtime.timed_out,
                started_at=started,
                details=details,
            )
        try:
            output = runtime.stdout_text()
            marker = "AKG_HEALTH_RESULT="
            position = output.rfind(marker)
            if position < 0:
                raise ValueError("health result marker is missing")
            runtime_details, _ = json.JSONDecoder().raw_decode(
                output[position + len(marker) :].lstrip()
            )
            if not isinstance(runtime_details, dict):
                raise ValueError("health JSON must be an object")
            count = int(runtime_details.get("count", 0))
        except (IndexError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return StageResult.infrastructure_error(
                "HEALTH_CHECK",
                message=f"health probe emitted invalid JSON: {exc}",
                error_type="HealthProtocolError",
                retryable=False,
                started_at=started,
                details=details,
            )
        healthy = bool(runtime_details.get("available")) and count > 0
        details.update(runtime_details)
        details["healthy"] = healthy
        if not smi.succeeded:
            details["warning"] = (
                "container npu-smi was unavailable, but the NPU runtime "
                "imported and synchronized successfully"
            )
        return StageResult(
            stage="HEALTH_CHECK",
            status=StageStatus.PASS if healthy else StageStatus.FAIL,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            details=details,
            error=(
                None
                if healthy
                else {"type": "NoAvailableDevice", "message": "torch reports no NPU"}
            ),
        )
