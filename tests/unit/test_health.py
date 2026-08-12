from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from ascend_kernel_lab.worker.health import AscendHealthChecker
from ascend_kernel_lab.worker.stage_runner import StageProcessResult


class _Runner:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, *_args: object, **_kwargs: object) -> StageProcessResult:
        self.calls += 1
        if self.calls == 1:
            return StageProcessResult(("npu-smi", "info"), 211, b"", b"", 0.1)
        return StageProcessResult(
            ("python",),
            0,
            b'AKG_HEALTH_RESULT={"available":true,"count":1}\nCANN noise',
            b"",
            0.1,
        )


class HealthTests(unittest.TestCase):
    def test_runtime_sync_is_authoritative_when_container_npu_smi_is_unusable(self) -> None:
        runner = _Runner()
        checker = AscendHealthChecker(runner=runner, work_dir=Path("/tmp"))  # type: ignore[arg-type]
        with mock.patch(
            "ascend_kernel_lab.worker.health.shutil.which", return_value="/usr/bin/npu-smi"
        ):
            result = checker.check()
        self.assertTrue(result.passed)
        self.assertFalse(result.details["npu_smi_usable"])
        self.assertTrue(result.details["healthy"])


if __name__ == "__main__":
    unittest.main()
