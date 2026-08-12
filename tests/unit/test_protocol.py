from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ascend_kernel_lab.protocol import harness_digest


class HarnessDigestTests(unittest.TestCase):
    def test_digest_is_sensitive_to_guard_and_profiler_semantics(self) -> None:
        baseline = harness_digest()
        self.assertEqual(harness_digest(), baseline)
        original_read_bytes = Path.read_bytes

        for relative in (
            "evaluation/source_guard.py",
            "profiling/parser.py",
            "profiling/runner.py",
        ):
            with self.subTest(relative=relative):

                def changed(path: Path, target: str = relative) -> bytes:
                    value = original_read_bytes(path)
                    if path.as_posix().endswith(target):
                        return value + b"\n# simulated release mutation\n"
                    return value

                with patch.object(Path, "read_bytes", changed):
                    self.assertNotEqual(harness_digest(), baseline)


if __name__ == "__main__":
    unittest.main()
