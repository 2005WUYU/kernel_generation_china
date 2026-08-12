from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate-hidden-seed.py"


class DeploymentScriptTests(unittest.TestCase):
    def _validate(self, value: str | None) -> subprocess.CompletedProcess[str]:
        environment = {} if value is None else {"AKG_HIDDEN_SEED": value}
        return subprocess.run(
            (sys.executable, str(VALIDATOR)),
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_hidden_seed_validator_enforces_128_to_256_bits_without_echo(self) -> None:
        valid = str(1 << 127)
        self.assertEqual(self._validate(valid).returncode, 0)

        for invalid in (None, "12x", "123", str(1 << 256)):
            with self.subTest(invalid=invalid):
                result = self._validate(invalid)
                self.assertEqual(result.returncode, 10)
                if invalid is not None:
                    self.assertNotIn(invalid, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
