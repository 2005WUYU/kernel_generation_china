from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ascend_kernel_lab.worker import StageRunner, clean_environment


class CleanEnvironmentTests(unittest.TestCase):
    def test_credentials_and_proxies_are_removed(self) -> None:
        clean = clean_environment(
            {
                "PATH": "/bin",
                "ASCEND_HOME_PATH": "/opt/ascend",
                "KIMI_API_KEY": "secret",
                "ANTHROPIC_AUTH_TOKEN": "secret",
                "HTTPS_PROXY": "http://proxy",
                "UNRELATED": "value",
            }
        )
        self.assertEqual(clean["PATH"], "/bin")
        self.assertEqual(clean["ASCEND_HOME_PATH"], "/opt/ascend")
        self.assertNotIn("KIMI_API_KEY", clean)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", clean)
        self.assertNotIn("HTTPS_PROXY", clean)
        self.assertNotIn("UNRELATED", clean)

    def test_unsafe_extra_environment_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clean_environment({}, {"API_TOKEN": "secret"})


class StageRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_payload_is_supplied_on_stdin_without_a_shell(self) -> None:
        runner = StageRunner()
        result = runner.run(
            (
                sys.executable,
                "-c",
                "import sys; print(sys.stdin.buffer.read().decode()); print(sys.argv[1])",
                "; touch should-not-exist",
            ),
            cwd=self.cwd,
            payload=b"payload",
            timeout_seconds=3,
        )
        self.assertTrue(result.succeeded, result.stderr_text())
        self.assertIn("payload", result.stdout_text())
        self.assertIn("; touch should-not-exist", result.stdout_text())
        self.assertFalse((self.cwd / "should-not-exist").exists())

    def test_timeout_terminates_a_process_that_closed_its_pipes(self) -> None:
        runner = StageRunner(termination_grace_seconds=0.1)
        started = time.monotonic()
        result = runner.run(
            (
                sys.executable,
                "-c",
                "import os,time; os.close(1); os.close(2); time.sleep(30)",
            ),
            cwd=self.cwd,
            timeout_seconds=0.2,
        )
        self.assertTrue(result.timed_out)
        self.assertTrue(result.terminated)
        self.assertLess(time.monotonic() - started, 3)

    def test_timeout_kills_descendants_holding_output_pipes(self) -> None:
        runner = StageRunner(termination_grace_seconds=0.1)
        code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'])"
        )
        result = runner.run(
            (sys.executable, "-c", code),
            cwd=self.cwd,
            timeout_seconds=0.2,
        )
        self.assertTrue(result.timed_out)
        self.assertLess(result.duration_seconds, 3)

    def test_timeout_kills_descendant_after_group_leader_exits(self) -> None:
        runner = StageRunner(termination_grace_seconds=0.1)
        child = (
            "import os,pathlib,signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path('descendant.pid').write_text(str(os.getpid())); "
            "time.sleep(30)"
        )
        parent = (
            "import subprocess,sys; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}])"
        )
        result = runner.run(
            (sys.executable, "-c", parent),
            cwd=self.cwd,
            timeout_seconds=0.3,
        )
        self.assertTrue(result.timed_out)
        descendant_pid = int((self.cwd / "descendant.pid").read_text())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                os.kill(descendant_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            self.fail("descendant survived the process-group timeout cleanup")

    def test_combined_stdout_stderr_limit_terminates_stage(self) -> None:
        runner = StageRunner(maximum_output_bytes=1024, termination_grace_seconds=0.1)
        result = runner.run(
            (
                sys.executable,
                "-c",
                "import os; os.write(1,b'a'*800); os.write(2,b'b'*800)",
            ),
            cwd=self.cwd,
            timeout_seconds=3,
        )
        self.assertTrue(result.output_limit_exceeded)
        self.assertLessEqual(len(result.stdout) + len(result.stderr), 1024)
        self.assertFalse(result.succeeded)

    def test_cancel_current_terminates_process_group(self) -> None:
        runner = StageRunner(termination_grace_seconds=0.1)
        holder: list[object] = []

        def run() -> None:
            holder.append(
                runner.run(
                    (sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=self.cwd,
                    timeout_seconds=20,
                )
            )

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 3
        while not runner.cancel_current() and time.monotonic() < deadline:
            time.sleep(0.01)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(holder), 1)

    def test_oversized_stdin_is_rejected_before_launch(self) -> None:
        runner = StageRunner(maximum_stdin_bytes=3)
        with self.assertRaises(ValueError):
            runner.run(
                (sys.executable, "-c", "pass"),
                cwd=self.cwd,
                payload=b"four",
                timeout_seconds=1,
            )


if __name__ == "__main__":
    unittest.main()
