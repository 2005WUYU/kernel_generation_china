from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DeploymentPermissionContractTests(unittest.TestCase):
    def test_service_units_share_group_and_restart_only_infrastructure_failures(self) -> None:
        controller = (
            ROOT / "deploy/systemd/ascend-kernel-controller.service"
        ).read_text(encoding="utf-8")
        worker = (ROOT / "deploy/systemd/ascend-kernel-worker.service").read_text(
            encoding="utf-8"
        )

        for unit in (controller, worker):
            self.assertIn("Group=ascend-kernel\n", unit)
            self.assertIn("UMask=0007\n", unit)
            self.assertIn("ReadWritePaths=/opt/ascend-kernel-lab/runs\n", unit)

        self.assertIn("RestartPreventExitStatus=2 3 5 10 130\n", controller)
        self.assertNotIn("RestartPreventExitStatus=2 3 4", controller)
        self.assertIn("RestartPreventExitStatus=2 3 10\n", worker)
        self.assertNotIn("RestartPreventExitStatus=2 3 4", worker)

    def test_service_wrappers_set_group_collaboration_umask(self) -> None:
        for name in ("run-controller.sh", "run-worker.sh"):
            script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            active_umasks = [
                line.strip()
                for line in script.splitlines()
                if re.fullmatch(r"umask\s+\d+", line.strip())
            ]
            self.assertEqual(active_umasks, ["umask 0007"])

    def test_install_uses_traversable_non_group_writable_shared_venv(self) -> None:
        script = (ROOT / "scripts/install-910c.sh").read_text(encoding="utf-8")
        self.assertIn("umask 0022", script)
        self.assertIn("AKG_SHARED_GROUP", script)
        self.assertIn("AKG_VENV_OWNER", script)
        self.assertIn('-exec chown "$VENV_OWNER:$SHARED_GROUP"', script)
        self.assertIn('-exec chown -h "$VENV_OWNER:$SHARED_GROUP"', script)
        self.assertNotIn('chown -R "$VENV_OWNER:$SHARED_GROUP"', script)
        self.assertIn("-type d -exec chmod 0755", script)
        self.assertIn("-type f ! -perm -0100 -exec chmod 0644", script)
        self.assertIn("install -d -m 2770", script)
        self.assertIn("AKG_ALLOW_NON_GIT_SOURCE_FOR_TESTING", script)
        self.assertGreaterEqual(script.count("status --porcelain=v1"), 2)

    def test_remote_install_fences_clean_commit_before_and_after_install(self) -> None:
        script = (ROOT / "scripts/remote-install-910c.sh").read_text(encoding="utf-8")
        self.assertGreaterEqual(script.count("git rev-parse --verify HEAD"), 2)
        self.assertGreaterEqual(script.count("git status --porcelain=v1"), 2)
        self.assertIn("clone is dirty before installation", script)
        self.assertIn("clone changed during installation", script)


if __name__ == "__main__":
    unittest.main()
