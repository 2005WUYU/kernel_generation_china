from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTAINER = ROOT / "deploy/container"


class ContainerDeploymentContractTests(unittest.TestCase):
    def test_shell_files_are_syntactically_valid(self) -> None:
        paths = [
            CONTAINER / "controller-entrypoint.sh",
            CONTAINER / "worker-entrypoint.sh",
            ROOT / "scripts/build-container-images.sh",
            ROOT / "scripts/run-containers-910c.sh",
        ]
        for path in paths:
            with self.subTest(path=path):
                process = subprocess.run(
                    ["sh", "-n", str(path)], capture_output=True, text=True, check=False
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("set -eu", path.read_text(encoding="utf-8"))

    def test_build_context_cannot_include_checkout_or_secrets(self) -> None:
        script = (ROOT / "scripts/build-container-images.sh").read_text(encoding="utf-8")
        self.assertEqual(script.count('"$PROJECT_ROOT/deploy/container"'), 2)
        self.assertNotIn('    "$PROJECT_ROOT"\n', script)
        for name in ("Dockerfile.controller", "Dockerfile.worker"):
            dockerfile = (CONTAINER / name).read_text(encoding="utf-8")
            self.assertNotIn("COPY .", dockerfile)
            self.assertNotIn("deploy/container/", dockerfile)
        dockerignore = (CONTAINER / ".dockerignore").read_text(encoding="utf-8")
        self.assertTrue(dockerignore.startswith("*\n"))
        self.assertNotIn("!*.env", dockerignore)

    def test_images_never_install_accelerator_stack(self) -> None:
        worker = (CONTAINER / "Dockerfile.worker").read_text(encoding="utf-8")
        for forbidden in ("pip install", "torch==", "torch_npu==", "triton==", "apt-get"):
            self.assertNotIn(forbidden, worker)
        self.assertIn("import torch, torch_npu, triton", worker)
        self.assertIn("import yaml", worker)

    def test_worker_is_networkless_secretless_and_single_npu(self) -> None:
        runner = (ROOT / "scripts/run-containers-910c.sh").read_text(encoding="utf-8")
        entrypoint = (CONTAINER / "worker-entrypoint.sh").read_text(encoding="utf-8")
        self.assertIn("--runtime=ascend", runner)
        self.assertIn("--network none", runner)
        self.assertIn("--read-only", runner)
        self.assertIn("--cap-drop ALL", runner)
        self.assertIn("--security-opt no-new-privileges", runner)
        self.assertIn("probe|baseline", runner)
        self.assertIn("{{.State.Health.Status}}", runner)
        worker_block = runner.split("start-worker)", 1)[1].split(";;", 1)[0]
        self.assertNotIn('"$CONTROLLER_ENV_FILE"', worker_block)
        self.assertIn("reject_model_environment", entrypoint)
        self.assertIn("torch.npu.device_count() != 1", entrypoint)
        self.assertIn("*','*", entrypoint)

    def test_checkout_and_hidden_seed_are_fail_closed(self) -> None:
        for name in ("controller-entrypoint.sh", "worker-entrypoint.sh"):
            content = (CONTAINER / name).read_text(encoding="utf-8")
            self.assertIn("rev-parse --show-toplevel", content)
            self.assertIn("rev-parse --verify HEAD", content)
            self.assertIn("status --porcelain=v1 --untracked-files=all", content)
            self.assertIn("validate-hidden-seed.py", content)

    def test_base_and_claude_versions_are_immutable(self) -> None:
        script = (ROOT / "scripts/build-container-images.sh").read_text(encoding="utf-8")
        controller = (CONTAINER / "Dockerfile.controller").read_text(encoding="utf-8")
        self.assertIn("sha256:", script)
        self.assertIn("docker image inspect", script)
        self.assertIn("CONTROLLER_NODE_BASE", script)
        self.assertIn("CONTROLLER_NODE_BASE", controller)
        self.assertIn("CLAUDE_CODE_VERSION", script)
        self.assertIn("CLAUDE_CODE_INTEGRITY", script)
        self.assertIn("dist.integrity", controller)
        self.assertIn("--pull=false", script)
        self.assertNotIn("CLAUDE_CODE_VERSION=latest", script)

    def test_arbitrary_runtime_users_get_private_writable_home(self) -> None:
        runner = (ROOT / "scripts/run-containers-910c.sh").read_text(encoding="utf-8")
        self.assertEqual(runner.count("--env HOME=/tmp/akg-home"), 3)
        self.assertIn('--user "$WORKER_UID:$SHARED_GID"', runner)
        self.assertIn('--user "$CONTROLLER_UID:$SHARED_GID"', runner)
        self.assertIn('--group-add "$NPU_DEVICE_GID"', runner)
        for name in ("controller-entrypoint.sh", "worker-entrypoint.sh"):
            content = (CONTAINER / name).read_text(encoding="utf-8")
            self.assertIn('mkdir -p "$HOME"', content)
            self.assertIn('chmod 0700 "$HOME"', content)


if __name__ == "__main__":
    unittest.main()
