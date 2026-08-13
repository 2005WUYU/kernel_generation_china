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
            ROOT / "scripts/watch-experiment-910c.sh",
        ]
        for path in paths:
            with self.subTest(path=path):
                process = subprocess.run(
                    ["sh", "-n", str(path)], capture_output=True, text=True, check=False
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertIn("set -eu", path.read_text(encoding="utf-8"))

    def test_experiment_watcher_tracks_multicard_rounds_without_secrets(self) -> None:
        watcher = (ROOT / "scripts/watch-experiment-910c.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("exp_910c_deepseek_v4_pro_cold_sft_v1", watcher)
        self.assertIn("0,2,4,6,8,10,12,14", watcher)
        self.assertIn('f"round_{round_number:02d}"', watcher)
        self.assertIn("sleep 5", watcher)
        self.assertIn("docker top", watcher)
        self.assertIn("final_result.json", watcher)
        self.assertIn("ELAPSED=$(elapsed_time)", watcher)
        self.assertIn("FINAL SUMMARY elapsed=$(elapsed_time)", watcher)
        self.assertIn("hidden_geo=", watcher)
        self.assertIn("TOTAL passed=", watcher)
        self.assertIn('error_type == "StageTimeout"', watcher)
        self.assertIn('return f"{label}超时"', watcher)
        self.assertIn('status == "RETRY_WAIT"', watcher)
        self.assertIn('return f"{label}执行中"', watcher)
        self.assertIn("只生成了 $finals/10 个最终结果", watcher)
        self.assertIn('docker logs --tail 100 "$CONTROLLER_NAME"', watcher)
        self.assertNotIn("AUTH_TOKEN", watcher)
        self.assertNotIn("hidden.env", watcher.lower())
        self.assertNotIn("hidden_seed", watcher.lower())
        self.assertNotIn("final_evaluation", watcher.lower())
        self.assertEqual(watcher.count("prompt.json"), 1)

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
        self.assertIn('names=("torch", "torch_npu", "triton")', worker)
        self.assertIn("find_spec(name)", worker)
        self.assertNotIn("import torch, torch_npu, triton", worker)
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
        self.assertIn("AKG_DEVICE_LOCK_ROOT", runner)
        self.assertIn("/usr/local/Ascend/driver/lib64/driver", runner)
        self.assertEqual(
            runner.count('--env "LD_LIBRARY_PATH=$WORKER_LD_LIBRARY_PATH"'),
            2,
        )
        self.assertEqual(
            runner.count('--volume "$LOCK_ROOT:/var/lock/ascend-kernel-lab:rw"'),
            2,
        )
        self.assertIn("stop the project Worker before probe or baseline", runner)
        self.assertIn("device lock root must have AKG_SHARED_GID", runner)
        maintenance_block = runner.split("    probe|baseline)", 1)[1].split(
            "        ;;", 1
        )[0]
        self.assertIn("--tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777", maintenance_block)
        self.assertNotIn("\n             shift\n", maintenance_block)
        self.assertIn('exec "$@"\' sh "$@"', maintenance_block)
        worker_block = runner.split("start-worker|start-workers)", 1)[1].split(
            ";;", 1
        )[0]
        self.assertNotIn('"$CONTROLLER_ENV_FILE"', worker_block)
        self.assertIn("AKG_DEVICE_IDS:-0,2,4,6,8,10,12,14", runner)
        self.assertIn('start_worker_container "$physical_device"', worker_block)
        self.assertIn('ASCEND_VISIBLE_DEVICES=$physical_device', runner)
        self.assertIn('--env DEVICE_ID=0', runner)
        self.assertIn(
            "AKG_DEVICE_LOCK_ROOT=/var/lock/ascend-kernel-lab/device-$physical_device",
            runner,
        )
        self.assertIn('AKG_CONFIG_PATH=$CONTAINER_CONFIG_PATH', runner)
        self.assertNotIn("experiment_910c_kimi_k3.yaml", runner)
        self.assertIn("reject_model_environment", entrypoint)
        self.assertIn("torch.npu.device_count() != 1", entrypoint)
        self.assertIn("*','*", entrypoint)

    def test_controller_explicitly_uses_runc_when_ascend_is_host_default(self) -> None:
        runner = (ROOT / "scripts/run-containers-910c.sh").read_text(encoding="utf-8")
        init_block = runner.split("    init)", 1)[1].split("    ;;", 1)[0]
        controller_block = runner.split("    start-controller)", 1)[1].split(
            "    ;;", 1
        )[0]
        self.assertIn("--runtime=runc", init_block)
        self.assertNotIn("--runtime=ascend", init_block)
        self.assertIn("--runtime=runc", controller_block)
        self.assertNotIn("--runtime=ascend", controller_block)

    def test_controller_is_oneshot_while_worker_restarts_after_failure(self) -> None:
        runner = (ROOT / "scripts/run-containers-910c.sh").read_text(encoding="utf-8")
        worker_block = runner.split("start_worker_container()", 1)[1].split(
            "\n}", 1
        )[0]
        controller_block = runner.split("    start-controller)", 1)[1].split(
            "        ;;", 1
        )[0]

        self.assertIn("--restart on-failure:3", worker_block)
        self.assertIn("--restart=no", controller_block)
        self.assertNotIn("--restart on-failure", controller_block)

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
        self.assertIn("CLAUDE_CODE_ARCHIVE", script)
        self.assertIn("CLAUDE_CODE_SHA256", script)
        self.assertIn("sha256sum", controller)
        self.assertIn("tar -tvzf", controller)
        self.assertIn("CLAUDE_CODE_VERSION", script)
        self.assertNotIn("npm view", controller)
        self.assertNotIn("npm install", controller)
        self.assertEqual(script.count("--network=none"), 2)
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
