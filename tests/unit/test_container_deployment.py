from __future__ import annotations

import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTAINER = ROOT / "deploy/container"


class ContainerDeploymentContractTests(unittest.TestCase):
    @staticmethod
    def _watcher_python(function_name: str) -> str:
        watcher = (ROOT / "scripts/watch-experiment-910c.sh").read_text(
            encoding="utf-8"
        )
        function = watcher.split(f"{function_name}() {{", 1)[1]
        return function.split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

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
        self.assertIn("未运行(Repair耗尽/无正确Seed)", watcher)
        self.assertIn("未运行(Host瓶颈早停)", watcher)
        self.assertIn("空槽(Repair提前通过)", watcher)
        self.assertNotIn("预算未用或早停", watcher)
        self.assertIn("FINAL=$finals/$task_total", watcher)
        self.assertIn('"$finals" -eq "$task_total"', watcher)
        self.assertIn("只生成了 $finals/$task_total 个最终结果", watcher)
        self.assertIn('docker logs --tail 100 "$CONTROLLER_NAME"', watcher)
        self.assertNotIn("AUTH_TOKEN", watcher)
        self.assertNotIn("hidden.env", watcher.lower())
        self.assertNotIn("hidden_seed", watcher.lower())
        self.assertNotIn("final_evaluation", watcher.lower())
        self.assertEqual(watcher.count("prompt.json"), 1)

    def test_experiment_watcher_prints_exact_round_and_terminal_outcomes(self) -> None:
        experiment_id = "watcher-output"
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_root = project_root / "runs" / experiment_id
            self._write_json(
                run_root / "experiment.json",
                {
                    "experiment": {
                        "rounds_per_task": 5,
                        "maximum_repair_rounds": 3,
                    }
                },
            )

            def final(
                task_id: str,
                *,
                status: str,
                termination: str,
                repairs: int,
                optimizations: int,
            ) -> None:
                self._write_json(
                    run_root / "tasks" / task_id / "final_result.json",
                    {
                        "status": status,
                        "termination_reason": termination,
                        "repair_rounds": repairs,
                        "optimization_rounds": optimizations,
                    },
                )

            def evaluated_round(
                task_id: str,
                round_number: int,
                *,
                phase: str,
                phase_index: int,
                overall_status: str,
                repair_attempt: int = 0,
            ) -> None:
                round_root = (
                    run_root / "tasks" / task_id / f"round_{round_number:02d}"
                )
                self._write_json(
                    round_root / "prompt.json",
                    {
                        "metadata": {
                            "phase": phase,
                            "phase_index": phase_index,
                        },
                        "user_prompt": {
                            "feedback_state": {
                                "repair_attempt": repair_attempt,
                            }
                        },
                    },
                )
                self._write_json(
                    round_root / "evaluation_result.json",
                    {
                        "overall_status": overall_status,
                        "trajectory_phase": phase,
                        "phase_index": phase_index,
                        "repair_attempt": repair_attempt,
                    },
                )

            final(
                "repair_failed",
                status="repair_exhausted",
                termination="repair_exhausted",
                repairs=3,
                optimizations=0,
            )
            evaluated_round(
                "repair_failed",
                3,
                phase="repair",
                phase_index=3,
                overall_status="correctness_failed",
            )

            final(
                "host_stopped",
                status="passed",
                termination="host_dispatch_limited",
                repairs=1,
                optimizations=0,
            )
            evaluated_round(
                "host_stopped",
                1,
                phase="repair",
                phase_index=1,
                overall_status="correct",
            )

            final(
                "repair_early",
                status="passed",
                termination="optimization_round_budget_completed",
                repairs=1,
                optimizations=5,
            )
            for round_number in range(1, 7):
                phase = "repair" if round_number == 1 else "optimization"
                phase_index = 1 if round_number == 1 else round_number - 1
                evaluated_round(
                    "repair_early",
                    round_number,
                    phase=phase,
                    phase_index=phase_index,
                    overall_status="correct",
                )

            final(
                "slot_repaired",
                status="passed",
                termination="optimization_round_budget_completed",
                repairs=1,
                optimizations=1,
            )
            evaluated_round(
                "slot_repaired",
                1,
                phase="repair",
                phase_index=1,
                overall_status="correct",
            )
            evaluated_round(
                "slot_repaired",
                2,
                phase="optimization",
                phase_index=1,
                overall_status="correctness_failed",
            )
            evaluated_round(
                "slot_repaired",
                3,
                phase="optimization_repair",
                phase_index=1,
                repair_attempt=1,
                overall_status="correct",
            )

            candidate = run_root / "tasks" / "queue_failure" / "round_01" / "candidate.py"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text("def run(): pass\n", encoding="utf-8")

            database = project_root / "runs" / "metadata.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE evaluation_jobs (
                        experiment_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        round_number INTEGER NOT NULL,
                        stage TEXT NOT NULL,
                        status TEXT NOT NULL,
                        last_error_json TEXT,
                        result_json TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO evaluation_jobs (
                        experiment_id, task_id, round_number, stage, status,
                        last_error_json, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        "queue_failure",
                        1,
                        "CORRECTNESS",
                        "SUCCEEDED",
                        None,
                        json.dumps({"status": "fail", "passed": False}),
                    ),
                )

            program = self._watcher_python("task_snapshot")
            process = subprocess.run(
                [
                    "python3",
                    "-",
                    str(database),
                    str(run_root),
                    experiment_id,
                    "repair_failed",
                    "host_stopped",
                    "repair_early",
                    "slot_repaired",
                    "queue_failure",
                    "unfinished",
                ],
                input=program,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            lines = {
                line.split(" ", 1)[0]: line for line in process.stdout.splitlines()
            }

            self.assertIn(
                "FINAL=repair_exhausted STOP=repair_exhausted ROUNDS=REP3/OPT0",
                lines["repair_failed"],
            )
            self.assertIn(
                "R03[REP3]=完成[correctness_failed]", lines["repair_failed"]
            )
            self.assertIn(
                "R04=未运行(Repair耗尽/无正确Seed)", lines["repair_failed"]
            )
            self.assertIn(
                "FINAL=passed STOP=host_dispatch_limited ROUNDS=REP1/OPT0",
                lines["host_stopped"],
            )
            self.assertIn(
                "R02=未运行(Host瓶颈早停)", lines["host_stopped"]
            )
            self.assertIn(
                "FINAL=passed STOP=optimization_round_budget_completed "
                "ROUNDS=REP1/OPT5",
                lines["repair_early"],
            )
            self.assertIn("R07=空槽(Repair提前通过)", lines["repair_early"])
            self.assertIn("R08=空槽(Repair提前通过)", lines["repair_early"])
            self.assertIn(
                "ROUNDS=REP1/OPT1/OPTREP1", lines["slot_repaired"]
            )
            self.assertIn(
                "R03[OPT1-REP1]=完成[correct]", lines["slot_repaired"]
            )
            self.assertIn("R01=CORRECTNESS失败", lines["queue_failure"])
            self.assertIn("R01=未开始", lines["unfinished"])

    def test_experiment_watcher_prefers_durable_single_task_set(self) -> None:
        experiment_id = "single-task"
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            run_root = project_root / "runs" / experiment_id
            self._write_json(
                run_root / "experiment.json",
                {
                    "experiment": {
                        "tasks": ["k01_vector_add", "k02_bias_gelu"]
                    }
                },
            )
            database = project_root / "runs" / "metadata.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    """
                    CREATE TABLE tasks (
                        experiment_id TEXT NOT NULL,
                        task_id TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO tasks (experiment_id, task_id) VALUES (?, ?)",
                    (experiment_id, "k01_vector_add"),
                )

            process = subprocess.run(
                [
                    "python3",
                    "-",
                    str(database),
                    str(run_root),
                    experiment_id,
                    "fallback_task",
                ],
                input=self._watcher_python("task_ids"),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.splitlines(), ["k01_vector_add"])

            with sqlite3.connect(database) as connection:
                connection.execute("DELETE FROM tasks")
            fallback_process = subprocess.run(
                [
                    "python3",
                    "-",
                    str(database),
                    str(run_root),
                    experiment_id,
                    "fallback_task",
                ],
                input=self._watcher_python("task_ids"),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                fallback_process.returncode, 0, fallback_process.stderr
            )
            self.assertEqual(
                fallback_process.stdout.splitlines(),
                ["k01_vector_add", "k02_bias_gelu"],
            )

    def test_experiment_watcher_final_summary_includes_termination_and_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            self._write_json(
                run_root / "tasks" / "passed_task" / "final_result.json",
                {
                    "status": "passed",
                    "termination_reason": "host_dispatch_limited",
                    "repair_rounds": 1,
                    "optimization_rounds": 0,
                    "best_round": 1,
                    "hidden_correctness_passed": True,
                    "speedup_geomean": 1.25,
                    "minimum_speedup": 1.1,
                },
            )
            for number, phase in (
                (1, "repair"),
                (2, "optimization"),
                (3, "optimization_repair"),
            ):
                self._write_json(
                    run_root
                    / "tasks"
                    / "passed_task"
                    / f"round_{number:02d}"
                    / "evaluation_result.json",
                    {"trajectory_phase": phase},
                )
            self._write_json(
                run_root / "tasks" / "failed_task" / "final_result.json",
                {
                    "status": "repair_exhausted",
                    "termination_reason": "repair_exhausted",
                    "repair_rounds": 3,
                    "optimization_rounds": 0,
                    "best_round": None,
                },
            )

            program = self._watcher_python("print_final_summary")
            process = subprocess.run(
                [
                    "python3",
                    "-",
                    str(run_root),
                    "passed_task",
                    "failed_task",
                ],
                input=program,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn(
                "passed_task status=passed termination=host_dispatch_limited "
                "REP=1 OPT=0",
                process.stdout,
            )
            self.assertIn("OPT_REPAIR=1 ACTUAL=3", process.stdout)
            self.assertIn(
                "failed_task status=repair_exhausted termination=repair_exhausted "
                "REP=3 OPT=0",
                process.stdout,
            )
            self.assertIn("TOTAL passed=1 failed=1 tasks=2", process.stdout)

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
