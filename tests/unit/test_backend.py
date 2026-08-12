from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ascend_kernel_lab.backend import (
    AscendTritonBackend,
    Backend,
    FakeBackend,
    StageResult,
    StageStatus,
)
from ascend_kernel_lab.domain import EvaluationStage
from ascend_kernel_lab.profiling import MsprofRunner
from ascend_kernel_lab.tasks import TaskRegistry
from ascend_kernel_lab.tasks.runtime import hidden_cases_from_template
from ascend_kernel_lab.worker import StageProcessResult
from ascend_kernel_lab.worker.stage_entry import _candidate_guard

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RecordingRunner:
    def __init__(self, *, passed: bool = True) -> None:
        self.passed = passed
        self.calls: list[dict[str, object]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        payload: bytes = b"",
        env: Mapping[str, str] | None = None,
        timeout_seconds: float,
    ) -> StageProcessResult:
        request = json.loads(payload)
        self.calls.append(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "request": request,
                "env": dict(env or {}),
                "timeout": timeout_seconds,
            }
        )
        stage = request["stage"]
        if stage == "baseline":
            details = {
                "passed": True,
                "per_case": [
                    {
                        "case_id": case["id"],
                        "weight": case.get("weight", 1.0),
                        "pytorch_eager_us": 12.0,
                        "torch_compile_us": None,
                        "official_us": None,
                    }
                    for case in request["cases"]
                ],
                "torch_compile_available": False,
                "official_available": False,
                "unavailable_reasons": {"torch_compile": "test", "official": "test"},
            }
        else:
            details = {"passed": self.passed, "compiled": self.passed}
        result = {
            "schema_version": "ascend_isolated_stage_v1",
            "stage": stage,
            "passed": self.passed,
            "details": details,
        }
        (cwd / "stage_result.json").write_text(json.dumps(result), encoding="utf-8")
        return StageProcessResult(
            argv=tuple(argv),
            returncode=0 if self.passed else 2,
            stdout=b"",
            stderr=b"",
            duration_seconds=0.01,
        )

    def cancel_current(self) -> bool:
        return False


class AvailableMsprofRunner(MsprofRunner):
    def __init__(self) -> None:
        super().__init__("msprof-test-double")

    def available(self) -> bool:
        return True


class ProfileEvidenceRunner(RecordingRunner):
    def __init__(self, csv_text: str, *, driver_passed: bool = True) -> None:
        super().__init__(passed=driver_passed)
        self.csv_text = csv_text
        self.driver_passed = driver_passed

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        payload: bytes = b"",
        env: Mapping[str, str] | None = None,
        timeout_seconds: float,
    ) -> StageProcessResult:
        request = json.loads(payload)
        self.calls.append(
            {
                "argv": tuple(argv),
                "cwd": cwd,
                "request": request,
                "env": dict(env or {}),
                "timeout": timeout_seconds,
            }
        )
        output_argument = next(
            value for value in argv if value.startswith("--output=")
        )
        raw_root = Path(output_argument.removeprefix("--output="))
        raw_root.mkdir(parents=True, exist_ok=True)
        (raw_root / "op.csv").write_text(self.csv_text, encoding="utf-8")
        (cwd / "stage_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "ascend_isolated_stage_v1",
                    "stage": "profile",
                    "passed": self.driver_passed,
                    "details": {"passed": self.driver_passed},
                }
            ),
            encoding="utf-8",
        )
        return StageProcessResult(
            argv=tuple(argv),
            returncode=0,
            stdout=b"",
            stderr=b"",
            duration_seconds=0.01,
        )


class StageResultTests(unittest.TestCase):
    def test_round_trip_and_derived_properties(self) -> None:
        started = datetime.now(timezone.utc)
        result = StageResult.success(
            EvaluationStage.COMPILE,
            started_at=started,
            details={"compiled": True},
            artifacts={"log": "stdout.log"},
        )
        restored = StageResult.from_dict(result.to_dict())
        self.assertTrue(restored.passed)
        self.assertEqual(restored.stage, EvaluationStage.COMPILE)
        self.assertEqual(restored.details, {"compiled": True})
        self.assertGreaterEqual(restored.duration_seconds, 0)

    def test_retryable_is_limited_to_infrastructure_outcomes(self) -> None:
        with self.assertRaises(ValueError):
            StageResult(
                stage=EvaluationStage.CORRECTNESS,
                status=StageStatus.FAIL,
                retryable=True,
            )


class CandidateRuntimeGuardTests(unittest.TestCase):
    def test_tensor_dunder_and_compute_methods_are_blocked_then_restored(self) -> None:
        class Tensor:
            def __add__(self, other: object) -> str:
                del other
                return "original-add"

            def exp(self) -> str:
                return "original-exp"

            def matmul(self, other: object) -> str:
                del other
                return "original-matmul"

        class Functional:
            @staticmethod
            def gelu(value: object) -> object:
                return value

        class NN:
            functional = Functional()

        class Torch:
            Tensor: type[Any]
            nn = NN()

            @staticmethod
            def add(left: object, right: object) -> tuple[object, object]:
                return left, right

        Torch.Tensor = Tensor

        tensor = Tensor()
        torch = Torch()
        with _candidate_guard(torch):
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                tensor + tensor
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                tensor.exp()
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                tensor.matmul(tensor)
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                torch.add(tensor, tensor)
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                torch.nn.functional.gelu(tensor)
        self.assertEqual(tensor + tensor, "original-add")
        self.assertEqual(tensor.exp(), "original-exp")
        self.assertEqual(tensor.matmul(tensor), "original-matmul")


class FakeBackendTests(unittest.TestCase):
    def test_protocol_and_scripted_results(self) -> None:
        failure = StageResult.failure(EvaluationStage.COMPILE)
        fake = FakeBackend({EvaluationStage.COMPILE: [failure]})
        self.assertIsInstance(fake, Backend)
        task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")
        with tempfile.TemporaryDirectory() as temporary:
            result = fake.compile(
                Path(temporary) / "candidate.py",
                task,
                task.correctness_cases[:1],
                Path(temporary),
            )
            baselines = fake.measure_baselines(
                task, task.benchmark_cases[:1], Path(temporary)
            )
        self.assertFalse(result.passed)
        self.assertEqual(baselines["per_case"][0]["pytorch_eager_us"], 12.0)


class AscendBackendHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = TaskRegistry(PROJECT_ROOT / "task_specs").load("k01_vector_add")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _candidate(self) -> Path:
        path = self.root / "candidate.py"
        path.write_text(
            """import torch
import triton
import triton.language as tl

@triton.jit
def generated_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))

def custom_op(x, y):
    out = torch.empty_like(x)
    generated_kernel[(1,)](x, y, out, n=x.numel())
    return out
""",
            encoding="utf-8",
        )
        return path

    def test_source_check_uses_fail_closed_guard(self) -> None:
        backend = AscendTritonBackend(lock_root=self.root / "locks")
        candidate = self._candidate()
        good = backend.source_check(candidate, self.task)
        candidate.write_text("import os\ndef custom_op(x): return x\n", encoding="utf-8")
        bad = backend.source_check(candidate, self.task)
        self.assertTrue(good.passed)
        self.assertFalse(bad.passed)
        self.assertTrue(bad.details["findings"])

    def test_compile_creates_an_isolated_cache_and_parses_result(self) -> None:
        runner = RecordingRunner()
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            lock_root=self.root / "locks",
        )
        result = backend.compile(
            self._candidate(),
            self.task,
            self.task.correctness_cases[:1],
            self.root / "artifacts",
        )
        self.assertTrue(result.passed)
        call = runner.calls[0]
        request = cast(dict[str, Any], call["request"])
        environment = cast(dict[str, str], call["env"])
        self.assertEqual(request["stage"], "compile")
        self.assertEqual(environment["TRITON_ALWAYS_COMPILE"], "1")
        self.assertTrue(Path(result.artifacts["triton_cache"]).is_dir())
        self.assertTrue(Path(result.artifacts["candidate"]).is_file())
        private_work = cast(Path, call["cwd"])
        self.assertEqual(stat.S_IMODE(private_work.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(private_work.parent.stat().st_mode), 0o700)
        published = Path(result.artifacts["attempt_dir"])
        published_mode = stat.S_IMODE(published.stat().st_mode)
        self.assertEqual(published_mode & 0o777, 0o770)
        if sys.platform != "darwin":
            self.assertEqual(published_mode & stat.S_ISGID, stat.S_ISGID)
        self.assertIn("published", published.parts)
        for raw_path in result.artifacts.values():
            path = Path(raw_path)
            mode = stat.S_IMODE(path.stat().st_mode)
            if path.is_dir():
                self.assertEqual(mode & 0o777, 0o770)
                if sys.platform != "darwin":
                    self.assertEqual(mode & stat.S_ISGID, stat.S_ISGID)
            else:
                self.assertEqual(mode, 0o660)
            self.assertNotIn(private_work, path.parents)
        manifest_path = Path(result.artifacts["artifact_manifest"])
        manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(
            manifest_path.name,
            f"artifact_manifest.{manifest_digest}.json",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["schema_version"],
            "ascend_stage_artifact_manifest_v1",
        )
        self.assertTrue(manifest["files"])
        for entry in manifest["files"]:
            self.assertEqual(
                set(entry),
                {"relative_path", "sha256", "size_bytes", "type"},
            )
            controlled = published / entry["relative_path"]
            value = controlled.read_bytes()
            self.assertEqual(entry["size_bytes"], len(value))
            self.assertEqual(entry["sha256"], hashlib.sha256(value).hexdigest())
        self.assertNotIn(
            manifest_path.name,
            {entry["relative_path"] for entry in manifest["files"]},
        )

    def test_baseline_protocol_requires_b0_and_reports_optional_layers(self) -> None:
        runner = RecordingRunner()
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            lock_root=self.root / "locks",
        )
        result = backend.measure_baselines(
            self.task,
            self.task.benchmark_cases[:1],
            self.root / "baseline-artifacts",
        )
        self.assertEqual(result["per_case"][0]["pytorch_eager_us"], 12.0)
        self.assertFalse(result["torch_compile_available"])
        self.assertFalse(result["official_available"])

    def test_strong_baseline_comparisons_are_attached_in_parent(self) -> None:
        stage = StageResult.success(
            EvaluationStage.BENCHMARK,
            details={
                "passed": True,
                "per_case": [
                    {
                        "case_id": "bench_1",
                        "weight": 2.0,
                        "candidate": {"median_us": 5.0},
                    },
                    {
                        "case_id": "bench_2",
                        "weight": 1.0,
                        "candidate": {"median_us": 10.0},
                    },
                ],
            },
        )
        snapshot = {
            "identity_sha256": "a" * 64,
            "per_case": [
                {
                    "case_id": "bench_1",
                    "torch_compile_us": 10.0,
                    "official_us": 4.0,
                },
                {
                    "case_id": "bench_2",
                    "torch_compile_us": 5.0,
                    "official_us": None,
                },
            ],
        }
        result = AscendTritonBackend._attach_strong_baselines(stage, snapshot)
        per_case = cast(list[dict[str, Any]], result.details["per_case"])
        self.assertEqual(per_case[0]["speedup_vs_compile"], 2.0)
        self.assertEqual(per_case[0]["speedup_vs_official"], 0.8)
        self.assertEqual(per_case[1]["speedup_vs_compile"], 0.5)
        self.assertIsNone(per_case[1]["speedup_vs_official"])
        self.assertAlmostEqual(
            cast(float, result.details["geomean_speedup_vs_compile"]),
            (2.0 * 2.0 * 0.5) ** (1 / 3),
        )
        self.assertEqual(result.details["official_comparison_case_count"], 1)

    def test_profile_is_explicitly_unavailable_without_msprof(self) -> None:
        backend = AscendTritonBackend(
            msprof_runner=MsprofRunner("definitely-not-a-real-msprof"),
            lock_root=self.root / "locks",
        )
        result = backend.profile(
            self._candidate(),
            self.task,
            self.task.profile_cases,
            self.root / "profile-artifacts",
        )
        self.assertEqual(result.status, StageStatus.UNAVAILABLE)
        self.assertFalse(result.details["profile_available"])

    def test_kernel_patterns_are_specific_and_reject_broad_names(self) -> None:
        pattern_text = AscendTritonBackend._kernel_patterns(self._candidate())
        self.assertEqual(len(pattern_text), 1)
        pattern = re.compile(pattern_text[0])
        self.assertIsNotNone(pattern.search("generated_kernel"))
        self.assertIsNotNone(pattern.search("generated_kernel_01234567"))
        self.assertIsNone(pattern.search("framework_generated_kernel"))
        self.assertIsNone(pattern.search("generated_kernel_epilogue"))

        broad = self.root / "broad.py"
        for name in ("k", "_", "kernel"):
            with self.subTest(name=name):
                broad.write_text(
                    f"import triton\n@triton.jit\ndef {name}(x):\n    return x\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "too broad"):
                    AscendTritonBackend._kernel_patterns(broad)

    def test_profile_driver_failure_cannot_pass_with_csv_evidence(self) -> None:
        runner = ProfileEvidenceRunner(
            "Kernel Name,Task Duration(us),Vector Ratio\n"
            "generated_kernel,10,50%\n",
            driver_passed=False,
        )
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            msprof_runner=AvailableMsprofRunner(),
            lock_root=self.root / "locks",
        )
        result = backend.profile(
            self._candidate(),
            self.task,
            self.task.profile_cases,
            self.root / "driver-failure-profile",
        )
        self.assertEqual(result.status, StageStatus.FAIL)
        self.assertFalse(result.details["profile_available"])

    def test_duration_only_profile_fails_mandatory_pipeline_gate(self) -> None:
        runner = ProfileEvidenceRunner(
            "Kernel Name,Task Duration(us)\n"
            "generated_kernel,10\n"
            "framework_kernel,1\n"
        )
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            msprof_runner=AvailableMsprofRunner(),
            lock_root=self.root / "locks",
        )
        result = backend.profile(
            self._candidate(),
            self.task,
            self.task.profile_cases,
            self.root / "duration-only-profile",
        )
        self.assertEqual(result.status, StageStatus.UNAVAILABLE)
        self.assertFalse(result.details["profile_available"])
        self.assertEqual(
            result.details["missing_mandatory_groups"], ["pipe_utilization"]
        )

    def test_profile_pass_requires_every_mandatory_group(self) -> None:
        runner = ProfileEvidenceRunner(
            "Kernel Name,Task Duration(us),Vector Ratio\n"
            "generated_kernel,10,50%\n"
            "framework_kernel,1,1%\n"
        )
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            msprof_runner=AvailableMsprofRunner(),
            lock_root=self.root / "locks",
        )
        result = backend.profile(
            self._candidate(),
            self.task,
            self.task.profile_cases,
            self.root / "complete-profile",
        )
        self.assertEqual(result.status, StageStatus.PASS)
        self.assertEqual(result.details["missing_mandatory_groups"], [])
        argv = cast(tuple[str, ...], runner.calls[0]["argv"])
        self.assertIn("--kernel-name=generated_kernel", argv)

    def test_hidden_profile_passes_cases_only_over_stdin(self) -> None:
        assert self.task.root is not None
        hidden_cases = tuple(
            case
            for case in hidden_cases_from_template(
                self.task.root,
                secret_seed=44_321,
                count_correctness=20,
                count_benchmark=6,
            )
            if case.kind == "benchmark"
        )[:1]
        runner = RecordingRunner()
        backend = AscendTritonBackend(
            runner=runner,  # type: ignore[arg-type]
            msprof_runner=AvailableMsprofRunner(),
            lock_root=self.root / "locks",
        )
        result = backend.profile(
            self._candidate(),
            self.task,
            hidden_cases,
            self.root / "hidden-profile-artifacts",
        )
        self.assertEqual(len(runner.calls), 1)
        call = runner.calls[0]
        request = cast(dict[str, Any], call["request"])
        work = cast(Path, call["cwd"])
        self.assertEqual(request["stage"], "profile")
        self.assertTrue(request["settings"]["redact_case_details"])
        self.assertFalse(any(work.rglob("stage_input.json")))
        self.assertEqual(
            (work / "profile_driver.py").read_text(encoding="utf-8"),
            "from ascend_kernel_lab.worker.stage_entry import main\n"
            "raise SystemExit(main([]))\n",
        )
        persisted = b"\n".join(
            path.read_bytes()
            for path in work.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        hidden_snapshot = json.dumps(
            [case.to_dict() for case in hidden_cases], sort_keys=True
        ).encode("utf-8")
        self.assertNotIn(hidden_snapshot, persisted)
        self.assertNotIn(b'"params"', persisted)
        self.assertNotIn(b'"seed"', persisted)
        self.assertNotIn(b'"shape"', persisted)
        self.assertEqual(result.artifacts, {})
        self.assertFalse(
            any(
                path.name.startswith("artifact_manifest.")
                for path in (self.root / "hidden-profile-artifacts").rglob("*")
            )
        )


if __name__ == "__main__":
    unittest.main()
