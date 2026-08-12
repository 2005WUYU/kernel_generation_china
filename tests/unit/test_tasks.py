from __future__ import annotations

import unittest
from pathlib import Path

from ascend_kernel_lab.tasks.loader import CaseSpec, TaskRegistry, TaskSpecError
from ascend_kernel_lab.tasks.runtime import hidden_cases_from_template, validate_hidden_seed

ROOT = Path(__file__).resolve().parents[2]


class TaskRegistryTests(unittest.TestCase):
    def test_all_builtin_tasks_are_complete(self) -> None:
        registry = TaskRegistry(ROOT / "task_specs")
        self.assertEqual(len(registry.ids()), 10)
        for spec in registry.load_many():
            self.assertGreaterEqual(len(spec.correctness_cases), 8)
            self.assertGreaterEqual(len(spec.benchmark_cases), 3)
            self.assertEqual(spec.entry_point, "custom_op")
            self.assertNotIn("hidden", str(spec.public_prompt_view()).lower())
            self.assertEqual(len(spec.digest()), 64)
            assert spec.root is not None
            for filename in ("reference.py", "input_generator.py", "output_validator.py", "baseline.py"):
                self.assertTrue((spec.root / filename).is_file())

    def test_hidden_cases_are_secret_seeded_and_not_persisted(self) -> None:
        root = ROOT / "task_specs" / "k01_vector_add"
        first = hidden_cases_from_template(root, secret_seed=1234)
        again = hidden_cases_from_template(root, secret_seed=1234)
        other = hidden_cases_from_template(root, secret_seed=5678)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(sum(case.kind == "correctness" for case in first), 20)
        self.assertEqual(sum(case.kind == "benchmark" for case in first), 6)
        self.assertFalse((root / "hidden_cases.jsonl").exists())

    def test_hidden_suite_kind_is_independent_for_queue_reconstruction(self) -> None:
        root = ROOT / "task_specs" / "k09_gemm"
        combined = hidden_cases_from_template(
            root, secret_seed=1234, count_correctness=20, count_benchmark=6
        )
        correctness_only = hidden_cases_from_template(
            root, secret_seed=1234, count_correctness=20, count_benchmark=0
        )
        benchmark_only = hidden_cases_from_template(
            root, secret_seed=1234, count_correctness=0, count_benchmark=6
        )
        self.assertEqual(tuple(case for case in combined if case.kind == "correctness"), correctness_only)
        self.assertEqual(tuple(case for case in combined if case.kind == "benchmark"), benchmark_only)

    def test_production_hidden_seed_policy_is_bounded_and_high_entropy(self) -> None:
        self.assertEqual(validate_hidden_seed(1 << 127), 1 << 127)
        with self.assertRaises(ValueError):
            validate_hidden_seed(1234)
        with self.assertRaises(ValueError):
            validate_hidden_seed(1 << 256)
        self.assertEqual(
            validate_hidden_seed(1234, allow_insecure_for_testing=True), 1234
        )

    def test_case_schema_rejects_unknown_fields(self) -> None:
        with self.assertRaises(TaskSpecError):
            CaseSpec.from_dict({
                "id": "x", "kind": "correctness", "dtype": "float16", "params": {"n": 1},
                "unexpected": True,
            })

    def test_case_schema_rejects_coercions_and_unbounded_values(self) -> None:
        base = {
            "id": "c01",
            "kind": "correctness",
            "dtype": "float16",
            "params": {"n": 1},
        }
        invalid = (
            {**base, "params": {"n": True}},
            {**base, "seed": "7"},
            {**base, "weight": float("nan")},
            {**base, "address_offset": -1},
            {**base, "noncontiguous": 1},
            {**base, "distribution": "adversarial-code"},
            {**base, "id": "../escape"},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(TaskSpecError):
                CaseSpec.from_dict(value)

    def test_path_traversal_task_id_is_rejected(self) -> None:
        registry = TaskRegistry(ROOT / "task_specs")
        with self.assertRaises(TaskSpecError):
            registry.load("../secrets")


if __name__ == "__main__":
    unittest.main()
