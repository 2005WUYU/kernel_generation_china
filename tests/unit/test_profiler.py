from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.profiling.parser import MsprofParser

ROOT = Path(__file__).resolve().parents[2]


class MsprofParserTests(unittest.TestCase):
    def test_percent_columns_and_coverage(self) -> None:
        parser = MsprofParser((r"^generated_",))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "nested"
            target.mkdir()
            target.joinpath("op.csv").write_bytes((ROOT / "tests/fixtures/msprof/op_summary_v1.csv").read_bytes())
            summary = parser.parse(temporary)
        self.assertTrue(summary.profile_available)
        self.assertEqual(summary.kernel_count, 1)
        self.assertAlmostEqual(summary.candidate_kernel_coverage or 0, 18 / 18.4)
        self.assertAlmostEqual(summary.pipeline["scalar_ratio"] or 0, 0.18)
        self.assertEqual(summary.memory["gm_read_gbps"], None)
        self.assertTrue(any(item.type == "scalar_operations_high" for item in summary.observations))

    def test_nanoseconds_and_multiple_kernels(self) -> None:
        parser = MsprofParser((r"^generated_",))
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "op.csv").write_bytes((ROOT / "tests/fixtures/msprof/op_summary_v2.csv").read_bytes())
            summary = parser.parse(temporary)
        self.assertEqual(summary.kernel_count, 2)
        self.assertAlmostEqual(summary.scheduling["device_execution_us"] or 0, 30.0)
        self.assertAlmostEqual(summary.candidate_kernel_coverage or 0, 30 / 31)

    def test_mixed_csv_schemas_are_normalized_per_file(self) -> None:
        parser = MsprofParser((r"^generated_",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v1.csv").write_bytes(
                (ROOT / "tests/fixtures/msprof/op_summary_v1.csv").read_bytes()
            )
            (root / "v2.csv").write_bytes(
                (ROOT / "tests/fixtures/msprof/op_summary_v2.csv").read_bytes()
            )
            (root / "metadata.csv").write_text(
                "Label,Value\nversion,1\n", encoding="utf-8"
            )
            summary = parser.parse(root)
        self.assertTrue(summary.profile_available)
        self.assertEqual(summary.kernel_count, 3)
        self.assertAlmostEqual(summary.scheduling["device_execution_us"] or 0, 48.0)
        self.assertAlmostEqual(summary.candidate_kernel_coverage or 0, 48 / 49.4)
        self.assertEqual(len(summary.source_files), 2)
        self.assertEqual(len(summary.readable_source_files), 3)

    def test_percent_sign_controls_ratio_scaling_at_boundaries(self) -> None:
        parser = MsprofParser((r"^generated_",))
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "op.csv").write_text(
                "Kernel Name,Task Duration(us),Vector Ratio,Cube Ratio,"
                "Scalar Ratio,MTE2 Ratio\n"
                "generated_ratio_kernel,1,0.5%,1%,0.5,1\n",
                encoding="utf-8",
            )
            summary = parser.parse(temporary)
        self.assertAlmostEqual(summary.pipeline["vector_ratio"] or 0, 0.005)
        self.assertAlmostEqual(summary.pipeline["cube_ratio"] or 0, 0.01)
        self.assertAlmostEqual(summary.pipeline["scalar_ratio"] or 0, 0.5)
        self.assertAlmostEqual(summary.pipeline["mte2_ratio"] or 0, 1.0)

    def test_no_csv_is_explicitly_unavailable(self) -> None:
        parser = MsprofParser(("generated",))
        with tempfile.TemporaryDirectory() as temporary:
            summary = parser.parse(temporary)
        self.assertFalse(summary.profile_available)
        self.assertIsNotNone(summary.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
