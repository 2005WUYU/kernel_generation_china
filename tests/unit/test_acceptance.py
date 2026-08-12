from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.acceptance import aggregate_acceptance


class AcceptanceTests(unittest.TestCase):
    @staticmethod
    def _doctor(ready: bool = True) -> dict[str, object]:
        return {
            "ready": ready,
            "checks": [
                {
                    "name": "git_release",
                    "status": "pass" if ready else "fail",
                }
            ],
        }

    def test_missing_hardware_evidence_remains_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = aggregate_acceptance(
                evidence_root=temporary,
                experiment_id="exp",
                harness_git_commit="abc",
                doctor_report=self._doctor(),
                verification_report={"passed": True},
            )
            self.assertFalse(report["passed"])
            statuses = {item["gate"]: item["status"] for item in report["gates"]}
            self.assertEqual(statuses["G3"], "pending")
            self.assertEqual(statuses["G8"], "pending")

    def test_all_version_bound_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "G0": "g0_release.json",
                "G3": "g3_worker_smoke.json",
                "G4": "g4_crash_concurrency.json",
                "G5": "g5_functional_security.json",
                "G6": "g6_measurement.json",
                "G7": "g7_real_model.json",
                "G8": "g8_reboot_upgrade.json",
            }
            for gate, filename in files.items():
                (root / filename).write_text(
                    json.dumps(
                        {
                            "schema_version": "ascend_acceptance_gate_v1",
                            "gate": gate,
                            "experiment_id": "exp",
                            "harness_git_commit": "abc",
                            "status": "pass",
                            "checks": [{"name": "check", "status": "pass"}],
                        }
                    ),
                    encoding="utf-8",
                )
            report = aggregate_acceptance(
                evidence_root=root,
                experiment_id="exp",
                harness_git_commit="abc",
                doctor_report=self._doctor(),
                verification_report={"passed": True},
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["status_counts"], {"pass": 9, "fail": 0, "pending": 0})

    def test_version_mismatch_fails_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "g0_release.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_acceptance_gate_v1",
                        "gate": "G0",
                        "experiment_id": "exp",
                        "harness_git_commit": "wrong",
                        "status": "pass",
                        "checks": [{"name": "tests", "status": "pass"}],
                    }
                ),
                encoding="utf-8",
            )
            report = aggregate_acceptance(
                evidence_root=root,
                experiment_id="exp",
                harness_git_commit="abc",
                doctor_report=self._doctor(),
                verification_report=None,
            )
            gate = next(item for item in report["gates"] if item["gate"] == "G0")
            self.assertEqual(gate["status"], "fail")

    def test_json_credential_in_evidence_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "g0_release.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ascend_acceptance_gate_v1",
                        "gate": "G0",
                        "experiment_id": "exp",
                        "harness_git_commit": "abc",
                        "status": "pass",
                        "checks": [
                            {
                                "name": "unsafe",
                                "status": "pass",
                                "api_key": "not-allowed-here",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = aggregate_acceptance(
                evidence_root=root,
                experiment_id="exp",
                harness_git_commit="abc",
                doctor_report=self._doctor(),
                verification_report=None,
            )
            gate = next(item for item in report["gates"] if item["gate"] == "G0")
            self.assertEqual(gate["status"], "fail")
            self.assertIn("credential", gate["summary"])


if __name__ == "__main__":
    unittest.main()
