"""Strict aggregation of release, hardware, recovery, and security gate evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AcceptanceGate:
    gate: str
    title: str
    status: str
    source: str
    evidence_sha256: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_EVIDENCE_FILES = {
    "G0": ("release", "g0_release.json"),
    "G3": ("worker smoke and fault recovery", "g3_worker_smoke.json"),
    "G4": ("crash, lease, idempotency, and device concurrency", "g4_crash_concurrency.json"),
    "G5": ("functional kernels and sandbox/security corpus", "g5_functional_security.json"),
    "G6": ("measurement stability and profiler coverage", "g6_measurement.json"),
    "G7": ("real AIPing/Kimi five-round experiment", "g7_real_model.json"),
    "G8": ("reboot, drain, upgrade, and rollback", "g8_reboot_upgrade.json"),
}
_CREDENTIAL = re.compile(
    r"(?i)[\"']?(authorization|api[_-]?key|auth[_-]?token|password|secret)"
    r"[\"']?\s*[:=]\s*[\"']?(?!<redacted>)[^\s\"'}]+"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evidence_gate(
    evidence_root: Path,
    gate: str,
    title: str,
    filename: str,
    *,
    experiment_id: str,
    harness_git_commit: str,
) -> AcceptanceGate:
    path = evidence_root / filename
    if path.is_symlink() or not path.is_file():
        return AcceptanceGate(
            gate, title, "pending", str(path), None, "required evidence file is absent"
        )
    try:
        raw = path.read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError("evidence exceeds 16 MiB")
        text = raw.decode("utf-8")
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise ValueError("evidence root is not an object")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return AcceptanceGate(
            gate,
            title,
            "fail",
            str(path),
            None,
            f"invalid evidence: {exc}",
        )
    digest = _sha256(raw)
    errors: list[str] = []
    if value.get("schema_version") != "ascend_acceptance_gate_v1":
        errors.append("schema_version mismatch")
    if value.get("gate") != gate:
        errors.append("gate identifier mismatch")
    if value.get("experiment_id") != experiment_id:
        errors.append("experiment_id mismatch")
    if value.get("harness_git_commit") != harness_git_commit:
        errors.append("harness_git_commit mismatch")
    checks = value.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("non-empty checks array is required")
    elif any(not isinstance(item, Mapping) for item in checks):
        errors.append("checks must contain objects")
    elif any(item.get("status") != "pass" for item in checks):
        errors.append("one or more evidence checks did not pass")
    if value.get("status") != "pass":
        errors.append("gate status is not pass")
    if _CREDENTIAL.search(text):
        errors.append("possible credential in evidence")
    checks_count = len(checks) if isinstance(checks, list) else 0
    return AcceptanceGate(
        gate,
        title,
        "fail" if errors else "pass",
        str(path),
        digest,
        "; ".join(errors) if errors else f"{checks_count} recorded checks passed",
    )


def aggregate_acceptance(
    *,
    evidence_root: Path | str,
    experiment_id: str,
    harness_git_commit: str,
    doctor_report: Mapping[str, Any],
    verification_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require explicit evidence for every destructive or hardware-specific gate."""

    root = Path(evidence_root).expanduser().resolve()
    gates: list[AcceptanceGate] = []
    for gate, (title, filename) in _EVIDENCE_FILES.items():
        value = _evidence_gate(
            root,
            gate,
            title,
            filename,
            experiment_id=experiment_id,
            harness_git_commit=harness_git_commit,
        )
        if (
            gate == "G7"
            and value.status == "pass"
            and (verification_report is None or verification_report.get("passed") is not True)
        ):
            value = AcceptanceGate(
                value.gate,
                value.title,
                "fail",
                value.source,
                value.evidence_sha256,
                "G7 evidence exists but verify-run did not pass",
            )
        gates.append(value)

    checks = doctor_report.get("checks", [])
    git_ready = any(
        isinstance(item, Mapping)
        and item.get("name") == "git_release"
        and item.get("status") == "pass"
        for item in checks
    )
    gates.insert(
        1,
        AcceptanceGate(
            "G1",
            "fresh clean clone identity",
            "pass" if git_ready else "fail",
            "doctor.git_release",
            None,
            "clean committed checkout" if git_ready else "checkout is dirty or unversioned",
        ),
    )
    doctor_ready = doctor_report.get("ready") is True
    gates.insert(
        2,
        AcceptanceGate(
            "G2",
            "deployment doctor and feature prerequisites",
            "pass" if doctor_ready else "fail",
            "doctor",
            None,
            "all mandatory doctor checks passed" if doctor_ready else "doctor has mandatory failures",
        ),
    )
    status_counts = {
        status: sum(gate.status == status for gate in gates)
        for status in ("pass", "fail", "pending")
    }
    return {
        "schema_version": "ascend_acceptance_report_v1",
        "experiment_id": experiment_id,
        "harness_git_commit": harness_git_commit,
        "passed": status_counts["pass"] == len(gates),
        "status_counts": status_counts,
        "evidence_root": str(root),
        "gates": [gate.to_dict() for gate in gates],
        "doctor": dict(doctor_report),
        "run_verification": dict(verification_report) if verification_report else None,
    }


__all__ = ["AcceptanceGate", "aggregate_acceptance"]
