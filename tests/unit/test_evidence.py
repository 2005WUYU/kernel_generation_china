from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ascend_kernel_lab.evidence import (
    EvidenceIntegrityError,
    validate_artifact_map,
    verify_stage_artifact_manifest,
)


class EvidenceIntegrityTests(unittest.TestCase):
    def _tree(self, root: Path) -> tuple[Path, Path]:
        published = root / "exp/tasks/k01/round_01/published/compile"
        ir = published / "ir"
        ir.mkdir(parents=True)
        candidate = published / "candidate.py"
        kernel = ir / "kernel.ttir.mlir"
        candidate.write_text("def custom_op(x): return x\n", encoding="utf-8")
        kernel.write_text("module {}\n", encoding="utf-8")
        entries = []
        for path in (candidate, kernel):
            payload = path.read_bytes()
            entries.append(
                {
                    "relative_path": path.relative_to(published).as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "type": "application/octet-stream",
                }
            )
        value = {
            "schema_version": "ascend_stage_artifact_manifest_v1",
            "files": entries,
        }
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()
        manifest = published / f"artifact_manifest.{digest}.json"
        manifest.write_bytes(payload)
        return manifest, kernel

    def test_verifies_manifest_filename_and_complete_evidence_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, kernel = self._tree(root)

            summary = verify_stage_artifact_manifest(manifest, artifact_root=root)
            mapped = validate_artifact_map(
                {
                    "artifact_manifest": str(manifest),
                    "ir": str(kernel.parent),
                    "kernel": str(kernel),
                },
                artifact_root=root,
            )

            self.assertEqual(summary.file_count, 2)
            self.assertEqual(mapped, summary)

    def test_rejects_tamper_and_undeclared_addition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, kernel = self._tree(root)
            kernel.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceIntegrityError, "hash or size"):
                verify_stage_artifact_manifest(manifest, artifact_root=root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _kernel = self._tree(root)
            (manifest.parent / "unlisted.log").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceIntegrityError, "disagree"):
                verify_stage_artifact_manifest(manifest, artifact_root=root)

    def test_rejects_content_address_and_map_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _kernel = self._tree(root)
            renamed = manifest.with_name(f"artifact_manifest.{'0' * 64}.json")
            manifest.rename(renamed)
            with self.assertRaisesRegex(EvidenceIntegrityError, "digest"):
                verify_stage_artifact_manifest(renamed, artifact_root=root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _kernel = self._tree(root)
            outside = root / "outside.log"
            outside.write_text("outside", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceIntegrityError, "outside"):
                validate_artifact_map(
                    {
                        "artifact_manifest": str(manifest),
                        "escape": str(outside),
                    },
                    artifact_root=root,
                )


if __name__ == "__main__":
    unittest.main()
