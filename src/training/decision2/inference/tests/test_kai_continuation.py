"""The native Kai continuation collector must verify every ancestry hop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference.kai_continuation import PARENT_MANIFEST_SHA, verify_export


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    root: Path, parent_sha: str, parent_native: Path | None = None
) -> tuple[Path, Path, str]:
    root.mkdir()
    native = root / "selected-native"
    native.mkdir()
    weight = native / "weight.bin"
    weight.write_bytes(b"sample native weights")
    (native / "MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "decision.files.v1",
                "files": {
                    "weight.bin": {
                        "bytes": weight.stat().st_size,
                        "sha256": _sha(weight),
                    }
                },
            }
        )
        + "\n"
    )
    manifest_sha = _sha(native / "MANIFEST.json")
    identity = {"native_manifest_sha256": parent_sha}
    (root / "COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "COMPLETE_DECISION_FINETUNE",
                "native": str(native),
                "native_manifest_sha256": manifest_sha,
                "identity": identity,
            }
        )
        + "\n"
    )
    (root / "RUN.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "native_path": str(parent_native) if parent_native else "source",
            }
        )
        + "\n"
    )
    return root, native, manifest_sha


class KaiContinuationChainTest(unittest.TestCase):
    def test_sequential_export_requires_intact_parent_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            first_run, first_native, first_sha = _run(
                base / "first", PARENT_MANIFEST_SHA
            )
            next_run, next_native, next_sha = _run(
                base / "next", first_sha, first_native
            )
            with self.assertRaisesRegex(ValueError, "parent needs"):
                verify_export(next_run, next_native, next_sha)
            identity = verify_export(
                next_run, next_native, next_sha, first_run, first_native
            )
            self.assertEqual(identity["base_manifest_sha256"], first_sha)
            self.assertEqual(identity["root_base_manifest_sha256"], PARENT_MANIFEST_SHA)
            self.assertEqual(
                identity["parent_run_receipt_sha256"], _sha(first_run / "COMPLETE.json")
            )
            (first_native / "weight.bin").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "file differs"):
                verify_export(next_run, next_native, next_sha, first_run, first_native)


if __name__ == "__main__":
    unittest.main()
