"""Synthetic byte, parameter, path and privacy gates for baseline receipts."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from scripts.baseline_attestation_v3 import (
    build_attestation,
    main,
    verify_attestation,
)
from scripts.plan_final_eval import BASELINES

SOURCE_ROOT = Path(__file__).resolve().parents[2]


def safetensors(path: Path, *, shape: list[int] | None = None) -> None:
    shape = shape or [4]
    count = 1
    for dim in shape:
        count *= dim
    data = b"\0" * (count * 4)
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, len(data)],
            }
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


class BaselineAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = next(model for model in BASELINES if model.key == "eikos4b")
        self.model_root = self.root / "models"
        self.external_root = self.root / "external"
        self.package = self.model_root / self.model.model_dir
        self.package.mkdir(parents=True)
        (self.package / "config.json").write_text("{}\n")
        safetensors(self.package / "model.safetensors")
        self.receipt = self.root / "receipt.json"
        self.attestation = self.root / "attestation.json"

    def build(self, *, count: int = 4, model=None, calibration=None) -> dict:
        build_attestation(
            model or self.model,
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
            receipt_output=self.receipt,
            attestation_output=self.attestation,
            loaded_parameter_count=count,
            calibration_path=calibration,
        )
        return json.loads(self.attestation.read_text())

    def verify(self, item: dict, *, model=None) -> dict:
        return verify_attestation(
            item,
            model or self.model,
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )

    def test_build_is_exclusive_private_and_parameter_bound(self) -> None:
        item = self.build()
        receipt = self.verify(item)
        self.assertEqual(receipt["parameter_count"], 4)
        self.assertEqual(receipt["loaded_parameter_count"], 4)
        self.assertEqual(item["size_b"], 4e-9)
        self.assertEqual(receipt["weight_files"], ["model.safetensors"])
        self.assertEqual(self.receipt.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.attestation.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build()

    def test_rejects_wrong_loaded_count_before_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "Native loader count"):
            self.build(count=5)
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.attestation.exists())

    def test_package_mutation_and_extra_file_are_detected(self) -> None:
        item = self.build()
        (self.package / "config.json").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "package, runtime, or calibration"):
            self.verify(item)
        (self.package / "config.json").write_text("{}\n")
        (self.package / "extra.txt").write_text("new\n")
        with self.assertRaisesRegex(ValueError, "package, runtime, or calibration"):
            self.verify(item)

    def test_receipt_and_roster_tampering_are_detected(self) -> None:
        item = self.build()
        bad = dict(item, size_b=9.0)
        with self.assertRaisesRegex(ValueError, "differs from receipt"):
            self.verify(bad)
        self.receipt.write_text(self.receipt.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "receipt changed"):
            self.verify(item)

    def test_symlink_and_nonprivate_output_are_rejected(self) -> None:
        external = self.root / "external-file"
        external.write_text("outside")
        (self.package / "outside").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.build()
        (self.package / "outside").unlink()
        public_parent = self.root / "public"
        public_parent.mkdir(mode=0o755)
        with self.assertRaisesRegex(ValueError, "mode 0700"):
            build_attestation(
                self.model,
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
                receipt_output=public_parent / "receipt.json",
                attestation_output=self.attestation,
                loaded_parameter_count=4,
            )
        self.assertFalse((public_parent / "receipt.json").exists())

    def test_runtime_and_calibration_are_rehashed(self) -> None:
        model = replace(self.model, source_dir="native-runtime")
        runtime = self.external_root / model.source_dir
        runtime.mkdir(parents=True)
        runtime_file = runtime / "run.py"
        runtime_file.write_text("pass\n")
        calibration = self.package / "calibration.json"
        calibration.write_text("{}\n")
        item = self.build(model=model, calibration=calibration)
        self.verify(item, model=model)
        runtime_file.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "package, runtime, or calibration"):
            self.verify(item, model=model)

    def test_invalid_safetensors_and_duplicate_tensor_are_rejected(self) -> None:
        safetensors(self.package / "second.safetensors")
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            self.build()
        (self.package / "second.safetensors").unlink()
        (self.package / "model.safetensors").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "Invalid safetensors header"):
            self.build()

    def test_cli_emits_no_private_paths_or_contents(self) -> None:
        args = [
            "--key",
            self.model.key,
            "--source-root",
            str(SOURCE_ROOT),
            "--model-root",
            str(self.model_root),
            "--external-root",
            str(self.external_root),
            "--attestation-output",
            str(self.attestation),
        ]
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(
                main(
                    [
                        "build",
                        *args,
                        "--receipt-output",
                        str(self.receipt),
                        "--loaded-parameter-count",
                        "4",
                    ]
                ),
                0,
            )
            self.assertEqual(main(["verify", *args]), 0)
            (self.package / "config.json").write_text("changed")
            with self.assertRaises(SystemExit) as stopped:
                main(["verify", *args])
            with self.assertRaises(SystemExit) as malformed:
                main(["build", "--unknown", str(self.root)])
        self.assertEqual(stopped.exception.code, 2)
        self.assertEqual(malformed.exception.code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn(str(self.root), errors.getvalue())
        self.assertIn("Baseline attestation failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
