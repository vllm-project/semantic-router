"""Synthetic byte, parameter, path and privacy gates for baseline receipts."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from inference.kev import KEV_BASE_ID, KEV_BASE_REVISION
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


def named_safetensors(path: Path, tensors: dict[str, int]) -> None:
    offset = 0
    header = {}
    for name, count in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": [count],
            "data_offsets": [offset, offset + count * 4],
        }
        offset += count * 4
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


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


class KevCompositeAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = next(model for model in BASELINES if model.key == "kev")
        self.model_root = self.root / "models"
        self.external_root = self.root / "external"
        self.package = self.model_root / self.model.model_dir
        self.base = self.model_root / "Qwen3.5-4B-Base"
        self.runtime = self.external_root / self.model.source_dir
        self.package.mkdir(parents=True)
        self.base.mkdir(parents=True)
        (self.runtime / "kev").mkdir(parents=True)
        code = self.runtime / "kev/api.py"
        code.write_text("# pinned native runtime\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(self.runtime)], check=True)
        subprocess.run(["git", "-C", str(self.runtime), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(self.runtime),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "Pinned runtime fixture",
            ],
            check=True,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(self.runtime), "rev-parse", "HEAD"], text=True
        ).strip()
        for package, rev in (
            (self.package, self.model.revision),
            (self.base, KEV_BASE_REVISION),
        ):
            metadata = package / ".cache/huggingface/download/config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(rev + "\netag\n", encoding="utf-8")
        named_safetensors(self.package / "adapter_model.safetensors", {"lora": 3})
        named_safetensors(
            self.base / "model.safetensors",
            {"model.language_model.weight": 4, "model.visual.weight": 1},
        )
        torch.save(
            {
                "base": KEV_BASE_ID,
                "base_revision": KEV_BASE_REVISION,
                "temperature": 2.4,
                "head": {"weight": torch.zeros(2)},
            },
            self.package / "head.pt",
        )
        self.provenance = self.package / "provenance.json"
        self.provenance.write_text(
            json.dumps(
                {
                    "git_commit": revision,
                    "config": {"base": KEV_BASE_ID, "base_revision": KEV_BASE_REVISION},
                    "measured_checkpoint": {
                        "adapter_sha256": hashlib.sha256(
                            (self.package / "adapter_model.safetensors").read_bytes()
                        ).hexdigest()
                    },
                    "source_hashes": {
                        "kev/api.py": hashlib.sha256(code.read_bytes()).hexdigest()
                    },
                }
            ),
            encoding="utf-8",
        )
        self.count_path = self.root / "native-count.json"
        self.count_path.write_text(
            json.dumps(
                {
                    "schema_version": "decision2-native-loaded-parameter-count/1",
                    "model_id": self.model.model_id,
                    "revision": self.model.revision,
                    "count_method": "sum(p.numel() for p in inference.kev.load_native(...)[0].parameters())",
                    "parameter_count": 6,
                    "backbone_parameter_count": 4,
                    "head_parameter_count": 2,
                    "native_dtype": "float32",
                    "source_files_verified": 1,
                    "runtime": {
                        "base_revision": KEV_BASE_REVISION,
                        "source_revision": revision,
                        "calibration_temperature": 2.4,
                        "inference_path": "Checkpoint.load/DecisionModel.probs/api.to_answers",
                    },
                }
            ),
            encoding="utf-8",
        )
        self.count_path.chmod(0o600)
        self.receipt = self.root / "receipt.json"
        self.attestation = self.root / "attestation.json"
        for target, value in (
            ("inference.kev.KEV_SOURCE_REVISION", revision),
            ("scripts.baseline_attestation_v3.KEV_BASE_TEXT_PARAMETERS", 4),
            ("scripts.baseline_attestation_v3.KEV_BASE_ALL_PARAMETERS", 5),
            ("scripts.baseline_attestation_v3.KEV_HEAD_PARAMETERS", 2),
            ("scripts.baseline_attestation_v3.KEV_LOADED_PARAMETERS", 6),
        ):
            replacement = patch(target, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def build(self) -> dict:
        build_attestation(
            self.model,
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
            receipt_output=self.receipt,
            attestation_output=self.attestation,
            loaded_parameter_count=6,
            native_loaded_count_receipt_path=self.count_path,
        )
        return json.loads(self.attestation.read_text(encoding="utf-8"))

    def verify(self, item: dict) -> dict:
        return verify_attestation(
            item,
            self.model,
            source_root=SOURCE_ROOT,
            model_root=self.model_root,
            external_root=self.external_root,
        )

    def test_composite_binds_all_files_and_native_loader_count(self) -> None:
        item = self.build()
        receipt = self.verify(item)
        self.assertEqual(receipt["parameter_count"], 6)
        self.assertEqual(receipt["base_parameter_count"], 5)
        self.assertEqual(receipt["base_text_parameter_count"], 4)
        self.assertEqual(receipt["head_parameter_count"], 2)
        self.assertEqual(receipt["adapter_parameter_count"], 3)
        self.assertIn("kev/api.py", receipt["runtime_files"])
        self.assertIn("model.safetensors", receipt["base_files"])
        self.assertEqual(receipt["calibration_sha256"], receipt["files"]["head.pt"])
        self.assertEqual(item["size_b"], 6e-9)
        self.assertEqual(
            set(item),
            {
                "key",
                "model_id",
                "revision",
                "size_b",
                "native_model_sha256",
                "adapter_sha256",
                "calibration_sha256",
                "receipt_path",
                "receipt_sha256",
            },
        )

    def test_base_package_and_author_runtime_mutations_fail_closed(self) -> None:
        item = self.build()
        (self.base / "extra.txt").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(
            ValueError, "package, runtime, or calibration differs"
        ):
            self.verify(item)
        (self.base / "extra.txt").unlink()
        (self.runtime / "kev/api.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.verify(item)

    def test_loader_receipt_or_head_mutation_fails_closed(self) -> None:
        item = self.build()
        self.count_path.write_text(self.count_path.read_text() + " ")
        with self.assertRaisesRegex(
            ValueError, "package, runtime, or calibration differs"
        ):
            self.verify(item)
        self.count_path.write_text(self.count_path.read_text().rstrip())
        torch.save(
            {
                "base": KEV_BASE_ID,
                "base_revision": KEV_BASE_REVISION,
                "temperature": 2.4,
                "head": {"weight": torch.zeros(3)},
            },
            self.package / "head.pt",
        )
        with self.assertRaisesRegex(ValueError, "count differs from pinned components"):
            self.verify(item)

    def test_missing_or_public_native_loader_receipt_fails(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "needs the native loader count receipt"
        ):
            build_attestation(
                self.model,
                source_root=SOURCE_ROOT,
                model_root=self.model_root,
                external_root=self.external_root,
                receipt_output=self.receipt,
                attestation_output=self.attestation,
                loaded_parameter_count=6,
            )
        self.count_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "not private"):
            self.build()
        self.assertFalse(self.receipt.exists())
        self.assertFalse(self.attestation.exists())


if __name__ == "__main__":
    unittest.main()
