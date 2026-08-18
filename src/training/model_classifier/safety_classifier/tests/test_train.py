"""Dependency-light tests for safety training orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.training.model_classifier.safety_classifier import train
from src.training.model_classifier.safety_classifier.config import (
    load_contract,
    task_contract,
)


def release_args(**overrides: object) -> SimpleNamespace:
    values = {
        "max_steps": -1,
        "num_train_epochs": None,
        "learning_rate": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ReleaseEligibilityTest(unittest.TestCase):
    def test_canonical_eight_device_bf16_run_is_eligible(self) -> None:
        self.assertTrue(train._release_eligible(release_args(), 8, True))

    def test_every_training_override_makes_run_ineligible(self) -> None:
        overrides = {
            "max_steps": 1,
            "num_train_epochs": 1.0,
            "learning_rate": 1e-4,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
        }
        for name, value in overrides.items():
            with self.subTest(name=name):
                self.assertFalse(
                    train._release_eligible(release_args(**{name: value}), 8, True)
                )

    def test_world_size_and_precision_are_release_requirements(self) -> None:
        self.assertFalse(train._release_eligible(release_args(), 1, True))
        self.assertFalse(train._release_eligible(release_args(), 8, False))

    def test_explicit_source_commit_is_validated(self) -> None:
        commit = "a" * 40
        with mock.patch.dict(os.environ, {train.SOURCE_COMMIT_ENV: commit}, clear=True):
            self.assertEqual(train._source_commit(), commit)
        with (
            mock.patch.dict(
                os.environ,
                {train.SOURCE_COMMIT_ENV: "not-a-commit"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, train.SOURCE_COMMIT_ENV),
        ):
            train._source_commit()


class TrainingReceiptTest(unittest.TestCase):
    def test_manifest_schema_and_batch_accounting_remain_stable(self) -> None:
        contract = load_contract()
        with tempfile.TemporaryDirectory() as directory:
            source_manifest = Path(directory) / "data_manifest.json"
            source_manifest.write_bytes(b'{"stable":true}\n')
            runtime = train._TrainingRuntime(
                stack={},
                torch=None,
                world_size=8,
                per_device_batch=8,
                gradient_accumulation=1,
                use_bf16=True,
                source_commit="a" * 40,
                output_root=Path(directory),
            )
            model = SimpleNamespace(config=SimpleNamespace(reference_compile=False))
            environment = {
                "HSA_NO_SCRATCH_RECLAIM": "1",
                "NCCL_MAX_NCHANNELS": "8",
                "NCCL_P2P_DISABLE": "1",
            }
            with (
                mock.patch.object(
                    train, "_package_versions", return_value={"torch": "pinned"}
                ),
                mock.patch.dict(os.environ, environment, clear=True),
            ):
                manifest = train._build_training_manifest(
                    contract,
                    "level1",
                    source_manifest,
                    model,
                    runtime,
                    True,
                )

        self.assertEqual(
            set(manifest),
            {
                "schema_version",
                "task",
                "contract_sha256",
                "data_manifest_sha256",
                "source_commit",
                "base_model",
                "taxonomy_version",
                "world_size",
                "global_train_batch_size",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "precision",
                "reference_compile",
                "distributed_runtime",
                "release_eligible",
                "python",
                "packages",
            },
        )
        self.assertEqual(manifest["global_train_batch_size"], 64)
        self.assertEqual(manifest["precision"], "bf16")
        self.assertEqual(manifest["distributed_runtime"], environment)
        self.assertEqual(
            manifest["data_manifest_sha256"],
            hashlib.sha256(b'{"stable":true}\n').hexdigest(),
        )

    def test_receipt_writer_preserves_artifact_layout(self) -> None:
        contract = load_contract()
        task = task_contract(contract, "level1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_dir = root / "adapter"
            adapter_dir.mkdir()
            runtime = train._TrainingRuntime(
                stack={},
                torch=None,
                world_size=8,
                per_device_batch=8,
                gradient_accumulation=1,
                use_bf16=True,
                source_commit="a" * 40,
                output_root=root,
            )
            args = release_args()
            args.task = "level1"
            with mock.patch.object(
                train, "_build_training_manifest", return_value={"schema_version": 1}
            ):
                train._write_run_receipts(
                    contract,
                    task,
                    args,
                    runtime,
                    root / "data",
                    {"task": "level1"},
                    SimpleNamespace(config=SimpleNamespace(reference_compile=False)),
                    {"test": {"accuracy": 1.0}},
                    adapter_dir,
                )

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {
                    "adapter",
                    "metrics.json",
                    "data_manifest.json",
                    "training_manifest.json",
                    "reconstruction_contract.json",
                },
            )
            mapping = json.loads(
                (adapter_dir / "label_mapping.json").read_text(encoding="utf-8")
            )
            self.assertEqual(mapping["label2id"], task["label2id"])
            self.assertEqual(mapping["taxonomy_version"], "legacy-9-v1")


if __name__ == "__main__":
    unittest.main()
