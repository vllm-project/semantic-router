"""Regression tests for the dependency-light refactor seams."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier import data, data_contract
from src.training.model_classifier.safety_classifier.evaluate import (
    _prediction_record,
)
from src.training.model_classifier.safety_classifier.export import _export_runtime


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    float32 = "float32"

    def __init__(self, available: bool) -> None:
        self.cuda = _FakeCuda(available)

    @staticmethod
    def device(name: str) -> str:
        return name


class RefactorContractTest(unittest.TestCase):
    def test_data_errors_remain_available_from_original_module(self) -> None:
        self.assertIs(data.DataPreparationError, data_contract.DataPreparationError)
        self.assertIs(data.SchemaError, data_contract.SchemaError)
        self.assertIs(
            data.InsufficientClassSamplesError,
            data_contract.InsufficientClassSamplesError,
        )

    def test_jsonl_loader_raises_original_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.jsonl"
            source.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(data.SchemaError):
                data.load_jsonl(source)

    def test_prediction_record_schema_and_text_redaction(self) -> None:
        row = {
            "text": "fixture",
            "sample_id": "sample-1",
            "source": "unit",
            "is_multitarget": False,
        }
        record = _prediction_record(
            row,
            0,
            reference=0,
            prediction=1,
            scores=[0.25, 0.75],
            names=["safe", "unsafe"],
            include_text=False,
        )

        self.assertEqual(record["sample_id"], "sample-1")
        self.assertEqual(record["reference_label"], "safe")
        self.assertEqual(record["prediction_label"], "unsafe")
        self.assertTrue(record["strict_single_target"])
        self.assertEqual(
            record["text_sha256"],
            hashlib.sha256(b"fixture").hexdigest(),
        )
        self.assertNotIn("text", record)
        json.dumps(record, allow_nan=False)

    def test_export_runtime_uses_fp32_on_cpu_and_accelerator(self) -> None:
        self.assertEqual(
            _export_runtime(_FakeTorch(available=True), use_cpu=False),
            ("float32", "cuda:0"),
        )
        self.assertEqual(
            _export_runtime(_FakeTorch(available=True), use_cpu=True),
            ("float32", "cpu"),
        )
        self.assertEqual(
            _export_runtime(_FakeTorch(available=False), use_cpu=False),
            ("float32", "cpu"),
        )


if __name__ == "__main__":
    unittest.main()
