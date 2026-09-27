"""Fail-closed checks for the direct full-checkpoint product path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from publication.full_product import _row
from publication.full_runtime_api import _inventory, _tensor_count


class FullPublicationTest(unittest.TestCase):
    def test_hub_metadata_requires_pinned_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitattributes").write_text("unexpected metadata\n")
            with self.assertRaises(ValueError):
                _inventory(root)

    def test_safetensors_parameter_count_uses_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.safetensors"
            header = json.dumps(
                {"weight": {"shape": [2, 3], "dtype": "F32", "data_offsets": [0, 24]}}
            ).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(24))
            self.assertEqual(_tensor_count(path), 6)
            path.write_bytes((4).to_bytes(8, "little") + b"xxxx")
            with self.assertRaises(ValueError):
                _tensor_count(path)

    def test_product_row_uses_frozen_typed_and_task_metrics(self) -> None:
        typed = {
            "schema_version": "typed-decision-report/2",
            "items": 1600,
            "model": {"id": "research/qwen3-06b-official-full466"},
            "macro_family_accuracy": 0.25,
            "by_type": {
                "choice": {"accuracy_all": 0.2},
                "noul": {"accuracy_all": 0.3},
                "score": {"accuracy_all": 0.25},
            },
            "overall": {"valid_n": 2000},
        }
        css = {
            "score_schema_version": "css-transfer-score/2",
            "tasks": {
                f"task_{index}": {"macro_f1_all": 0.4, "invalid_or_missing_n": 1}
                for index in range(15)
            },
        }
        public = {
            "score_version": "jevarena-jevbench-public-score/1",
            "items": 231,
            "correct": 143,
            "valid": 231,
            "tiers": {},
        }
        row = _row("new", typed, css, public)
        self.assertAlmostEqual(row["score"], 100 * (0.25 * 0.4) ** 0.5)
        self.assertEqual(row["css_invalid"], 15)
        typed["model"]["id"] = "another/model"
        with self.assertRaises(ValueError):
            _row("new", typed, css, public)


if __name__ == "__main__":
    unittest.main()
