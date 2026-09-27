import json
import tempfile
import unittest
from pathlib import Path

from training.eikos.baseline_parity import check_baseline_parity
from training.model.data import file_sha256


class TestBaselineParity(unittest.TestCase):
    def test_ignores_gold_but_checks_native_identity_and_probs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "old.jsonl"
            current = root / "new.jsonl"
            row = {
                "id": "x",
                "source_input_sha256": "abc",
                "keys": ["A", "B"],
                "prediction_key": "A",
                "probabilities": {"A": 0.75, "B": 0.25},
                "gold_key": "B",
                "correct": False,
            }
            original.write_text(json.dumps(row) + "\n")
            changed = dict(row, gold_key="A", correct=True)
            current.write_text(json.dumps(changed) + "\n")
            receipt = check_baseline_parity(
                original, current, reference_sha256=file_sha256(original)
            )
            self.assertEqual(receipt["rows"], 1)
            changed["probabilities"] = {"A": 0.74, "B": 0.26}
            current.write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "numeric"):
                check_baseline_parity(
                    original, current, reference_sha256=file_sha256(original)
                )

    def test_wrong_reference_digest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.jsonl"
            path.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                check_baseline_parity(path, path, reference_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
