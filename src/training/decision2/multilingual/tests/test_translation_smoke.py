import json
from pathlib import Path
import tempfile
import unittest

from multilingual.translation_smoke import (
    _load_public_references,
    _numbers,
    _translated_row,
)
from training.model.data import INPUT_FIELDS, digest


class TranslationSmokeTest(unittest.TestCase):
    def test_numeric_invariant_normalizes_arabic_digits(self):
        self.assertEqual(_numbers("Pay 12 then 3.5"), _numbers("ادفع ١٢ ثم ٣.٥"))
        self.assertNotEqual(_numbers("Pay 12"), _numbers("ادفع ١٣"))

    def test_translated_row_preserves_group_label_and_option_keys(self):
        row = {
            "id": "source-1",
            "group_id": "source-group",
            "source": "fixture",
            "state": "The event happened yesterday.",
            "instructions": "Choose a class.",
            "task_type": "choice",
            "options": [
                {"key": "a", "description": "The event happened."},
                {"key": "b", "description": "The event did not happen."},
            ],
            "label": 0,
            "family": "fixture",
            "language": "en",
            "split": "train",
            "evaluation_role": "train",
            "render_template": "fixture",
            "audit_metadata": {},
        }
        row["input_sha256"] = digest({key: row[key] for key in INPUT_FIELDS})
        translations = {
            "The event happened yesterday.": "El evento ocurrió ayer.",
            "Choose a class.": "Elige una clase.",
            "The event happened.": "El evento ocurrió.",
            "The event did not happen.": "El evento no ocurrió.",
        }
        result = _translated_row(row, "es", translations)
        self.assertEqual(result["group_id"], row["group_id"])
        self.assertEqual(result["label"], 0)
        self.assertEqual([item["key"] for item in result["options"]], ["a", "b"])
        self.assertEqual(
            result["audit_metadata"]["parent_input_sha256"], row["input_sha256"]
        )
        self.assertEqual(
            result["input_sha256"], digest({key: result[key] for key in INPUT_FIELDS})
        )

    def test_final_named_path_is_rejected_without_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sealed-final-prompts.jsonl"
            path.write_text(json.dumps({"id": "secret", "state": "hidden"}))
            with self.assertRaisesRegex(ValueError, "No sealed FINAL"):
                _load_public_references([path])


if __name__ == "__main__":
    unittest.main()
