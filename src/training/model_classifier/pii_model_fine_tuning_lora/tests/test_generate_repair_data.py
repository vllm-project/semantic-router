import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_long_data import concatenate, place_needle
from generate_repair_data import TYPES, entity_value, generated_split
from span_data import assert_split_isolation, validated_spans


class GeneratorTests(unittest.TestCase):
    def test_generated_splits_have_unique_texts_and_no_cross_split_identity(self):
        splits = {
            name: generated_split(name, 24, 91) for name in ("train", "dev", "test")
        }
        assert_split_isolation(splits)
        labels = {
            "O": 0,
            **{
                prefix + entity_type: 1
                for entity_type in TYPES
                for prefix in ("B-", "I-")
            },
        }
        for records in splits.values():
            self.assertEqual(
                len(records), len({record["full_text"] for record in records})
            )
            for record in records:
                validated_spans(record, labels)

    def test_generated_card_and_iban_checksums_are_valid(self):
        for split in ("train", "dev", "test"):
            for index in range(8):
                card = entity_value("CREDIT_CARD", split, index)
                checksum = 0
                for position, digit in enumerate(reversed(card)):
                    value = int(digit) * (2 if position % 2 else 1)
                    checksum += value - 9 if value > 9 else value  # noqa: PLR2004
                self.assertEqual(checksum % 10, 0)
                iban = entity_value("IBAN_CODE", split, index)
                rearranged = iban[4:] + iban[:4]
                digits = "".join(
                    str(ord(char) - 55) if char.isalpha() else char
                    for char in rearranged
                )
                self.assertEqual(int(digits) % 97, 1)

    def test_needle_positions_preserve_unicode_span(self):
        needle = {
            "full_text": "记录 a@b.io",
            "spans": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "entity_value": "a@b.io",
                    "start_position": 3,
                    "end_position": 9,
                }
            ],
        }
        for position in ("head", "middle", "tail"):
            result = place_needle(needle, "普通句子", position, 100, len)
            self.assertLessEqual(len(result["full_text"]), 100)
            self.assertGreater(len(result["full_text"]), 90)
            span = result["spans"][0]
            self.assertEqual(
                result["full_text"][span["start_position"] : span["end_position"]],
                "a@b.io",
            )
        with self.assertRaises(ValueError):
            place_needle(needle, "text", "tail", 2, len)

    def test_concatenation_keeps_identical_entities_at_distinct_positions(self):
        record = {
            "full_text": "a@b.io",
            "spans": [
                {
                    "entity_type": "EMAIL_ADDRESS",
                    "entity_value": "a@b.io",
                    "start_position": 0,
                    "end_position": 6,
                }
            ],
        }
        result = concatenate([record, record])
        self.assertEqual([span["start_position"] for span in result["spans"]], [0, 7])
        self.assertEqual(result["full_text"], "a@b.io\na@b.io")


if __name__ == "__main__":
    unittest.main()
