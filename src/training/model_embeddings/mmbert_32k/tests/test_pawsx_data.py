"""Regression for the translated PAWS-X NS placeholders seen in pinned data."""

import unittest

from src.training.model_embeddings.mmbert_32k.pawsx_data import clean_pawsx_pairs


class PawsXDataTest(unittest.TestCase):
    def test_filters_both_sides_casefold_and_whitespace_with_counts(self):
        good = {
            "sentence1": " NS is a railway operator. ",
            "sentence2": "自然语言",
            "label": 0,
        }
        rows = [
            good,
            {"sentence1": " NS\t", "sentence2": "translated"},
            {"sentence1": "translated", "sentence2": "nS"},
            {"sentence1": "", "sentence2": "NS"},
            {"sentence1": "translated", "sentence2": "\n\t"},
        ]
        selected, counts = clean_pawsx_pairs(iter(rows))
        self.assertEqual(selected, [good])
        self.assertIs(selected[0], good)
        self.assertEqual(
            counts,
            {
                "input_pairs": 5,
                "kept_pairs": 1,
                "empty_sentence": 2,
                "untranslated_ns": 2,
            },
        )

    def test_missing_or_nontext_sentence_rejected_before_sampling(self):
        for row in ({"sentence1": "ok"}, {"sentence1": None, "sentence2": "ok"}):
            with self.subTest(row=row), self.assertRaisesRegex(
                ValueError, "must be a string"
            ):
                clean_pawsx_pairs([row])


if __name__ == "__main__":
    unittest.main()
