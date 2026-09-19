"""Reviewed requests preserve development and reject conflicts or leakage."""

import unittest

from src.training.model_classifier.user_feedback_classifier.vela_natural_negatives import (
    append_reviewed_negatives,
)


def row(identifier, text, label="NO_FEEDBACK", group=None):
    return {
        "id": identifier,
        "text": text,
        "label": label,
        "group_id": group or identifier,
        "source_split": "train",
    }


class NaturalNegativeTests(unittest.TestCase):
    def test_append_preserves_inputs_and_skips_same_label_duplicates(self):
        previous = [row("one", "Translate this supplied paragraph.")]
        addition = row("new", "Explain photosynthesis.")
        train, accepted, duplicates = append_reviewed_negatives(
            previous,
            [row("dev", "Find the area of this triangle.")],
            {"train": [row("copy", previous[0]["text"].upper()), addition]},
        )
        self.assertEqual(len(train), 2)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(duplicates[0]["existing_id"], "one")
        self.assertNotIn("source", addition)
        self.assertEqual(len(previous), 1)

    def test_conflicting_previous_label_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            append_reviewed_negatives(
                [row("old", "Find the supplied error.", "WRONG_ANSWER")],
                [],
                {"train": [row("new", "Find the supplied error.")]},
            )

    def test_rejects_validation_review_or_nonnegative_label(self):
        with self.assertRaisesRegex(ValueError, "training only"):
            append_reviewed_negatives(
                [], [], {"train": [], "validation": [row("d", "d")]}
            )
        with self.assertRaisesRegex(ValueError, "NO_FEEDBACK"):
            append_reviewed_negatives([], [], {"train": [row("x", "x", "SAT")]})

    def test_translation_family_cannot_cross_into_development(self):
        with self.assertRaisesRegex(ValueError, "group"):
            append_reviewed_negatives(
                [],
                [row("dev", "Different wording", group="one-family")],
                {"train": [row("new", "Independent wording", group="one-family")]},
            )


if __name__ == "__main__":
    unittest.main()
