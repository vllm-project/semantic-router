"""The training filter cannot manufacture feedback from a new task."""

import hashlib
import unittest

from src.training.model_classifier.user_feedback_classifier.vela_current_turn import (
    refine_rows,
    reviewed_source_rows,
    weak_exclusion_reason,
)


def record(identifier, text, label="NEED_CLARIFICATION", group=None):
    return {
        "id": identifier,
        "group_id": group or identifier,
        "text": text,
        "label": label,
        "source": "wildfeedback_conservative_v3",
        "source_state": "FEEDBACK",
    }


class CurrentTurnTests(unittest.TestCase):
    def test_new_subject_confusion_does_not_imply_feedback(self):
        self.assertIsNotNone(
            weak_exclusion_reason(
                record("x", "I don't understand photosynthesis. Teach me.")
            )
        )
        self.assertIsNone(weak_exclusion_reason(record("x", "What did you mean?")))
        self.assertIsNone(weak_exclusion_reason(record("x", "I don't understand.")))

    def test_supplied_transformation_is_not_asserted_wrong_answer(self):
        self.assertIsNotNone(
            weak_exclusion_reason(
                record(
                    "x", "Check this sentence: Our result is incorrect.", "WRONG_ANSWER"
                )
            )
        )
        self.assertIsNone(
            weak_exclusion_reason(
                record("x", "Your answer used the wrong unit.", "WRONG_ANSWER")
            )
        )

    def test_satisfaction_source_state_is_not_enough(self):
        row = record("x", "Translate: Thank you for the clear answer.", "SAT")
        self.assertIsNotNone(weak_exclusion_reason(row))
        self.assertIsNone(
            weak_exclusion_reason(record("y", "That is helpful, thanks.", "SAT"))
        )

    def test_quoted_new_subject_is_excluded_before_deictic_matching(self):
        for text in ['Explain this "carbon cycle"', "Explain this `x = y + 1`"]:
            with self.subTest(text=text):
                self.assertIsNotNone(weak_exclusion_reason(record("x", text)))
        self.assertIsNone(
            weak_exclusion_reason(
                record("x", 'Your answer says "conserved". What did you mean?')
            )
        )

    def test_group_override_removes_derived_variants_without_spreading_gold(self):
        old = [record("old", "Long original text", group="family")]
        reviewed = [
            {
                **record(
                    "reviewed", "A complete new question", "NO_FEEDBACK", "family"
                ),
                "source": "reviewed",
            }
        ]
        rows, excluded = refine_rows(old, [], reviewed, {"family"})
        self.assertEqual(rows, reviewed)
        self.assertEqual(len(excluded), 1)
        self.assertEqual(old[0]["label"], "NEED_CLARIFICATION")

    def test_review_must_match_exact_source_and_training_partition(self):
        source = [
            {
                "UtterranceId": 0,
                "Role": "User",
                "Content": "Correct answer.",
                "State": "FEEDBACK",
            }
        ]
        item = {
            "id": "wildfeedback:0",
            "group_id": "wildfeedback:conversation:0",
            "text_sha256": "incorrect",
            "reviewed_label": "SAT",
            "reason": "Explicit approval",
        }
        with self.assertRaisesRegex(ValueError, "changed|outside"):
            reviewed_source_rows(source, {"items": [item], "reviewer": "test"})
        item["text_sha256"] = hashlib.sha256(b"Correct answer.").hexdigest()
        item["group_id"] = "wrong-family"
        with self.assertRaisesRegex(ValueError, "changed|outside"):
            reviewed_source_rows(source, {"items": [item], "reviewer": "test"})


if __name__ == "__main__":
    unittest.main()
