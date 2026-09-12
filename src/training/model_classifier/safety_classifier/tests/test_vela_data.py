import unittest

from src.training.model_classifier.safety_classifier.vela_data import (
    LABELS,
    hazard_target,
    partial_target,
)


class VelaHazardDataTests(unittest.TestCase):
    def test_dialogue_unsafe_category_is_not_prompt_supervision(self):
        row = {
            "prompt_label": "unsafe",
            "response_label": "unsafe",
            "response": "text",
            "violated_categories": "Violence",
        }
        self.assertIsNotNone(hazard_target(row)[2])

    def test_safe_topic_is_not_a_hazard(self):
        row = {
            "prompt_label": "safe",
            "response": None,
            "violated_categories": "Violence, PII/Privacy",
        }
        target, mask, reason = hazard_target(row)
        self.assertEqual(target, [0] * len(LABELS))
        self.assertEqual(mask, [1] * len(LABELS))
        self.assertIsNone(reason)

    def test_multiple_categories_are_retained(self):
        row = {
            "prompt_label": "unsafe",
            "response": None,
            "violated_categories": "Violence, PII/Privacy",
        }
        target, _, _ = hazard_target(row)
        self.assertEqual(sum(target), 2)

    def test_missing_annotation_remains_masked(self):
        target, mask = partial_target(["violence"], ["violence", "privacy"])
        self.assertEqual(sum(target), 1)
        self.assertEqual(sum(mask), 2)
        self.assertEqual(mask[LABELS.index("self_harm")], 0)

    def test_unknown_source_taxonomy_fails(self):
        with self.assertRaises(ValueError):
            hazard_target(
                {"prompt_label": "safe", "violated_categories": "New category"}
            )


if __name__ == "__main__":
    unittest.main()
