import unittest

from src.training.model_classifier.safety_classifier.vela_cultureguard import (
    capped,
    group_for,
    merge_annotations,
)


class CultureGuardDataTests(unittest.TestCase):
    def test_translation_cannot_cross_original_split(self):
        original = {"id": ("validation", "aegis:group")}
        self.assertEqual(group_for({"id": "id"}, "validation", original), "aegis:group")
        with self.assertRaises(ValueError):
            group_for({"id": "id"}, "train", original)

    def test_disagreeing_duplicate_category_is_unknown(self):
        first = {
            "label": "unsafe",
            "targets": [1, 1],
            "label_mask": [1, 1],
            "hazard_exclusion": None,
        }
        second = {**first, "targets": [1, 0]}
        merged = merge_annotations(first, second)
        self.assertEqual(merged["targets"], [1, 0])
        self.assertEqual(merged["label_mask"], [1, 0])

    def test_refusal_variant_restores_category_attribution(self):
        first = {
            "hazard_exclusion": "unsafe response",
            "targets": None,
            "label_mask": None,
        }
        second = {"hazard_exclusion": None, "targets": [1], "label_mask": [1]}
        self.assertEqual(merge_annotations(first, second), second)

    def test_sampling_caps_each_language_and_label(self):
        rows = [{"id": str(i), "language": "en", "label": "safe"} for i in range(20)]
        rows += [{"id": "zh", "language": "zh", "label": "unsafe"}]
        selected = capped(rows, 2)
        self.assertEqual(len(selected), 3)
        self.assertIn(rows[-1], selected)


if __name__ == "__main__":
    unittest.main()
