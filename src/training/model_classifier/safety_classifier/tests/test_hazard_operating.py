"""Operating-point selection must respect the deployed any-label FP rule."""

import importlib.util
import unittest

from src.training.model_classifier.safety_classifier.vela_hazard_operating import (
    select_operating_point,
)


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
    "NumPy and scikit-learn are optional for dependency-light data checks",
)
class HazardOperatingTests(unittest.TestCase):
    def test_budget_selects_category_quality_instead_of_any_risk(self):
        rows = [
            {"label": "safe", "targets": [0, 0], "label_mask": [1, 1]},
            {"label": "unsafe", "targets": [1, 0], "label_mask": [1, 0]},
            {"label": "unsafe", "targets": [0, 1], "label_mask": [1, 1]},
        ]
        # The wrong category on the first unsafe row cannot count as a hit.
        result = select_operating_point(
            rows,
            [[0.1, 0.1], [0.2, 0.9], [0.1, 0.8]],
            ["one", "two"],
            false_positive_budget=0,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["threshold"], 0.2)
        self.assertEqual(result["known_positive_category_hit_recall"], 1.0)
        self.assertEqual(result["safe_false_positives"], 0)
        self.assertEqual(result["selection_score"], 1.0)
        self.assertEqual(
            result["metrics_all_labels"]["per_label"]["two"]["negative_support"], 1
        )

    def test_rare_head_still_counts_toward_any_label_false_alarms(self):
        rows = [
            {"label": "safe", "targets": [0, 0], "label_mask": [1, 1]},
            {"label": "safe", "targets": [0, 0], "label_mask": [1, 1]},
            {"label": "unsafe", "targets": [1, 0], "label_mask": [1, 1]},
            {"label": "unsafe", "targets": [1, 1], "label_mask": [1, 1]},
        ]
        result = select_operating_point(
            rows,
            [[0.1, 0.95], [0.1, 0.1], [0.9, 0.1], [0.8, 0.7]],
            ["supported", "rare"],
            false_positive_budget=0,
            minimum_support=2,
        )
        self.assertEqual(result["selection_labels"], ["supported"])
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["selection_score"], 0.0)
        self.assertEqual(len(result["metrics_all_labels"]["per_label"]), 2)

    def test_independent_scores_are_not_added(self):
        rows = [
            {"label": "safe", "targets": [0, 0], "label_mask": [1, 1]},
            {"label": "unsafe", "targets": [1, 1], "label_mask": [1, 1]},
        ]
        result = select_operating_point(
            rows, [[0.4, 0.4], [0.5, 0.5]], ["one", "two"], false_positive_budget=0
        )
        self.assertEqual(result["threshold"], 0.5)
        self.assertEqual(result["selection_score"], 1.0)

    def test_no_feasible_threshold_does_not_claim_success(self):
        rows = [
            {"label": "safe", "targets": [0], "label_mask": [1]},
            {"label": "unsafe", "targets": [1], "label_mask": [1]},
        ]
        result = select_operating_point(
            rows, [[1.0], [1.0]], ["one"], false_positive_budget=0
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["selection_score"], -1.0)
        self.assertNotIn("threshold", result)

    def test_invalid_budget_and_unobserved_safe_gold_are_rejected(self):
        rows = [
            {"label": "safe", "targets": [0], "label_mask": [1]},
            {"label": "unsafe", "targets": [1], "label_mask": [1]},
        ]
        for value in [-0.1, 1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                select_operating_point(
                    rows, [[0.1], [0.8]], ["one"], false_positive_budget=value
                )
        rows[0]["label_mask"] = [0]
        with self.assertRaisesRegex(ValueError, "fully observed safe"):
            select_operating_point(rows, [[0.1], [0.8]], ["one"])


if __name__ == "__main__":
    unittest.main()
