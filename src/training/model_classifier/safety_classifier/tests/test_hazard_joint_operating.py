"""Independent thresholds must satisfy every declared safe-union constraint."""

import importlib.util
import unittest

from src.training.model_classifier.safety_classifier.vela_hazard_joint import (
    safe_group_indices,
    select_joint_operating_point,
)


def row(identity, targets, *, mask=None):
    observed = mask or [1] * len(targets)
    return {
        "id": identity,
        "label": "unsafe" if any(targets) else "safe" if all(observed) else "unknown",
        "targets": targets,
        "label_mask": observed,
    }


@unittest.skipUnless(
    importlib.util.find_spec("numpy") and importlib.util.find_spec("sklearn"),
    "NumPy and scikit-learn are optional for dependency-light data checks",
)
class JointOperatingTests(unittest.TestCase):
    def test_distinct_scales_need_independent_thresholds(self):
        rows = [row("s", [0, 0]), row("a", [1, 0]), row("b", [0, 1])]
        result = select_joint_operating_point(
            rows,
            [[0.8, 0.1], [0.9, 0.1], [0.8, 0.2]],
            ["a", "b"],
            false_positive_budget=0,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["selection_score"], 1)
        self.assertGreater(result["thresholds"][0], result["thresholds"][1])
        self.assertEqual(result["metrics"]["safe_any_hazard_rate"], 0)

    def test_small_group_budget_cannot_hide_in_pooled_budget(self):
        rows = [row(f"s{i}", [0]) for i in range(20)] + [row("u", [1])]
        probabilities = [[0.9]] + [[0.1]] * 19 + [[0.8]]
        pooled = select_joint_operating_point(rows, probabilities, ["risk"])
        constrained = select_joint_operating_point(
            rows, probabilities, ["risk"], safe_groups={"boundary": ["s0"]}
        )
        self.assertGreater(pooled["selection_score"], 0)
        self.assertEqual(constrained["selection_score"], 0)
        self.assertEqual(constrained["limits"][1]["max_fp"], 0)
        self.assertEqual(constrained["metrics"]["safe_any_hazard_rate"], 0)

    def test_union_budget_counts_different_head_false_alarms(self):
        rows = [row(f"s{i}", [0, 0]) for i in range(20)]
        rows += [row("a", [1, 0]), row("b", [0, 1])]
        scores = [[0.8, 0.1], [0.1, 0.8]] + [[0.1, 0.1]] * 18 + [[0.8, 0.1], [0.1, 0.8]]
        result = select_joint_operating_point(rows, scores, ["a", "b"])
        self.assertLessEqual(result["metrics"]["safe_any_hazard_rate"], 0.05)
        self.assertEqual(result["limits"][0]["max_fp"], 1)
        self.assertAlmostEqual(result["thresholds"][0], 0.8, places=6)
        self.assertEqual(result["thresholds"][1], 1)

    def test_unsupported_label_remains_in_union_and_taxonomy_score(self):
        rows = [row("s", [0, 0]), row("u", [1, 0], mask=[1, 0])]
        result = select_joint_operating_point(
            rows,
            [[0.1, 0.95], [0.8, 0.99]],
            ["supported", "unknown"],
            false_positive_budget=0,
        )
        self.assertEqual(result["unsupported_labels"], ["unknown"])
        self.assertEqual(result["thresholds"][1], 1)
        self.assertEqual(result["selection_score"], 0.5)
        self.assertEqual(result["metrics"]["per_label"]["unknown"]["unknown"], 1)

    def test_no_feasible_threshold_is_explicit(self):
        result = select_joint_operating_point(
            [row("s", [0]), row("u", [1])],
            [[1], [1]],
            ["risk"],
            false_positive_budget=0,
        )
        self.assertFalse(result["feasible"])
        self.assertEqual(result["selection_score"], -1)
        self.assertNotIn("thresholds", result)

    def test_safe_groups_reject_missing_partial_and_duplicate_ids(self):
        rows = [row("s", [0]), row("u", [1]), row("partial", [0, 0], mask=[1, 0])]
        for groups in (
            {"x": []},
            {"x": ["missing"]},
            {"x": ["s", "s"]},
            {"x": ["u"]},
            {"x": ["partial"]},
            {"all_safe": ["s"]},
            [],
        ):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                safe_group_indices(rows, groups)
        with self.assertRaises(ValueError):
            safe_group_indices([row("s", [0]), row("s", [0])])

    def test_nonfinite_out_of_range_and_changed_support_contract_rejected(self):
        rows = [row("s", [0]), row("u", [1])]
        for value in (float("nan"), float("inf"), -1e-9, 1 + 1e-9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_joint_operating_point(rows, [[value], [0.9]], ["risk"])
        with self.assertRaisesRegex(ValueError, "minimum support 1"):
            select_joint_operating_point(
                rows, [[0.1], [0.9]], ["risk"], minimum_support=2
            )


if __name__ == "__main__":
    unittest.main()
