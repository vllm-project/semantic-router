"""Unit tests for dependency-free safety-classifier metrics."""

from __future__ import annotations

import math
import unittest

from src.training.model_classifier.safety_classifier.metrics import (
    binary_classification_metrics,
    classification_metrics,
    compute_level1_metrics,
    compute_level2_metrics,
    freeze_binary_threshold,
    multiclass_classification_metrics,
    predictions_from_scores,
)


class ArrayLike:
    """Small NumPy stand-in that verifies the public ``tolist`` protocol."""

    def __init__(self, values: object) -> None:
        self.values = values

    def tolist(self) -> object:
        return self.values


class BinaryMetricsTest(unittest.TestCase):
    def test_complete_binary_report(self) -> None:
        report = binary_classification_metrics(
            ArrayLike([0, 0, 1, 1]),
            ArrayLike([0, 1, 0, 1]),
            unsafe_scores=ArrayLike([0.1, 0.8, 0.4, 0.9]),
        )

        self.assertEqual(report["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(report["support"], {"total": 4, "safe": 2, "unsafe": 2})
        for name in (
            "accuracy",
            "precision_unsafe",
            "recall_unsafe",
            "f1_unsafe",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "false_positive_rate",
            "false_negative_rate",
        ):
            self.assertEqual(report[name], 0.5, name)
        self.assertAlmostEqual(report["auroc"], 0.75)
        self.assertAlmostEqual(report["auprc"], 5.0 / 6.0)

    def test_auc_ties_are_grouped(self) -> None:
        report = binary_classification_metrics(
            [0, 1, 0, 1],
            unsafe_scores=[0.5, 0.5, 0.5, 0.5],
        )
        self.assertAlmostEqual(report["auroc"], 0.5)
        self.assertAlmostEqual(report["auprc"], 0.5)

    def test_undefined_auc_is_explicit(self) -> None:
        report = binary_classification_metrics([0, 0], unsafe_scores=[0.1, 0.2])
        self.assertIsNone(report["auroc"])
        self.assertIsNone(report["auprc"])

    def test_threshold_rule_is_greater_than_or_equal(self) -> None:
        self.assertEqual(predictions_from_scores([0.49, 0.5, 0.51], 0.5), [0, 1, 1])

    def test_threshold_freeze_maximizes_specificity_at_recall_floor(self) -> None:
        labels = [0, 0] + [1] * 20
        scores = [0.2, 0.3, 0.1] + [0.8] * 19
        frozen = freeze_binary_threshold(labels, scores, minimum_recall=0.95)

        self.assertEqual(frozen["threshold"], 0.8)
        self.assertEqual(frozen["recall_unsafe"], 0.95)
        self.assertEqual(frozen["specificity"], 1.0)
        self.assertIsNone(frozen["fallback"])
        self.assertIn("maximum specificity", frozen["tie_break"])

    def test_threshold_tie_prefers_recall_before_threshold(self) -> None:
        frozen = freeze_binary_threshold(
            [0, 1, 1],
            [0.1, 0.8, 0.9],
            minimum_recall=0.5,
        )
        self.assertEqual(frozen["threshold"], 0.8)
        self.assertEqual(frozen["recall_unsafe"], 1.0)

    def test_threshold_fallback_without_unsafe_examples(self) -> None:
        frozen = freeze_binary_threshold([0, 0], [0.1, 0.2])
        self.assertGreater(frozen["threshold"], 0.2)
        self.assertEqual(frozen["specificity"], 1.0)
        self.assertIsNotNone(frozen["fallback"])

    def test_level1_trainer_helper_accepts_logits_and_probabilities(self) -> None:
        logits = compute_level1_metrics([0, 1], [[3.0, 1.0], [1.0, 3.0]])
        probabilities = compute_level1_metrics([0, 1], [[0.9, 0.1], [0.1, 0.9]])
        self.assertEqual(logits["accuracy"], 1.0)
        self.assertEqual(probabilities["accuracy"], 1.0)
        self.assertTrue(all(isinstance(value, float) for value in logits.values()))
        self.assertNotIn("confusion_matrix", logits)

    def test_level1_trainer_helper_uses_frozen_comparison_on_tie(self) -> None:
        report = compute_level1_metrics([1], [[0.0, 0.0]])
        self.assertEqual(report["accuracy"], 1.0)

    def test_binary_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            binary_classification_metrics([0], [0, 1])
        with self.assertRaisesRegex(ValueError, "only 0"):
            binary_classification_metrics([2], [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            binary_classification_metrics([0], unsafe_scores=[math.nan])


class MulticlassMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [0, 1, 2, 2]
        self.scores = [
            [0.9, 0.05, 0.05],
            [0.6, 0.3, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
        ]

    def test_complete_multiclass_report_and_top2(self) -> None:
        report = multiclass_classification_metrics(
            self.labels,
            class_scores=self.scores,
            class_names=["a", "b", "c"],
        )

        self.assertEqual(report["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["balanced_accuracy"], 0.5)
        self.assertEqual(report["top2_accuracy"], 0.75)
        self.assertAlmostEqual(report["precision_macro"], 0.5)
        self.assertAlmostEqual(report["recall_macro"], 0.5)
        self.assertAlmostEqual(report["f1_macro"], 4.0 / 9.0)
        self.assertAlmostEqual(report["f1_weighted"], 0.5)
        self.assertEqual(report["per_class"]["c"]["support"], 2)

    def test_strict_single_target_subset(self) -> None:
        report = multiclass_classification_metrics(
            self.labels,
            class_scores=self.scores,
            class_names=["a", "b", "c"],
            strict_single_target_mask=ArrayLike([True, True, False, False]),
        )
        strict = report["strict_single_target"]
        self.assertEqual(strict["support"]["total"], 2)
        self.assertEqual(strict["accuracy"], 0.5)
        self.assertEqual(strict["top2_accuracy"], 1.0)
        self.assertEqual(strict["coverage"], 0.5)

    def test_balanced_accuracy_uses_observed_reference_classes(self) -> None:
        report = multiclass_classification_metrics(
            [0, 0],
            [0, 0],
            class_names=["present", "absent"],
        )
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["recall_macro"], 0.5)

    def test_level2_trainer_helper_is_flat(self) -> None:
        report = compute_level2_metrics(
            self.labels,
            self.scores,
            class_names=["a", "b", "c"],
        )
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["top2_accuracy"], 0.75)
        self.assertTrue(all(isinstance(value, float) for value in report.values()))
        self.assertNotIn("per_class", report)

    def test_generic_classification_metrics_retains_full_report(self) -> None:
        report = classification_metrics(
            "level2",
            self.labels,
            self.scores,
            class_names=["a", "b", "c"],
        )
        self.assertIn("confusion_matrix", report)
        self.assertIn("per_class", report)

    def test_top2_ties_prefer_lower_class_id(self) -> None:
        report = multiclass_classification_metrics(
            [2],
            class_scores=[[0.1, 0.8, 0.1]],
            class_names=["a", "b", "c"],
        )
        self.assertEqual(report["top2_accuracy"], 0.0)

    def test_multiclass_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            multiclass_classification_metrics(
                [0, 1],
                class_scores=[[1.0, 0.0]],
                class_names=["a", "b"],
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            multiclass_classification_metrics([2], [0], class_names=["a", "b"])


if __name__ == "__main__":
    unittest.main()
