"""Calibration changes real logits and cannot pass by omitting a required slice."""

# Optional numerical dependencies are loaded only in their guarded tests.
# ruff: noqa: PLC0415

import copy
import importlib.util
import unittest
from types import SimpleNamespace

from src.training.model_classifier.user_feedback_classifier.calibrate_vela_bias import (
    adjusted_probabilities,
    apply_head_bias,
    gate_results,
    probability_metrics,
    screen_grid,
)
from src.training.model_classifier.user_feedback_classifier.vela_contract import (
    VELA_ID2LABEL,
    VELA_LABEL2ID,
)


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "NumPy is optional for data tests"
)
class BiasScreeningTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"label": label, "source": "reviewed", "length_bucket": "256"}
            for label in VELA_LABEL2ID
        ]
        self.probabilities = [
            [0.99 if index == target else 0.0025 for index in range(5)]
            for target in range(5)
        ]
        self.probabilities[4] = [0.51, 0, 0, 0, 0.49]

    def test_smallest_feasible_bias_and_complete_grid(self):
        selected, trials = screen_grid(self.rows, self.probabilities, ["256"])
        self.assertEqual(selected, 0.125)
        self.assertEqual(len(trials), 9)
        self.assertFalse(trials[0]["gates"]["no_feedback_recall"])
        self.assertTrue(all(trials[1]["gates"].values()))

    def test_unrecoverable_probability_does_not_expand_grid(self):
        self.probabilities[4] = [1, 0, 0, 0, 0]
        selected, trials = screen_grid(self.rows, self.probabilities, ["256"])
        self.assertIsNone(selected)
        self.assertEqual(trials[-1]["bias"], 1)

    def test_required_length_and_class_support_cannot_pass_vacuously(self):
        values = adjusted_probabilities(self.probabilities, 0.125)
        metrics = probability_metrics(self.rows, values)
        self.assertFalse(gate_results(metrics)["lengths"])
        self.assertFalse(gate_results(metrics, [])["lengths"])
        metrics["per_label"]["SAT"]["support"] = 0
        self.assertFalse(gate_results(metrics, ["256"])["class_support"])

    def test_invalid_or_nonfinite_probabilities_rejected(self):
        for values in ([], [[1, 0]], [[float("nan"), 0, 0, 0, 1]], [[1, 1, 0, 0, 0]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                adjusted_probabilities(values, 0.125)
        for bias in (-0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                adjusted_probabilities(self.probabilities, bias)

    def test_probability_input_not_mutated_and_other_odds_preserved(self):
        import numpy as np

        before = copy.deepcopy(self.probabilities)
        result = adjusted_probabilities(self.probabilities, 0.625)
        self.assertEqual(self.probabilities, before)
        np.testing.assert_allclose(result.sum(-1), 1, atol=1e-7)
        self.assertAlmostEqual(result[0, 0] / result[0, 1], 396, places=3)


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch is optional for data tests"
)
class ActualHeadTests(unittest.TestCase):
    def test_real_linear_head_matches_logit_shift_and_preserves_other_weights(self):
        import torch

        torch.manual_seed(19)
        classifier = torch.nn.Linear(8, 5, dtype=torch.float32)
        model = SimpleNamespace(
            classifier=classifier,
            config=SimpleNamespace(
                label2id=dict(VELA_LABEL2ID),
                id2label=dict(VELA_ID2LABEL),
                problem_type="single_label_classification",
            ),
        )
        inputs = torch.randn(7, 8)
        before_logits = classifier(inputs).detach().clone()
        before_weight = classifier.weight.detach().clone()
        before_bias, after_bias = apply_head_bias(model, 0.625)
        expected = before_logits.clone()
        expected[:, 4] += 0.625
        torch.testing.assert_close(classifier(inputs), expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(before_weight, classifier.weight))
        self.assertEqual(before_bias[:4], after_bias[:4])
        self.assertEqual(after_bias[4], float(torch.tensor(before_bias[4]) + 0.625))
        model.config.label2id["NO_FEEDBACK"] = 3
        with self.assertRaises(ValueError):
            apply_head_bias(model, 0.125)


if __name__ == "__main__":
    unittest.main()
