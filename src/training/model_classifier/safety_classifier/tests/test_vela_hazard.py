import importlib.util
import random
import unittest

# Heavy numerical dependencies are optional for corpus-only checks.
# ruff: noqa: PLC0415
from src.training.model_classifier.safety_classifier.train_vela_hazard import (
    grouped_pools,
    sample_grouped,
)
from src.training.model_classifier.safety_classifier.vela_hazard import (
    calibrate_thresholds,
    masked_loss,
    score_predictions,
)


class VelaHazardSamplingTests(unittest.TestCase):
    def test_source_and_length_balance_with_safe_only_bucket(self):
        large_count = 1000
        rows = [
            {"targets": [1, 0], "source": "large", "length_bucket": "short"}
            for _ in range(large_count)
        ] + [{"targets": [0, 0], "source": "small", "length_bucket": "32768"}]
        pools = grouped_pools(rows, 2, True, True)
        rng = random.Random(7)
        count = sum(sample_grouped(pools, rng) == large_count for _ in range(2000))
        self.assertGreater(count, 850)
        self.assertLess(count, 1150)


@unittest.skipUnless(
    importlib.util.find_spec("sklearn"), "Scikit-learn is optional for data-only checks"
)
class VelaHazardMetricsTests(unittest.TestCase):
    def test_unknown_scores_do_not_change_observed_metrics(self):
        rows = [
            {"targets": [1, 0], "label_mask": [1, 0], "label": "unsafe"},
            {"targets": [0, 0], "label_mask": [1, 1], "label": "safe"},
        ]
        result = score_predictions(rows, [[0.9, 0.99], [0.1, 0.1]], ["a", "b"])
        self.assertEqual(result["observed_exact_match"], 1.0)
        self.assertEqual(result["per_label"]["b"]["unknown"], 1)
        self.assertEqual(result["safe_any_hazard_rate"], 0.0)

    def test_thresholds_fit_only_observed_and_require_support(self):
        rows = [
            {"targets": [1, 0], "label_mask": [1, 0]},
            {"targets": [0, 0], "label_mask": [1, 1]},
        ]
        thresholds, details = calibrate_thresholds(
            rows, [[0.2, 0.99], [0.1, 0.1]], ["a", "b"], minimum=1
        )
        self.assertAlmostEqual(thresholds[0], 0.2)
        self.assertEqual(thresholds[1], 0.5)
        self.assertEqual(details["b"]["positive"], 0)

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            score_predictions(
                [{"targets": [1], "label_mask": [1]}], [[float("nan")]], ["a"]
            )


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Torch is optional for data-only checks"
)
class VelaHazardLossTests(unittest.TestCase):
    def test_unknown_label_has_zero_gradient(self):
        import torch

        logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        loss = masked_loss(
            logits, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])
        )
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.0)
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_accumulated_loss_matches_full_batch(self):
        import torch

        logits = torch.randn(16, 4, requires_grad=True)
        target = torch.randint(0, 2, (16, 4)).float()
        mask = torch.ones_like(target)
        mask[::2, 2:] = 0
        direct = masked_loss(logits, target, mask)
        accumulated = sum(
            masked_loss(logits[i : i + 2], target[i : i + 2], mask[i : i + 2], 8)
            for i in range(0, 16, 2)
        )
        torch.testing.assert_close(direct, accumulated)
        torch.testing.assert_close(
            torch.autograd.grad(direct, logits, retain_graph=True)[0],
            torch.autograd.grad(accumulated, logits)[0],
        )

    def test_empty_supervision_is_rejected(self):
        import torch

        with self.assertRaises(ValueError):
            masked_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
