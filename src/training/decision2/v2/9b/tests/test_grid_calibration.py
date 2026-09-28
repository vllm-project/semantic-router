import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def summary(macro, brier):
    return {"select": {"family_macro_accuracy": macro, "family_macro_brier": brier}}


@unittest.skipUnless(HAS_TORCH, "grid imports the torch arm registry")
class LayerRuleTest(unittest.TestCase):
    def test_macro_then_brier_then_later_layer(self):
        from clm9b.grid import choose_layer

        self.assertEqual(
            choose_layer(
                {16: summary(0.8, 0.1), 24: summary(0.9, 0.2), 32: summary(0.85, 0.05)}
            ),
            24,
        )
        self.assertEqual(
            choose_layer(
                {16: summary(0.9, 0.1), 24: summary(0.9, 0.05), 32: summary(0.9, 0.2)}
            ),
            24,
        )
        self.assertEqual(
            choose_layer(
                {16: summary(0.9, 0.1), 24: summary(0.9, 0.1), 32: summary(0.9, 0.1)}
            ),
            32,
        )


@unittest.skipUnless(HAS_TORCH, "score calibration imports the trainer")
class ScoreMetricTest(unittest.TestCase):
    def test_perfect_and_uniform(self):
        from clm9b.score_calibration import metrics

        perfect = metrics(
            [([1.0, 0.0, 0.0], 0), ([0.0, 1.0, 0.0], 1), ([0.0, 0.0, 1.0], 2)]
        )
        self.assertEqual(perfect["accuracy"], 1.0)
        self.assertAlmostEqual(perfect["brier"], 0.0)
        self.assertAlmostEqual(perfect["rps"], 0.0)
        self.assertAlmostEqual(perfect["ece_10"], 0.0)
        self.assertEqual(perfect["recall_by_level"], [1.0, 1.0, 1.0])
        tied = metrics([([1 / 3, 1 / 3, 1 / 3], 1)])
        self.assertEqual(tied["accuracy"], 0.0)
        self.assertEqual(
            tied["confusion_gold_by_predicted"], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )
        self.assertAlmostEqual(tied["expected_score_mae"], 0.0)

    def test_rps_penalizes_distance(self):
        from clm9b.score_calibration import metrics

        near = metrics([([0.0, 1.0, 0.0], 2)])["rps"]
        far = metrics([([1.0, 0.0, 0.0], 2)])["rps"]
        self.assertLess(near, far)


if __name__ == "__main__":
    unittest.main()
