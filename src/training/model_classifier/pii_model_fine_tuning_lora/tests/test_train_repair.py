import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_repair import checkpoint_score


class CheckpointSelectionTests(unittest.TestCase):
    def test_each_length_gets_equal_weight_despite_different_entity_density(self):
        metrics = {
            "micro": {"f1": 0.95},
            "breakdowns": {
                "length_bucket": {
                    "4096": {"micro": {"f1": 0.9, "support": 10000}},
                    "32768": {"micro": {"f1": 0.3, "support": 10}},
                }
            },
        }
        self.assertAlmostEqual(checkpoint_score(metrics, "micro-f1"), 0.95)
        self.assertAlmostEqual(checkpoint_score(metrics, "length-macro-f1"), 0.6)

    def test_missing_length_evidence_cannot_select_a_checkpoint(self):
        with self.assertRaises(ValueError):
            checkpoint_score({"breakdowns": {"length_bucket": {}}}, "length-macro-f1")


if __name__ == "__main__":
    unittest.main()
