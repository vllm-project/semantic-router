import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_repair import checkpoint_score, save_adapter


class CheckpointSelectionTests(unittest.TestCase):
    def test_adapter_receipt_keeps_the_actual_continuation_base(self):
        config = SimpleNamespace(base_model_name_or_path="old", revision="old")
        model = Mock(peft_config={"default": config})
        tokenizer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            save_adapter(
                model,
                tokenizer,
                Path(directory),
                "organization/selected-base",
                "immutable-revision",
                {"O": 0},
            )
            self.assertEqual(
                config.base_model_name_or_path, "organization/selected-base"
            )
            self.assertEqual(config.revision, "immutable-revision")

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
