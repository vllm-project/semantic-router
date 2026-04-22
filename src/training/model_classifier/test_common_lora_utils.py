import unittest  # noqa: I001
from unittest.mock import patch

import sys
from pathlib import Path

# Add parent directory so common_lora_utils can be imported directly in
# environments where the training tree is not installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_lora_utils import select_training_split  # noqa: E402, RUF100


class SelectTrainingSplitTests(unittest.TestCase):
    def test_prefers_train_split_over_test(self):
        dataset = {
            "test": {"question": ["test-question"]},
            "train": {"question": ["train-question"]},
        }

        split_name, split_data = select_training_split(dataset, "demo-dataset")

        self.assertEqual(split_name, "train")
        self.assertEqual(split_data["question"], ["train-question"])

    def test_falls_back_to_non_test_split_when_train_missing(self):
        dataset = {
            "validation": {"question": ["validation-question"]},
            "test": {"question": ["test-question"]},
        }

        split_name, split_data = select_training_split(dataset, "demo-dataset")

        self.assertEqual(split_name, "validation")
        self.assertEqual(split_data["question"], ["validation-question"])

    def test_uses_test_split_only_as_last_resort(self):
        dataset = {
            "test": {"question": ["test-question"]},
        }

        split_name, split_data = select_training_split(dataset, "demo-dataset")

        self.assertEqual(split_name, "test")
        self.assertEqual(split_data["question"], ["test-question"])

    def test_raises_error_when_no_usable_splits(self):
        dataset = {}

        with self.assertRaises(ValueError) as ctx:
            select_training_split(dataset, "empty-dataset")

        self.assertIn("empty-dataset", str(ctx.exception))

    @patch("common_lora_utils.logger")
    def test_fallback_is_deterministic_across_non_test_splits(self, _mock_logger):
        dataset = {
            "zeta": {"question": ["z"]},
            "alpha": {"question": ["a"]},
            "test": {"question": ["t"]},
        }

        split_name, _ = select_training_split(dataset, "demo-dataset")

        self.assertEqual(split_name, "alpha")


if __name__ == "__main__":
    unittest.main()
