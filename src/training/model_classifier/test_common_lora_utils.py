import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_lora_utils import select_training_split


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


if __name__ == "__main__":
    unittest.main()
