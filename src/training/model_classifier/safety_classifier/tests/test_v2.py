"""The v2 recipe is explicit and leaves historical v1 reproducible."""

import unittest
from pathlib import Path

from src.training.model_classifier.safety_classifier.config import (
    distributed_batch_parameters,
    load_contract,
)
from src.training.model_classifier.safety_classifier.tests.test_train import (
    release_args,
)
from src.training.model_classifier.safety_classifier.train import _release_eligible


class SafetyV2Test(unittest.TestCase):
    def test_versions_keep_distinct_training_lengths_and_final_test_policy(self):
        v1 = load_contract()
        v2 = load_contract(Path(__file__).parents[1] / "configs" / "training-v2.json")
        self.assertEqual(v1["model"]["max_length"], 512)
        self.assertEqual(v2["model"]["max_length"], 2048)
        self.assertFalse(v2["training"]["evaluate_test_after_training"])
        self.assertEqual(v1["tasks"], v2["tasks"])
        self.assertEqual(v1["base_model"], v2["base_model"])
        self.assertEqual(distributed_batch_parameters(v2, 1), (8, 8))
        self.assertTrue(_release_eligible(release_args(), 1, True, v2))
        self.assertFalse(_release_eligible(release_args(), 1, True, v1))
        self.assertFalse(_release_eligible(release_args(max_steps=1), 1, True, v2))
