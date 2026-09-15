# ruff: noqa: PLC0415

import unittest

from src.training.model_classifier.user_feedback_classifier.data_contract import (
    ID2LABEL,
    LABEL2ID,
)
from src.training.model_classifier.user_feedback_classifier.vela_contract import (
    VELA_ID2LABEL,
    VELA_LABEL2ID,
    checkpoint_labels,
)


class VelaContractTests(unittest.TestCase):
    def test_five_class_applicability_counts_all_four_feedback_false_positives(self):
        from src.training.model_classifier.user_feedback_classifier.vela_applicability import (
            summarize,
        )

        result = summarize(
            [
                {"group_id": "a", "prediction": "NO_FEEDBACK", "confidence": 0.95},
                {"group_id": "b", "prediction": "SAT", "confidence": 0.95},
            ],
            five_class=True,
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(
            result["thresholds"]["0.8"]["high_confidence_feedback_fraction"], 0.5
        )

    def test_only_exact_legacy_and_vela_mapping_are_accepted(self):
        self.assertEqual(checkpoint_labels(ID2LABEL, LABEL2ID), ID2LABEL)
        self.assertEqual(checkpoint_labels(VELA_ID2LABEL, VELA_LABEL2ID), VELA_ID2LABEL)
        with self.assertRaises(ValueError):
            checkpoint_labels({**ID2LABEL, 4: "OTHER"}, VELA_LABEL2ID)
        with self.assertRaises(ValueError):
            checkpoint_labels(VELA_ID2LABEL, {**VELA_LABEL2ID, "SAT": 1})


if __name__ == "__main__":
    unittest.main()
