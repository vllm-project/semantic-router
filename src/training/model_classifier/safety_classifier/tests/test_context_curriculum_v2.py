"""Long training selection must cover risks, not only the unsafe super-label."""

import unittest

from src.training.model_classifier.safety_classifier.vela_data import LABELS
from src.training.model_classifier.vela_context_curriculum_v2 import (
    select_payloads,
    strata,
)


class ContextCurriculumTests(unittest.TestCase):
    def test_each_observed_hazard_gets_its_own_stratum(self):
        rows = [
            {
                "id": f"risk:{index}",
                "group_id": f"family:{index}",
                "text": f"Training payload {index}",
                "language": "en",
                "label": "unsafe",
                "targets": [int(i == index) for i in range(len(LABELS))],
                "label_mask": [1] * len(LABELS),
            }
            for index in range(len(LABELS))
        ]
        selected, coverage = select_payloads(rows, "hazard", 1, len)
        self.assertEqual(len(selected), len(LABELS))
        self.assertEqual(coverage, {label + ":en": 1 for label in LABELS})

    def test_unknown_hazard_is_not_a_positive_training_stratum(self):
        row = {
            "language": "en",
            "label": "unsafe",
            "targets": [1] * len(LABELS),
            "label_mask": [0] * len(LABELS),
        }
        self.assertEqual(strata(row, "hazard"), [])

    def test_topic_variants_do_not_count_as_independent_families(self):
        rows = [
            {
                "id": str(index),
                "group_id": "one-family",
                "text": f"Variant {index}",
                "label": "benign",
                "language": "en",
            }
            for index in range(20)
        ]
        selected, coverage = select_payloads(rows, "sequence", 8, len)
        self.assertEqual(len(selected), 1)
        self.assertEqual(coverage["benign:en"], 1)
