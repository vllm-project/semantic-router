"""Domain labels from Routing Preview must score the same in both evaluators."""

import unittest
from unittest.mock import patch

from src.training.model_eval import signal_eval


class DomainSignalEvaluationTest(unittest.TestCase):
    def test_custom_and_registered_domain_scores_match(self):
        dataset = [
            {"question": q, "category": "math"} for q in ("hit", "miss", "empty")
        ]
        custom = [
            {"question": row["question"], "answer": row["category"]} for row in dataset
        ]
        domains = {"hit": ["math"], "miss": ["physics"], "empty": []}

        def preview(question, endpoint, timeout):
            return {
                "decision_result": {"matched_signals": {"domains": domains[question]}}
            }

        with (
            patch.object(signal_eval, "call_routing_preview", side_effect=preview),
            patch.object(
                signal_eval,
                "load_dataset_by_id",
                return_value=(dataset, signal_eval.DATASET_REGISTRY["mmlu-pro-en"]),
            ),
        ):
            builtin = signal_eval.evaluate_dataset("mmlu-pro-en", "preview", None, 1)
            result = signal_eval.evaluate_custom_dataset(custom, "domain", "preview", 1)

        self.assertEqual(
            (builtin["correct"], builtin["incorrect"], builtin["skipped"]), (1, 1, 1)
        )
        self.assertEqual(result["details"], builtin["details"])
        self.assertEqual(
            (result["correct"], result["incorrect"], result["skipped"]), (1, 1, 1)
        )
