"""V2 normalization permits typography changes while retaining rule meaning."""

import unittest

from training.data.audit_sharc_policy_source_v2 import inventory, normalized


def _row(item: str, label: str, snippet: str, scenario: str) -> dict:
    return {
        "utterance_id": item,
        "tree_id": "rule-tree",
        "source_url": "https://example.org/rule",
        "snippet": snippet,
        "question": "Can I apply?",
        "scenario": scenario,
        "history": [],
        "answer": label,
    }


class SharcV2Tests(unittest.TestCase):
    def test_typography_only_pair_and_no_free_text_report(self) -> None:
        result = inventory(
            [
                _row("yes", "Yes", "  A person MAY apply. ", "All records ready"),
                _row("no", "No", "a person may   apply?", "Missing record"),
                _row("other", "Do you have a record?", "A person may apply!", ""),
            ]
        )
        self.assertEqual(result["candidate_choice_tree_pairs"], 1)
        self.assertEqual(result["answer_classes"], {"No": 1, "Other": 1, "Yes": 1})
        self.assertNotIn("Do you have a record?", str(result))

    def test_negation_remains_distinct(self) -> None:
        result = inventory(
            [
                _row("yes", "Yes", "A person may apply.", "Ready"),
                _row("no", "No", "A person may not apply.", "Not ready"),
            ]
        )
        self.assertEqual(result["candidate_choice_tree_pairs"], 0)
        self.assertFalse(result["training_admitted"])
        self.assertNotEqual(normalized("may apply"), normalized("may not apply"))
