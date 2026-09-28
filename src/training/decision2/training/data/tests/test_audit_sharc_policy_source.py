"""The source inventory must preserve original rule groups and input roles."""

import unittest

from training.data.audit_sharc_policy_source import _visible, inventory


def _row(tree: str, item: str, answer: str, scenario: str) -> dict:
    return {
        "utterance_id": item,
        "tree_id": tree,
        "source_url": "https://example.org/rule",
        "snippet": "A completed application qualifies.",
        "question": "Does this application qualify?",
        "scenario": scenario,
        "history": [],
        "evidence": [{"private": "must never become input"}],
        "answer": answer,
    }


class SharcSourceInventoryTests(unittest.TestCase):
    def test_only_visible_contrasting_states_form_one_pair_per_tree(self) -> None:
        rows = [
            _row("one", "yes", "Yes", "The application is complete."),
            _row("one", "no", "No", "The application is incomplete."),
            _row("one", "other", "No", "The application is incomplete."),
            _row("two", "same-a", "Yes", "Same visible state"),
            _row("two", "same-b", "No", "Same visible state"),
        ]
        result = inventory(rows)
        self.assertEqual(result["candidate_choice_tree_pairs"], 1)
        self.assertEqual(result["distinct_trees"], 2)
        self.assertEqual(result["answer_classes"], {"No": 3, "Yes": 2})
        self.assertFalse(result["training_admitted"])
        self.assertNotIn("must never become input", _visible(rows[0]))

    def test_inconsistent_rule_and_duplicate_ids_are_not_candidate_pairs(self) -> None:
        yes = _row("one", "yes", "Yes", "Complete")
        no = _row("one", "no", "No", "Incomplete")
        no["snippet"] = "A different rule."
        result = inventory([yes, no, yes])
        self.assertEqual(result["candidate_choice_tree_pairs"], 0)
        self.assertEqual(result["inconsistent_rule_trees"], 1)
        self.assertEqual(result["malformed"], {"repeated_utterance_id": 1})
