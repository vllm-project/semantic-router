"""Release-split lineage and strict group quarantine checks."""

from __future__ import annotations

import unittest

from training.data.build_nox4b_no_mnli import is_mnli_origin
from training.data.build_rights_clean_v1 import (
    _exclude_context_groups,
    _is_excluded,
    _set_role,
)


def row(identifier: str, group: str, state: str, source: str = "internal") -> dict:
    return {
        "id": identifier,
        "group_id": group,
        "state": state,
        "source": source,
        "input_sha256": identifier,
        "audit_metadata": {},
    }


class RightsCleanTests(unittest.TestCase):
    def test_nested_mnli_lineage_and_tweeteval_are_excluded(self):
        inherited = row("a", "g1", "A", "legacy:stage3_replay")
        inherited["audit_metadata"]["original_source"] = {
            "dataset": "nyu-mll/multi_nli"
        }
        self.assertTrue(is_mnli_origin(inherited))
        self.assertTrue(_is_excluded(inherited))
        self.assertTrue(_is_excluded(row("b", "g2", "B", "tweeteval_train:irony")))
        self.assertFalse(_is_excluded(row("c", "g3", "C", "legacy:cosmos_qa")))

    def test_exact_context_quarantines_complete_counterfactual_group(self):
        groups = {
            "pair": [
                row("a", "pair", "A clerk filed the blue note."),
                row("b", "pair", "A clerk filed the red note."),
            ],
            "safe": [row("c", "safe", "Three lanterns stand behind the door.")],
        }
        protected = [row("holdout", "other", "A clerk filed the blue note.")]
        eligible, audit = _exclude_context_groups(groups, protected)
        self.assertEqual(set(eligible), {"safe"})
        self.assertEqual(audit["rejected_rows"]["exact_context"], 2)

    def test_holdout_role_has_no_training_flag(self):
        sample = row("a", "g", "A text")
        sample["audit_metadata"] = {"oracle": "true"}
        converted = _set_role(sample, "cal")
        self.assertEqual(converted["split"], "cal")
        self.assertEqual(converted["evaluation_role"], "calibrate")
        self.assertTrue(converted["audit_metadata"]["holdout_only"])
        self.assertEqual(sample["source"], "internal")


if __name__ == "__main__":
    unittest.main()
