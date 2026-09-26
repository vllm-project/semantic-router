"""Fail-closed checks for the v5 semantic feasibility builders."""

from __future__ import annotations

import unittest

from jev_arena.authored_v5_dev import _read_visible, _resolve_worlds, build_short
from jev_arena.authored_v5_reference import (
    evaluate_reference,
    governing_reference,
    swap_priority,
)
from jev_arena.authored_v5_registry import PAIRS, evaluate_current, prove_conflict

SECRET = bytes(range(32))
SCENE = (
    "The records office has received signed entries from three separate authorities. "
    "A current policy governs this identified case, and an older policy remains "
    "attached for the audit history. The owner must make a decision from the "
    "scoped source documents before the next operations meeting starts."
)


def spec(pair_id, challenge, facts, **extra):
    return {
        "slug": f"unit-{pair_id}-{challenge}",
        "pair_id": pair_id,
        "challenge": challenge,
        "facts": facts,
        "scene": SCENE,
        "field_notes": ["A separate office signed this source entry."] * 3,
        **extra,
    }


class AuthoredV5Test(unittest.TestCase):
    def test_all_twelve_policies_execute_both_versions_and_disagree(self):
        cases = {
            "procurement-benefit": {
                "costs": {"Oak": 30, "Pine": 20},
                "benefits": {"Oak": 90, "Pine": 70},
                "qualified_ids": ["Oak", "Pine"],
            },
            "delivery-speed": {
                "lead_days": {"Oak": 2, "Pine": 4},
                "reliability": {"Oak": 70, "Pine": 90},
                "approved_ids": ["Oak", "Pine"],
            },
            "grant-impact": {
                "community_impact": {"Oak": 90, "Pine": 70},
                "cofund_pct": {"Oak": 20, "Pine": 60},
                "eligible_ids": ["Oak", "Pine"],
            },
            "incident-exposure": {
                "exposure_reduced": {"Oak": 90, "Pine": 70},
                "start_hours": {"Oak": 4, "Pine": 2},
                "cleared_ids": ["Oak", "Pine"],
            },
            "committee-quorum": {"signed_votes": 2, "quorum": 2, "committee_size": 4},
            "inventory-buffer": {
                "remaining_stock": 18,
                "reserve_min": 15,
                "demand_next_day": 4,
            },
            "expense-receipt": {
                "expense": 50,
                "approved_budget": 60,
                "receipt_verified": False,
            },
            "incident-escalation": {
                "severity": 4,
                "mitigation_signed": True,
                "incident_signed": True,
            },
            "audit-ratio": {"passed": 2, "reviewed": 3, "audit_signed": True},
            "restoration-lateness": {
                "late_days": 2,
                "deadline_signed": True,
                "closure_signed": True,
            },
            "risk-product": {"likelihood": 3, "impact": 2, "assessment_signed": True},
            "quality-critical": {
                "passed_checks": 3,
                "critical_failures": 1,
                "review_signed": True,
            },
        }
        self.assertEqual(set(cases), set(PAIRS))
        for pair_id, facts in cases.items():
            with self.subTest(pair_id=pair_id):
                proof = prove_conflict(PAIRS[pair_id], facts)
                self.assertTrue(proof["priority_swap_changes_output"])
                self.assertNotEqual(proof["current_output"], proof["archived_output"])
                self.assertEqual(
                    evaluate_reference(pair_id, facts), proof["current_output"]
                )
                self.assertEqual(
                    evaluate_reference(pair_id, facts, archived=True),
                    proof["archived_output"],
                )

    def test_priority_text_swap_changes_answer_without_changing_facts(self):
        facts = {"signed_votes": 2, "quorum": 2, "committee_size": 4}
        row = spec("committee-quorum", "rule_precedence", facts)
        prompt, _, trace = build_short(row, SECRET)
        state = prompt["state"]
        self.assertTrue(governing_reference(state, row["pair_id"], facts))
        self.assertFalse(
            governing_reference(
                swap_priority(state, row["pair_id"]), row["pair_id"], facts
            )
        )
        self.assertFalse(trace["priority_swap_output"])
        with self.assertRaisesRegex(ValueError, "Visible policy text differs"):
            governing_reference(
                state.replace("strictly exceed", "always exceed"), row["pair_id"], facts
            )

    def test_partial_worlds_must_be_legal_and_nontrivially_invariant(self):
        facts = {"signed_votes": 5, "quorum": 4, "committee_size": 7}
        row = spec(
            "committee-quorum",
            "partial_evidence",
            facts,
            disputed_field="quorum",
            alternative_value=5,
            outcome_class="invariant",
        )
        prompt, target, trace = build_short(row, SECRET)
        pair = PAIRS[row["pair_id"]]
        documents = _read_visible(prompt["state"], pair)
        target_id = (
            prompt["questions"]["decision"]["instructions"].split()[3].rstrip(",")
        )
        first, second, sources = _resolve_worlds(
            documents, target_id, pair, disputed="quorum"
        )
        self.assertEqual(first, facts)
        self.assertEqual(second["quorum"], 5)
        self.assertEqual(len(set(sources.values())), 3)
        self.assertTrue(evaluate_current(pair, first))
        self.assertTrue(evaluate_current(pair, second))
        self.assertTrue(target["answer"]["noul"])
        self.assertTrue(trace["partial_invariant_nonfallback"])
        self.assertNotIn("source_group", str(prompt))
        invalid = {**row, "alternative_value": 8}
        with self.assertRaisesRegex(ValueError, "domain-inconsistent"):
            build_short(invalid, SECRET)
        mislabeled = {**row, "outcome_class": "unresolved"}
        with self.assertRaisesRegex(ValueError, "Declared partial-evidence outcome"):
            build_short(mislabeled, SECRET)

    def test_near_case_requires_material_but_scoped_distractor(self):
        facts = {"signed_votes": 2, "quorum": 2, "committee_size": 4}
        related = {"signed_votes": 1, "quorum": 2, "committee_size": 4}
        row = spec(
            "committee-quorum",
            "near_distractor",
            facts,
            related_facts=related,
            related_note="A different committee kept its own signed attendance record.",
        )
        prompt, target, trace = build_short(row, SECRET)
        self.assertTrue(target["answer"]["noul"])
        self.assertFalse(trace["decoy_output"])
        self.assertEqual(len(_read_visible(prompt["state"], PAIRS[row["pair_id"]])), 6)
        same_result = {**row, "related_facts": facts}
        with self.assertRaisesRegex(ValueError, "does not change the answer"):
            build_short(same_result, SECRET)


if __name__ == "__main__":
    unittest.main()
