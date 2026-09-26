"""Semantic regression checks for the v6 pilot redesign."""

from __future__ import annotations

import unittest

from jev_arena.authored_v6_pilot import _check_counterfactuals, _question, build_short
from jev_arena.authored_v6_policies import (
    POLICIES,
    aggregate,
    evaluate,
    governing,
    reference,
    swap_priority,
)

SECRET = bytes(range(32))
SCENE = (
    "A service owner has three separately signed sources for this exact case. "
    "The current policy was approved after an archived version, and the clerk "
    "retained both in the file. Two admissible records for one source field "
    "remain under review. The decision must use the explicit world aggregation "
    "rule in the signed policy without borrowing a neighboring case's facts."
)
SOURCE_NOTES = [
    {
        "intro": "An accountable office signed this case-specific source after an independent check.",
        "closing": "This dated entry belongs to the target decision rather than an adjacent file.",
    }
] * 3


class AuthoredV6Test(unittest.TestCase):
    def test_all_six_current_and_archived_rules_are_independently_executable(self):
        cases = {
            "benefit-per-cost": {
                "costs": {"Oak": 50, "Pine": 40, "Rose": 60},
                "benefits": {"Oak": 95, "Pine": 75, "Rose": 85},
                "qualified_ids": ["Oak", "Pine", "Rose"],
            },
            "reliable-delivery": {
                "lead_days": {"Oak": 3, "Pine": 4, "Rose": 6},
                "reliability": {"Oak": 85, "Pine": 95, "Rose": 90},
                "approved_ids": ["Oak", "Pine", "Rose"],
            },
            "certified-quorum": {"signed_votes": 3, "quorum": 3, "committee_size": 5},
            "safe-stock-release": {
                "remaining_stock": 10,
                "reserve_min": 8,
                "demand_next_day": 3,
            },
            "signed-audit-ratio": {"passed": 3, "reviewed": 5, "audit_signed": True},
            "signed-risk-product": {
                "likelihood": 3,
                "impact": 2,
                "assessment_signed": True,
            },
        }
        self.assertEqual(set(cases), set(POLICIES))
        for policy_id, facts in cases.items():
            with self.subTest(policy_id=policy_id):
                policy = POLICIES[policy_id]
                current = evaluate(policy, facts)
                archived = evaluate(policy, facts, archived=True)
                self.assertNotEqual(current, archived)
                self.assertEqual(current, reference(policy, facts))
                self.assertEqual(archived, reference(policy, facts, archived=True))
                state = f"Current signed rule: {policy.current_text}\nArchived rule: {policy.archived_text}"
                self.assertEqual(governing(state, policy, facts), current)
                self.assertEqual(
                    governing(swap_priority(state, policy), policy, facts), archived
                )

    def test_all_three_source_values_must_change_the_current_decision(self):
        policy = POLICIES["benefit-per-cost"]
        facts = {
            "costs": {"Oak": 50, "Pine": 40, "Rose": 60},
            "benefits": {"Oak": 95, "Pine": 75, "Rose": 85},
            "qualified_ids": ["Oak", "Pine", "Rose"],
        }
        interventions = {
            "costs": {"value": {"Oak": 70, "Pine": 40, "Rose": 60}},
            "benefits": {"value": {"Oak": 75, "Pine": 75, "Rose": 85}},
            "qualified_ids": {"value": ["Pine", "Rose"]},
        }
        for row in interventions.values():
            row["rationale"] = (
                "A separately signed correction could alter this exact field "
                "while preserving the other two approved target sources."
            )
        proof = _check_counterfactuals(policy, facts, interventions)
        self.assertEqual(set(proof), set(policy.fields))
        redundant = {
            **interventions,
            "costs": {
                **interventions["costs"],
                "value": {"Oak": 49, "Pine": 40, "Rose": 60},
            },
        }
        with self.assertRaisesRegex(ValueError, "not causally necessary"):
            _check_counterfactuals(policy, facts, redundant)

    def test_partial_evidence_has_explicit_type_specific_semantics(self):
        self.assertEqual(
            aggregate(POLICIES["benefit-per-cost"], ["Oak", "Pine"]), "hold"
        )
        self.assertFalse(aggregate(POLICIES["certified-quorum"], [True, False]))
        self.assertEqual(aggregate(POLICIES["signed-audit-ratio"], [3, 2]), 2)
        facts = {"signed_votes": 3, "quorum": 3, "committee_size": 5}
        row = {
            "slug": "unit-quorum-worlds",
            "policy_id": "certified-quorum",
            "challenge": "partial_evidence",
            "facts": facts,
            "scene": SCENE,
            "source_notes": SOURCE_NOTES,
            "disputed_field": "quorum",
            "alternative_value": 4,
            "world_relation": "sensitive",
        }
        prompt, target, trace = build_short(row, SECRET)
        self.assertFalse(target["answer"]["noul"])
        self.assertEqual(trace["first_output"], True)
        self.assertEqual(trace["second_output"], False)
        self.assertIn("every world", prompt["state"])

    def test_invariant_partial_must_not_be_default(self):
        row = {
            "slug": "unit-false-worlds",
            "policy_id": "safe-stock-release",
            "challenge": "partial_evidence",
            "facts": {"remaining_stock": 3, "reserve_min": 8, "demand_next_day": 3},
            "scene": SCENE,
            "source_notes": SOURCE_NOTES,
            "disputed_field": "remaining_stock",
            "alternative_value": 4,
            "world_relation": "invariant",
        }
        with self.assertRaisesRegex(ValueError, "nondefault"):
            build_short(row, SECRET)

    def test_choice_order_depends_on_secret_and_option_keys_not_answer(self):
        policy = POLICIES["benefit-per-cost"]
        facts = {
            "costs": {"Oak": 50, "Pine": 40, "Rose": 60},
            "benefits": {"Oak": 95, "Pine": 75, "Rose": 85},
            "qualified_ids": ["Oak", "Pine", "Rose"],
        }
        changed_answer = {**facts, "benefits": {"Oak": 70, "Pine": 95, "Rose": 85}}
        original_order = list(_question(policy, facts, "case-id", SECRET)["criteria"])
        changed_order = list(
            _question(policy, changed_answer, "case-id", SECRET)["criteria"]
        )
        self.assertEqual(original_order, changed_order)
        self.assertEqual(set(original_order), {"Oak", "Pine", "Rose", "hold"})


if __name__ == "__main__":
    unittest.main()
