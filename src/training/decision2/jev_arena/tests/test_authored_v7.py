"""Regression checks for the v7 evidence and shortcut fixes."""

from __future__ import annotations

import unittest

from jev_arena.authored_v7_pilot import (
    build_item,
    ordered_value,
    parse_sources,
    question,
)
from jev_arena.authored_v7_policies import POLICIES, aggregate, evaluate, reference

SECRET = bytes(range(32))
SCENE = (
    "A safety office collected separately signed records for a single operational "
    "decision after changing its governing procedure. The older procedure remains "
    "visible in the file for audit, while the current signed procedure governs the "
    "request. Staff cannot replace a missing source with an assumption drawn from "
    "an adjacent case. They must use the surviving admissible evidence envelope and "
    "apply the stated rule across every allowed completion."
)
PROSE = (
    "This accountable office signed a case-specific source after checking the "
    "underlying operational register. The field belongs to the named decision "
    "file and is independent of other offices' registers. Its signatory did not "
    "compute a final decision or infer an absent field. Historical planning "
    "documents may use similar names, but they do not override this signed "
    "source for the exact case identifier."
)


class AuthoredV7Test(unittest.TestCase):
    def test_missing_source_is_physically_absent_and_worlds_are_aggregated(self):
        spec = {
            "slug": "unit-missing-capacity",
            "policy_id": "safe-exit",
            "challenge": "missing_source",
            "facts": {
                "occupied_levels": 6,
                "inspected_exits": 3,
                "certified_capacity": 60,
            },
            "missing_field": "certified_capacity",
            "admissible_values": [60, 72],
            "scene": SCENE,
            "source_prose": dict.fromkeys(POLICIES["safe-exit"].fields, PROSE),
        }
        prompt, target, trace = build_item(spec, SECRET)
        self.assertFalse(target["answer"]["noul"])
        self.assertEqual(trace["world_outputs"], [False, True])
        self.assertEqual(trace["missing_source_count"], 0)
        self.assertNotIn(
            "certified_capacity",
            {row["field"] for row in parse_sources(prompt["state"])},
        )
        self.assertIn("MISSING FIELD certified_capacity", prompt["state"])

    def test_choice_display_order_tracks_answer_independent_source_order(self):
        policy = POLICIES["backup-routing"]
        facts = {
            "outage_risk": {"Aster": 5, "Birch": 2, "Clover": 1},
            "restore_hours": {"Aster": 1, "Birch": 4, "Clover": 7},
            "available": ["Aster", "Birch", "Clover"],
        }
        changed = {**facts, "outage_risk": {"Aster": 1, "Birch": 2, "Clover": 5}}
        source_order = list(ordered_value(SECRET, "unit-order", facts["outage_risk"]))
        first = list(question(policy, facts, "case", SECRET, "unit-order")["criteria"])
        second = list(
            question(policy, changed, "case", SECRET, "unit-order")["criteria"]
        )
        self.assertEqual(first[:-1], source_order)
        self.assertEqual(first, second)
        self.assertNotEqual(evaluate(policy, facts), evaluate(policy, changed))

    def test_current_and_archived_oracles_cover_all_policy_archetypes(self):
        samples = {
            "relief-supplier": {
                "costs": {"A": 4, "B": 3, "C": 8},
                "capacity": {"A": 8, "B": 3, "C": 9},
                "certified": ["A", "B", "C"],
            },
            "cold-chain-dispatch": {
                "drive_minutes": {"A": 20, "B": 30, "C": 10},
                "seal_ok": {"A": 1, "B": 1, "C": 0},
                "authorized": ["A", "B", "C"],
            },
            "grant-panel": {
                "merit": {"A": 70, "B": 60, "C": 50},
                "matching_funds": {"A": 55, "B": 90, "C": 65},
                "eligible": ["A", "B", "C"],
            },
            "backup-routing": {
                "outage_risk": {"A": 3, "B": 2, "C": 1},
                "restore_hours": {"A": 1, "B": 2, "C": 4},
                "available": ["A", "B", "C"],
            },
            "sterile-release": {"signoffs": 2, "open_defects": 0, "seal_intact": True},
            "safe-exit": {
                "occupied_levels": 4,
                "inspected_exits": 2,
                "certified_capacity": 48,
            },
            "supplier-renewal": {
                "delivery_pct": 93,
                "open_complaints": 1,
                "bond_current": True,
            },
            "incident-severity": {"exposure": 4, "containment": 2, "signed": True},
            "inspection-deficit": {"critical": 1, "major": 1, "reviewer_signed": True},
            "readiness-evidence": {
                "verified_stages": 3,
                "blocked_stages": 1,
                "executive_signed": True,
            },
            "recovery-progress": {
                "restored_sites": 4,
                "overdue_tasks": 1,
                "verified": True,
            },
            "resilience-grid": {
                "independent_feeds": 2,
                "tested_islands": 2,
                "verified": True,
            },
        }
        self.assertEqual(set(samples), set(POLICIES))
        for name, facts in samples.items():
            with self.subTest(name=name):
                policy = POLICIES[name]
                self.assertEqual(evaluate(policy, facts), reference(policy, facts))
                self.assertEqual(
                    evaluate(policy, facts, archived=True),
                    reference(policy, facts, archived=True),
                )

    def test_type_specific_world_aggregation(self):
        self.assertEqual(aggregate(POLICIES["cold-chain-dispatch"], ["A", "B"]), "hold")
        self.assertFalse(aggregate(POLICIES["safe-exit"], [True, False]))
        self.assertEqual(aggregate(POLICIES["resilience-grid"], [4, 4]), 4)
        self.assertEqual(aggregate(POLICIES["incident-severity"], [0, 2]), 0)


if __name__ == "__main__":
    unittest.main()
