"""Focused contracts for authored v10 semantics and source necessity."""

from __future__ import annotations

import copy
import unittest

from jev_arena.authored_v10_pilot import (
    OPS,
    build_item,
    evaluate,
    parse_sources,
    reference,
    source_blocks,
)

SECRET = bytes(range(32))
SCENE = (
    "The council records the selected proposal for a local facilities project. "
    "Several offices have different responsibilities for tallying votes, "
    "checking the compliance file, and reviewing objections. Their records "
    "are written independently so that an office cannot silently substitute "
    "its judgment for another. The chair applies the published rule to the "
    "case scope shown below and retains the supporting documents for audit."
)
SOURCE_TEXT = (
    "This office maintains its own register for the case scope named in the "
    "header. Its staff recorded the method used to collect this portion of "
    "the file and the authority responsible for later correction. Other "
    "offices maintain separate portions. The structured data line below is "
    "the attestation; this narrative does not establish another office's result."
)


def council_spec() -> dict:
    votes = {"Aster": 5, "Birch": 4, "Cedar": 3}
    facts = {"votes": votes, "vetoed": ["Aster"], "audited": list(votes)}
    return {
        "slug": "test-council",
        "operation_id": "council-proposal-vote",
        "mechanism": "ordinary",
        "scene": SCENE,
        "facts": facts,
        "documents": [
            {
                "slug": field,
                "field": field,
                "value": value,
                "form": "register",
                "title": f"Office record for {field}",
                "text": SOURCE_TEXT,
            }
            for field, value in facts.items()
        ],
        "essential": {field: field for field in facts},
        "domains": {
            "votes": [votes, {"Aster": 5, "Birch": 1, "Cedar": 6}],
            "vetoed": [["Aster"], []],
            "audited": [list(votes), ["Aster", "Cedar"]],
        },
    }


class AuthoredV10Test(unittest.TestCase):
    def test_independent_policy_paths_agree_for_all_twelve_operations(self):
        fixtures = {
            "floodgate-dispatch": {
                "capacity": {"A": 10, "B": 8, "C": 12},
                "risk": {"A": 2, "B": 3, "C": 1},
                "cleared": ["A", "B", "C"],
                "demand": 9,
            },
            "art-loan-courier": {
                "transit_hours": {"A": 8, "B": 4, "C": 6},
                "insured": ["A", "B", "C"],
                "handling": {"A": 3, "B": 2, "C": 4},
                "fragility": 3,
            },
            "radio-channel-assignment": {
                "interference": {"A": 1, "B": 2, "C": 3},
                "reserved": ["A"],
                "bandwidth": {"A": 8, "B": 7, "C": 6},
                "minimum_bandwidth": 6,
            },
            "council-proposal-vote": council_spec()["facts"],
            "forest-crossing-clearance": {
                "load": 10,
                "rated_capacity": 10,
                "closure_active": False,
                "emergency_override": False,
                "inspector_signed": False,
            },
            "patient-data-consent": {
                "consent_scopes": ["care", "research"],
                "requested_scopes": ["care"],
                "revoked_scopes": [],
                "review_signed": True,
            },
            "aquifer-alert-corroboration": {
                "readings": {"s1": 5, "s2": 6, "s3": 1},
                "validated_sensors": ["s1", "s2"],
                "threshold": 5,
            },
            "polling-ledger-reconciliation": {
                "issued": 100,
                "cast": 98,
                "spoiled": 2,
                "seal_ok": True,
            },
            "satellite-channel-readiness": {
                "working_channels": ["a", "b"],
                "required_channels": ["a", "b", "c"],
                "tested_channels": ["a", "b"],
                "ground_signed": True,
            },
            "hospital-surge-capacity": {
                "staffed_beds": 8,
                "oxygen_beds": 6,
                "isolation_beds": 5,
                "request_beds": 10,
            },
            "storm-barrier-stage": {"surge_day": 10, "ready_day": 8, "drill_delay": 3},
            "biosample-custody": {
                "received": 5,
                "verified": 4,
                "unmatched": 1,
                "review_signed": True,
            },
        }
        self.assertEqual(set(fixtures), set(OPS))
        for name, facts in fixtures.items():
            with self.subTest(name=name):
                self.assertEqual(
                    evaluate(OPS[name], facts), reference(OPS[name], facts)
                )
                self.assertEqual(
                    evaluate(OPS[name], facts, archived=True),
                    reference(OPS[name], facts, archived=True),
                )

    def test_every_claimed_source_has_two_output_completions_after_deletion(self):
        prompt, target, proof, ablations = build_item(council_spec(), SECRET)
        self.assertEqual(target["answer"]["choice"], "Birch")
        self.assertEqual(len(ablations), 3)
        self.assertEqual(len(source_blocks(prompt["state"])), 3)
        self.assertTrue(
            all(row["distinct_outputs"] >= 2 for row in proof["sensitivity"].values())
        )
        scope = prompt["state"].split("Target scope: ", 1)[1].split(".", 1)[0]
        for row in ablations:
            self.assertEqual(len(parse_sources(row["state"], scope)), 2)

    def test_rejects_redundant_source_and_factual_lead(self):
        spec = council_spec()
        spec["domains"]["vetoed"] = [["Aster"], ["Aster", "Cedar"]]
        with self.assertRaisesRegex(ValueError, "single provable outcome"):
            build_item(spec, SECRET)
        spec = copy.deepcopy(council_spec())
        spec["scene"] += " The chair was approved."
        with self.assertRaisesRegex(ValueError, "result-bearing prose"):
            build_item(spec, SECRET)


if __name__ == "__main__":
    unittest.main()
