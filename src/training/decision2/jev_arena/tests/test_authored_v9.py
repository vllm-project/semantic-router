"""Mechanical tests for v9 evidence joins and source-deletion semantics."""

from __future__ import annotations

import unittest

from jev_arena.authored_v9_pilot import build_item, derive_facts, parse_documents

SECRET = bytes(range(32))
SCENE = (
    "The incident chair must decide using signed records from offices with different "
    "responsibilities. Earlier versions of some records remain available for audit. "
    "The material describes a target operational file and related work that shares "
    "staff and equipment but has its own authority. A decision taken from the packet "
    "must respect the governing source links and the effective policy version."
)
BODY = (
    "This controlled record describes an independently checked part of the operational "
    "case. Staff examined the raw register, reconciled its scope with the responsible "
    "office, and signed the attestation for the named file. The account explains what "
    "was measured and which office can amend it; no final decision was made here. "
) * 3


def doc(slug, kind, field=None, value=None, role="target", **extras):
    result = {
        "slug": slug,
        "kind": kind,
        "role": role,
        "format": "memo",
        "title": f"Signed record {slug}",
        "text": BODY,
    }
    if field is not None:
        result["field"] = field
    if value is not None:
        result["value"] = value
    return {**result, **extras}


class AuthoredV9Test(unittest.TestCase):
    def test_amendment_is_essential_and_not_latest_value_shortcut(self):
        facts = {
            "outage_risk": {"Aster": 1, "Birch": 1, "Cedar": 3, "Dune": 0},
            "restore_hours": {"Aster": 8, "Birch": 2, "Cedar": 1, "Dune": 0},
            "available": ["Aster", "Birch", "Cedar"],
        }
        alt_risk = {"Aster": 0, "Birch": 1, "Cedar": 3, "Dune": 0}
        spec = {
            "slug": "test-amendment",
            "policy_id": "backup-routing",
            "mechanism": "long_amendment",
            "scene": SCENE,
            "facts": facts,
            "documents": [
                doc("risk_old", "revision_fact", "outage_risk", alt_risk, revision="A"),
                doc(
                    "risk_new",
                    "revision_fact",
                    "outage_risk",
                    facts["outage_risk"],
                    revision="B",
                ),
                doc("amendment", "amendment", "outage_risk", revision="B"),
                doc("hours", "fact", "restore_hours", facts["restore_hours"]),
                doc("roster", "fact", "available", facts["available"]),
            ],
            "essential": {
                "risk_new": "outage_risk",
                "amendment": "outage_risk",
                "hours": "restore_hours",
                "roster": "available",
            },
            "counterfactuals": {
                "risk_new": alt_risk,
                "amendment": alt_risk,
                "hours": {"Aster": 1, "Birch": 8, "Cedar": 1, "Dune": 0},
                "roster": ["Aster", "Cedar"],
            },
        }
        prompt, target, proof, ablations = build_item(spec, SECRET)
        self.assertEqual(target["answer"]["choice"], "Birch")
        self.assertEqual(len(ablations), 4)
        self.assertEqual(proof["ablation_count"], 4)
        self.assertEqual(len(parse_documents(prompt["state"])), 5)
        criteria = list(prompt["questions"]["decision"]["criteria"])
        self.assertEqual(
            [label for label in criteria if label != "hold"],
            list(proof["visible_facts"]["outage_risk"]),
        )
        amendment_ablation = next(
            row
            for row in ablations
            if row["omitted_field"] == "outage_risk"
            and "ACCEPT outage_risk" not in row["state"]
        )
        self.assertNotIn(
            "outage_risk",
            derive_facts(
                parse_documents(amendment_ablation["state"]),
                parse_documents(prompt["state"])[0]["file"],
                "long_amendment",
            ),
        )

    def test_alias_mapping_is_required_for_boolean_source(self):
        spec = {
            "slug": "test-alias",
            "policy_id": "substation-restart",
            "mechanism": "long_alias",
            "scene": SCENE,
            "facts": {"relay_tests": 3, "isolated_faults": 2, "operator_signed": True},
            "documents": [
                doc("tests", "fact", "relay_tests", 3),
                doc("isolation", "fact", "isolated_faults", 2, role="alias"),
                doc("alias_map", "alias"),
                doc("signature", "fact", "operator_signed", True),
            ],
            "essential": {
                "tests": "relay_tests",
                "isolation": "isolated_faults",
                "alias_map": "isolated_faults",
                "signature": "operator_signed",
            },
            "counterfactuals": {
                "tests": 2,
                "isolation": 1,
                "alias_map": 1,
                "signature": False,
            },
        }
        prompt, target, _, ablations = build_item(spec, SECRET)
        self.assertTrue(target["answer"]["noul"])
        self.assertEqual(len(ablations), 4)
        self.assertEqual(len(parse_documents(prompt["state"])), 4)

    def test_timeline_cutoff_is_required_for_score_source(self):
        spec = {
            "slug": "test-timeline",
            "policy_id": "readiness-evidence",
            "mechanism": "long_timeline",
            "scene": SCENE,
            "facts": {
                "verified_stages": 3,
                "blocked_stages": 1,
                "executive_signed": True,
            },
            "documents": [
                doc("verified", "fact", "verified_stages", 3),
                doc(
                    "updates",
                    "timeline",
                    "blocked_stages",
                    events=[
                        {"day": 10, "value": 2},
                        {"day": 20, "value": 1},
                        {"day": 30, "value": 0},
                    ],
                ),
                doc("cutoff", "cutoff", day=25),
                doc("signature", "fact", "executive_signed", True),
            ],
            "essential": {
                "verified": "verified_stages",
                "updates": "blocked_stages",
                "cutoff": "blocked_stages",
                "signature": "executive_signed",
            },
            "counterfactuals": {
                "verified": 1,
                "updates": 2,
                "cutoff": 2,
                "signature": False,
            },
        }
        prompt, target, _, ablations = build_item(spec, SECRET)
        self.assertEqual(target["answer"]["score"], 2)
        self.assertEqual(len(ablations), 4)
        self.assertEqual(len(parse_documents(prompt["state"])), 4)

    def test_new_choice_policy_uses_same_candidate_order_in_all_sources(self):
        facts = {
            "arrival_minutes": {"Aster": 5, "Birch": 8, "Cedar": 9},
            "draft_meters": {"Aster": 9, "Birch": 8, "Cedar": 8},
            "licensed": ["Aster", "Birch", "Cedar"],
        }
        spec = {
            "slug": "test-berth",
            "policy_id": "port-berth-arrival",
            "mechanism": "ordinary",
            "scene": SCENE,
            "facts": facts,
            "documents": [
                doc("arrival", "fact", "arrival_minutes", facts["arrival_minutes"]),
                doc("draft", "fact", "draft_meters", facts["draft_meters"]),
                doc("licensed", "fact", "licensed", facts["licensed"]),
            ],
            "essential": {
                "arrival": "arrival_minutes",
                "draft": "draft_meters",
                "licensed": "licensed",
            },
            "counterfactuals": {
                "arrival": {"Aster": 5, "Birch": 10, "Cedar": 9},
                "draft": {"Aster": 9, "Birch": 9, "Cedar": 8},
                "licensed": ["Aster", "Cedar"],
            },
        }
        prompt, target, proof, ablations = build_item(spec, SECRET)
        self.assertEqual(target["answer"]["choice"], "Birch")
        self.assertEqual(len(ablations), 3)
        labels = list(prompt["questions"]["decision"]["criteria"])
        for field in ("arrival_minutes", "draft_meters", "licensed"):
            self.assertEqual(
                [label for label in labels if label != "hold"],
                list(proof["visible_facts"][field]),
            )

    def test_numeric_intro_and_document_inventory_are_rejected(self):
        spec = {
            "slug": "test-leak",
            "policy_id": "substation-restart",
            "mechanism": "ordinary",
            "scene": SCENE + " Three tests were completed.",
            "facts": {"relay_tests": 3, "isolated_faults": 2, "operator_signed": True},
            "documents": [],
        }
        with self.assertRaisesRegex(ValueError, "numeric answer cues"):
            build_item(spec, SECRET)
        spec["scene"] = SCENE + " The attached record is conclusive."
        with self.assertRaisesRegex(ValueError, "inventories evidence"):
            build_item(spec, SECRET)


if __name__ == "__main__":
    unittest.main()
