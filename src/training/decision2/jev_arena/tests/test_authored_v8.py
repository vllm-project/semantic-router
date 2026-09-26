"""Mechanical v8 pilot checks; independent editorial review is separate."""

from __future__ import annotations

import unittest

from jev_arena.authored_v8_audit import remove_document
from jev_arena.authored_v8_pilot import (
    ARCHIVE_POLICY,
    build_item,
    evaluate,
    parse_documents,
    reference,
    selected,
)

SECRET = bytes(range(32))
PROSE = (
    "The registrar compared the physical file with the activity log before signing "
    "this entry. This statement applies to the marked file alone; adjacent files "
    "carry separate evidence, and the archived policy text is retained for audit."
)


def doc(
    role: str, field: str, value: object, day: int, status: str = "CURRENT"
) -> dict[str, object]:
    return {
        "role": role,
        "day": day,
        "status": status,
        "field": field,
        "value": value,
        "format": "memo",
        "title": "Signed register extract",
        "text": PROSE,
    }


class AuthoredV8Test(unittest.TestCase):
    def test_ablation_removes_one_document_without_reviving_void_source(self):
        state = (
            "Decision file\n\n"
            "DOCUMENT aaaaaaaaaaaa | FORMAT memo | FILE cccccccccccc | DAY 1 | STATUS VOID\n"
            "Old record\nATTESTED verified_copies = 1\n\n"
            "DOCUMENT bbbbbbbbbbbb | FORMAT letter | FILE cccccccccccc | DAY 2 | STATUS CURRENT\n"
            "New record\nATTESTED verified_copies = 3"
        )
        reduced = remove_document(state, "bbbbbbbbbbbb")
        self.assertEqual(len(parse_documents(reduced)), 1)
        self.assertEqual(selected(parse_documents(reduced), "cccccccccccc")[0], {})

    def test_void_same_case_source_does_not_revive_after_current_ablation(self):
        rows = [
            {
                "doc_id": "old",
                "case_id": "target",
                "day": 1,
                "status": "VOID",
                "field": "verified_copies",
                "value": 1,
            },
            {
                "doc_id": "new",
                "case_id": "target",
                "day": 2,
                "status": "CURRENT",
                "field": "verified_copies",
                "value": 3,
            },
        ]
        self.assertEqual(selected(rows, "target")[0], {"verified_copies": 3})
        self.assertEqual(selected(rows[:1], "target")[0], {})

    def test_fourth_noul_policy_has_independent_reference(self):
        facts = {
            "verified_copies": 3,
            "independent_sites": 2,
            "recovery_minutes": 42,
        }
        self.assertTrue(evaluate(ARCHIVE_POLICY, facts))
        self.assertEqual(
            evaluate(ARCHIVE_POLICY, facts), reference(ARCHIVE_POLICY, facts)
        )
        for key, value in (
            ("verified_copies", 2),
            ("independent_sites", 1),
            ("recovery_minutes", 46),
        ):
            changed = {**facts, key: value}
            self.assertFalse(evaluate(ARCHIVE_POLICY, changed))
            self.assertEqual(
                evaluate(ARCHIVE_POLICY, changed), reference(ARCHIVE_POLICY, changed)
            )

    def test_missing_source_stays_absent_despite_neighbor_value(self):
        facts = {
            "verified_copies": 3,
            "independent_sites": 2,
            "recovery_minutes": 40,
        }
        spec = {
            "slug": "unit-missing",
            "policy_id": "archive-readiness",
            "challenge": "missing_source",
            "scene": (
                "A records officer closed the signed restoration packet after comparing "
                "two independently controlled storage locations. A neighboring file was "
                "delivered with it because both sites use the same contractor, but its "
                "time trial belongs to another exercise. The target trial sheet was not "
                "signed before the decision deadline, so only explicit allowed outcomes "
                "from the remaining evidence envelope can be used."
            ),
            "facts": facts,
            "missing_field": "recovery_minutes",
            "admissible_values": [40, 50],
            "documents": [
                doc("target", "verified_copies", 3, 20),
                doc("target", "independent_sites", 2, 21),
                doc("neighbor", "recovery_minutes", 30, 22),
            ],
        }
        prompt, target, proof = build_item(spec, SECRET)
        self.assertFalse(target["answer"]["noul"])
        self.assertEqual(proof["world_outputs"], [True, False])
        parsed = parse_documents(prompt["state"])
        target_case = proof["case_id"]
        self.assertNotIn("recovery_minutes", selected(parsed, target_case)[0])


if __name__ == "__main__":
    unittest.main()
