from __future__ import annotations

import importlib.util
import unittest

from external_index021.score import acos_review_f1
from external_index021.selection import canonical, select


def home_row(name: str, state: dict) -> dict:
    return {
        "id": name,
        "state": state,
        "questions": {"a": {"type": "choice"}},
        "expected": {"a": "yes"},
        "_evaluation": {"run_id": name, "catalog_id": 9, "group_id": name},
    }


def retrieval_row(number: int, name: str, group: str, relevant: bool) -> dict:
    return {
        "id": name,
        "_evaluation": {"run_id": name, "catalog_id": number, "group_id": group},
        "scoring": {
            "retrieved_ids": ["doc"],
            "scorable_ids": ["doc"],
            "qrels": {"doc": int(relevant)},
        },
    }


class SelectionTests(unittest.TestCase):
    def test_answerable_retrieval_and_provisional_home_choice(self) -> None:
        rows = [
            retrieval_row(2, "tool-good", "tool-A", True),
            retrieval_row(2, "tool-bad", "tool-B", False),
            retrieval_row(36, "bright-good", "bright-A", True),
            retrieval_row(36, "bright-bad", "bright-B", False),
        ]
        a = home_row("h-a", {"request": "keep"})
        b = home_row("h-b", {"request": "keep"})
        dev = home_row("h-dev-overlap", {"request": "seen in dev"})
        rows.extend((a, b, dev))
        ref = ({r["id"]: r for r in (a, b, dev)}, {canonical(dev["state"])})
        first = select(
            rows, home_policy="first", home_reference=ref, require_full_counts=False
        )
        last = select(
            rows, home_policy="last", home_reference=ref, require_full_counts=False
        )
        self.assertEqual(first.scoreable, 3)
        self.assertEqual(
            first.dropped,
            {"toolret": 1, "bright": 1, "home_dev_overlap": 1, "home_duplicate": 1},
        )
        self.assertIn("h-a", first.keep_run_ids)
        self.assertIn("h-b", last.keep_run_ids)
        self.assertNotEqual(first.keep_ids_sha256, last.keep_ids_sha256)
        self.assertEqual(first.status, "provisional_home_copy")
        explicit = select(
            rows,
            home_policy="explicit",
            home_keep_run_ids={"h-b"},
            home_reference=ref,
            require_full_counts=False,
        )
        self.assertEqual(explicit.keep_run_ids, last.keep_run_ids)

    def test_inconsistent_retrieval_chunks_rejected(self) -> None:
        a = retrieval_row(2, "a", "g", True)
        b = retrieval_row(2, "b", "g", False)
        with self.assertRaisesRegex(ValueError, "disagree"):
            select([a, b], home_reference=({}, set()), require_full_counts=False)

    def test_acos_review_f1_and_missing_review(self) -> None:
        def row(rid, group, gold):
            return {
                "_evaluation": {"run_id": rid, "group_id": group, "catalog_id": 38},
                "questions": {"q": {"type": "choice"}},
                "expected": {"q": gold},
            }

        rows = [row("correct", "review-1", "yes"), row("missing", "review-2", "yes")]
        results = {
            "correct": {
                "status": "ok",
                "response": {"answers": {"q": {"choice": "yes"}}},
            }
        }
        out = acos_review_f1(rows, results)
        self.assertEqual(out["raw"], 0.5)
        self.assertEqual(out["coverage"], 0.5)

    @unittest.skipUnless(
        importlib.util.find_spec("decision_index"), "pinned public kit is not installed"
    )
    def test_pinned_home_generator_yields_published_48_plus_24_drops(self) -> None:
        from decision_index.suite.build.home_appliance import make_row

        rows = []
        for household in range(8, 24):
            for offset in range(10):
                row = make_row(household, offset)
                row["_evaluation"] = {
                    "catalog_id": 9,
                    "group_id": row["id"],
                    "run_id": "9:Home-Appliance:" + row["id"],
                }
                rows.append(row)
        first = select(rows, home_policy="first", require_full_counts=False)
        last = select(rows, home_policy="last", require_full_counts=False)
        self.assertEqual(first.scoreable, 88)
        self.assertEqual(first.dropped["home_dev_overlap"], 48)
        self.assertEqual(first.dropped["home_duplicate"], 24)
        self.assertEqual(len(first.keep_run_ids ^ last.keep_run_ids), 48)


if __name__ == "__main__":
    unittest.main()
