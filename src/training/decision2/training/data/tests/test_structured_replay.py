from __future__ import annotations

import unittest

from training.data.build_structured_replay import filter_groups


def row(
    item_id: str,
    group: str,
    state: str,
    *,
    source: str = "legacy:stage4-general-composition-v2",
) -> dict:
    return {
        "id": item_id,
        "group_id": group,
        "state": state,
        "input_sha256": "hash-" + item_id,
        "family": "stage4_scope",
        "source": source,
    }


class StructuredReplayTests(unittest.TestCase):
    def test_whole_group_quarantine_and_gold_free_context(self) -> None:
        source = [
            row("a1", "a", "first pair"),
            row("a2", "a", "reserved context"),
            row("b1", "b", "near neighbor"),
            row("c1", "c", "safe text"),
            row("d1", "d", "unreviewed source", source="legacy:nyu-mll/multi_nli"),
        ]
        protected = [{"id": "heldout-1", "state": "reserved context"}]

        def near_search(
            left: list[dict], right: list[dict], *, collect_left_ids: bool
        ) -> dict:
            self.assertTrue(collect_left_ids)
            self.assertEqual({row["id"] for row in left}, {"b1", "c1"})
            self.assertEqual([row["state"] for row in right], ["reserved context"])
            return {"count": 1, "examples": [], "left_ids": ["b1"], "method": "fixture"}

        selected, audit = filter_groups(
            source, protected, near_duplicate_search=near_search
        )
        self.assertEqual([row["id"] for row in selected], ["c1"])
        self.assertEqual(
            audit["rejected_rows_by_reason"],
            {"exact_context": 2, "near_context": 1, "outside_structured_source": 1},
        )
        self.assertEqual(audit["selected_groups"], 1)

    def test_exact_id_and_input_reject_complete_group(self) -> None:
        source = [
            row("same", "group1", "different text"),
            row("neighbor", "group1", "other text"),
            row("unique", "group2", "safe"),
        ]
        protected = [
            {
                "id": "same",
                "group_id": "other",
                "input_sha256": "other-hash",
                "state": "heldout",
            }
        ]

        def no_near(
            left: list[dict], right: list[dict], *, collect_left_ids: bool
        ) -> dict:
            return {"count": 0, "examples": [], "left_ids": [], "method": "fixture"}

        selected, audit = filter_groups(
            source, protected, near_duplicate_search=no_near
        )
        self.assertEqual([row["id"] for row in selected], ["unique"])
        self.assertEqual(audit["rejected_rows_by_reason"]["id_group_or_input"], 2)


if __name__ == "__main__":
    unittest.main()
