"""Contract tests for the private Score checkpoint-selection packet builder."""

from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from training.data import score_three_level_select as subject
from training.data import score_three_level_select_oracle as oracle


class ThreeLevelScoreSelectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.specs, cls.rows = subject.generate(b"a" * 32, b"b" * 32)

    def test_preregistered_groups_and_language_balance(self) -> None:
        self.assertEqual(len(self.specs), 80)
        self.assertEqual(len(self.rows), 240)
        self.assertEqual(
            collections.Counter((s["operation"], s["locale"]) for s in self.specs),
            collections.Counter(
                {
                    (op, locale): count
                    for op in subject.OPS
                    for locale, count in (("en", 16), ("zh", 4))
                }
            ),
        )
        for source in self.specs:
            self.assertEqual(len(source["variants"]), 3)
            self.assertEqual(
                {
                    oracle.score(source["operation"], variant["facts"])
                    for variant in source["variants"]
                },
                {0, 1, 2},
            )
            self.assertEqual(
                len({variant["row_id"] for variant in source["variants"]}), 3
            )

    def test_same_source_counterfactuals_and_no_count_only_quorum(self) -> None:
        for source in self.specs:
            facts = [variant["facts"] for variant in source["variants"]]
            if source["operation"] == "waiver_precedence":
                self.assertEqual(len({item["review_day"] for item in facts}), 1)
                self.assertEqual(len({item["archived_notice"] for item in facts}), 1)
            elif source["operation"] == "inclusive_coverage":
                self.assertEqual(len({tuple(item["request"]) for item in facts}), 1)
                self.assertEqual(
                    len({tuple(item["archived_window"]) for item in facts}), 1
                )
            elif source["operation"] == "independent_quorum":
                self.assertEqual(
                    {
                        sum(report["signed"] for report in item["reports"])
                        for item in facts
                    },
                    {3},
                )
                self.assertEqual(
                    len(
                        {
                            tuple(report["lineage"] for report in item["reports"])
                            for item in facts
                        }
                    ),
                    1,
                )
            else:
                for name in ("first", "second"):
                    self.assertEqual(
                        len(
                            {
                                tuple(
                                    item["pools"][name][field]
                                    for field in (
                                        "capacity",
                                        "committed",
                                        "reserve_floor",
                                    )
                                )
                                for item in facts
                            }
                        ),
                        1,
                    )

    def test_reviewer_packet_is_gold_free_and_deterministic(self) -> None:
        packet = subject._reviewer_packet(self.rows, b"b" * 32)
        self.assertEqual(packet, subject._reviewer_packet(self.rows, b"b" * 32))
        self.assertEqual(len(packet), 240)
        self.assertTrue(
            all(
                "label" not in item and "gold" not in item and "target" not in item
                for item in packet
            )
        )
        self.assertEqual(len({item["review_id"] for item in packet}), 240)
        self.assertEqual(len({item["group_id"] for item in packet}), 80)

    def test_overlap_blocks_a_whole_shared_source(self) -> None:
        row = self.rows[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            path.write_text(
                json.dumps({"id": "reference", "state": row["state"]}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "overlap blocks"):
                subject.audit_overlap(
                    self.rows[:3],
                    [{"role": "reference", "path": str(path), "gold_free": True}],
                )


if __name__ == "__main__":
    unittest.main()
