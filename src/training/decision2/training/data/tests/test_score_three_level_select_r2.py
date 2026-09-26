"""Independent construction and shortcut checks for Score SELECT r2."""

from __future__ import annotations

import collections
import copy
import unittest

from training.data import score_three_level_select_r2 as candidate
from training.data import score_three_level_select_r2_oracle as oracle
from training.data import score_three_level_select_r2_render_oracle as visible_oracle


class ThreeLevelScoreSelectR2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources, cls.rows = candidate.generate(
            b"unit-seed".ljust(32, b"x"), b"unit-salt".ljust(32, b"y")
        )

    def test_groups_languages_and_visible_answer_agree(self) -> None:
        self.assertEqual((len(self.sources), len(self.rows)), (80, 240))
        self.assertEqual(
            collections.Counter(
                (source["operation"], source["locale"]) for source in self.sources
            ),
            {(operation, "en"): 16 for operation in candidate.OPS}
            | {(operation, "zh"): 4 for operation in candidate.OPS},
        )
        by_id = {row["id"]: row for row in self.rows}
        for source in self.sources:
            self.assertEqual(len(source["variants"]), 3)
            self.assertEqual(
                {
                    oracle.score(source["operation"], variant["facts"])
                    for variant in source["variants"]
                },
                {0, 1, 2},
            )
            for variant in source["variants"]:
                row = by_id[variant["row_id"]]
                self.assertEqual(
                    visible_oracle.score(
                        source["operation"], row["state"], source["locale"]
                    ),
                    row["label"],
                )

    def test_allocation_and_quorum_shortcut_invariants(self) -> None:
        audit = candidate.shortcut_audit(self.sources)
        self.assertEqual(set(audit), set(candidate.OPS))
        self.assertTrue(
            all(result["best_shallow_correct"] <= 40 for result in audit.values())
        )
        for source in self.sources:
            if source["operation"] == "allocation_caps":
                facts = [variant["facts"] for variant in source["variants"]]
                for pool in ("first", "second"):
                    margins = {
                        item["pools"][pool]["capacity"]
                        - item["pools"][pool]["committed"]
                        - item["pools"][pool]["request"]
                        - item["pools"][pool]["reserve_floor"]
                        for item in facts
                    }
                    self.assertEqual(len(margins), 2)
                    self.assertTrue(any(value < 0 for value in margins))
                    self.assertTrue(any(value >= 0 for value in margins))
            elif source["operation"] == "independent_quorum":
                for variant in source["variants"]:
                    self.assertEqual(
                        sum(item["signed"] for item in variant["facts"]["reports"]), 3
                    )

    def test_r1_three_margin_shortcut_is_blocked(self) -> None:
        altered = copy.deepcopy(self.sources)
        source = next(
            item for item in altered if item["operation"] == "allocation_caps"
        )
        facts_by_level = {
            oracle.score("allocation_caps", variant["facts"]): variant["facts"]
            for variant in source["variants"]
        }
        # The r1 defect: one pool takes negative, zero and positive margins.
        first = [facts_by_level[level]["pools"]["first"] for level in range(3)]
        for margin, pool in zip((-1, 0, 1), first):
            pool["request"] = (
                pool["capacity"] - pool["committed"] - pool["reserve_floor"] - margin
            )
        with self.assertRaisesRegex(ValueError, "one-field three-level shortcut"):
            candidate.shortcut_audit(altered)


if __name__ == "__main__":
    unittest.main()
