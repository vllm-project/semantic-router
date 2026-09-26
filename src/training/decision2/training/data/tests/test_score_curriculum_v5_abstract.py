"""Check Score v5 abstract arithmetic without rendering candidate examples."""

from __future__ import annotations

import unittest

from training.data import score_curriculum_v5_abstract as abstract


class ScoreCurriculumV5AbstractTests(unittest.TestCase):
    def test_recompute_selected_weighted_bands_independently(self) -> None:
        for index in range(abstract.GROUPS_PER_FAMILY):
            group = abstract._group(index)
            if group is None:
                continue
            totals = [
                sum(weight * mark for weight, mark in zip(group.weights, plan.marks))
                for plan in group.plans
            ]
            self.assertEqual(totals, [plan.total for plan in group.plans])
            self.assertLess(totals[0], totals[1])
            self.assertLess(totals[1], totals[2])
            lower, upper = totals[1], totals[2]
            self.assertEqual(
                [0 if total < lower else 1 if total < upper else 2 for total in totals],
                [0, 1, 2],
            )
            self.assertEqual(
                [sum(plan.marks) for plan in group.plans],
                [sum(group.plans[0].marks)] * 3,
            )
            self.assertEqual(set(group.target_positions), {0, 1, 2})
            self.assertTrue(
                all(
                    len({plan.marks[position] for plan in group.plans}) <= 2
                    for position in range(5)
                )
            )


if __name__ == "__main__":
    unittest.main()
