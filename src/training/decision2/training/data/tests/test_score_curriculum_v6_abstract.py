"""Independent set-algebra checks for the prospective Score v6 source join."""

from __future__ import annotations

import itertools
import unittest

from training.data import score_curriculum_v6_abstract as abstract


class ScoreCurriculumV6AbstractTests(unittest.TestCase):
    def test_all_source_patterns_and_two_source_oracle(self) -> None:
        templates = abstract.eligible_templates()
        self.assertEqual(len(templates), 6)
        for index in range(abstract.GROUP_COUNT):
            group = abstract._group(index, templates)
            self.assertEqual(len(set(group.source_a)), 2)
            self.assertEqual(len(set(group.source_b)), 2)
            for level, (a, b) in enumerate(zip(group.source_a, group.source_b)):
                a_claims = {claim for claim in range(4) if a & (1 << claim)}
                b_claims = {claim for claim in range(4) if b & (1 << claim)}
                self.assertEqual(len(a_claims), 2)
                self.assertEqual(len(b_claims), 2)
                self.assertEqual(len(a_claims & b_claims), level)
                self.assertEqual(abstract.oracle(a, b), level)
                alternatives = [
                    sum(1 << claim for claim in pair)
                    for pair in itertools.combinations(range(4), 2)
                ]
                self.assertEqual(
                    {abstract.oracle(a, replacement) for replacement in alternatives},
                    {0, 1, 2},
                )
                self.assertEqual(
                    {abstract.oracle(replacement, b) for replacement in alternatives},
                    {0, 1, 2},
                )


if __name__ == "__main__":
    unittest.main()
