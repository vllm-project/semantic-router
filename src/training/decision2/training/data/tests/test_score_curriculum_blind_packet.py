"""Gold-free Score curriculum packet contract."""

from __future__ import annotations

import collections
import unittest

from training.data import build_score_curriculum as curriculum
from training.data import score_curriculum_blind_packet as blind


class ScoreCurriculumBlindPacketTests(unittest.TestCase):
    def test_deterministic_complete_gold_free_packet(self) -> None:
        rows = curriculum.generate()
        first, groups = blind.select(rows)
        second, second_groups = blind.select(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(groups, second_groups)
        self.assertEqual(len(first), 144)
        self.assertEqual(len(groups), 48)
        self.assertEqual(
            collections.Counter(row["family"] for row in first),
            {f"score_{family}": 36 for family in curriculum.FAMILIES},
        )
        self.assertTrue(all(set(row) == set(blind.PACKET_FIELDS) for row in first))
        self.assertTrue(
            all("label" not in row and "audit_metadata" not in row for row in first)
        )


if __name__ == "__main__":
    unittest.main()
