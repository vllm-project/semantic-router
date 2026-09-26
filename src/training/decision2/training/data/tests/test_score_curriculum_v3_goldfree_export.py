"""Privacy checks for the full Score v3 TRAIN overlap projection."""

from __future__ import annotations

import json
import unittest

from training.data import score_curriculum_v3_goldfree_export as exporter


class ScoreV3GoldfreeExportTests(unittest.TestCase):
    def test_projection_omits_gold_and_original_identifiers(self) -> None:
        rows = [
            {
                "id": f"private-source-{group}-{label}",
                "group_id": f"private-group-{group}",
                "family": "score_weighted_points",
                "language": "en",
                "state": {"case": f"case-{group}", "marks": [label]},
                "instructions": "Compute the weighted result.",
                "options": [{"label": "A", "text": "low"}],
                "label": label,
                "source": "private-lineage",
            }
            for group in range(320)
            for label in range(3)
        ]
        projected = exporter.project(rows, b"x" * 32)
        self.assertEqual(len(projected), 960)
        self.assertEqual(len({row["review_id"] for row in projected}), 960)
        self.assertEqual(len({row["group_id"] for row in projected}), 320)
        self.assertTrue(
            all(set(row) == set(exporter.PUBLIC_FIELDS) for row in projected)
        )
        serialized = json.dumps(projected, ensure_ascii=False)
        self.assertNotIn("private-source-", serialized)
        self.assertNotIn("private-group-", serialized)
        self.assertNotIn("private-lineage", serialized)
        self.assertNotIn('"label": 0', serialized)
        self.assertEqual(projected, exporter.project(rows, b"x" * 32))


if __name__ == "__main__":
    unittest.main()
