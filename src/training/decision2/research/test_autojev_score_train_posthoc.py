"""CPU-only fixed-artifact slice contracts."""

from __future__ import annotations

import unittest

from research.autojev_score_train_posthoc import slice_metrics
from research.test_autojev_score_train_audit import score_row


class PosthocSliceTest(unittest.TestCase):
    def test_language_and_level_groups_reuse_sealed_probabilities(self) -> None:
        rows = [score_row(0), score_row(1)]
        rows[1]["language"] = "zh"
        rows[1]["options"] = [
            {"key": str(i), "description": f"level {i}"} for i in range(5)
        ]
        outputs = [
            {
                "id": row["id"],
                "input_sha256": row["input_sha256"],
                "group_id": row["group_id"],
                "source": row["source"],
                "family": row["family"],
                "level_count": len(row["options"]),
                "probabilities": {
                    str(i): 1.0 / len(row["options"])
                    for i in range(len(row["options"]))
                },
            }
            for row in rows
        ]
        stats = slice_metrics(rows, outputs)
        self.assertEqual(stats["en|3"]["n"], 1)
        self.assertEqual(stats["zh|4-8"]["n"], 1)
        self.assertEqual(stats["all|all"]["n"], 2)
        outputs[0]["input_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity"):
            slice_metrics(rows, outputs)


if __name__ == "__main__":
    unittest.main()
