"""Group isolation and no-truncation tests for the prospective 9B subset."""

from __future__ import annotations

import unittest

from training.data.filter_native_length import choose_groups


class NativeLengthFilterTests(unittest.TestCase):
    def test_one_long_variant_rejects_its_complete_group(self) -> None:
        rows = [
            {"id": "a:0", "group_id": "a", "task_type": "score", "language": "en"},
            {"id": "a:1", "group_id": "a", "task_type": "score", "language": "en"},
            {"id": "b:0", "group_id": "b", "task_type": "choice", "language": "zh"},
        ]
        kept, summary = choose_groups(rows, [100, 4100, 4096], 4096)
        self.assertEqual([row["id"] for row in kept], ["b:0"])
        self.assertEqual(summary["overlength_rows"], 1)
        self.assertEqual(summary["removed_rows"], 2)
        self.assertEqual(summary["removed_groups"], 1)
        self.assertEqual(summary["retained_native_tokens"], 4096)
        self.assertEqual(summary["retained_by_type"], {"choice": 1})

    def test_empty_and_misaligned_input_rejected(self) -> None:
        row = {"id": "a", "group_id": "a", "task_type": "choice", "language": "en"}
        with self.assertRaises(ValueError):
            choose_groups([row], [4097], 4096)
        with self.assertRaises(ValueError):
            choose_groups([row], [], 4096)
        with self.assertRaises(ValueError):
            choose_groups([row], [0], 4096)


if __name__ == "__main__":
    unittest.main()
