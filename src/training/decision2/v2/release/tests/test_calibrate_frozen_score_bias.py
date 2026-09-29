"""calibrate_frozen --score-bias: CAL Score logits get the package's offsets (stdlib)."""

from __future__ import annotations

import unittest

from v2.release.calibrate_frozen import with_score_bias


class ScoreBiasCalTest(unittest.TestCase):
    def test_only_score_rows_with_an_offset_row_change(self):
        records = [
            {
                "id": "a",
                "task_type": "score",
                "label": 1,
                "logits": [0.0, 1.0, 2.0, 3.0, 4.0],
            },
            {"id": "b", "task_type": "score", "label": 0, "logits": [0.0, 1.0, 2.0]},
            {
                "id": "c",
                "task_type": "choice",
                "label": 0,
                "logits": [0.0, 1.0, 2.0, 3.0, 4.0],
            },
        ]
        biased = with_score_bias(records, {5: [0.5, 0.0, 0.0, 0.0, -0.5]})
        self.assertEqual(biased[0]["logits"], [0.5, 1.0, 2.0, 3.0, 3.5])
        self.assertEqual(biased[1:], records[1:])
        self.assertEqual(records[0]["logits"], [0.0, 1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
