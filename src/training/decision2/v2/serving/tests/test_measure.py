from __future__ import annotations

import unittest

from v2.serving.measure import (
    PanelTally,
    latency_summary,
    payload_sha256,
    quantile,
    slot_drifts,
)

PROMPT = {
    "id": "p1",
    "state": "s",
    "questions": {"a": {"type": "noul"}, "b": {"type": "choice"}},
}


def stored(answers):
    return {
        "id": "p1",
        "answers": answers,
        "source_input_sha256": payload_sha256(PROMPT),
    }


class MeasureTest(unittest.TestCase):
    def test_nearest_rank_quantiles(self) -> None:
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        self.assertEqual(quantile(values, 0.5), 3.0)
        self.assertEqual(quantile(values, 0.95), 5.0)
        self.assertIsNone(quantile([], 0.5))
        summary = latency_summary([0.001, 0.002, 0.003])
        self.assertEqual((summary["n"], summary["p50_ms"]), (3, 2.0))

    def test_slot_drift_separates_probabilities_and_score_values(self) -> None:
        left = {
            "a": {"type": "noul", "noul": 0.61},
            "s": {
                "type": "score",
                "score": 1.2,
                "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
            },
        }
        right = {
            "a": {"type": "noul", "noul": 0.40},
            "s": {
                "type": "score",
                "score": 1.0,
                "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
            },
        }
        rows = {row["qid"]: row for row in slot_drifts(left, right)}
        self.assertTrue(rows["a"]["changed"])
        self.assertAlmostEqual(rows["a"]["prob_drift"], 0.21)
        self.assertIsNone(rows["a"]["score_drift"])
        self.assertAlmostEqual(rows["s"]["prob_drift"], 0.15)
        self.assertAlmostEqual(rows["s"]["score_drift"], 0.2)

    def test_panel_tally_counts_changes_errors_and_input_mismatch(self) -> None:
        tally = PanelTally("t")
        same = {
            "a": {"type": "noul", "noul": 0.9},
            "b": {
                "type": "choice",
                "choice": "x",
                "probabilities": {"x": 0.7, "y": 0.3},
            },
        }
        moved = {"a": {"type": "noul", "noul": 0.2}, "b": same["b"]}
        tally.add(PROMPT, {"answers": same}, stored(same))
        tally.add(PROMPT, {"answers": moved}, stored(same))
        tally.add(PROMPT, None, stored(same))
        tally.add(
            PROMPT, {"answers": same}, {**stored(same), "source_input_sha256": "0" * 64}
        )
        summary = tally.summary()
        self.assertEqual(summary["prompts"], 4)
        self.assertEqual(summary["category_changes"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["missing"], 2)
        self.assertEqual(summary["input_mismatch"], 1)
        self.assertEqual(summary["changed_by_type"], {"noul": 1})
        self.assertAlmostEqual(summary["prob_drift"]["max"], 0.7)


if __name__ == "__main__":
    unittest.main()
