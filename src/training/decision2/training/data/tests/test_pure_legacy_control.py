from __future__ import annotations

import unittest

from training.data import build_pure_legacy_control as control


def row(row_id: str, task_type: str, language: str) -> dict:
    return {"id": row_id, "task_type": task_type, "language": language}


class PureLegacyControlTests(unittest.TestCase):
    def test_matches_joint_strata_and_exact_total_when_available(self) -> None:
        targets = [
            row("target-a", "choice", "en"),
            row("target-b", "choice", "en"),
            row("target-c", "noul", "zh"),
        ]
        candidates = [
            row("old-a", "choice", "en"),
            row("old-b", "choice", "en"),
            row("old-c", "choice", "en"),
            row("old-d", "noul", "zh"),
            row("old-e", "noul", "zh"),
        ]
        lengths = {
            "target-a": 100,
            "target-b": 110,
            "target-c": 50,
            "old-a": 100,
            "old-b": 110,
            "old-c": 1000,
            "old-d": 50,
            "old-e": 500,
        }
        selected, receipt = control.choose_replacements(
            targets, candidates, lengths, "stable"
        )
        self.assertEqual({x["id"] for x in selected}, {"old-a", "old-b", "old-d"})
        self.assertEqual(
            receipt["target_replacement_tokens"], receipt["selected_replacement_tokens"]
        )
        self.assertEqual(receipt["unmatchable_strata"], [])
        self.assertTrue(all(x["row_count_difference"] == 0 for x in receipt["strata"]))

    def test_unavailable_joint_stratum_fails_closed(self) -> None:
        targets = [row("target", "score", "zh")]
        candidates = [row("old", "choice", "zh")]
        lengths = {"target": 100, "old": 100}
        with self.assertRaisesRegex(ValueError, "Unmatchable task/language strata"):
            control.choose_replacements(targets, candidates, lengths, "stable")


if __name__ == "__main__":
    unittest.main()
