"""Synthetic-only tests for frozen ANLI TRAIN quarantine accounting."""

from __future__ import annotations

import json
import unittest

from training.data.audit_anli_score_diagnostic import normalize
from training.data.audit_anli_score_train_quarantine import quarantine_profile
from training.data.audit_anli_score_train_source import TrainRow


def row(position: int, premise: str, hypothesis: str, label: int) -> TrainRow:
    return TrainRow(1, position, premise, hypothesis, label, False)


class AnliTrainQuarantineTests(unittest.TestCase):
    def test_exact_dev_and_conflict_quarantine_whole_selected_groups(self) -> None:
        dev_premise = "A distinctive synthetic premise for the dev collision."
        conflict_premise = "A distinct synthetic premise for conflicting labels."
        clean_premise = "A unique synthetic clean premise."
        source = [
            row(0, dev_premise, "A claim repeated in open development.", 0),
            row(1, dev_premise, "A different claim in the same group.", 2),
            row(2, conflict_premise, "An identical claim with a conflict.", 0),
            row(3, conflict_premise, "An identical claim with a conflict.", 2),
            row(4, clean_premise, "A unique clean claim.", 1),
        ]
        result = quarantine_profile(
            source, source, {normalize("A claim repeated in open development.")}
        )
        self.assertEqual(result["selected_rows"], 5)
        self.assertEqual(result["selected_groups"], 3)
        self.assertEqual(result["selected_exact_dev_affected_rows"], 2)
        self.assertEqual(result["selected_source_conflict_rows"], 2)
        self.assertEqual(
            result["fixed_sample_after_exact_dev_and_conflict_quarantine_upper_bound"],
            {"rows": 1, "groups": 1},
        )
        self.assertEqual(result["actual_admitted_rows"], 0)
        self.assertNotIn(dev_premise, json.dumps(result))
        self.assertNotIn(conflict_premise, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
