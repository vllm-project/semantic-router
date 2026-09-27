"""Synthetic contracts for the distinct 27B pooled-quota schedule candidate."""

from __future__ import annotations

import unittest

from training.data.audit_27b_teacher_admission import AdmissionHold
from training.data.plan_27b_pooled_schedule import plan


def row(identifier: str, task: str, source: str, group: str) -> dict:
    return {
        "id": identifier,
        "input_sha256": identifier,
        "task_type": task,
        "source": source,
        "group_id": group,
        "language": "en",
    }


class PooledScheduleContracts(unittest.TestCase):
    def test_minimal_source_shift_and_stable_complete_groups(self) -> None:
        rows = [
            row("s0", "score", "s", "sg"),
            row("s1", "score", "s", "sg"),
            row("a0", "choice", "a", "ag0"),
            row("a1", "choice", "a", "ag0"),
            row("a2", "choice", "a", "ag1"),
            row("a3", "choice", "a", "ag1"),
            row("b0", "choice", "b", "bg0"),
            row("b1", "choice", "b", "bg1"),
            row("n0", "noul", "n", "ng0"),
            row("n1", "noul", "n", "ng1"),
        ]
        lengths = {item["id"]: 9 for item in rows}
        result = plan(rows, lengths, total=6, min_score=2)
        reverse = plan(list(reversed(rows)), lengths, total=6, min_score=2)
        self.assertEqual(result["status"], "SCHEDULE_CANDIDATE_ONLY")
        self.assertEqual(result["by_type"], {"choice": 2, "noul": 2, "score": 2})
        self.assertEqual(result["source_quotas"]["choice"], {"a": 1, "b": 1})
        self.assertEqual(result["source_l1_total"], 2)
        self.assertEqual(
            result["source_realized"]["choice"], reverse["source_realized"]["choice"]
        )
        self.assertEqual(result["schedule_sha256"], reverse["schedule_sha256"])
        self.assertEqual(result["raw_token_exposure"], 54)
        self.assertEqual(result["padded_token_exposure"], 96)
        ids = {item["id"] for item in result["schedule"]}
        for left, right in (("a0", "a1"), ("a2", "a3"), ("s0", "s1")):
            self.assertEqual(left in ids, right in ids)

    def test_partial_score_group_fails_before_schedule(self) -> None:
        rows = [
            row("s0", "score", "s", "sg"),
            row("s1", "score", "s", "sg"),
            row("c0", "choice", "c", "cg"),
            row("n0", "noul", "n", "ng"),
        ]
        with self.assertRaisesRegex(AdmissionHold, "PARTIAL_SCORE_GROUP"):
            plan(
                rows,
                {"s0": 9, "s1": 5000, "c0": 9, "n0": 9},
                total=3,
                min_score=1,
            )

    def test_unreachable_type_total_stays_hold(self) -> None:
        rows = [
            row("s0", "score", "s", "sg"),
            row("s1", "score", "s", "sg"),
            row("c0", "choice", "c", "cg"),
            row("c1", "choice", "c", "cg"),
            row("n0", "noul", "n", "ng"),
            row("n1", "noul", "n", "ng"),
        ]
        with self.assertRaisesRegex(AdmissionHold, "POOLED_WHOLE_GROUP_QUOTA"):
            plan(rows, {item["id"]: 9 for item in rows}, total=4, min_score=2)


if __name__ == "__main__":
    unittest.main()
