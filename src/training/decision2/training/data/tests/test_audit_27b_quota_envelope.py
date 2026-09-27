"""Gold-free synthetic group-capacity tests for the 27B next-arm audit."""

from __future__ import annotations

import unittest

from training.data.audit_27b_quota_envelope import _best_source_counts, evaluate
from training.data.audit_27b_teacher_admission import AdmissionHold


def row(identifier: str, task: str, source: str, group: str) -> dict:
    return {
        "id": identifier,
        "task_type": task,
        "source": source,
        "group_id": group,
        "language": "en",
    }


class QuotaEnvelopeContracts(unittest.TestCase):
    def test_strict_source_quota_can_fail_when_type_total_is_feasible(self) -> None:
        rows = [
            row("s0", "score", "score-source", "score-group"),
            row("s1", "score", "score-source", "score-group"),
            row("a0", "choice", "a", "a-group"),
            row("a1", "choice", "a", "a-group"),
            row("b0", "choice", "b", "b-group"),
            row("b1", "choice", "b", "b-group"),
            row("c0", "choice", "c", "c-group"),
            row("n0", "noul", "n", "n0"),
            row("n1", "noul", "n", "n1"),
            row("n2", "noul", "n", "n2"),
        ]
        result = evaluate(rows, {item["id"]: 10 for item in rows}, total=8, min_score=2)
        self.assertEqual(result["status"], "FEASIBLE_CAPACITY_ONLY")
        self.assertEqual(result["score_admitted_rows"], 2)
        self.assertEqual(result["type_profile"]["choice"]["target"], 3)
        self.assertEqual(result["strict_infeasible_source_buckets"], 2)
        self.assertEqual(result["minimum_l1_source_quota_shift"], 2)
        self.assertEqual(result["groups_spanning_task_types"], 0)
        self.assertNotIn("group_id", result)

    def test_global_group_target_still_fails_closed_when_unreachable(self) -> None:
        groups = {
            "a": [[row("a0", "choice", "a", "g0"), row("a1", "choice", "a", "g0")]],
            "b": [[row("b0", "choice", "b", "g1"), row("b1", "choice", "b", "g1")]],
        }
        with self.assertRaisesRegex(AdmissionHold, "POOLED_WHOLE_GROUP_QUOTA"):
            _best_source_counts(groups, {"a": 1, "b": 2}, 3)

    def test_context_cap_reports_partial_score_group(self) -> None:
        rows = [
            row("s0", "score", "s", "shared"),
            row("s1", "score", "s", "shared"),
            row("c0", "choice", "c", "c0"),
            row("n0", "noul", "n", "n0"),
        ]
        result = evaluate(
            rows,
            {"s0": 10, "s1": 5000, "c0": 10, "n0": 10},
            total=3,
            min_score=1,
        )
        self.assertEqual(result["score_partial_groups_at_context_limit"], 1)
        self.assertEqual(result["admitted_rows"], 3)


if __name__ == "__main__":
    unittest.main()
