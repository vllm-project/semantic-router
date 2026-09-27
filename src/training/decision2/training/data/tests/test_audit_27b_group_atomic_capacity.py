"""Synthetic near-evidence and whole-group capacity contracts."""

from __future__ import annotations

import unittest

from training.data.audit_27b_group_atomic_capacity import (
    choose_group_atomic_witness,
    state_near_ids,
)


def row(identifier: str, group: str, task: str, source: str) -> dict:
    keys = ["0", "1", "2"] if task == "score" else ["0", "1"]
    return {
        "id": identifier,
        "source": source,
        "group_id": group,
        "task_type": task,
        "input_sha256": identifier + "-input",
        "state": "Unique case state " + identifier,
        "options": [{"key": key} for key in keys],
        "label": 0,
    }


class GroupAtomicContracts(unittest.TestCase):
    def test_near_state_id_filter_is_not_generic_instruction_match(self) -> None:
        train = [
            {
                "id": "a",
                "state": "The shipment for item A arrived Monday after approval.",
            },
            {"id": "b", "state": "The unrelated invoice B was rejected Friday."},
        ]
        protected = [
            {
                "id": "p",
                "state": "The shipment for item A arrived Monday after approval.",
            }
        ]
        matched, pairs = state_near_ids(train, protected)
        self.assertEqual(matched, {"a"})
        self.assertEqual(pairs, {("a", "p")})

    def test_exact_budget_witness_never_splits_cross_type_group(self) -> None:
        score_source = "decision2_programmatic_original_v1"
        human_source = "google_goemotions_official_train"
        train = [row(f"s{i}", f"sg{i}", "score", score_source) for i in range(460)]
        for i in range(100):
            train.append(row(f"c{i}", f"hg{i}", "choice", human_source))
            train.append(row(f"n{i}", f"hg{i}", "noul", human_source))
        train.extend(
            row(f"single{i}", f"single{i}", "choice", human_source) for i in range(2000)
        )
        lengths = {item["id"]: 500 for item in train}
        vectors = {
            item["id"]: {
                option["key"]: float(option["key"] == "0") for option in item["options"]
            }
            for item in train
        }
        report = choose_group_atomic_witness(
            train,
            lengths,
            set(),
            vectors,
            total=2560,
            target_raw_tokens=1_280_000,
            target_padded_tokens=1_280_000,
            seeds=1,
        )
        self.assertEqual(report["status"], "PASS_CAPACITY_WITNESS_ONLY")
        self.assertEqual(report["summary"]["rows"], 2560)
        selected = {item["id"] for item in report["private_schedule"]}
        for i in range(100):
            self.assertEqual(f"c{i}" in selected, f"n{i}" in selected)
        self.assertGreaterEqual(
            report["summary"]["teacher_mask"]["score_by_level_count"]["3"], 40
        )
        excluded = choose_group_atomic_witness(
            train,
            lengths,
            {"c0"},
            vectors,
            total=2560,
            target_raw_tokens=1_280_000,
            target_padded_tokens=1_280_000,
            seeds=1,
        )
        self.assertEqual(excluded["status"], "PASS_CAPACITY_WITNESS_ONLY")
        self.assertEqual(excluded["summary"]["near_excluded_groups"], 1)
        excluded_selected = {item["id"] for item in excluded["private_schedule"]}
        self.assertFalse({"c0", "n0"} & excluded_selected)

        insufficient_score = choose_group_atomic_witness(
            train,
            lengths,
            {"s0"},
            vectors,
            total=2560,
            target_raw_tokens=1_280_000,
            target_padded_tokens=1_280_000,
            seeds=1,
        )
        self.assertEqual(insufficient_score["status"], "HOLD_CAPACITY_UPPER_BOUND")
        self.assertEqual(insufficient_score["summary"]["eligible_score_rows"], 459)


if __name__ == "__main__":
    unittest.main()
