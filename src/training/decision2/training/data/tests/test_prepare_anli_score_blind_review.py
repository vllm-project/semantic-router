"""Synthetic-only tests for the private ANLI Score blind-review packet."""

from __future__ import annotations

import json
import unittest

from training.data.audit_anli_score_diagnostic import normalize
from training.data.audit_anli_score_train_source import TrainRow
from training.data.prepare_anli_score_blind_review import build_packet


def source_rows(groups_per_round: int = 3) -> list[TrainRow]:
    rows = []
    for round_id in (1, 2, 3):
        for group_index in range(groups_per_round):
            premise = f"Synthetic evidence for round {round_id}, group {group_index}."
            for label in (0, 1, 2):
                rows.append(
                    TrainRow(
                        round=round_id,
                        position=group_index * 3 + label,
                        premise=premise,
                        hypothesis=(
                            f"A synthetic claim in round {round_id}, group "
                            f"{group_index}, relation {label}."
                        ),
                        label=label,
                        reason_present=False,
                    )
                )
    return rows


class BlindReviewPacketTests(unittest.TestCase):
    def test_group_complete_native_inputs_and_separate_answer_key(self) -> None:
        rows = source_rows()
        packet, key, counts = build_packet(rows, rows, set())
        self.assertEqual(len(packet["groups"]), 9)
        self.assertEqual(sum(counts[f"r{n}_rows"] for n in (1, 2, 3)), 27)
        self.assertEqual(len(key["answers"]), 27)
        self.assertNotIn("mapped_native_score", json.dumps(packet))
        self.assertNotIn('"label"', json.dumps(packet))
        self.assertNotIn('"reason"', json.dumps(packet))
        self.assertEqual(
            {answer["mapped_native_score"] for answer in key["answers"]},
            {0, 1, 2},
        )
        for group in packet["groups"]:
            self.assertEqual(len(group["items"]), 3)
            for item in group["items"]:
                self.assertEqual(
                    set(item["request"]),
                    {"id", "state", "instructions", "options", "task_type"},
                )
                self.assertEqual(item["request"]["task_type"], "score")
                self.assertEqual(
                    [choice["key"] for choice in item["request"]["options"]],
                    ["0", "1", "2"],
                )
        self.assertEqual((packet, key, counts), build_packet(rows, rows, set()))

    def test_open_dev_collision_quarantines_entire_group(self) -> None:
        rows = source_rows(4)
        group = rows[0].group
        packet, _, counts = build_packet(rows, rows, {normalize(rows[1].hypothesis)})
        self.assertEqual(sum(counts[f"r{n}_rows"] for n in (1, 2, 3)), 27)
        self.assertFalse(
            any(
                group == normalize(item["request"]["state"]["evidence"])
                for group_items in packet["groups"]
                for item in group_items["items"]
            )
        )

    def test_conflicting_source_pair_is_not_reviewed(self) -> None:
        rows = source_rows(4)
        conflicting = TrainRow(1, 100, rows[0].premise, rows[0].hypothesis, 2, False)
        packet, _, _ = build_packet(rows, rows + [conflicting], set())
        self.assertFalse(
            any(
                rows[0].group == normalize(item["request"]["state"]["evidence"])
                for group_items in packet["groups"]
                for item in group_items["items"]
            )
        )

    def test_missing_three_relation_groups_fails_closed(self) -> None:
        rows = source_rows(2)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            build_packet(rows, rows, set())


if __name__ == "__main__":
    unittest.main()
