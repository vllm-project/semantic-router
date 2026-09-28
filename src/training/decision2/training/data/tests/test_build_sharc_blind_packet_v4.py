"""Blind-order and target-isolation tests for the corrected ShARC packet."""

import json
import unittest

from training.data.build_sharc_blind_packet import select_pairs
from training.data.build_sharc_blind_packet_v4 import make_packet


def _row(index: int, label: str) -> dict:
    return {
        "utterance_id": f"source-{index}-{label}",
        "tree_id": f"tree-{index}",
        "source_url": f"https://example.org/source/{index}",
        "snippet": f"Permit rule {index}: a valid permit is required.",
        "question": "May I apply?",
        "scenario": "I have a valid permit." if label == "Yes" else "I lack a permit.",
        "history": [],
        "answer": label,
        "evidence": ["target never copied"],
    }


class SharcV4Tests(unittest.TestCase):
    def test_hidden_id_order_balanced_and_not_index_parity(self) -> None:
        rows = [
            row
            for index in range(26)
            for row in (_row(index, "Yes"), _row(index, "No"))
        ]
        selected, _ = select_pairs(rows, set())
        packet, mapping = make_packet(selected)
        positions = [
            blind["states"][0]["scenario"] == yes["scenario"]
            for blind, (yes, _) in zip(packet["pairs"], selected)
        ]
        self.assertEqual(sum(positions), 12)
        self.assertNotEqual(positions, [index % 2 == 0 for index in range(24)])
        self.assertEqual(
            [row["blind_id"] for row in packet["pairs"]],
            [f"P{index:02d}" for index in range(1, 25)],
        )
        self.assertEqual(len({row["source_url"] for row in mapping["pairs"]}), 24)
        self.assertEqual(packet["schema"], "decision2-sharc-blind-review/4")

    def test_no_target_or_extra_history_fields(self) -> None:
        yes, no = _row(0, "Yes"), _row(0, "No")
        yes["history"] = [
            {
                "follow_up_question": "Do you have a permit?",
                "follow_up_answer": "Yes",
                "answer": "HIDDEN_TARGET",
            }
        ]
        packet, mapping = make_packet([(yes, no), (_row(1, "Yes"), _row(1, "No"))])
        payload = json.dumps(packet)
        self.assertNotIn("HIDDEN_TARGET", payload)
        self.assertNotIn('"answer"', payload)
        self.assertNotIn('"evidence"', payload)
        self.assertNotIn("utterance_id", payload)
        self.assertNotIn("source_url", payload)
        self.assertNotIn("tree_id", payload)
        self.assertNotIn('"answer"', json.dumps(mapping))
        self.assertEqual(len(mapping["pairs"][0]["exact_snippet_sha256"]), 64)

    def test_odd_packet_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "even packet"):
            make_packet([(_row(0, "Yes"), _row(0, "No"))])


if __name__ == "__main__":
    unittest.main()
