"""CPU-only tests for prospective ShARC review packet construction."""

import json
import stat
import tempfile
import unittest
from pathlib import Path

from training.data.build_sharc_blind_packet import (
    _write_private,
    make_packet,
    select_pairs,
)


def _row(index: int, label: str, *, url: str | None = None) -> dict:
    return {
        "utterance_id": f"{index:038d}{label.lower()}",
        "tree_id": f"tree-{index}",
        "source_url": url or f"https://example.org/rule/{index}",
        "snippet": f"A person may apply to program {index} with a valid permit.",
        "question": "May I apply?",
        "scenario": f"I {'have' if label == 'Yes' else 'lack'} a valid permit.",
        "history": [],
        "answer": label,
        "evidence": [{"never": "show this"}],
    }


class SharcBlindPacketTests(unittest.TestCase):
    def test_distinct_urls_determinism_and_no_targets(self) -> None:
        rows = [
            row
            for index in range(26)
            for row in (_row(index, "Yes"), _row(index, "No"))
        ]
        selected, aggregate = select_pairs(rows, set())
        reversed_selected, _ = select_pairs(list(reversed(rows)), set())
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 24)
        self.assertEqual(aggregate["selected_source_urls"], 24)
        self.assertEqual(aggregate["eligible_source_urls"], 26)
        packet, mapping = make_packet(selected)
        self.assertEqual(len(packet["pairs"]), 24)
        self.assertEqual(len(mapping["pairs"]), 24)
        self.assertEqual(len(mapping["pairs"][0]["exact_snippet_sha256"]), 64)
        self.assertEqual(len(mapping["pairs"][0]["normalized_snippet_sha256"]), 64)
        self.assertEqual(
            sum(
                blind["states"][0]["scenario"] == pair[0]["scenario"]
                for blind, pair in zip(packet["pairs"], selected)
            ),
            12,
        )
        packet_json = json.dumps(packet).lower()
        self.assertNotIn('"answer"', packet_json)
        self.assertNotIn('"evidence"', packet_json)
        self.assertNotIn("utterance_id", packet_json)
        self.assertNotIn("source_url", packet_json)
        self.assertNotIn("tree_id", packet_json)
        self.assertNotIn("never", packet_json)
        self.assertNotIn('"answer"', json.dumps(mapping))
        self.assertNotIn('"evidence"', json.dumps(mapping))

    def test_exclusion_and_state_difference(self) -> None:
        rows = [_row(i, label) for i in range(24) for label in ("Yes", "No")]
        rows[3]["scenario"] = rows[2]["scenario"]
        rows[3]["history"] = rows[2]["history"]
        excluded = {rows[5]["utterance_id"]}
        with self.assertRaisesRegex(ValueError, "distinct eligible source URLs"):
            select_pairs(rows, excluded)

    def test_source_url_is_sampling_unit(self) -> None:
        rows = []
        for index in range(25):
            url = "https://example.org/shared" if index < 2 else None
            rows.extend((_row(index, "Yes", url=url), _row(index, "No", url=url)))
        selected, aggregate = select_pairs(rows, set())
        self.assertEqual(aggregate["eligible_source_urls"], 24)
        self.assertEqual(len({pair[0]["source_url"] for pair in selected}), 24)

    def test_history_copies_only_publisher_visible_fields(self) -> None:
        yes, no = _row(0, "Yes"), _row(0, "No")
        yes["history"] = [
            {
                "follow_up_question": "Do you have a permit?",
                "follow_up_answer": "Yes",
                "answer": "HIDDEN_LABEL",
            }
        ]
        packet, _ = make_packet([(yes, no)])
        self.assertNotIn("HIDDEN_LABEL", json.dumps(packet))
        self.assertIn("follow_up_answer", json.dumps(packet))

    def test_private_files_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.json"
            digest = _write_private(path, {"pairs": []})
            self.assertEqual(len(digest), 64)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                _write_private(path, {"pairs": []})


if __name__ == "__main__":
    unittest.main()
