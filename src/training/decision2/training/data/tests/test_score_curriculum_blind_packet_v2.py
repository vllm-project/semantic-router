"""Blind packet aliases must not disclose TRAIN answer IDs."""

from __future__ import annotations

import argparse
import collections
import tempfile
import unittest
from pathlib import Path

from training.data import build_pilot as pilot
from training.data import build_score_curriculum as curriculum
from training.data import score_curriculum_blind_packet_v2 as blind


class ScoreCurriculumBlindPacketV2Tests(unittest.TestCase):
    def test_aliases_hide_source_ids_and_preserve_complete_groups(self) -> None:
        rows = curriculum.generate()
        first, mapping, groups = blind.select(rows, b"a" * 32)
        repeat, repeat_map, repeat_groups = blind.select(
            list(reversed(rows)), b"a" * 32
        )
        other, _, _ = blind.select(rows, b"b" * 32)
        self.assertEqual((first, mapping, groups), (repeat, repeat_map, repeat_groups))
        self.assertNotEqual(
            [row["review_id"] for row in first],
            [row["review_id"] for row in other],
        )
        self.assertEqual(len(first), 144)
        self.assertEqual(len(groups), 48)
        self.assertEqual(len({row["review_id"] for row in first}), 144)
        self.assertEqual(
            collections.Counter(row["group_id"] for row in first),
            dict.fromkeys(groups, 3),
        )
        self.assertTrue(all(set(row) == set(blind.PUBLIC_FIELDS) for row in first))
        self.assertTrue(all(row["review_id"].startswith("sbr-") for row in first))
        self.assertTrue(
            all("source_id" not in row and "label" not in row for row in first)
        )
        self.assertEqual(
            {row["review_id"] for row in first},
            {row["review_id"] for row in mapping},
        )
        self.assertTrue(
            all(
                row["source_id"].rsplit("_", 1)[-1] in {"0", "1", "2"}
                for row in mapping
            )
        )

    def test_rejects_missing_private_salt(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-byte"):
            blind.select([], b"short")

    def test_build_keeps_public_packet_separate_from_private_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            train.write_bytes(pilot.jsonl_bytes(curriculum.generate()))
            salt = root / "private-salt.bin"
            salt.write_bytes(b"s" * 32)
            packet_dir = root / "review"
            private_map = root / "private" / "id-map.jsonl"
            manifest = blind.build(
                argparse.Namespace(
                    train=train,
                    expected_train_sha256=pilot.sha_file(train),
                    private_salt=salt,
                    output_dir=packet_dir,
                    map_output=private_map,
                )
            )
            packet = (packet_dir / "packet.jsonl").read_text()
            self.assertEqual(manifest["schema_version"], blind.VERSION)
            self.assertEqual(manifest["packet_rows"], 144)
            self.assertTrue(private_map.exists())
            self.assertNotIn("source_id", packet)
            self.assertNotIn('"label"', packet)
            self.assertNotIn('"id"', packet)
            self.assertEqual(
                manifest["packet_sha256"], pilot.sha_file(packet_dir / "packet.jsonl")
            )


if __name__ == "__main__":
    unittest.main()
