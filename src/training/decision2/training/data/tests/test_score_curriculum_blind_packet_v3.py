"""The Score v3 reviewer sees only opaque row/group aliases and prompts."""

from __future__ import annotations

import argparse
import collections
import json
import re
import tempfile
import unittest
from pathlib import Path

from training.data import build_pilot as pilot
from training.data import build_score_curriculum_v3 as curriculum
from training.data import score_curriculum_blind_packet_v3 as blind


class ScoreCurriculumBlindPacketV3Tests(unittest.TestCase):
    def test_opaque_row_and_group_aliases_are_deterministic_per_salt(self) -> None:
        rows = curriculum.generate()
        first, mapping, group_count = blind.select(rows, b"a" * 32)
        repeat, repeat_map, repeat_count = blind.select(list(reversed(rows)), b"a" * 32)
        other, _, _ = blind.select(rows, b"b" * 32)
        self.assertEqual(
            (first, mapping, group_count), (repeat, repeat_map, repeat_count)
        )
        self.assertEqual((len(first), group_count), (144, 48))
        self.assertNotEqual(
            [row["review_id"] for row in first],
            [row["review_id"] for row in other],
        )
        self.assertEqual(len({row["review_id"] for row in first}), 144)
        self.assertEqual(len({row["group_id"] for row in first}), 48)
        self.assertEqual(
            set(collections.Counter(row["group_id"] for row in first).values()), {3}
        )
        self.assertTrue(all(set(row) == set(blind.PUBLIC_FIELDS) for row in first))
        self.assertTrue(
            all(re.fullmatch(r"sbr3-[0-9a-f]{20}", row["review_id"]) for row in first)
        )
        self.assertTrue(
            all(re.fullmatch(r"sbg3-[0-9a-f]{20}", row["group_id"]) for row in first)
        )
        self.assertTrue(
            all("source_id" not in row and "label" not in row for row in first)
        )
        self.assertEqual(
            {row["review_id"] for row in first},
            {row["review_id"] for row in mapping},
        )

    def test_build_keeps_private_join_outside_public_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            train.write_bytes(pilot.jsonl_bytes(curriculum.generate()))
            private_dir = root / "private"
            private_dir.mkdir(mode=0o700)
            salt = private_dir / "salt.bin"
            salt.write_bytes(b"s" * 32)
            salt.chmod(0o600)
            packet_dir = root / "review"
            map_file = private_dir / "id-map.jsonl"
            manifest = blind.build(
                argparse.Namespace(
                    train=train,
                    expected_train_sha256=pilot.sha_file(train),
                    private_salt=salt,
                    output_dir=packet_dir,
                    map_output=map_file,
                )
            )
            packet_text = (packet_dir / "packet.jsonl").read_text()
            self.assertEqual(manifest["schema_version"], blind.VERSION)
            self.assertEqual(
                (manifest["packet_rows"], manifest["selected_group_count"]), (144, 48)
            )
            self.assertTrue(map_file.exists())
            self.assertNotIn("source_id", packet_text)
            self.assertNotIn("source_group_id", packet_text)
            self.assertNotIn('"label"', packet_text)
            self.assertEqual(
                manifest["packet_sha256"], pilot.sha_file(packet_dir / "packet.jsonl")
            )
            self.assertEqual(manifest["private_map_sha256"], pilot.sha_file(map_file))
            self.assertEqual(
                set(json.loads(packet_text.splitlines()[0])), set(blind.PUBLIC_FIELDS)
            )

    def test_rejects_short_salt(self) -> None:
        with self.assertRaisesRegex(ValueError, "32-byte"):
            blind.select([], b"short")


if __name__ == "__main__":
    unittest.main()
