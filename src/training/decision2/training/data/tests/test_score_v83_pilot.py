"""CPU tests for the prospective Score v8.3 data-quality pilot."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.data import score_v83_pilot as pilot


class ScoreV83PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.secret = bytes(range(32))  # Test-only fixture, never a candidate seed.

    def test_complete_source_disjoint_triplets(self) -> None:
        train = pilot.build(self.secret, "train")
        select = pilot.build(self.secret, "select")
        self.assertEqual((len(train), len(select)), (36, 18))
        self.assertEqual(
            Counter(row["label"] for row in train + select), {0: 18, 1: 18, 2: 18}
        )
        self.assertEqual(len({row["group_id"] for row in train + select}), 18)
        self.assertFalse(
            {row["audit_metadata"]["case"] for row in train}
            & {row["audit_metadata"]["case"] for row in select}
        )
        for row in train + select:
            self.assertEqual(
                pilot.rendered_oracle(
                    row["state"],
                    row["audit_metadata"]["mechanism"],
                    row["audit_metadata"]["case"],
                ),
                row["label"],
            )

    def test_private_blind_packets_and_immutable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            seed = folder / "seed.bin"
            seed.write_bytes(self.secret)
            seed.chmod(0o600)
            output = folder / "candidate"
            manifest = pilot.write(seed, output)
            self.assertEqual(manifest["status"], "PENDING_INDEPENDENT_BLIND_REVIEW")
            for role, expected in (("train", 12), ("select", 6)):
                packet = [
                    json.loads(line)
                    for line in (output / f"{role}-blind-packet.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(packet), expected)
                self.assertTrue(all(len(group["items"]) == 3 for group in packet))
                self.assertTrue(
                    all(
                        "label" not in item
                        for group in packet
                        for item in group["items"]
                    )
                )
                self.assertEqual(
                    os.stat(output / f"{role}-sealed-key.json").st_mode & 0o077, 0
                )
            with self.assertRaises(ValueError):
                pilot.write(seed, output)

    def test_fails_on_wrong_seed_or_role(self) -> None:
        with self.assertRaises(ValueError):
            pilot.build(b"short", "train")
        with self.assertRaises(ValueError):
            pilot.build(self.secret, "cal")


if __name__ == "__main__":
    unittest.main()
