"""Row construction: contract, balanced gold positions, image cap."""

import collections
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from d25.omni.common import vision_format
from d25.omni.data import rows


class RowTest(unittest.TestCase):
    def test_choice_rows_balance_gold_and_cap_pixels(self):
        positions = collections.Counter()
        with tempfile.TemporaryDirectory() as tmp:
            big = Image.new("RGB", (2400, 1600), (10, 120, 200))
            for i in range(400):
                item = rows.Item(
                    source="test",
                    family="unit",
                    skill="spatial",
                    images=[big] if i == 0 else [],
                    image_kinds=["jpeg"] if i == 0 else [],
                    question="Pick one.",
                    options=["gold", "b", "c", "d"],
                    gold=0,
                )
                row = rows.make_row(item, i, tmp, "CC0-1.0", "train")
                positions[row["label"]] += 1
                self.assertEqual(
                    row["question"]["criteria"][row["meta"]["gold"]], "gold"
                )
                if i == 0:
                    with Image.open(Path(tmp) / row["images"][0]) as stored:
                        self.assertLessEqual(
                            stored.width * stored.height, vision_format.MAX_PIXELS
                        )
        self.assertEqual(sum(positions.values()), 400)
        for position in range(4):
            self.assertGreater(positions[position], 70)

    def test_noul_and_semantic_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            noul = rows.make_row(
                rows.Item("t", "n", "moderation", [], [], "Hateful?", noul=True),
                0,
                tmp,
                "CC-BY-4.0",
                "train",
            )
            self.assertEqual((noul["target"], noul["label"]), ([0.0, 1.0], 1))
            keyed = rows.make_row(
                rows.Item(
                    "t",
                    "k",
                    "spatial",
                    [],
                    [],
                    "Where?",
                    options=["left", "right"],
                    gold=1,
                    keys=["left", "right"],
                    fixed_order=True,
                ),
                1,
                tmp,
                "MIT",
                "train",
            )
            self.assertEqual(list(keyed["question"]["criteria"]), ["left", "right"])
            self.assertEqual(keyed["meta"]["gold"], "right")

    def test_numeric_distractors_are_distinct(self):
        rng = rows.rng_for("x")
        values = rows.numeric_distractors(12, rng, 5)
        self.assertEqual(len(set(values)), 5)
        self.assertNotIn(12, values)


if __name__ == "__main__":
    unittest.main()
