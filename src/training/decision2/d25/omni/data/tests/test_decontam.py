"""Planted-positive checks for the Omni text and image decontamination."""

import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from d25.omni.data import decontam

WORDS = (
    "river stone market garden window yellow quiet engine ladder pocket violin orbit candle forest "
    "silver bridge thunder pencil harbor meadow lantern copper falcon velvet canyon marble whisper "
    "glacier saddle tunnel compass ember willow cobalt mosaic prism raven sapphire tundra"
).split()


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def kit_row(i: int, state: str, options: list[str]) -> dict:
    return {
        "id": f"bench:{i}",
        "state": state,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Which option is correct?",
                "criteria": {chr(65 + k): o for k, o in enumerate(options)},
            }
        },
    }


def photo(seed: int, size=(320, 240)) -> Image.Image:
    """A photo stand-in: smooth random field (1/f-like) plus a few shapes."""
    rng = np.random.default_rng(seed)
    w, h = size
    coarse = rng.uniform(0, 255, (6, 8, 3)).astype(np.uint8)
    image = Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC)
    detail = Image.fromarray(rng.uniform(0, 255, (h // 8, w // 8, 3)).astype(np.uint8))
    image = Image.blend(image, detail.resize((w, h), Image.Resampling.BICUBIC), 0.35)
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        x0, y0 = int(rng.integers(0, w - 40)), int(rng.integers(0, h - 40))
        x1, y1 = x0 + int(rng.integers(20, 120)), y0 + int(rng.integers(20, 100))
        fill = tuple(int(c) for c in rng.integers(0, 255, 3))
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(
            (x0, y0, x1, y1), fill=fill
        )
    return image.filter(ImageFilter.GaussianBlur(0.8))


class TextRuleTest(unittest.TestCase):
    def setUp(self):
        rng = random.Random(1)
        self.template = (
            "Considering the relative positions of the two objects in the image "
            "provided, where is the first object located?"
        )
        self.rows = [
            kit_row(
                i,
                sentence(rng, 25) + " " + self.template,
                [sentence(rng, 3) for _ in range(4)],
            )
            for i in range(200)
        ]
        self.index = decontam.TextIndex()
        for row in self.rows:
            for units, options in decontam.protected_units(row):
                self.index.add(units, options)
        self.index.finalize()
        self.clean = [[sentence(rng, 30), sentence(rng, 8)] for _ in range(300)]

    def test_template_is_boilerplate(self):
        exact, coverage = self.index.score([self.template])
        self.assertFalse(exact)
        self.assertEqual(coverage, 0.0)
        self.assertGreater(self.index.boilerplate, 0)

    def test_planted_items_are_caught_and_clean_rows_pass(self):
        report = decontam.calibrate_text(
            self.index, self.rows, self.clean, n_planted=200
        )
        self.assertEqual(report["planted_caught"], report["planted"])
        self.assertEqual(report["clean_flagged"], 0)
        self.assertLessEqual(report["threshold"], 0.5)

    def test_exact_state_match_after_normalisation(self):
        state = self.rows[3]["state"].upper().replace(" ", "  ")
        exact, _ = self.index.score([state])
        self.assertTrue(exact)

    def test_option_set_key_is_order_free(self):
        rng = random.Random(5)
        options = [sentence(rng, 4) for _ in range(4)]
        index = decontam.TextIndex()
        index.add(
            ["unrelated"],
            decontam.option_key(kit_row(0, "x", options)["questions"]["q"]),
        )
        index.finalize()
        shuffled = {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {
                chr(65 + k): o.upper() for k, o in enumerate(reversed(options))
            },
        }
        exact, _ = index.score(["something else"], decontam.option_key(shuffled))
        self.assertTrue(exact)

    def test_image_text_is_checked(self):
        row = {
            "state": "",
            "question": {
                "type": "choice",
                "instructions": "Pick.",
                "criteria": {"A": None, "B": None},
            },
            "meta": {"image_text": ["Shown: " + self.rows[7]["state"]]},
        }
        exact, coverage = self.index.score(decontam.training_units(row))
        self.assertGreater(coverage, 0.5)


class ImageRuleTest(unittest.TestCase):
    def test_hashes_match_imagehash(self):
        try:
            import imagehash
        except ImportError:
            self.skipTest("imagehash not installed")
        for seed in range(12):
            image = photo(seed, (257 + seed * 13, 191 + seed * 7))
            self.assertEqual(
                f"{decontam.phash64(image):016x}", str(imagehash.phash(image))
            )
            self.assertEqual(
                f"{decontam.dhash64(image):016x}", str(imagehash.dhash(image))
            )

    def test_trim_recovers_padding(self):
        image = photo(3)
        padded = Image.new("RGB", (400, 300), (255, 255, 255))
        padded.paste(image, (40, 30))
        self.assertEqual(decontam.trim_borders(padded).size, image.size)

    def test_planted_copies_caught_with_low_false_positives(self):
        planted = [photo(seed) for seed in range(40)]
        clean = [photo(1000 + seed) for seed in range(200)]
        report = decontam.calibrate_images(planted, clean)
        self.assertEqual(report["planted_caught"], report["planted_copies"])
        self.assertLessEqual(report["false_positive_rate"], 0.02)
        rule = decontam.Rule(**report["rule"])
        bank = decontam.ImageIndex(
            [
                {
                    "sha256": "x",
                    "phash64": decontam.phash64(im),
                    "dhash64": decontam.dhash64(im),
                    "benchmark": "b",
                    "split": "s",
                    "item_id": i,
                }
                for i, im in enumerate(planted)
            ]
        )
        buffer = io.BytesIO()
        planted[5].resize((200, 150)).save(buffer, format="JPEG", quality=60)
        copy = Image.open(io.BytesIO(buffer.getvalue()))
        self.assertEqual(bank.match(decontam.image_hashes(copy), rule), 5)

    def test_signed_and_hex_bank_values(self):
        value = (1 << 63) + 12345
        self.assertEqual(decontam.as_u64(value - (1 << 64)), value)
        self.assertEqual(decontam.as_u64(f"{value:016x}"), value)

    def test_sscd_hook(self):
        bank = np.eye(4, dtype=np.float32)
        train = np.array([[1, 0.01, 0, 0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
        self.assertEqual(decontam.sscd_matches(train, bank).tolist(), [0])


class HoldoutTest(unittest.TestCase):
    def test_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "holdouts.json"
            path.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {"hf_id": "org/whole", "scope": "all"},
                            {
                                "hf_id": "org/split",
                                "scope": "splits",
                                "splits": ["dev"],
                            },
                            {
                                "hf_id": "org/rows",
                                "scope": "rows",
                                "splits": ["train"],
                                "slice": {
                                    "name": "rows",
                                    "key": "q",
                                    "mod": 2,
                                    "keep": 1,
                                },
                            },
                        ],
                        "ids": ["item-9"],
                        "image_sha256": ["abc"],
                    }
                )
            )
            holdouts = decontam.Holdouts.load([path, Path(tmp) / "missing.json"])
        self.assertEqual(holdouts.files[str(Path(tmp) / "missing.json")], "missing")

        def row(**meta):
            return {"meta": {"source_split": "train", **meta}}

        self.assertEqual(holdouts.row_dropped(row(), ["org/whole"]), "holdout_source")
        self.assertEqual(
            holdouts.row_dropped(row(source_split="dev"), ["org/split"]),
            "holdout_split",
        )
        self.assertIsNone(holdouts.row_dropped(row(), ["org/split"]))
        keys = [k for k in range(40) if decontam.slice_selected("rows", k, 2, 1)]
        self.assertEqual(
            holdouts.row_dropped(row(source_key=keys[0]), ["org/rows"]), "holdout_rows"
        )
        self.assertEqual(
            holdouts.row_dropped(row(source_id="item-9"), ["x"]), "holdout_item"
        )
        self.assertEqual(
            holdouts.row_dropped(row(image_sha256=["abc"]), ["x"]), "holdout_image"
        )


if __name__ == "__main__":
    unittest.main()
