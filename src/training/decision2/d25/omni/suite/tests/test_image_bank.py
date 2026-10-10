"""Image-bank hashing: parity with ``imagehash``, tile rule, robustness, and a small end-to-end bank."""

import io
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from d25.omni.suite import image_bank as bank


def synthetic(seed: int, size=(320, 240)) -> Image.Image:
    rng = random.Random(seed)
    im = Image.new("RGB", size, tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = rng.randrange(size[0]), rng.randrange(size[1])
        x1, y1 = x0 + rng.randrange(20, 160), y0 + rng.randrange(20, 120)
        draw.rectangle(
            (x0, y0, x1, y1), fill=tuple(rng.randrange(256) for _ in range(3))
        )
    noise = np.random.default_rng(seed).integers(
        0, 24, (size[1], size[0], 3), dtype=np.uint8
    )
    return Image.fromarray(
        np.clip(np.asarray(im, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    )


def png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class HashTest(unittest.TestCase):
    def test_parity_with_imagehash(self):
        try:
            import imagehash
        except ImportError:
            self.skipTest("imagehash not installed")
        for seed in range(40):
            im = synthetic(seed, (200 + seed * 7, 150 + seed * 3))
            self.assertEqual(bank.phash64(im), int(str(imagehash.phash(im)), 16), seed)
            self.assertEqual(bank.dhash64(im), int(str(imagehash.dhash(im)), 16), seed)

    def test_near_duplicates_stay_close_and_others_far(self):
        im = synthetic(1, (640, 480))
        resized = im.resize((320, 240), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        recoded = Image.open(io.BytesIO(buf.getvalue()))
        for variant in (resized, recoded):
            self.assertLessEqual(
                bank.hamming(bank.phash64(im), bank.phash64(variant)), 6
            )
        others = [
            bank.hamming(bank.phash64(im), bank.phash64(synthetic(s, (640, 480))))
            for s in range(2, 12)
        ]
        self.assertGreater(min(others), 10)

    def test_tiles_cover_tall_and_wide_images(self):
        self.assertEqual(bank.tiles(1280, 1280), [])
        self.assertEqual(bank.tiles(1280, 2500), [])
        t = bank.tiles(1280, 5136)
        self.assertEqual(t[0], (0, 0, 1280, 1280))
        self.assertEqual(t[-1], (0, 5136 - 1280, 1280, 5136))
        self.assertTrue(all(b[3] - b[1] == 1280 for b in t))
        self.assertTrue(all(b[1] - a[1] <= 640 for a, b in zip(t, t[1:])))
        wide = bank.tiles(3000, 1000)
        self.assertEqual(wide[0], (0, 0, 1000, 1000))

    def test_tile_record_matches_a_viewport_crop(self):
        tall = synthetic(5, (400, 1400))
        recs = bank.records(png(tall), "B", "s", "x", "k")
        crop = tall.crop((0, 600, 400, 1000))
        best = min(bank.hamming(r["phash64"], bank.phash64(crop)) for r in recs[1:])
        self.assertLessEqual(best, 8)
        self.assertEqual(recs[0]["region"], "")
        self.assertEqual(len({r["sha256"] for r in recs}), 1)


class EndToEndTest(unittest.TestCase):
    def test_zip_task_and_merge(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zpath = root / "pool.zip"
            with zipfile.ZipFile(zpath, "w") as z:
                for i in range(3):
                    z.writestr(f"val/{i}.png", png(synthetic(i)))
                z.writestr("val/dup.png", png(synthetic(0)))
                z.writestr("val/readme.txt", "x")
            task = (
                "pool__val",
                "pool",
                "zip",
                str(zpath),
                {"benchmark": "Pool", "split": "val", "prefix": "val/"},
            )
            first = bank.run_task(task, str(root / "out"))
            again = bank.run_task(task, str(root / "out"))
            self.assertEqual(first["rows"], 3)
            self.assertTrue(again["skipped"])
            summary = bank.merge(root / "out")
            table = pq.read_table(root / "out" / "bank.parquet")
            self.assertEqual(table.column_names, list(bank.COLUMNS))
            self.assertEqual(summary["distinct_sha256"], 3)
            self.assertEqual(str(table.schema.field("phash64").type), "uint64")
            self.assertEqual(
                json.loads((root / "out" / "BANK.json").read_text())["rows"], 3
            )


if __name__ == "__main__":
    unittest.main()
