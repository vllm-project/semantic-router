"""Layout of the product-card banner for every released size.

Needs matplotlib (with numpy and Pillow), the Inter fonts in ``$DEV2_CARD_FONTS``
and the repository logo; skipped otherwise.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import tempfile
import unittest
from pathlib import Path

from v2.release import card_assets, layout

LOGO = (
    Path(card_assets.__file__).resolve().parents[5]
    / "website/static/img/artworks/vllm-sr-logo.dark.png"
)
FONTS = Path(os.environ.get("DEV2_CARD_FONTS", ""))
TEXT = ("logo", "eyebrow", "codename", "size", "tagline", "dot")
# Clear space between the last text and the V-mark, as a fraction of the banner width.
VMARK_MARGIN = 0.08


def _ready() -> bool:
    return (
        all(importlib.util.find_spec(m) for m in ("matplotlib", "numpy", "PIL"))
        and LOGO.is_file()
        and FONTS.is_dir()
        and all((FONTS / f"{n}.ttf").is_file() for n in card_assets.FONTS)
    )


def _disjoint(a, b) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


@unittest.skipUnless(_ready(), "needs matplotlib, the Inter fonts and the logo")
class BannerLayoutTest(unittest.TestCase):
    def test_every_size_renders_without_overlap(self):
        renderer = card_assets.Renderer(LOGO, FONTS)
        with tempfile.TemporaryDirectory() as scratch:
            for tier in layout.TIERS:
                name = layout.tier_name(tier)
                with self.subTest(name=name):
                    path = Path(scratch) / f"{tier}.png"
                    boxes = renderer.banner(name, path)
                    self.assertTrue(path.stat().st_size > 0)
                    canvas = boxes["canvas"]
                    for key in TEXT:
                        x0, y0, x1, y1 = boxes[key]
                        self.assertTrue(
                            canvas[0] <= x0 < x1 <= canvas[2]
                            and canvas[1] <= y0 < y1 <= canvas[3],
                            key,
                        )
                    for a, b in itertools.combinations(TEXT, 2):
                        self.assertTrue(_disjoint(boxes[a], boxes[b]), (a, b))
                    right = max(boxes[k][2] for k in TEXT)
                    self.assertLess(right, boxes["vmark"][0] - VMARK_MARGIN * canvas[2])
                    self.assertEqual(boxes["vmark"][2], canvas[2])
                    # The codename is the focal point: the largest element.
                    height = lambda k: boxes[k][3] - boxes[k][1]
                    self.assertTrue(
                        all(
                            height("codename") > height(k)
                            for k in TEXT[:-1]
                            if k != "codename"
                        )
                    )

    def test_banner_refuses_other_names(self):
        renderer = card_assets.Renderer(LOGO, FONTS)
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                renderer.banner("DEV2.0-4B", Path(scratch) / "x.png")


if __name__ == "__main__":
    unittest.main()
