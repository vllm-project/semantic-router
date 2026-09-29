from __future__ import annotations

import unittest

from v2.eval import panels


class PanelsTest(unittest.TestCase):
    def test_sealed_panels_are_only_verified_by_name(self):
        default = panels.expected_files()
        self.assertNotIn(panels.SEALED["sealed-c1"]["prompts"], default)
        self.assertIn(panels.FORMAL["typed-final"]["prompts"], default)
        named = panels.expected_files(["sealed-c1"])
        self.assertEqual(
            named,
            {
                "goldfree/sealed-c1.prompts.jsonl": panels.SEALED["sealed-c1"][
                    "prompts_sha256"
                ],
                "gold/sealed-c1.gold.jsonl": panels.SEALED["sealed-c1"]["gold_sha256"],
            },
        )
        self.assertIn("sealed-c1", panels.ALL)

    def test_ht_dev_is_a_development_panel_verified_by_default(self):
        entry = panels.DEVELOPMENT["ht-dev"]
        self.assertNotIn("ht-dev", panels.FORMAL)
        self.assertEqual(entry["originals"], 3240)
        default = panels.expected_files()
        self.assertEqual(
            default["goldfree/ht-dev.prompts.jsonl"], entry["prompts_sha256"]
        )
        self.assertEqual(default["gold/ht-dev.gold.jsonl"], entry["gold_sha256"])
        self.assertEqual(
            panels.path(panels.DEFAULT_ROOT, "ht-dev", "gold"),
            panels.DEFAULT_ROOT / "gold/ht-dev.gold.jsonl",
        )

    def test_score5_dev_is_a_development_panel_verified_by_default(self):
        entry = panels.DEVELOPMENT["score5-dev"]
        self.assertNotIn("score5-dev", panels.FORMAL)
        self.assertEqual(entry["originals"], 500)
        default = panels.expected_files()
        self.assertEqual(
            default["goldfree/score5-dev.prompts.jsonl"], entry["prompts_sha256"]
        )
        self.assertEqual(default["gold/score5-dev.gold.jsonl"], entry["gold_sha256"])
        self.assertEqual(
            panels.path(panels.DEFAULT_ROOT, "score5-dev", "gold"),
            panels.DEFAULT_ROOT / "gold/score5-dev.gold.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
