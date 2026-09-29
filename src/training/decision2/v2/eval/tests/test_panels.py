from __future__ import annotations

import unittest
from pathlib import Path

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

    def test_hs1_dev_is_a_development_panel_verified_by_default(self):
        entry = panels.DEVELOPMENT["hs1-dev"]
        self.assertNotIn("hs1-dev", panels.FORMAL)
        self.assertEqual(entry["originals"], 2396)
        default = panels.expected_files()
        self.assertEqual(
            default["goldfree/hs1-dev.prompts.jsonl"], entry["prompts_sha256"]
        )
        self.assertEqual(default["gold/hs1-dev.gold.jsonl"], entry["gold_sha256"])

    def test_install_keeps_goldfree_readable_and_gold_private(self):
        import hashlib
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompts, gold = tmp_path / "p.jsonl", tmp_path / "g.jsonl"
            prompts.write_text("{}\n")
            gold.write_text("[]\n")
            digest = lambda path: hashlib.sha256(
                path.read_bytes()
            ).hexdigest()  # noqa: E731
            wanted = {
                "goldfree/x.prompts.jsonl": digest(prompts),
                "gold/x.gold.jsonl": digest(gold),
            }
            root = tmp_path / "root"
            with mock.patch.object(panels, "expected_files", return_value=wanted):
                panels.install(root, {"p": prompts, "g": gold})
            mode = lambda path: path.stat().st_mode & 0o777  # noqa: E731
            self.assertEqual(mode(root / "goldfree"), 0o755)
            self.assertEqual(mode(root / "goldfree/x.prompts.jsonl"), 0o644)
            self.assertEqual(mode(root / "gold"), 0o700)
            self.assertEqual(mode(root / "gold/x.gold.jsonl"), 0o600)

    def test_score5t_dev_is_a_development_panel_verified_by_default(self):
        entry = panels.DEVELOPMENT["score5t-dev"]
        self.assertNotIn("score5t-dev", panels.FORMAL)
        self.assertEqual(entry["originals"], 800)
        default = panels.expected_files()
        self.assertEqual(
            default["goldfree/score5t-dev.prompts.jsonl"], entry["prompts_sha256"]
        )
        self.assertEqual(default["gold/score5t-dev.gold.jsonl"], entry["gold_sha256"])
        self.assertEqual(
            panels.path(panels.DEFAULT_ROOT, "score5t-dev", "gold"),
            panels.DEFAULT_ROOT / "gold/score5t-dev.gold.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
