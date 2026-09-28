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


if __name__ == "__main__":
    unittest.main()
