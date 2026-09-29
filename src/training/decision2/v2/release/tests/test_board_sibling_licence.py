"""A BF16 sibling takes the licence of its FP8 Decision Index board entry (stdlib)."""

from __future__ import annotations

import unittest

from v2.release import licence
from v2.release.tests.test_release import ROSTER


class BoardSiblingLicenceTest(unittest.TestCase):
    def setUp(self):
        self.roster = licence.load_roster(ROSTER)

    def check(self, board: str | None) -> dict:
        return licence.card_eligibility(
            {
                "repo_id": "caiovicentino1/Eikos-27B",
                "family": "peer",
                "board_entry": board,
            },
            self.roster,
        )

    def test_bf16_sibling_takes_its_fp8_board_entry_licence(self):
        decision = self.check("caiovicentino1/Eikos-27B-FP8")
        self.assertTrue(decision["eligible"])
        self.assertEqual(decision["licence"], "mit")
        self.assertIn("board entry caiovicentino1/Eikos-27B-FP8", decision["reason"])
        self.assertFalse(self.check(None)["eligible"])
        self.assertFalse(self.check("denis-pplx/autojev-27b")["eligible"])


if __name__ == "__main__":
    unittest.main()
