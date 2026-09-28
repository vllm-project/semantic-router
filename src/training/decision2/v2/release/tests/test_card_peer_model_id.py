"""A peer report without a model id takes the spec's repo_id for the chart licence lookup (stdlib)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release import card
from v2.release.tests.test_release import REPORTS, ROSTER, facts


def entries(decider_repo: str | None = "Mapika/decider-2b") -> list[dict]:
    decider = {
        "key": "decider2b",
        "role": "peer",
        "report": str(REPORTS / "decider2b.json"),
    }
    if decider_repo:
        decider["repo_id"] = decider_repo
    return [
        {
            "key": "cand",
            "role": "candidate",
            "report": str(REPORTS / "bosun17b.json"),
            "label": "DEV2.0-2B",
        },
        {
            "key": "sol1",
            "role": "own-1.0",
            "report": str(REPORTS / "sol1.json"),
            "repo_id": "llm-semantic-router/Decision-1.0-Sol-2B",
            "label": "Decision 1.0 Sol",
        },
        decider,
    ]


class PeerModelIdTest(unittest.TestCase):
    def build(self, items: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            out, banner = Path(scratch) / "pkg", Path(scratch) / "banner.png"
            banner.write_bytes(b"\x89PNG\r\n\x1a\n")
            card.build_card(
                entries=items,
                roster=ROSTER,
                paired=None,
                facts=facts(),
                text={"tagline": "A decision model.", "limitations": []},
                banner=banner,
                work=Path(scratch) / "work",
                output=out,
            )
            charts = json.loads(
                (Path(scratch) / "work/charts-receipt.json").read_text()
            )
            readme = (out / "README.md").read_text()
            return {"charts": charts, "readme": readme}

    def test_fixture_report_names_no_model_id(self):
        report = json.loads((REPORTS / "decider2b.json").read_text())
        self.assertIsNone(report["model"]["model_id"])

    def test_repo_id_fills_the_missing_model_id(self):
        result = self.build(entries())
        self.assertEqual(
            result["charts"]["model_id_overrides"], {"Decider 2B": "Mapika/decider-2b"}
        )
        self.assertIn("Decider 2B", result["readme"])

    def test_without_repo_id_the_peer_stays_off_the_card(self):
        result = self.build(entries(decider_repo=None))
        self.assertEqual(result["charts"]["model_id_overrides"], {})
        self.assertNotIn("Decider 2B", result["readme"])


if __name__ == "__main__":
    unittest.main()
