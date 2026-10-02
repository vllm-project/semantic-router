"""A peer report without a model id takes the spec's repo_id for the card licence lookup (stdlib)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.release.tests.test_release import REPORTS, build_test_card, facts


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
            "label": "Decision-2.0-Sol-2B",
        },
        {
            "key": "sol1",
            "role": "own-1.0",
            "report": str(REPORTS / "sol1.json"),
            "repo_id": "vllm-sr/Decision-1.0-Sol-2B",
            "label": "Decision 1.0 Sol",
        },
        decider,
    ]


class PeerModelIdTest(unittest.TestCase):
    def build(self, items: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as scratch:
            values = {**facts(), "model_name": "Decision-2.0-Sol-2B"}
            return build_test_card(Path(scratch), items, values)["readme"]

    def test_fixture_report_names_no_model_id(self):
        report = json.loads((REPORTS / "decider2b.json").read_text())
        self.assertIsNone(report["model"]["model_id"])

    def test_repo_id_fills_the_missing_model_id(self):
        self.assertIn("Decider 2B", self.build(entries()))

    def test_without_repo_id_the_peer_stays_off_the_card(self):
        self.assertNotIn("Decider 2B", self.build(entries(decider_repo=None)))


if __name__ == "__main__":
    unittest.main()
