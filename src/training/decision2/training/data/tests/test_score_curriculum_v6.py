"""Independent checks for the frozen Score v6 candidate construction."""

from __future__ import annotations

import collections
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.data import build_score_curriculum_v6 as curriculum
from training.data import build_pilot as pilot
from training.data import score_curriculum_v6_abstract as abstract
from training.model.data import validate_row


def independent_intersection(state: dict) -> int:
    """Count membership directly from four eligible names, not builder oracle."""
    universe = state["eligible_claims"]
    documents = state["documents"]
    contents = []
    for document in documents:
        style = document["format"]
        if style == "checklist":
            entries = document["checked"]
        elif style == "table":
            entries = [
                item["claim"] for item in document["rows"] if item["status"] == "active"
            ]
        elif style == "memo":
            entries = document["current_attestations"].split("; ")
        elif style == "ticket":
            entries = [document["line_one"], document["line_two"]]
        else:
            raise AssertionError(style)
        assert len(entries) == 2 and len(set(entries)) == 2
        contents.append(set(entries))
    return sum(name in contents[0] and name in contents[1] for name in universe)


def independent_route(state: dict) -> int:
    links = [(edge["from"], edge["to"]) for edge in state["links"]]
    frontier = [(state["start"], (state["start"],))]
    lengths = []
    while frontier:
        current, path = frontier.pop()
        if current == state["finish"]:
            lengths.append(len(path) - 1)
        frontier.extend(
            (target, (*path, target))
            for source, target in links
            if source == current and target not in path
        )
    return 0 if not lengths else (2 if min(lengths) <= 2 else 1)


def independent_streak(state: dict) -> int:
    on_time = {item["day"] for item in state["days"] if item["on_time"]}
    longest = max(
        (
            length
            for start in range(1, 13)
            for length in range(1, 14 - start)
            if set(range(start, start + length)) <= on_time
        ),
        default=0,
    )
    return 0 if longest <= 2 else (1 if longest == 3 else 2)


def independent_obligation(state: dict) -> int:
    newest = {}
    for item in state["events"]:
        if item["scope"] != "core":
            continue
        name = item["control"]
        if name not in newest or item["timestamp"] > newest[name]["timestamp"]:
            newest[name] = item
    statuses = {item["assessment"] for item in newest.values()}
    return 0 if "rejected" in statuses else (1 if "unresolved" in statuses else 2)


class ScoreV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = curriculum.generate()

    def test_complete_balanced_groups_and_oracles(self) -> None:
        self.assertEqual(len(self.rows), 972)
        groups = collections.defaultdict(list)
        independent = {
            "evidence_intersection": independent_intersection,
            "obligation_review": independent_obligation,
            "route_depth": independent_route,
            "timely_streak": independent_streak,
        }
        for row in self.rows:
            validate_row(row, "train")
            family = row["family"].removeprefix("score_")
            self.assertEqual(independent[family](row["state"]), row["label"])
            self.assertEqual(row["source"], curriculum.SOURCE)
            groups[row["group_id"]].append(row)
        self.assertEqual(len(groups), 324)
        for variants in groups.values():
            self.assertEqual({row["label"] for row in variants}, {0, 1, 2})
            self.assertEqual(len({row["instructions"] for row in variants}), 1)
            self.assertEqual(len({str(row["options"]) for row in variants}), 1)
        for family in curriculum.FAMILIES:
            rows = [row for row in self.rows if row["family"] == f"score_{family}"]
            self.assertEqual(
                collections.Counter(row["label"] for row in rows), {0: 81, 1: 81, 2: 81}
            )

    def test_two_source_necessity_and_shallow_gates(self) -> None:
        rows = [
            row for row in self.rows if row["family"] == "score_evidence_intersection"
        ]
        for row in rows:
            state = row["state"]
            universe = state["eligible_claims"]
            sources = [curriculum._document_claims(doc) for doc in state["documents"]]
            alternatives = [set(pair) for pair in itertools.combinations(universe, 2)]
            for fixed in sources:
                self.assertEqual(
                    {len(fixed & alternate) for alternate in alternatives}, {0, 1, 2}
                )
        audit = curriculum.shortcut_audit(self.rows)
        evidence = audit["evidence_intersection"]
        self.assertEqual(evidence["oracle_disagreements"], 0)
        self.assertEqual(
            set(evidence["formats"]), {"checklist", "table", "memo", "ticket"}
        )
        self.assertLessEqual(
            max(item["correct"] for item in evidence["features"].values()), 162
        )
        self.assertTrue(
            all(item["perfect_groups"] == 0 for item in evidence["features"].values())
        )
        for family in ("obligation_review", "route_depth", "timely_streak"):
            self.assertEqual(audit[family]["count_only_correct"], 81)

    def test_protected_roster_rejects_gold_and_accepts_opaque_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet = root / "packet.jsonl"
            item = {
                "review_id": "opaque",
                "group_id": "opaque-group",
                "family": "score_example",
                "language": "en",
                "state": {"record": "private prompt"},
                "instructions": "Read the record.",
                "options": [{"key": "0", "description": "none"}],
            }
            packet.write_text(json.dumps(item) + "\n", encoding="utf-8")
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    [
                        {
                            "role": "opaque",
                            "path": str(packet),
                            "sha256": pilot.sha_file(packet),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with patch.object(curriculum, "REQUIRED_PROTECTED_ROLES", {"opaque"}):
                references, receipts = curriculum._load_protected(inventory)
                self.assertEqual(
                    references["opaque"][0]["instructions"], "Read the record."
                )
                self.assertEqual(receipts[0]["rows"], 1)
                item["label"] = 1
                packet.write_text(json.dumps(item) + "\n", encoding="utf-8")
                inventory.write_text(
                    json.dumps(
                        [
                            {
                                "role": "opaque",
                                "path": str(packet),
                                "sha256": pilot.sha_file(packet),
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "unexpected fields"):
                    curriculum._load_protected(inventory)


if __name__ == "__main__":
    unittest.main()
