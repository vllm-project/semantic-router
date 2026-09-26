"""Contract checks for the independent Score TRAIN curriculum."""

from __future__ import annotations

import collections
import difflib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from training.data import build_score_curriculum as curriculum
from training.model.data import validate_row


class ScoreCurriculumTests(unittest.TestCase):
    def test_balanced_oracles_and_complete_groups(self) -> None:
        rows = curriculum.generate()
        self.assertEqual(len(rows), 960)
        self.assertEqual(len({row["id"] for row in rows}), 960)
        self.assertEqual(len({row["input_sha256"] for row in rows}), 960)
        by_family = collections.defaultdict(list)
        by_group = collections.defaultdict(list)
        for row in rows:
            validate_row(row, "train")
            self.assertEqual(
                row["label"],
                curriculum.oracle(row["family"].removeprefix("score_"), row["state"]),
            )
            by_family[row["family"]].append(row)
            by_group[row["group_id"]].append(row)
        self.assertEqual(
            set(by_family), {f"score_{name}" for name in curriculum.FAMILIES}
        )
        for family_rows in by_family.values():
            self.assertEqual(
                collections.Counter(row["label"] for row in family_rows),
                {0: 80, 1: 80, 2: 80},
            )
            self.assertEqual(
                collections.Counter(row["language"] for row in family_rows),
                {"en": 180, "zh": 60},
            )
        self.assertEqual(len(by_group), 320)
        self.assertTrue(
            all(
                {row["label"] for row in group} == {0, 1, 2}
                for group in by_group.values()
            )
        )

    def test_obligation_and_route_counterfactuals(self) -> None:
        rows = curriculum.generate()
        for family in ("score_obligation_review", "score_route_depth"):
            group = sorted(
                (row for row in rows if row["family"] == family),
                key=lambda row: row["group_id"],
            )
            first = group[0]["group_id"]
            variants = [row for row in group if row["group_id"] == first]
            self.assertEqual({row["label"] for row in variants}, {0, 1, 2})
            self.assertEqual(
                {row["instructions"] for row in variants}, {variants[0]["instructions"]}
            )
            self.assertEqual(
                {json.dumps(row["options"], sort_keys=True) for row in variants},
                {json.dumps(variants[0]["options"], sort_keys=True)},
            )

    def test_hand_computed_oracle_boundaries(self) -> None:
        review = {
            "reviews": [
                {"scope": "core", "assessment": "accepted"},
                {"scope": "informational", "assessment": "rejected"},
            ]
        }
        self.assertEqual(curriculum.oracle("obligation_review", review), 2)
        review["reviews"][0]["assessment"] = "unresolved"
        self.assertEqual(curriculum.oracle("obligation_review", review), 1)
        review["reviews"][0]["assessment"] = "rejected"
        self.assertEqual(curriculum.oracle("obligation_review", review), 0)
        weighted = {"signals": [{"weight": 2, "mark": 4}], "lower": 8, "upper": 12}
        self.assertEqual(curriculum.oracle("weighted_points", weighted), 1)
        weighted["signals"][0]["mark"] = 6
        self.assertEqual(curriculum.oracle("weighted_points", weighted), 2)
        weighted["signals"][0]["mark"] = 3
        self.assertEqual(curriculum.oracle("weighted_points", weighted), 0)
        route = {
            "start": "a",
            "finish": "d",
            "links": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
                {"from": "c", "to": "d"},
                {"from": "c", "to": "b"},
            ],
        }
        self.assertEqual(curriculum.oracle("route_depth", route), 1)
        route["links"].append({"from": "b", "to": "d"})
        self.assertEqual(curriculum.oracle("route_depth", route), 2)
        route["links"] = []
        self.assertEqual(curriculum.oracle("route_depth", route), 0)
        streak = {
            "days": [
                {"day": 0, "on_time": True},
                {"day": 2, "on_time": True},
                {"day": 1, "on_time": False},
                {"day": 3, "on_time": True},
                {"day": 4, "on_time": False},
                {"day": 5, "on_time": True},
            ]
        }
        self.assertEqual(curriculum.oracle("timely_streak", streak), 1)
        streak["days"][4]["on_time"] = True
        streak["days"][5]["on_time"] = True
        self.assertEqual(curriculum.oracle("timely_streak", streak), 2)

    def test_cross_group_level_zero_contexts_are_not_near_clones(self) -> None:
        groups = collections.defaultdict(list)
        for row in curriculum.generate():
            if row["label"] == 0:
                groups[row["family"]].append(
                    json.dumps(
                        row["state"],
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        for family, states in groups.items():
            closest = max(
                difflib.SequenceMatcher(None, left, right).ratio()
                for left, right in itertools.combinations(states, 2)
            )
            self.assertLess(closest, 0.94, family)

    def test_protected_reader_rejects_gold_and_incomplete_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = root / "typed-dev.prompts.jsonl"
            prompt.write_text(
                json.dumps({"id": "x", "state": {}, "questions": {}}) + "\n"
            )
            inventory = root / "sources.json"
            inventory.write_text(
                json.dumps([{"role": "typed_dev", "path": str(prompt)}])
            )
            with self.assertRaisesRegex(ValueError, "Protected roles"):
                curriculum._load_protected(inventory)
            others = ["css_pilot", "css15_goldfree"]
            inventory.write_text(
                json.dumps(
                    [
                        {"role": role, "path": str(prompt)}
                        for role in ["typed_dev", *others]
                    ]
                )
            )
            references, receipt = curriculum._load_protected(inventory)
            self.assertEqual(set(references), {"typed_dev", *others})
            self.assertEqual(len(receipt), 3)
            prompt.write_text(
                json.dumps({"id": "x", "state": {}, "questions": {}, "gold": 2}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "non-prompt fields"):
                curriculum._load_protected(inventory)


if __name__ == "__main__":
    unittest.main()
