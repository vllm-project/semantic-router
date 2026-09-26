"""Contract checks for the repaired TRAIN-only Score curriculum."""

from __future__ import annotations

import collections
import difflib
import itertools
import json
import unittest

from training.data import build_score_curriculum_v2 as curriculum
from training.model.data import validate_row


def _independent_shortest_path(state: dict) -> int | None:
    """Enumerate simple routes rather than using the production BFS."""
    edges = collections.defaultdict(list)
    for link in state["links"]:
        edges[link["from"]].append(link["to"])
    paths = [(state["start"],)]
    distances = []
    while paths:
        path = paths.pop()
        if path[-1] == state["finish"]:
            distances.append(len(path) - 1)
            continue
        paths.extend(
            (*path, next_node) for next_node in edges[path[-1]] if next_node not in path
        )
    return min(distances) if distances else None


def _independent_longest_run(state: dict) -> int:
    by_day = {record["day"]: bool(record["on_time"]) for record in state["days"]}
    return max(
        (
            length
            for start in range(1, 9)
            for length in range(1, 10 - start)
            if all(by_day[day] for day in range(start, start + length))
        ),
        default=0,
    )


class ScoreCurriculumV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = curriculum.generate()
        cls.groups = collections.defaultdict(list)
        for row in cls.rows:
            cls.groups[row["group_id"]].append(row)

    def test_balanced_contract_and_source_version(self) -> None:
        self.assertEqual(len(self.rows), 960)
        self.assertEqual(len(self.groups), 320)
        self.assertEqual(len({row["id"] for row in self.rows}), 960)
        self.assertEqual(len({row["input_sha256"] for row in self.rows}), 960)
        for row in self.rows:
            validate_row(row, "train")
            self.assertEqual(row["source"], curriculum.SOURCE)
            self.assertTrue(row["render_template"].endswith("_v2"))
            self.assertEqual(
                row["label"],
                curriculum.oracle(row["family"].removeprefix("score_"), row["state"]),
            )
        for family in curriculum.FAMILIES:
            rows = [row for row in self.rows if row["family"] == f"score_{family}"]
            self.assertEqual(
                collections.Counter(row["label"] for row in rows), {0: 80, 1: 80, 2: 80}
            )
            self.assertEqual(
                collections.Counter(row["language"] for row in rows),
                {"en": 180, "zh": 60},
            )
            self.assertEqual(len({row["instructions"] for row in rows}), 80)
            self.assertGreaterEqual(
                len({json.dumps(row["options"], ensure_ascii=False) for row in rows}), 5
            )
        self.assertTrue(
            all(
                {row["label"] for row in group} == {0, 1, 2}
                for group in self.groups.values()
            )
        )

    def test_route_graphs_require_traversal_not_counts(self) -> None:
        report = curriculum.shortcut_audit(self.rows)
        self.assertEqual(report["route_depth"]["count_only_correct"], 80)
        for group in self.groups.values():
            if group[0]["family"] != "score_route_depth":
                continue
            states = {row["label"]: row["state"] for row in group}
            self.assertEqual(
                {_independent_shortest_path(state) for state in states.values()},
                {None, 2, 3},
            )
            self.assertEqual(
                {
                    level: _independent_shortest_path(state)
                    for level, state in states.items()
                },
                {0: None, 1: 3, 2: 2},
            )
            node_sets = [
                {end for link in state["links"] for end in (link["from"], link["to"])}
                for state in states.values()
            ]
            self.assertEqual(node_sets[0], node_sets[1])
            self.assertEqual(node_sets[1], node_sets[2])
            self.assertEqual(
                {
                    curriculum._route_shortcut_features(state)
                    for state in states.values()
                },
                {(8, 8, (4, 4), 1, 1, 1, 1, 4, 4, False)},
            )

    def test_streak_order_changes_with_constant_counts(self) -> None:
        report = curriculum.shortcut_audit(self.rows)
        self.assertEqual(report["timely_streak"]["count_only_correct"], 80)
        for group in self.groups.values():
            if group[0]["family"] != "score_timely_streak":
                continue
            states = {row["label"]: row["state"] for row in group}
            runs = {
                level: _independent_longest_run(state)
                for level, state in states.items()
            }
            self.assertEqual(runs[0], 1)
            self.assertIn(runs[1], (2, 3))
            self.assertEqual(runs[2], 4)
            self.assertEqual(
                len(
                    {
                        curriculum._streak_shortcut_features(state)
                        for state in states.values()
                    }
                ),
                1,
            )
            self.assertTrue(
                all(
                    sum(
                        record["on_time"]
                        for record in state["days"]
                        if record["day"] > 0
                    )
                    == 4
                    for state in states.values()
                )
            )

    def test_cross_group_same_level_states_are_not_near_clones(self) -> None:
        by_cell = collections.defaultdict(list)
        for row in self.rows:
            by_cell[(row["family"], row["label"])].append(
                json.dumps(
                    row["state"],
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        for cell, states in by_cell.items():
            closest = max(
                difflib.SequenceMatcher(None, left, right).ratio()
                for left, right in itertools.combinations(states, 2)
            )
            self.assertLess(closest, 0.94, cell)


if __name__ == "__main__":
    unittest.main()
