"""Visual layout contracts for the first-release v3 card figures."""

from __future__ import annotations

import re
import unittest
import xml.etree.ElementTree as ET
from itertools import pairwise

from publication.render_arena_v3 import (
    pareto_svg,
    public_pareto_svg,
    task_matrix_svg,
)


def _row(
    key: str, label: str, group: str, rank: int, score: float, size: float
) -> dict:
    return {
        "key": key,
        "label": label,
        "group": group,
        "rank": rank,
        "score": score,
        "size_b": size,
        "pareto_frontier": rank == 1,
        "axes": {"typed": 0.75, "transfer": 0.52},
        "task_scores": {
            "typed": {"choice": 0.972, "noul": 0.819, "score": 0.418},
            "transfer": {f"task_{index:02d}": 0.5 for index in range(15)},
        },
    }


class ArenaV3LayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _row("new", "Decision 2.0 4B", "decision2", 1, 62.67, 4.205751296),
            _row("kev", "Kev 4B", "open", 2, 59.23, 4.207062528),
            _row("old", "Decision 1.0 Nox", "decision1", 3, 56.47, 4.208383488),
        ]

    def _assert_spaced_labels(self, source: str, minimum: float) -> None:
        root = ET.fromstring(source)
        labels = [
            element
            for element in root.iter()
            if element.tag.endswith("text")
            and "".join(element.itertext()) in {row["label"] for row in self.rows}
        ]
        self.assertEqual(len(labels), 3)
        ys = sorted(float(label.attrib["y"]) for label in labels)
        self.assertTrue(all(b - a >= minimum for a, b in pairwise(ys)))
        self.assertTrue(
            any(
                element.attrib.get("class") == "label-leader" for element in root.iter()
            )
        )
        self.assertIn("4.21", source)  # Actual 4.2B-size tick, not just 5B.

    def test_near_tied_arena_pareto_labels_remain_separate(self) -> None:
        source = pareto_svg(
            {"schema_version": "jevarena-ranking/3", "models": self.rows}
        )
        self._assert_spaced_labels(source, 18)

    def test_near_tied_public_pareto_labels_remain_separate(self) -> None:
        public_rows = [dict(row) for row in self.rows]
        for row, score in zip(public_rows, (84.42, 75.76, 74.89)):
            row["score"] = score
        source = public_pareto_svg(
            {"schema_version": "jevarena-jevbench-public-rank/1", "models": public_rows}
        )
        self._assert_spaced_labels(source, 17)
        self.assertIn("not an official JevBench rank", source)

    def test_all_eighteen_tasks_are_legible_at_card_width(self) -> None:
        self.rows[2]["task_scores"]["typed"]["score"] = 0.445
        source = task_matrix_svg(
            {"schema_version": "jevarena-ranking/3", "models": self.rows}
        )
        root = ET.fromstring(source)
        self.assertLessEqual(int(root.attrib["width"]), 950)
        self.assertEqual(source.count("Human transfer · tasks"), 2)
        headers = [
            element.attrib["aria-label"]
            for element in root.iter()
            if element.tag.endswith("text") and "aria-label" in element.attrib
        ]
        self.assertEqual(len(headers), 18)
        self.assertIn("Typed / Score", headers)
        self.assertIn("Transfer / task 14", headers)
        numbers = [
            "".join(element.itertext())
            for element in root.iter()
            if element.tag.endswith("text")
            and re.fullmatch(r"\d{1,3}\.\d", "".join(element.itertext()))
        ]
        self.assertEqual(len(numbers), 54)
        self.assertIn("41.8", numbers)
        self.assertIn("44.5", numbers)


if __name__ == "__main__":
    unittest.main()
