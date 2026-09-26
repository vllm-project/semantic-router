"""Charts expose score scope and keep rank/Pareto/matrix inputs consistent."""

from __future__ import annotations

import unittest

from jev_arena.render import matrix_svg, pareto_svg, ranking_svg


class RenderTest(unittest.TestCase):
    def test_public_rank_and_pareto_mark_scope_and_frontier(self) -> None:
        report = {
            "schema_version": "jevarena-jevbench-public-rank/1",
            "models": [
                {
                    "key": "old",
                    "rank": 2,
                    "label": "Decision 1",
                    "group": "decision1",
                    "score": 65.0,
                    "size_b": 2.0,
                    "pareto_frontier": False,
                },
                {
                    "key": "new",
                    "rank": 1,
                    "label": "Decision 2",
                    "group": "decision2",
                    "score": 75.0,
                    "size_b": 2.0,
                    "pareto_frontier": True,
                },
            ],
        }
        rank = ranking_svg(report)
        pareto = pareto_svg(report)
        self.assertIn("JevBench public-only", rank)
        self.assertIn("not an official JevBench rank", rank)
        self.assertIn("Decision 2", pareto)
        self.assertIn("filled = Pareto frontier", pareto)
        self.assertIn("#176b5b", pareto)

    def test_arena_matrix_uses_all_predeclared_axes(self) -> None:
        report = {
            "schema_version": "jevarena-ranking/1",
            "phase": "release",
            "models": [
                {
                    "key": "a",
                    "rank": 1,
                    "label": "A",
                    "group": "decision2",
                    "score": 60.0,
                    "size_b": 4.0,
                    "pareto_frontier": True,
                    "axes": {
                        "typed": 0.8,
                        "transfer": 0.5,
                        "authored": 0.7,
                        "robustness": 0.6,
                    },
                },
            ],
        }
        matrix = matrix_svg(report)
        for axis in ("Typed", "Transfer", "Authored", "Robustness"):
            self.assertIn(axis, matrix)
        self.assertIn("JevArena RELEASE", matrix)
        self.assertIn("50.0%", matrix)


if __name__ == "__main__":
    unittest.main()
