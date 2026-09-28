"""CPU contracts for the bounded own-Kai TRAIN-only teacher screen."""

from __future__ import annotations

import unittest

from research.small06_own_kai_teacher_probe import aggregate, question, roster


def row(kind: str, part: str, index: int) -> dict:
    if kind == "noul":
        options = [
            {"key": "false", "description": None},
            {"key": "true", "description": None},
        ]
    else:
        count = (
            3
            if kind == "score" and part == "other_three"
            else 5 if kind == "score" else 2
        )
        options = [
            {
                "key": str(i) if kind == "score" else f"option_{i}",
                "description": None if kind == "score" and i == 0 else f"meaning {i}",
            }
            for i in range(count)
        ]
    source = (
        "legacy:stage4-general-composition-v2" if part == "stage4" else "independent"
    )
    return {
        "id": f"{kind}/{part}/{index}",
        "group_id": f"{kind}/{part}/{index}",
        "source": source,
        "task_type": kind,
        "state": "state",
        "instructions": "decide",
        "options": options,
        "label": 1,
        "input_sha256": "0" * 64,
    }


class OwnKaiTeacherProbeTest(unittest.TestCase):
    def test_fixed_roster_has_independent_groups_and_score_levels(self) -> None:
        rows = [
            row(kind, part, index)
            for kind, part, count in (
                ("choice", "stage4", 20),
                ("choice", "other", 20),
                ("noul", "stage4", 20),
                ("noul", "other", 20),
                ("score", "stage4", 20),
                ("score", "other_three", 12),
                ("score", "other_non_three", 12),
            )
            for index in range(count)
        ]
        selected = roster(rows)
        self.assertEqual(len(selected), 96)
        self.assertEqual(len({item["group_id"] for item in selected}), 96)
        self.assertEqual(
            sum(
                item["task_type"] == "score" and len(item["options"]) == 3
                for item in selected
            ),
            8,
        )
        self.assertEqual(
            [item["id"] for item in selected],
            [item["id"] for item in roster(list(reversed(rows)))],
        )

    def test_native_renderer_preserves_noul_and_ordinal_score(self) -> None:
        self.assertNotIn("criteria", question(row("noul", "other", 0)))
        score = question(row("score", "other_three", 0))
        self.assertEqual(score["criteria"], ["0", "meaning 1", "meaning 2"])
        self.assertEqual(
            list(question(row("choice", "other", 0))["criteria"]),
            ["option_0", "option_1"],
        )

    def test_aggregate_checks_all_native_probability_keys(self) -> None:
        rows = [
            row("choice", "other", 0),
            row("noul", "other", 1),
            row("score", "other_three", 2),
        ]

        def decide(_state: str, item: dict) -> dict:
            if item["type"] == "choice":
                answer = {
                    "type": "choice",
                    "probabilities": {"option_0": 0.1, "option_1": 0.9},
                }
            elif item["type"] == "noul":
                answer = {"type": "noul", "noul": 0.9}
            else:
                answer = {
                    "type": "score",
                    "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
                }
            return {"model": "Decision-1.0-Kai", "answers": {"decision": answer}}

        report = aggregate(rows, decide)
        self.assertEqual(
            [report[k]["valid"] for k in ("choice", "noul", "score")], [1, 1, 1]
        )
        self.assertEqual(
            [report[k]["correct"] for k in ("choice", "noul", "score")], [1, 1, 1]
        )
        with self.assertRaisesRegex(ValueError, "option keys"):
            aggregate(
                [rows[0]],
                lambda _state, _item: {
                    "model": "Decision-1.0-Kai",
                    "answers": {
                        "decision": {"type": "choice", "probabilities": {"wrong": 1.0}}
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
