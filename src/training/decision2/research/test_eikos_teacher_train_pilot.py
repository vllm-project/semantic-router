"""Gold-free contracts for the bounded external-teacher TRAIN screen."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from research.eikos_teacher_train_pilot import (
    aggregate,
    probabilities,
    question,
    roster,
    roster_sha256,
    write_once,
)


def row(kind: str, index: int, group: str | None = None) -> dict:
    keys = {
        "choice": ("left", "right"),
        "noul": ("true", "false"),
        "score": ("2", "0", "1"),
    }[kind]
    return {
        "id": f"{kind}-{index}",
        "group_id": group or f"{kind}-group-{index}",
        "input_sha256": f"sha-{kind}-{index}",
        "task_type": kind,
        "state": "A stated fact.",
        "instructions": "Choose from the offered meanings.",
        "options": [{"key": key, "description": f"meaning-{key}"} for key in keys],
        "label": 0,
    }


class FakeNative:
    def decide_all(self, *, state: str, questions: dict) -> dict:
        self.question = questions["decision"]
        kind = self.question["type"]
        answer = {"type": kind, "noul": 0.8}
        if kind == "choice":
            answer["probabilities"] = {"left": 0.7, "right": 0.3}
        elif kind == "score":
            answer["probabilities"] = {"0": 0.2, "1": 0.2, "2": 0.6}
        return {"decision": (answer, 1)}


class TeacherTrainPilotTest(unittest.TestCase):
    def test_roster_uses_groups_not_gold_labels(self) -> None:
        rows = [
            row(kind, index)
            for kind in ("choice", "noul", "score")
            for index in range(34)
        ]
        rows.append(row("choice", 99, rows[0]["group_id"]))
        original = roster(rows)
        changed = [{**item, "label": 1} for item in rows]
        self.assertEqual(roster_sha256(original), roster_sha256(roster(changed)))
        self.assertEqual(len(original), 96)
        self.assertEqual(len({item["group_id"] for item in original}), 96)

    def test_score_descriptions_follow_ordered_level_keys(self) -> None:
        self.assertEqual(
            question(row("score", 0))["criteria"],
            ["meaning-0", "meaning-1", "meaning-2"],
        )
        self.assertEqual(
            question(row("noul", 0))["criteria"],
            {"true": "meaning-true", "false": "meaning-false"},
        )

    def test_native_probability_contract_and_train_only_aggregates(self) -> None:
        fake = FakeNative()
        rows = [row(kind, 0) for kind in ("choice", "noul", "score")]
        report = aggregate(rows, fake)
        self.assertEqual(
            {kind: report[kind]["valid"] for kind in report},
            {kind: 1 for kind in report},
        )
        self.assertEqual(report["choice"]["correct"], 1)
        self.assertEqual(report["noul"]["correct"], 1)
        self.assertEqual(report["score"]["correct"], 1)
        self.assertNotIn("A stated fact.", str(report))
        with self.assertRaisesRegex(ValueError, "option keys"):
            probabilities(rows[0], {"type": "choice", "probabilities": {"wrong": 1.0}})

    def test_private_receipt_cannot_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "aggregate.json"
            write_once(destination, {"sampled_rows": 96})
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(ValueError, "fresh file"):
                write_once(destination, {"sampled_rows": 0})


if __name__ == "__main__":
    unittest.main()
