"""Exercise Decision-to-native mappings and fail-closed SELECT scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.gliner25.pilot import (
    question_and_target,
    score_select,
    select_training_rows,
)


def row(identifier: str, kind: str, label: int = 1) -> dict:
    options = [
        {"key": "A (unsafe)", "description": {"value": "first"}},
        {"key": "B", "description": "second"},
    ]
    if kind == "noul":
        options = [
            {"key": "false", "description": "No"},
            {"key": "true", "description": "Yes"},
        ]
    if kind == "score":
        options = [
            {"key": "2", "description": "high"},
            {"key": "0", "description": "low"},
            {"key": "1", "description": "medium"},
        ]
    return {
        "id": identifier,
        "task_type": kind,
        "state": "A context",
        "instructions": {"question": "Choose one"},
        "options": options,
        "label": label,
        "family": "sample",
    }


class PilotTest(unittest.TestCase):
    def test_question_mapping_preserves_native_semantics_without_gold(self):
        choice, key = question_and_target(row("c", "choice"))
        self.assertEqual(key, "B")
        self.assertEqual(choice["criteria"]["A (unsafe)"], '{"value":"first"}')
        self.assertNotIn("label", json.dumps(choice))

        noul, key = question_and_target(row("n", "noul", 0))
        self.assertEqual((noul["type"], key), ("noul", "no"))

        score, key = question_and_target(row("s", "score", 0))
        self.assertEqual(key, "2")
        self.assertEqual(score["criteria"], ["low", "medium", "high"])

    def test_task_quota_hash_selection_is_stable(self):
        pool = [row(f"c{i}", "choice") for i in range(5)]
        pool += [row(f"n{i}", "noul") for i in range(4)]
        pool += [row(f"s{i}", "score") for i in range(3)]
        quotas = {"choice": 3, "noul": 2, "score": 1}
        first = select_training_rows(pool, quotas)
        second = select_training_rows(list(reversed(pool)), quotas)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        with self.assertRaisesRegex(ValueError, "admissible score"):
            select_training_rows(pool, {"score": 4})

    def test_select_includes_invalids_and_uses_score_native_level(self):
        rows = [row("c", "choice"), row("n", "noul"), row("s", "score", 0)]
        predictions = [
            {"id": "c", "answer": {"type": "choice", "choice": "B"}},
            {"id": "n", "answer": {"type": "noul", "error": "context_overflow"}},
            {"id": "s", "answer": {"type": "score", "native_level": 2}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text("".join(json.dumps(x) + "\n" for x in predictions))
            result = score_select(rows, path)
            self.assertEqual((result["correct"], result["total"]), (2, 3))
            self.assertEqual(result["by_type"]["noul"]["correct"], 0)
            bad = predictions.copy()
            bad[1] = {"id": "other", "answer": bad[1]["answer"]}
            path.write_text("".join(json.dumps(x) + "\n" for x in bad))
            with self.assertRaisesRegex(ValueError, "IDs or order"):
                score_select(rows, path)


if __name__ == "__main__":
    unittest.main()
