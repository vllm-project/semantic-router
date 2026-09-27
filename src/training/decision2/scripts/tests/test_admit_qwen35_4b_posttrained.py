"""CPU-only tests for the bounded official 4B source admission."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.admit_qwen35_4b_posttrained import _probs, _train_args, compare


def record(identifier: str, task_type: str, probability: float) -> dict:
    if task_type == "noul":
        answer = {"type": "noul", "noul": probability}
        prediction = "true" if probability > 0.5 else "false"
    else:
        answer = {
            "type": task_type,
            "probabilities": {"a": probability, "b": 1 - probability},
        }
        prediction = "a" if probability > 0.5 else "b"
    return {
        "id": identifier,
        "task_type": task_type,
        "prompt_sha256": "p" * 64,
        "token_ids_sha256": "t" * 64,
        "prediction_key": prediction,
        "answer": answer,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AdmissionContractTests(unittest.TestCase):
    def test_zero_repeat_requires_matching_type_and_probability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = (Path(directory) / name for name in ("a", "b"))
            rows = [
                record("1", "choice", 0.7),
                record("2", "noul", 0.8),
                record("3", "score", 0.6),
            ]
            write_rows(first, rows)
            write_rows(second, rows)
            self.assertEqual(compare(first, second, 3)["status"], "PASS")
            rows[1] = record("2", "noul", 0.1)
            write_rows(second, rows)
            failed = compare(first, second, 3)
            self.assertEqual(failed["status"], "FAIL")
            self.assertEqual(failed["categorical_changes"], 1)

    def test_rejects_changed_answer_type_or_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first, second = (Path(directory) / name for name in ("a", "b"))
            a = record("1", "choice", 0.7)
            b = record("2", "noul", 0.8)
            write_rows(first, [a, b])
            changed = dict(a)
            changed["answer"] = {"type": "score", "probabilities": {"a": 0.7, "b": 0.3}}
            write_rows(second, [changed, b])
            with self.assertRaisesRegex(ValueError, "answer type"):
                compare(first, second, 2)
            write_rows(second, [b, a])
            with self.assertRaisesRegex(ValueError, "identity or order"):
                compare(first, second, 2)

    def test_one_update_uses_frozen_training_flags(self) -> None:
        lock = {
            "source_path": "/source",
            "data_paths": {
                "train": "/data/train",
                "select": "/data/select",
                "cal": "/data/cal",
            },
        }
        zero = _train_args(lock, Path("/out/zero-a"), zero=True)
        one = _train_args(lock, Path("/out/one"), zero=False)
        self.assertIn("--zero-step-only", zero)
        self.assertNotIn("--zero-step-only", one)
        self.assertEqual(one[one.index("--max-steps") + 1], "1")
        self.assertEqual(zero[zero.index("--max-steps") + 1], "466")
        self.assertEqual(one[one.index("--objective") + 1], "ce_brier")
        self.assertEqual(one[one.index("--init-kind") + 1], "posttrained")
        self.assertEqual(one[one.index("--accumulation") + 1], "16")

    def test_rejects_missing_or_nonfinite_probabilities(self) -> None:
        row = record("1", "choice", 0.7)
        row["answer"]["probabilities"]["a"] = float("nan")
        with self.assertRaisesRegex(ValueError, "probability"):
            _probs(row)


if __name__ == "__main__":
    unittest.main()
