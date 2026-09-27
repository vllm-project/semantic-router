"""Synthetic-only checks of the joint v3 paired confidence interval."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.generate import generate
from jev_arena.compare_v3 import _panel_digest, _task_f1, compare
from transfer.build import EVALUATION_TASKS, PANEL_VERSION, sha_file


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _answer(question: dict, gold: dict) -> dict:
    qtype = question["type"]
    if qtype == "choice":
        return {"type": "choice", "choice": gold["value"]}
    if qtype == "noul":
        return {"type": "noul", "noul": float(gold["value"])}
    return {"type": "score", "score": float(gold["value"])}


def _fixture(root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    # The generator is used only with a fixed test seed; this is not release gold.
    typed = generate("final", b"synthetic-comparator-fixture", groups_per_family=100)
    typed_left = []
    typed_right = []
    for index, row in enumerate(typed):
        prediction = {
            "id": row["id"],
            "source_input_sha256": row["provenance"]["payload_sha256"],
            "answers": {
                key: _answer(question, row["gold"][key])
                for key, question in row["questions"].items()
            },
        }
        typed_left.append(prediction)
        if index % 4 != 0:
            typed_right.append(prediction)
    css = []
    css_left = []
    css_right = []
    for task_index, task in enumerate(EVALUATION_TASKS):
        count = 437 if task_index < 7 else 436
        for item_index in range(count):
            label = ("a", "b")[item_index % 2]
            item_id = f"synthetic-css/{task}/{item_index}"
            input_sha = hashlib.sha256(item_id.encode()).hexdigest()
            row = {
                "id": item_id,
                "panel_version": PANEL_VERSION,
                "task": task,
                "role": "evaluation",
                "gold": label,
                "labels": ["a", "b"],
                "input_sha256": input_sha,
            }
            prediction = {
                "id": item_id,
                "source_input_sha256": input_sha,
                "answers": {
                    "label": {
                        "type": "choice",
                        "choice": label,
                        "probabilities": {
                            "a": float(label == "a"),
                            "b": float(label == "b"),
                        },
                    }
                },
            }
            css.append(row)
            css_left.append(prediction)
            if item_index % 4 != 0:
                css_right.append(prediction)
    paths = tuple(
        root / name
        for name in (
            "typed-gold.jsonl",
            "css-gold.jsonl",
            "left-typed.jsonl",
            "left-css.jsonl",
            "right-typed.jsonl",
            "right-css.jsonl",
        )
    )
    for path, rows in zip(
        paths, (typed, css, typed_left, css_left, typed_right, css_right)
    ):
        _write(path, rows)
    return paths


class PairedV3Test(unittest.TestCase):
    def test_fixed_universe_and_invalid_prediction_are_misses(self) -> None:
        class DrawBoth:
            positions = iter((0, 1))

            def randrange(self, _count):
                return next(self.positions)

        left, right = _task_f1([(0, 0, 0), (1, 1, -1)], 2, DrawBoth())
        self.assertEqual((left, right), (1.0, 0.5))

    def test_joint_paired_interval_is_bound_to_full_synthetic_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            result = compare(*paths, left_name="test-2.0", right_name="test-1.0")
            self.assertEqual(result["schema_version"], "jevarena-v3-paired-aggregate/1")
            self.assertEqual(result["coverage"]["typed_independent_groups"], 400)
            self.assertEqual(result["coverage"]["css_evaluation_items"], 6547)
            self.assertEqual(result["coverage"]["css_evaluation_tasks"], 15)
            self.assertEqual(result["replicates"], 5000)
            self.assertEqual(result["point"]["left"]["T"], 1.0)
            self.assertEqual(result["point"]["left"]["H"], 1.0)
            self.assertEqual(result["point"]["left"]["score"], 100.0)
            self.assertGreater(result["point"]["delta"]["score"], 0)
            self.assertGreater(result["ci95"]["low"], 0)
            self.assertLessEqual(result["ci95"]["low"], result["ci95"]["high"])
            self.assertEqual(result["typed_gold_sha256"], sha_file(paths[0]))
            self.assertEqual(result["css_gold_sha256"], sha_file(paths[1]))
            self.assertEqual(
                result["panel_sha256"],
                _panel_digest(sha_file(paths[0]), sha_file(paths[1])),
            )
            self.assertEqual(
                result["predictions_sha256"]["right"]["css"], sha_file(paths[5])
            )
            self.assertTrue(math.isfinite(result["axis_ci95"]["H"]["delta"]["high"]))

    def test_reject_insufficient_draws_before_opening_gold(self) -> None:
        missing = Path("does-not-exist.jsonl")
        with self.assertRaisesRegex(ValueError, "at least 5000"):
            compare(
                missing,
                missing,
                missing,
                missing,
                missing,
                missing,
                left_name="a",
                right_name="b",
                replicates=4999,
            )

    def test_identical_native_predictions_have_exact_zero_interval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            typed_gold, css_gold, left_typed, left_css, _, _ = _fixture(Path(temporary))
            result = compare(
                typed_gold,
                css_gold,
                left_typed,
                left_css,
                left_typed,
                left_css,
                left_name="test-2.0",
                right_name="test-1.0-clone",
            )
            self.assertEqual(result["point"]["delta"]["score"], 0.0)
            self.assertEqual(result["ci95"], {"low": 0.0, "high": 0.0})
            self.assertEqual(
                result["axis_ci95"]["T"]["delta"],
                {"low": 0.0, "high": 0.0},
            )
            self.assertEqual(
                result["axis_ci95"]["H"]["delta"],
                {"low": 0.0, "high": 0.0},
            )


if __name__ == "__main__":
    unittest.main()
