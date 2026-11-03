"""Paid-exploration accounting for end-of-attempt escalation (#5374).

Run from the repository root:
    python -m unittest discover -s src/training/model_selection/escalation_stream/tests -p 'test_*.py'
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.training.model_selection.escalation_stream.explore import (  # noqa: E402
    ALWAYS,
    apply_threshold,
    evaluate,
    fit_threshold,
    group_rows,
    hindsight_gain,
    load_scores,
    main,
    replay,
    stream_gains,
)


def row(row_id, task, cheap, strong, retry=None, score=None):
    attempts = {"cheap": {"ok": cheap[0], "cost": cheap[1]}, "strong": _attempt(strong)}
    if retry is not None:
        attempts["cheap#retry"] = _attempt(retry)
    return {"id": row_id, "task": task, "attempts": attempts, "score": score}


def _attempt(pair):
    return {"ok": pair[0], "cost": pair[1]}


def scores_of(rows):
    return {item["id"]: item["score"] for item in rows}


def play(rows, bits, value=2.0):
    order, grouped = group_rows(rows)
    return replay(grouped, order, scores_of(rows), "cheap", "strong", value, bits)


class ThresholdTest(unittest.TestCase):
    def test_empty_observations_never_escalate(self):
        self.assertTrue(math.isinf(fit_threshold([], {}, "cheap", "strong", 2.0)))

    def test_zero_exploration_pays_only_the_cheap_attempt(self):
        rows = [
            row("a", "a", (0.0, 0.1), (1.0, 0.5), score=0.99),
            row("b", "b", (0.0, 0.1), (1.0, 0.5), score=0.99),
        ]
        played = play(rows, [False, False])
        self.assertEqual(played.escalated, [False, False])
        self.assertEqual(played.cost, [0.1, 0.1])
        self.assertTrue(all(math.isinf(threshold) for threshold in played.thresholds))

    def test_forced_escalation_pays_both_and_moves_the_next_task(self):
        rows = [
            row("a", "a", (0.0, 0.1), (1.0, 0.5), score=0.2),
            row("b", "b", (0.0, 0.1), (1.0, 0.5), score=0.9),
        ]
        played = play(rows, [True, False])
        self.assertEqual(played.escalated, [True, True])
        self.assertEqual(played.cost, [0.6, 0.6])
        self.assertEqual(played.ok, [1.0, 1.0])
        self.assertTrue(math.isinf(played.thresholds[0]))
        self.assertEqual(played.thresholds[1], ALWAYS)
        self.assertAlmostEqual(played.gain(rows, 2.0, "cheap", "strong"), -0.1)

    def test_a_later_outcome_does_not_move_the_current_threshold(self):
        early = row("a", "a", (0.0, 0.1), (1.0, 0.5), score=0.2)
        later_ok = row("b", "b", (0.0, 0.1), (1.0, 0.5), score=0.9)
        later_bad = row("b", "b", (0.0, 0.1), (0.0, 0.5), score=0.9)
        bits = [True, False]
        first = play([early, later_ok], bits)
        second = play([early, later_bad], bits)
        self.assertEqual(first.thresholds[1], second.thresholds[1])
        self.assertEqual(first.escalated[1], second.escalated[1])
        failed = row("a", "a", (0.0, 0.1), (0.0, 0.5), score=0.2)
        blocked = play([failed, later_ok], bits)
        self.assertTrue(math.isinf(blocked.thresholds[1]))
        self.assertFalse(blocked.escalated[1])

    def test_rows_of_one_task_share_the_threshold_from_before_it(self):
        rows = [
            row("t1", "t", (0.0, 0.1), (1.0, 0.5), score=0.2),
            row("t2", "t", (0.0, 0.1), (1.0, 0.5), score=0.9),
            row("u", "u", (0.0, 0.1), (1.0, 0.5), score=0.9),
        ]
        played = play(rows, [True, False, False])
        self.assertEqual(played.escalated, [True, False, True])
        self.assertEqual(played.cost, [0.6, 0.1, 0.6])
        self.assertEqual(played.thresholds[2], ALWAYS)


class HindsightAndRetryTest(unittest.TestCase):
    def _pair(self, retry):
        return [
            row("a", "a", (0.0, 0.1), (1.0, 0.5), retry=retry[0], score=0.2),
            row("b", "b", (0.0, 0.1), (1.0, 0.5), retry=retry[1], score=0.9),
        ]

    def test_hindsight_sees_outcomes_exploration_has_not_paid_for(self):
        rows = self._pair(((0.0, 0.1), (0.0, 0.1)))
        scores = scores_of(rows)
        explored = stream_gains(
            rows, scores, "cheap", "strong", 2.0, 0.0, 1, 0, explore_bits=[False, False]
        )
        self.assertAlmostEqual(explored["explore_gain_mean"], -1.6)
        self.assertAlmostEqual(
            hindsight_gain(rows, scores, "cheap", "strong", 2.0, "strong"), -0.1
        )
        played = apply_threshold(
            rows,
            scores,
            "cheap",
            "strong",
            fit_threshold(rows, scores, "cheap", "strong", 2.0),
        )
        self.assertEqual(played.escalated, [True, True])
        self.assertEqual(played.thresholds[0], ALWAYS)

    def test_retry_matches_escalation_when_the_second_attempt_matches(self):
        rows = self._pair(((1.0, 0.5), (1.0, 0.5)))
        scores = scores_of(rows)
        gains = stream_gains(
            rows,
            scores,
            "cheap",
            "strong",
            2.0,
            0.0,
            1,
            0,
            retry="cheap#retry",
            explore_bits=[True, False],
        )
        self.assertAlmostEqual(
            gains["explore_gain_mean"], gains["retry_explore_gain_mean"]
        )

    def test_retry_falls_behind_when_the_second_attempt_fails(self):
        rows = self._pair(((0.0, 0.1), (0.0, 0.1)))
        gains = stream_gains(
            rows,
            scores_of(rows),
            "cheap",
            "strong",
            2.0,
            0.0,
            1,
            0,
            retry="cheap#retry",
            explore_bits=[True, False],
        )
        self.assertAlmostEqual(gains["explore_gain_mean"], -0.1)
        self.assertAlmostEqual(gains["retry_explore_gain_mean"], -1.65)
        self.assertGreater(gains["explore_gain_mean"], gains["retry_explore_gain_mean"])

    def test_same_seed_replays_the_same_stream(self):
        rows = [
            row(f"t{i}", f"t{i}", (float(i % 2), 0.1), (1.0, 0.4), score=i / 8)
            for i in range(8)
        ]
        scores = scores_of(rows)
        first = evaluate(rows, scores, "cheap", "strong", orders=5, seed=0, epsilon=0.5)
        second = evaluate(
            rows, scores, "cheap", "strong", orders=5, seed=0, epsilon=0.5
        )
        other = evaluate(rows, scores, "cheap", "strong", orders=5, seed=1, epsilon=0.5)
        self.assertEqual(first, second)
        self.assertNotEqual(first["values"], other["values"])


class ScoreLoaderTest(unittest.TestCase):
    def _write(self, directory, name, lines):
        path = Path(directory) / name
        path.write_text("".join(line + "\n" for line in lines))
        return path

    def test_conflicting_duplicates_name_the_file_and_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            forward = self._write(
                tmp,
                "forward.jsonl",
                ['{"id": "a", "score": 0.1}', '{"id": "a", "score": 0.9}'],
            )
            reverse = self._write(
                tmp,
                "reverse.jsonl",
                ['{"id": "a", "score": 0.9}', '{"id": "a", "score": 0.1}'],
            )
            with self.assertRaises(ValueError) as first:
                load_scores(forward)
            with self.assertRaises(ValueError) as second:
                load_scores(reverse)
        self.assertIn(f"{forward}:2:", str(first.exception))
        self.assertIn(f"{reverse}:2:", str(second.exception))

    def test_identical_duplicates_and_reordering_keep_one_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            identical = self._write(
                tmp,
                "same.jsonl",
                ['{"id": "a", "score": 0.1}', '{"id": "a", "score": 0.1}'],
            )
            left = self._write(
                tmp,
                "left.jsonl",
                ['{"id": "a", "score": 0.1}', '{"id": "b", "score": 0.2}'],
            )
            right = self._write(
                tmp,
                "right.jsonl",
                ['{"id": "b", "score": 0.2}', '{"id": "a", "score": 0.1}'],
            )
            self.assertEqual(load_scores(identical), {"a": 0.1})
            self.assertEqual(load_scores(left), load_scores(right))


class ReportCliTest(unittest.TestCase):
    def test_cli_writes_the_hindsight_and_explore_gains(self):
        rows = [
            row("a", "a", (0.0, 0.1), (1.0, 0.5), score=0.2),
            row("b", "b", (0.0, 0.1), (1.0, 0.5), score=0.9),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows_path = root / "rows.jsonl"
            scores_path = root / "scores.jsonl"
            out = root / "report.json"
            rows_path.write_text(
                "".join(
                    json.dumps({k: v for k, v in item.items() if k != "score"}) + "\n"
                    for item in rows
                )
            )
            scores_path.write_text(
                "".join(
                    json.dumps({"id": item["id"], "score": item["score"]}) + "\n"
                    for item in rows
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--rows",
                        str(rows_path),
                        "--scores",
                        str(scores_path),
                        "--cheap",
                        "cheap",
                        "--strong",
                        "strong",
                        "--epsilon",
                        "0",
                        "--orders",
                        "1",
                        "--values",
                        "2",
                        "--json",
                        str(out),
                    ]
                )
            report = json.loads(out.read_text())
        self.assertEqual(report["hindsight"], "in-sample")
        self.assertIsNone(report["retry"])
        self.assertAlmostEqual(report["values"]["2"]["explore_gain_mean"], -1.6)
        self.assertAlmostEqual(report["values"]["2"]["hindsight_gain"], -0.1)
        self.assertNotIn("retry_explore_gain_mean", report["values"]["2"])


if __name__ == "__main__":
    unittest.main()
