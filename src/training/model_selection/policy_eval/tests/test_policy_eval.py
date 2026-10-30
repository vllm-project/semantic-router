from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.training.model_selection.policy_eval.policy_eval import (
    Table,
    cross_validated_gain,
    frontier,
    judge_auroc,
    main,
    points,
    read_scores,
    static_mix,
    stream_gain,
)
from src.training.model_selection.policy_eval.prepare_terminal_bench import (
    STRONG,
    build_rows,
)


def row(rid, task, **attempts):
    return {
        "id": rid,
        "task": task,
        "attempts": {m: {"ok": ok, "cost": cost} for m, (ok, cost) in attempts.items()},
    }


class StaticMixTest(unittest.TestCase):
    def test_hull_skips_a_dominated_model_and_interpolates(self):
        # mid lies below the line from cheap to strong, so random mixing of the
        # two ends beats always running mid at mid's own cost.
        rows = [row("r", "t", cheap=(0.5, 0.1), mid=(0.55, 0.2), strong=(0.9, 0.5))]
        table = Table(rows, "cheap", "strong", baselines=["mid"])
        self.assertAlmostEqual(static_mix(table, 0.2), 0.6)
        self.assertEqual(static_mix(table, 0.05), 0.0)
        self.assertAlmostEqual(static_mix(table, 1.0), 0.9)


class PolicyCostTest(unittest.TestCase):
    def setUp(self):
        # The cheap model solves tasks 0 and 2, and the score ranks the failures first.
        self.rows = [
            row(f"r{i}", f"t{i}", cheap=(ok, 0.1), strong=(1.0, 1.0))
            for i, ok in enumerate([1.0, 0.0, 1.0, 0.0])
        ]
        self.scores = {"r0": 0.1, "r1": 0.9, "r2": 0.2, "r3": 0.8}

    def test_escalation_pays_for_both_attempts(self):
        pts = points(
            Table(self.rows, "cheap", "strong", scores=self.scores), "escalate"
        )
        self.assertEqual(frontier(pts, 0.601), 1.0)
        self.assertEqual(frontier(pts, 0.599), 0.75)
        self.assertEqual(frontier(pts, 0.349), 0.5)
        self.assertEqual(frontier(pts, 0.05), 0.0)

    def test_boundary_pays_for_one_attempt(self):
        boundary = {f"t{i}": s for i, s in enumerate(self.scores.values())}
        pts = points(Table(self.rows, "cheap", "strong", boundary=boundary), "boundary")
        self.assertEqual(frontier(pts, 0.551), 1.0)

    def test_cli_writes_the_same_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, scores, out = (
                Path(tmp) / name
                for name in ("rows.jsonl", "scores.jsonl", "report.json")
            )
            rows.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
            scores.write_text(
                "".join(
                    json.dumps({"id": k, "score": v}) + "\n"
                    for k, v in self.scores.items()
                )
            )
            args = [
                "--rows",
                str(rows),
                "--cheap",
                "cheap",
                "--strong",
                "strong",
                "--scores",
                str(scores),
                "--budgets",
                "0.601",
                "--folds",
                "2",
                "--bootstrap",
                "10",
                "--json",
                str(out),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                main(args)
            report = json.loads(out.read_text())
        self.assertEqual(report["policies"]["escalate"]["resolved_at_budget"], [1.0])
        self.assertEqual(report["score_auroc"]["overall"], 1.0)


class ThresholdSelectionTest(unittest.TestCase):
    def test_cross_validation_chooses_thresholds_out_of_fold(self):
        # Folds are tasks[i::2]: t0 and t2, then t1 and t3, each task with two attempts.
        # Every score is 0.9. Escalating pays in the second fold and loses in the
        # first, so a threshold learned on one fold is wrong for the other. A
        # threshold chosen with the held-out rows, or folds that split a task's
        # attempts, would never lose this much.
        rows = []
        for i, ok in enumerate([1.0, 0.0, 1.0, 0.0]):
            rows += [
                row(f"r{i}{k}", f"t{i}", cheap=(ok, 0.1), strong=(1.0, 0.5))
                for k in range(2)
            ]
        scores = {r["id"]: 0.9 for r in rows}
        gain = cross_validated_gain(
            Table(rows, "cheap", "strong", scores=scores), "escalate", 2.0, folds=2
        )
        self.assertAlmostEqual(gain.mean(), -1.05)

    def test_stream_threshold_comes_from_the_first_tasks_only(self):
        # Escalation helps on a* (the cheap attempt fails) and hurts on b*. Over
        # every order of four tasks, a threshold set from the first two gains 0 on
        # the remaining two on average, and a threshold set from all four would
        # gain about +0.083.
        rows = [
            row(t, t, cheap=(ok, 0.1), strong=(1.0, 0.5))
            for t, ok in [("a1", 0.0), ("a2", 0.0), ("b1", 1.0), ("b2", 1.0)]
        ]
        scores = {"a1": 0.9, "a2": 0.9, "b1": 0.1, "b2": 0.1}
        table = Table(rows, "cheap", "strong", scores=scores)
        self.assertAlmostEqual(
            stream_gain(table, "escalate", 2.0, 2, draws=3000), 0.0, delta=0.02
        )
        self.assertIsNone(stream_gain(table, "escalate", 2.0, 4))


class AurocTest(unittest.TestCase):
    def test_within_task_ignores_what_only_tells_tasks_apart(self):
        # The score only says task H is harder than task E.
        rows, scores = [], {}
        for task, score, outcomes in (
            ("E", 0.2, [1.0, 1.0, 0.0]),
            ("H", 0.8, [0.0, 0.0, 1.0]),
        ):
            for k, ok in enumerate(outcomes):
                rows.append(row(f"{task}{k}", task, cheap=(ok, 0.1), strong=(1.0, 1.0)))
                scores[f"{task}{k}"] = score
        overall, within, pairs = judge_auroc(
            Table(rows, "cheap", "strong", scores=scores)
        )
        self.assertAlmostEqual(overall, 6 / 9)
        self.assertEqual(within, 0.5)
        self.assertEqual(pairs, 4)


class ReadScoresTest(unittest.TestCase):
    def write(self, tmp, name, *records):
        path = Path(tmp) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_repeated_id_is_rejected_with_file_line_and_id(self):
        cases = {
            "low first": [0.1, 0.9],
            "high first": [0.9, 0.1],
            "identical": [0.5, 0.5],
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (first, second) in cases.items():
                with self.subTest(name):
                    path = self.write(
                        tmp,
                        "scores.jsonl",
                        {"id": "a", "score": first},
                        {"id": "b", "score": 0.3},
                        {"id": "a", "score": second},
                    )
                    with self.assertRaisesRegex(
                        ValueError, f"^{path}:3: duplicate id 'a'$"
                    ):
                        read_scores(path)

    def test_reordered_unique_file_gives_the_same_mapping(self):
        records = [{"id": k, "score": s} for k, s in [("a", 0.1), ("b", 0.9)]]
        with tempfile.TemporaryDirectory() as tmp:
            forward = read_scores(self.write(tmp, "f.jsonl", *records))
            backward = read_scores(self.write(tmp, "b.jsonl", *records[::-1]))
        self.assertEqual(forward, {"a": 0.1, "b": 0.9})
        self.assertEqual(backward, forward)

    def test_cli_rejects_a_repeated_id_in_either_score_file(self):
        rows = [
            row(f"r{i}", f"t{i}", cheap=(ok, 0.1), strong=(1.0, 1.0))
            for i, ok in enumerate([1.0, 0.0, 1.0, 0.0])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = [
                "--rows",
                str(self.write(tmp, "rows.jsonl", *rows)),
                "--cheap",
                "cheap",
                "--strong",
                "strong",
                "--folds",
                "2",
                "--bootstrap",
                "10",
            ]
            for flag, prefix in (("--scores", "r"), ("--boundary-scores", "t")):
                ids = [f"{prefix}{i}" for i in (0, 1, 2, 3, 1)]
                path = self.write(
                    tmp, "dup.jsonl", *({"id": k, "score": 0.5} for k in ids)
                )
                with self.subTest(flag), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        ValueError, f"^{path}:5: duplicate id '{prefix}1'$"
                    ):
                        main(base + [flag, str(path)])


class TerminalBenchRowsTest(unittest.TestCase):
    def test_rows_keep_the_recorded_trials(self):
        def record(
            model, task, reward, cents, trial, agent_step=True, agent="terminus-2"
        ):
            steps = json.dumps(
                [{"src": "user"}] + ([{"src": "agent"}] if agent_step else [])
            )
            return {
                "task_name": task,
                "agent": agent,
                "model": model,
                "reward": reward,
                "cost_cents": cents,
                "trial_name": trial,
                "steps": steps,
            }

        records = [
            record("deepseek-v3.2@deepseek", "t", 1.0, 2.0, "a"),
            record("deepseek-v3.2@deepseek", "t", 0.0, 4.0, "b", agent_step=False),
            record(
                "deepseek-v3.2@deepseek", "t", 0.0, 9.0, "c", agent="other-scaffold"
            ),
            record("deepseek-v3.2@deepseek", "u", 1.0, 1.0, "a"),
        ]
        for m in STRONG:
            records += [
                record(m, "t", 1.0, 40.0, "s1"),
                record(m, "t", 0.0, 60.0, "s2"),
            ]
        records += [record(STRONG[0], "u", 1.0, 50.0, "s1")]
        rows = build_rows(records)
        # Trial b has no agent step, task u lacks three strong models, and the
        # other scaffold is not terminus-2.
        self.assertEqual([r["id"] for r in rows], ["t::deepseek-v3.2::a"])
        attempts = rows[0]["attempts"]
        self.assertEqual(attempts["deepseek-v3.2"], {"ok": 1.0, "cost": 0.02})
        self.assertEqual(attempts["deepseek-v3.2#retry"], {"ok": 0.0, "cost": 0.04})
        self.assertEqual(attempts["claude-opus-4-6"], {"ok": 0.5, "cost": 0.5})


if __name__ == "__main__":
    unittest.main()
