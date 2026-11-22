from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.training.model_selection.policy_eval.candidate_eval import (
    CandidateTable,
    calibration,
    choose_global,
    choose_local,
    choose_utility,
    cross_validated_gain,
    knapsack,
    main,
    oracle_frontier,
    plug_in_gain,
    read_estimates,
)
from src.training.model_selection.policy_eval.policy_eval import Table
from src.training.model_selection.policy_eval.policy_eval import (
    oracle_frontier as pair_oracle_frontier,
)
from src.training.model_selection.policy_eval.prepare_sr_bench import build_rows


def table(outcomes, estimates, baselines=()):
    """outcomes: {task: {candidate: (ok, cost)}}; estimates: {task: {candidate: (p, c)}}."""
    rows = [
        {
            "id": t,
            "task": t,
            "attempts": {m: {"ok": ok, "cost": cost} for m, (ok, cost) in a.items()},
        }
        for t, a in outcomes.items()
    ]
    est = {
        (t, m): {"success_probability": p, "estimated_cost": c}
        for t, e in estimates.items()
        for m, (p, c) in e.items()
    }
    return rows, est, CandidateTable(rows, est, baselines=baselines)


class ChoiceTest(unittest.TestCase):
    def test_utility_trades_estimated_success_against_estimated_cost(self):
        p = np.array([[0.5, 0.9, 0.9]])
        c = np.array([[0.1, 0.5, 0.4]])
        # 2 * p - c: 0.9, 1.3, 1.4. The third has the same p as the second and costs less.
        self.assertEqual(choose_utility(p, c, 2.0).tolist(), [2])
        self.assertEqual(choose_utility(p, c, 0.0).tolist(), [0])
        self.assertEqual(choose_utility(p, c, np.inf).tolist(), [2])

    def test_local_takes_the_most_likely_candidate_within_the_ceiling(self):
        p = np.array([[0.5, 0.8, 0.9], [0.5, 0.8, 0.9]])
        c = np.array([[0.1, 0.3, 0.6], [0.4, 0.5, 0.6]])
        # The second row has nothing within 0.35, so it takes its cheapest estimate.
        self.assertEqual(choose_local(p, c, 0.35).tolist(), [1, 0])

    def test_global_moves_budget_from_an_easy_task_to_a_hard_one(self):
        # A 0.3 ceiling keeps both tasks on the cheap candidate. The same budget on
        # average buys the strong candidate for the hard task. With 30 units per
        # task both costs fall on the grid, so 0.1 + 0.5 fits exactly.
        p = np.array([[0.9, 0.95], [0.2, 0.9]])
        c = np.array([[0.1, 0.5], [0.1, 0.5]])
        self.assertEqual(choose_local(p, c, 0.3).tolist(), [0, 0])
        pick, infeasible = choose_global(p, c, 0.3, resolution=30)
        self.assertEqual(pick.tolist(), [0, 1])
        self.assertEqual(infeasible, 0)

    def test_rounding_up_can_exclude_an_upgrade_at_the_budget(self):
        # With 50 units per task a unit is 0.006. The 0.4 the strong candidate
        # adds rounds up to 67 units, and the 0.4 left after the cheap
        # assignment holds 66, so the exact fit is not taken.
        p = np.array([[0.9, 0.95], [0.2, 0.9]])
        c = np.array([[0.1, 0.5], [0.1, 0.5]])
        pick, infeasible = choose_global(p, c, 0.3, resolution=50)
        self.assertEqual((pick.tolist(), infeasible), ([0, 0], 0))

    def test_rounding_never_makes_the_cheapest_assignment_infeasible(self):
        # The cheapest estimates sum to exactly the budget of two tasks. Rounding
        # each cost up on its own (0.999 to 50 units of 0.02, 1.001 to 51) would
        # need 101 of 100 units and report the batch as infeasible.
        p = np.array([[0.5, 0.9], [0.5, 0.9]])
        c = np.array([[0.999, 1.5], [1.001, 1.5]])
        pick, infeasible = choose_global(p, c, 1.0, resolution=50)
        self.assertEqual((pick.tolist(), infeasible), ([0, 0], 0))
        # A cent less and the cheapest estimates no longer fit.
        self.assertEqual(choose_global(p, c, 0.99, resolution=50)[1], 1)

    def test_fit_does_not_depend_on_the_resolution(self):
        # A thousand cheapest estimates of 0.1 sum to slightly more than the
        # budget of 0.1 per task in floating point. That error must not decide
        # whether the batch fits, at any resolution.
        p = np.tile([0.5, 0.9], (1000, 1))
        c = np.tile([0.1, 0.3], (1000, 1))
        self.assertGreater(c.min(axis=1).sum(), 0.1 * 1000)
        for resolution in (50, 4000):
            pick, infeasible = choose_global(p, c, 0.1, resolution=resolution)
            self.assertEqual((pick.tolist(), infeasible), ([0] * 1000, 0))

    def test_global_takes_the_cheapest_estimate_when_no_assignment_fits(self):
        p = np.array([[0.5, 0.9]] * 3)
        c = np.array([[0.2, 0.5]] * 3)
        pick, infeasible = choose_global(p, c, 0.1, batch=2)
        self.assertEqual(pick.tolist(), [0, 0, 0])
        self.assertEqual(infeasible, 2)


class KnapsackTest(unittest.TestCase):
    def test_matches_exhaustive_search_on_rounded_costs(self):
        rng = np.random.default_rng(0)
        for _ in range(300):
            rows, k = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            p = rng.random((rows, k))
            c = rng.choice([0.0, 0.05, 0.1, 0.2, 0.35, 0.5], size=(rows, k))
            budget = float(rng.choice([0.05, 0.1, 0.2, 0.3]))
            resolution = int(rng.choice([1, 4, 20]))
            # What each candidate adds to its row's cheapest estimate, in units
            # of budget / resolution, against the budget the cheapest leave.
            base = c.min(axis=1)
            unit = budget / resolution
            units = np.ceil(np.round((c - base[:, None]) / unit, 9))
            slack = budget * rows - base.sum()
            pick = knapsack(p, c, budget, resolution)
            if slack < 0 and not np.isclose(slack, 0):
                self.assertIsNone(pick)
                continue
            capacity = np.floor(np.round(max(slack, 0) / unit, 9))
            rowsel = np.arange(rows)
            feasible = [
                (p[rowsel, list(a)].sum(), units[rowsel, list(a)].sum())
                for a in itertools.product(range(k), repeat=rows)
                if units[rowsel, list(a)].sum() <= capacity
            ]
            best = max(v for v, _ in feasible)
            fewest = min(u for v, u in feasible if np.isclose(v, best))
            self.assertAlmostEqual(p[rowsel, pick].sum(), best)
            self.assertEqual(units[rowsel, pick].sum(), fewest)
            self.assertLessEqual(c[rowsel, pick].sum(), budget * rows + 1e-9)

    def test_rounding_up_never_exceeds_the_budget_on_estimates(self):
        rng = np.random.default_rng(1)
        p, c = rng.random((40, 4)), rng.random((40, 4)) * 0.4
        for budget in (0.05, 0.1, 0.2):
            pick, _ = choose_global(p, c, budget, batch=10, resolution=7)
            for start in range(0, 40, 10):
                part = slice(start, start + 10)
                spent = c[part][np.arange(10), pick[part]].sum()
                cheapest = c[part].min(axis=1).sum()
                self.assertTrue(spent <= 10 * budget + 1e-12 or spent == cheapest)


class OracleTest(unittest.TestCase):
    def test_oracle_bounds_every_choice_of_candidate_per_task(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n, k = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            names = [f"m{j}" for j in range(k)]
            outcomes = {
                f"t{i}": {
                    m: (
                        float(rng.choice([0.0, 1.0, rng.random()])),
                        float(rng.random()),
                    )
                    for m in names
                }
                for i in range(n)
            }
            _, _, tab = table(
                outcomes, {t: dict.fromkeys(names, (0.5, 0.1)) for t in outcomes}
            )
            ok = np.stack([tab.ok[m] for m in names], axis=1)
            cost = np.stack([tab.cost[m] for m in names], axis=1)
            picks = [
                (cost[np.arange(n), a].mean(), ok[np.arange(n), a].mean())
                for a in map(list, itertools.product(range(k), repeat=n))
            ]
            for budget in np.linspace(0, 1.2, 13):
                best = max([y for x, y in picks if x <= budget] + [0.0])
                self.assertGreaterEqual(oracle_frontier(tab, budget), best - 1e-12)

    def test_two_candidates_match_the_pair_oracle(self):
        rng = np.random.default_rng(2)
        for _ in range(30):
            outcomes = {
                f"t{i}": {
                    "cheap": (float(rng.choice([0.0, 1.0])), float(rng.random() * 0.1)),
                    "strong": (float(rng.choice([0.0, 1.0])), float(rng.random())),
                }
                for i in range(6)
            }
            rows, _, tab = table(
                outcomes,
                {t: {"cheap": (0.5, 0.1), "strong": (0.5, 0.1)} for t in outcomes},
            )
            pair = Table(rows, "cheap", "strong")
            for budget in np.linspace(0, 1.0, 11):
                self.assertAlmostEqual(
                    oracle_frontier(tab, budget), pair_oracle_frontier(pair, budget)
                )


class UtilityGainTest(unittest.TestCase):
    def test_cross_validation_corrects_the_scale_of_uncalibrated_estimates(self):
        # B always solves and A never does, but the estimates are too close for
        # V=1 to pay for B: 0.4 - 0.1 > 0.6 - 0.5. A value chosen on other tasks
        # learns to pay for B.
        names = [f"t{i}" for i in range(6)]
        _, _, tab = table(
            {t: {"A": (0.0, 0.1), "B": (1.0, 0.5)} for t in names},
            {t: {"A": (0.4, 0.1), "B": (0.6, 0.5)} for t in names},
        )
        self.assertAlmostEqual(plug_in_gain(tab, 1.0).mean(), -0.6)
        self.assertAlmostEqual(cross_validated_gain(tab, 1.0, folds=3).mean(), 0.0)


class CalibrationTest(unittest.TestCase):
    def test_brier_ece_and_auroc(self):
        p = np.array([0.9, 0.9, 0.1, 0.1])
        ok = np.array([1.0, 0.0, 0.0, 0.0])
        brier, ece, area = calibration(p, ok)
        self.assertAlmostEqual(brier, (0.01 + 0.81 + 0.01 + 0.01) / 4)
        # Bin 0.9 holds half the rows at mean outcome 0.5, bin 0.1 the other half at 0.
        self.assertAlmostEqual(ece, 0.5 * 0.4 + 0.5 * 0.1)
        self.assertAlmostEqual(area, 5 / 6)


class InputTest(unittest.TestCase):
    def write(self, tmp, name, *records):
        path = Path(tmp) / name
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_repeated_estimate_is_rejected_with_file_and_line(self):
        record = {
            "id": "a",
            "candidate": "m",
            "success_probability": 0.5,
            "estimated_cost": 0.1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(
                tmp, "e.jsonl", record, {**record, "candidate": "n"}, record
            )
            with self.assertRaisesRegex(
                ValueError, f"^{path}:3: duplicate estimate for id 'a', candidate 'm'$"
            ):
                read_estimates(path)

    def test_out_of_range_estimate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ({"success_probability": 1.2}, {"estimated_cost": -0.1}):
                with self.subTest(bad):
                    path = self.write(
                        tmp,
                        "e.jsonl",
                        {
                            "id": "a",
                            "candidate": "m",
                            "success_probability": 0.5,
                            "estimated_cost": 0.1,
                            **bad,
                        },
                    )
                    with self.assertRaisesRegex(ValueError, f"^{path}:1: "):
                        read_estimates(path)

    def test_rows_without_every_outcome_and_estimate_are_skipped(self):
        rows, est, _ = table(
            {"t0": {"A": (1.0, 0.1), "B": (1.0, 0.5)}, "t1": {"A": (1.0, 0.1)}},
            {"t0": {"A": (0.5, 0.1), "B": (0.9, 0.5)}, "t1": {"A": (0.5, 0.1)}},
        )
        del est[("t0", "B")]
        rows.append({"id": "t2", "task": "t2", "attempts": rows[0]["attempts"]})
        est.update({("t2", "A"): est[("t0", "A")], ("t2", "B"): est[("t0", "A")]})
        tab = CandidateTable(rows, est, candidates=["A", "B"])
        self.assertEqual([r["id"] for r in tab.rows], ["t2"])
        self.assertEqual(tab.rows_read, 3)


class CliTest(unittest.TestCase):
    def test_cli_writes_the_report(self):
        outcomes = {
            "easy": {"A": (1.0, 0.1), "B": (1.0, 0.5), "mom": (1.0, 0.3)},
            "hard": {"A": (0.0, 0.1), "B": (1.0, 0.5), "mom": (0.0, 0.3)},
        }
        estimates = {
            "easy": {"A": (0.9, 0.1), "B": (0.95, 0.5)},
            "hard": {"A": (0.2, 0.1), "B": (0.9, 0.5)},
        }
        with tempfile.TemporaryDirectory() as tmp:
            rows, est, out = (
                Path(tmp) / n for n in ("rows.jsonl", "est.jsonl", "out.json")
            )
            rows.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": t,
                            "task": t,
                            "attempts": {
                                m: {"ok": o, "cost": c} for m, (o, c) in a.items()
                            },
                        }
                    )
                    + "\n"
                    for t, a in outcomes.items()
                )
            )
            est.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": t,
                            "candidate": m,
                            "success_probability": p,
                            "estimated_cost": c,
                        }
                    )
                    + "\n"
                    for t, e in estimates.items()
                    for m, (p, c) in e.items()
                )
            )
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                main(
                    [
                        "--rows",
                        str(rows),
                        "--estimates",
                        str(est),
                        "--baselines",
                        "mom",
                        "--budgets",
                        "0.3",
                        "--dp-resolution",
                        "30",
                        "--values",
                        "2",
                        "--folds",
                        "2",
                        "--bootstrap",
                        "10",
                        "--json",
                        str(out),
                    ]
                )
            report = json.loads(out.read_text())
        self.assertEqual(report["candidates"], ["A", "B"])
        self.assertEqual(report["baselines"], ["mom"])
        self.assertEqual(report["local"]["resolved"], [0.5])
        self.assertEqual(report["local"]["no_fit"], [0.0])
        self.assertEqual(report["global"]["resolved"], [1.0])
        self.assertAlmostEqual(report["global"]["cost"][0], 0.3)
        self.assertAlmostEqual(report["oracle"]["resolved_at_budget"][0], 1.0)
        self.assertIsNone(report["estimates"]["B"]["auroc"])
        self.assertIn("baseline, no estimates", printed.getvalue())


class SrBenchRowsTest(unittest.TestCase):
    def export(self, **overrides):
        target = {
            "id": "qwen3-8b-high",
            "model": "qwen3-8b",
            "kind": "single",
            "correct": True,
            "score": 1.0,
            "usage": {"output_tokens": 812},
            "cost_usd": 0.0021,
        }
        other = {
            **target,
            "id": "qwen3-8b-off",
            "correct": False,
            "score": 0.0,
            "cost_usd": 0.0004,
        }
        return {
            "kind": "training-matrix",
            "split": "dev",
            "cases": [
                {
                    "id": "q1",
                    "benchmark": "mmlu-pro",
                    "targets": [{**target, **overrides}, other],
                }
            ],
        }

    def test_rows_keep_outcome_cost_and_tokens_per_target(self):
        rows = build_rows(self.export())
        self.assertEqual(
            rows,
            [
                {
                    "id": "q1",
                    "task": "q1",
                    "attempts": {
                        "qwen3-8b-high": {
                            "ok": 1.0,
                            "cost": 0.0021,
                            "output_tokens": 812,
                        },
                        "qwen3-8b-off": {
                            "ok": 0.0,
                            "cost": 0.0004,
                            "output_tokens": 812,
                        },
                    },
                }
            ],
        )
        self.assertEqual(
            list(build_rows(self.export(), ["qwen3-8b-off"])[0]["attempts"]),
            ["qwen3-8b-off"],
        )

    def test_unpriced_or_missing_targets_are_errors(self):
        with self.assertRaisesRegex(ValueError, "no recorded cost"):
            build_rows(self.export(cost_usd=None))
        with self.assertRaisesRegex(ValueError, "has no target 'other'"):
            build_rows(self.export(), ["other"])
        with self.assertRaisesRegex(ValueError, "training-matrix"):
            build_rows({"kind": "offline-regrade", "cases": []})


if __name__ == "__main__":
    unittest.main()
