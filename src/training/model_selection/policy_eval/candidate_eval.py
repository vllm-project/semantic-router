"""Counterfactual evaluation of per-request candidate routing from per-candidate estimates.

CONCUR (Chen, Li, Roth, Cafarella, Madden, Andreas: "CONCUR: A Framework for
Continual Constrained and Unconstrained Routing", https://arxiv.org/abs/2512.09386)
routes each task to one of several candidates, each a model with a decoding
setting, from two estimates per candidate: the probability that it answers
correctly and its cost. This script replays those routing rules on recorded
outcomes, the way policy_eval.py replays escalation, without calling any model.

Rows use policy_eval.py's format: the recorded outcome and cost of every
candidate on one task. An estimate is one candidate's success_probability and
estimated_cost for one row. Candidates are estimated independently, so files
written per candidate can be concatenated.

Policies, over the candidates' estimates p (success) and c (cost):
  utility   per task, the candidate with the highest v * p - c, ties to the
            lower c; v is swept for the frontier, and for the utility report it
            is either the evaluation value V itself (plug-in) or chosen by
            task-grouped cross-validation
  local     per task, the highest p among candidates with c <= b, ties to the
            lower c; the lowest c when none fits (b is a ceiling per task)
  global    the assignment with the highest total p whose total c is at most b
            per task over a batch of tasks: a dynamic program over the cost
            each candidate adds to its task's cheapest estimate, rounded up to
            b / resolution
  oracle    per task, any candidate, knowing every outcome (an upper bound on
            every policy over these candidates at every budget)

Usage:
  python candidate_eval.py --rows rows.jsonl --estimates estimates.jsonl \
      [--candidates A,B,C] [--baselines MODEL] [--json out.json]
Rows: {"id", "task", "attempts": {name: {"ok": 0..1, "cost": dollars,
       "output_tokens": optional}}}.
Estimates: {"id", "candidate", "success_probability": 0..1,
            "estimated_cost": dollars, "expected_output_tokens": optional}.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

try:
    from .policy_eval import (
        BUDGETS,
        QUANTILES,
        SOLVED,
        VALUES,
        auroc,
        frontier,
        read_jsonl,
        single_utilities,
        static_mix,
        task_bootstrap,
    )
except ImportError:  # run as a script from this directory
    from policy_eval import (
        BUDGETS,
        QUANTILES,
        SOLVED,
        VALUES,
        auroc,
        frontier,
        read_jsonl,
        single_utilities,
        static_mix,
        task_bootstrap,
    )

# Batch size and cost resolution of the global dynamic program. The cost a
# candidate adds to its task's cheapest estimate is rounded up to units of
# budget / DP_RESOLUTION, so rounding can rule out an upgrade that would
# otherwise fit. Whether the cheapest assignment fits is decided before any
# rounding, up to a relative tolerance FIT_TOLERANCE for floating-point error.
DP_BATCH = 1000
DP_RESOLUTION = 50
FIT_TOLERANCE = 1e-9
CALIBRATION_BINS = 10
TIE = 1e-12


def read_estimates(path):
    """Map (row id, candidate) to its estimate, rejecting a repeated pair or a value out of range."""
    estimates = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["id"], r["candidate"])
            if key in estimates:
                raise ValueError(
                    f"{path}:{n}: duplicate estimate for id {key[0]!r}, candidate {key[1]!r}"
                )
            p, c = float(r["success_probability"]), float(r["estimated_cost"])
            if not 0.0 <= p <= 1.0 or c < 0.0:
                raise ValueError(
                    f"{path}:{n}: success_probability must be in [0, 1] and estimated_cost at least 0"
                )
            estimates[key] = r
    return estimates


class CandidateTable:
    """Rows that carry every candidate's outcome and estimate, as (rows, candidates) arrays."""

    def __init__(self, rows, estimates, candidates=None, baselines=()):
        if candidates is None:
            candidates = sorted({c for _, c in estimates})
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError("no candidates")
        self.rows_read = len(rows)
        singles = list(dict.fromkeys([*candidates, *baselines]))
        rows = [
            r
            for r in rows
            if all(m in r["attempts"] for m in singles)
            and all((r["id"], c) in estimates for c in candidates)
        ]
        if not rows:
            raise ValueError("no rows carry every candidate's outcome and estimate")
        self.rows, self.candidates, self.singles = rows, candidates, singles
        self.task = np.array([r["task"] for r in rows])
        self.ok = {
            m: np.array([float(r["attempts"][m]["ok"]) for r in rows]) for m in singles
        }
        self.cost = {
            m: np.array([float(r["attempts"][m]["cost"]) for r in rows])
            for m in singles
        }
        est = [[estimates[(r["id"], c)] for c in candidates] for r in rows]
        self.p = np.array([[float(e["success_probability"]) for e in es] for es in est])
        self.c = np.array([[float(e["estimated_cost"]) for e in es] for es in est])
        self.tokens = self._optional(
            [[r["attempts"][c].get("output_tokens") for c in candidates] for r in rows]
        )
        self.expected_tokens = self._optional(
            [[e.get("expected_output_tokens") for e in es] for es in est]
        )

    @staticmethod
    def _optional(values):
        """A float array, or None unless every value is present."""
        if any(v is None for row in values for v in row):
            return None
        return np.array(values, dtype=float)

    def outcome(self, pick):
        """Per-row (ok, cost) of the candidate each row picks, by column index."""
        ok = np.stack([self.ok[c] for c in self.candidates], axis=1)
        cost = np.stack([self.cost[c] for c in self.candidates], axis=1)
        rows = np.arange(len(self.rows))
        return ok[rows, pick], cost[rows, pick]

    def subset(self, mask):
        sub = object.__new__(CandidateTable)
        sub.rows_read = self.rows_read
        sub.rows = [r for r, keep in zip(self.rows, mask, strict=True) if keep]
        sub.candidates, sub.singles = self.candidates, self.singles
        sub.task = self.task[mask]
        sub.ok = {m: v[mask] for m, v in self.ok.items()}
        sub.cost = {m: v[mask] for m, v in self.cost.items()}
        sub.p, sub.c = self.p[mask], self.c[mask]
        sub.tokens = None if self.tokens is None else self.tokens[mask]
        sub.expected_tokens = (
            None if self.expected_tokens is None else self.expected_tokens[mask]
        )
        return sub


def pick_best(score, cost):
    """Per row, the column with the highest score; ties go to the lowest cost, then the first column.

    A row whose scores are all -inf gets its lowest-cost column.
    """
    best = score.max(axis=1, keepdims=True)
    tied = (score == best) | np.isclose(score, best, rtol=0.0, atol=TIE)
    return np.where(tied, cost, np.inf).argmin(axis=1)


def choose_utility(p, c, value):
    """Per row, the candidate with the highest value * p - c; value=inf ranks by p alone."""
    return pick_best(p if np.isinf(value) else value * p - c, c)


def choose_local(p, c, budget):
    """Per row, the highest p among candidates with c <= budget, or the lowest c when none fits."""
    return pick_best(np.where(c <= budget, p, -np.inf), c)


def utility_values(table, quantiles=QUANTILES):
    """0, the quantiles of the values at which some row's choice changes, and +inf.

    Between candidates i and j with p_i > p_j and c_i > c_j, the utility choice
    switches from j to i at v = (c_i - c_j) / (p_i - p_j).
    """
    dp = table.p[:, :, None] - table.p[:, None, :]
    dc = table.c[:, :, None] - table.c[:, None, :]
    up = (dp > 0) & (dc > 0)
    breaks = dc[up] / dp[up]
    qs = (
        np.unique(np.quantile(breaks, np.linspace(0, 1, quantiles)))
        if breaks.size
        else []
    )
    return np.r_[0.0, qs, np.inf]


def utility_points(table, quantiles=QUANTILES):
    out = []
    for v in utility_values(table, quantiles):
        ok, cost = table.outcome(choose_utility(table.p, table.c, v))
        out.append((cost.mean(), ok.mean(), v))
    return out


def best_value(table, value, quantiles=QUANTILES):
    """The decision value whose choices reach the highest value * resolved - cost on these rows."""
    return max(utility_points(table, quantiles), key=lambda p: value * p[1] - p[0])[2]


def plug_in_gain(table, value):
    """Per-row utility gain over the best single model when the policy uses value itself."""
    ok, cost = table.outcome(choose_utility(table.p, table.c, value))
    singles = single_utilities(table, value)
    best = max(singles, key=lambda m: singles[m].mean())
    return value * ok - cost - singles[best]


def cross_validated_gain(table, value, folds=5, quantiles=QUANTILES):
    """Per-row utility gain over the fold's best single model, the decision value chosen out of fold.

    Folds are tasks[i::folds] over the sorted task names, as in policy_eval.py,
    so every row of a task sits in one fold.
    """
    tasks = sorted(set(table.task.tolist()))
    gain = np.zeros(len(table.rows))
    for i in range(folds):
        held = np.isin(table.task, tasks[i::folds])
        train, test = table.subset(~held), table.subset(held)
        if not len(test.rows) or not len(train.rows):
            continue
        v = best_value(train, value, quantiles)
        ok, cost = test.outcome(choose_utility(test.p, test.c, v))
        singles = single_utilities(test, value)
        best = max(singles, key=lambda m: singles[m].mean())
        gain[held] = value * ok - cost - singles[best]
    return gain


def knapsack(p, c, budget, resolution=DP_RESOLUTION):
    """One candidate per row maximizing total p with total c at most budget per row, or None.

    Every row starts on its cheapest estimate, which is kept exact. None means
    that the cheapest estimates alone exceed the budget, which is decided before
    any rounding, up to a relative FIT_TOLERANCE for floating-point error. What
    a candidate adds to its row's cheapest estimate is rounded up to units of
    budget / resolution, and the units must fit in the budget left after the
    cheapest assignment. So the chosen assignment never exceeds the budget on
    the estimates, but rounding can rule out an upgrade that would otherwise
    fit. Among assignments with the same total p, the one with the fewest
    rounded units wins.
    """
    rows, k = p.shape
    base = c.min(axis=1)
    extra = c - base[:, None]
    total_budget, cheapest = budget * rows, base.sum()
    slack = total_budget - cheapest
    if slack < -FIT_TOLERANCE * max(abs(total_budget), abs(cheapest)):
        return None
    slack = max(slack, 0.0)
    if budget > 0:
        unit = budget / resolution
        capacity = int(np.floor(np.round(slack / unit, 9)))
        units = np.ceil(np.round(extra / unit, 9))
    else:
        capacity = 0
        units = np.where(extra > 0, 1, 0)
    units = np.minimum(units, capacity + 1).astype(np.int64)
    # total[u]: the highest total p of the rows so far using exactly u units
    # above their cheapest estimates, for u up to the most reachable so far.
    # Each row's cheapest candidate takes 0 units, so every row has a choice.
    total = np.zeros(1)
    choice = np.zeros(
        (rows, capacity + 1), dtype=np.int8 if k <= np.iinfo(np.int8).max else np.int16
    )
    for i in range(rows):
        fits = np.flatnonzero(units[i] <= capacity)
        reach = min(capacity, len(total) - 1 + int(units[i, fits].max()))
        best = np.full(reach + 1, -np.inf)
        for j in fits:
            w = int(units[i, j])
            span = min(len(total), reach + 1 - w)
            moved = total[:span] + p[i, j]
            better = moved > best[w : w + span]
            best[w : w + span][better] = moved[better]
            choice[i, w : w + span][better] = j
        total = best
    b = int(np.argmax(total))
    pick = np.empty(rows, dtype=int)
    for i in reversed(range(rows)):
        pick[i] = choice[i, b]
        b -= units[i, pick[i]]
    return pick


def choose_global(p, c, budget, batch=DP_BATCH, resolution=DP_RESOLUTION):
    """Per row, the knapsack choice over consecutive batches of rows, each with budget per row.

    A batch whose cheapest assignment exceeds its budget takes the lowest c on
    every row. Returns the choices and the number of such batches.
    """
    pick = np.empty(len(p), dtype=int)
    infeasible = 0
    for start in range(0, len(p), batch):
        part = slice(start, start + batch)
        chosen = knapsack(p[part], c[part], budget, resolution)
        if chosen is None:
            infeasible += 1
            chosen = c[part].argmin(axis=1)
        pick[part] = chosen
    return pick, infeasible


def oracle_frontier(table, budget):
    """The best resolve rate any choice of one candidate per task reaches at a mean budget per task.

    Each task starts on its cheapest candidate and moves up the upper concave
    hull of its candidates' (cost, outcome), the most resolve per extra dollar
    first across tasks. Interpolating between those moves gives the
    linear-programming bound, which no policy over these candidates passes.
    With two candidates it equals policy_eval.oracle_frontier.
    """
    ok = np.stack([table.ok[m] for m in table.candidates], axis=1)
    cost = np.stack([table.cost[m] for m in table.candidates], axis=1)
    base_cost = base_ok = 0.0
    gains, extras = [], []
    for y, x in zip(ok, cost, strict=True):
        order = np.lexsort((-y, x))
        hull = [order[0]]
        for j in order[1:]:
            if y[j] <= y[hull[-1]]:
                continue
            while len(hull) > 1:
                a, b = hull[-2], hull[-1]
                if (y[b] - y[a]) * (x[j] - x[b]) > (y[j] - y[b]) * (x[b] - x[a]):
                    break
                hull.pop()
            hull.append(j)
        base_cost += x[hull[0]]
        base_ok += y[hull[0]]
        for a, b in itertools.pairwise(hull):
            gains.append(y[b] - y[a])
            extras.append(x[b] - x[a])
    gains, extras = np.array(gains), np.array(extras)
    order = np.argsort(-(gains / extras), kind="stable") if len(gains) else []
    n = len(table.rows)
    xs = (base_cost + np.r_[0, np.cumsum(extras[order])]) / n
    ys = (base_ok + np.r_[0, np.cumsum(gains[order])]) / n
    return float(np.interp(budget, xs, ys)) if budget >= xs[0] else 0.0


def calibration(p, ok, bins=CALIBRATION_BINS):
    """Brier score, expected calibration error over equal-width bins, and AUROC for solving."""
    brier = float(np.mean((p - ok) ** 2))
    which = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(
        abs(p[which == b].mean() - ok[which == b].mean()) * (which == b).mean()
        for b in range(bins)
        if (which == b).any()
    )
    solved = ok >= SOLVED
    return brier, float(ece), auroc(p[solved], p[~solved])


def estimate_quality(table):
    report = {}
    for j, m in enumerate(table.candidates):
        brier, ece, area = calibration(table.p[:, j], table.ok[m])
        error = table.c[:, j] - table.cost[m]
        entry = {
            "brier": brier,
            "ece": ece,
            "auroc": area,
            "cost_mae": float(np.abs(error).mean()),
            "cost_bias": float(error.mean()),
            "tokens_mae": None,
        }
        if table.tokens is not None and table.expected_tokens is not None:
            entry["tokens_mae"] = float(
                np.abs(table.expected_tokens[:, j] - table.tokens[:, j]).mean()
            )
        report[m] = entry
    return report


def evaluate(
    table,
    budgets=BUDGETS,
    values=VALUES,
    folds=5,
    bootstrap=1000,
    seed=0,
    quantiles=QUANTILES,
    batch=DP_BATCH,
    resolution=DP_RESOLUTION,
):
    report = {
        "candidates": table.candidates,
        "baselines": [m for m in table.singles if m not in table.candidates],
        "rows": len(table.rows),
        "rows_read": table.rows_read,
        "tasks": len(set(table.task.tolist())),
        "single": {
            m: {
                "resolved": float(table.ok[m].mean()),
                "resolved_95": task_bootstrap(table.ok[m], table.task, bootstrap, seed),
                "cost": float(table.cost[m].mean()),
            }
            for m in table.singles
        },
        "estimates": estimate_quality(table),
        "budgets": list(budgets),
        "static_mix": [static_mix(table, b) for b in budgets],
    }
    pts = utility_points(table, quantiles)
    gain = {}
    for v in values:
        plug_in = plug_in_gain(table, v)
        fitted = cross_validated_gain(table, v, folds, quantiles)
        gain[str(v)] = {
            name: {
                "mean": float(g.mean()),
                "interval_95": task_bootstrap(g, table.task, bootstrap, seed),
            }
            for name, g in (("plug_in", plug_in), ("cross_validated", fitted))
        }
    report["utility"] = {
        "resolved_at_budget": [frontier(pts, b) for b in budgets],
        "gain": gain,
    }
    local = {"resolved": [], "cost": [], "over_budget": [], "no_fit": []}
    for b in budgets:
        ok, cost = table.outcome(choose_local(table.p, table.c, b))
        local["resolved"].append(float(ok.mean()))
        local["cost"].append(float(cost.mean()))
        local["over_budget"].append(float((cost > b).mean()))
        local["no_fit"].append(float((table.c > b).all(axis=1).mean()))
    report["local"] = local
    glob = {"resolved": [], "cost": [], "infeasible_batches": []}
    for b in budgets:
        pick, infeasible = choose_global(table.p, table.c, b, batch, resolution)
        ok, cost = table.outcome(pick)
        glob["resolved"].append(float(ok.mean()))
        glob["cost"].append(float(cost.mean()))
        glob["infeasible_batches"].append(infeasible)
    report["global"] = {**glob, "batch": batch, "resolution": resolution}
    report["oracle"] = {
        "resolved_at_budget": [oracle_frontier(table, b) for b in budgets]
    }
    return report


def fmt(x, spec, prefix=""):
    return "n/a" if x is None else prefix + format(x, spec)


def render(report):
    single, est = report["single"], report["estimates"]
    dropped = report["rows_read"] - report["rows"]
    lines = [
        f"== {len(report['candidates'])} candidates: {report['rows']} rows over {report['tasks']} tasks"
        + (
            f" ({dropped} of {report['rows_read']} rows skipped: a candidate's outcome or estimate is missing)"
            if dropped
            else ""
        ),
        f"   {'candidate':24s} {'alone [95% interval]':20s} {'cost':>8s}  |"
        f" {'Brier':>6s} {'ECE':>6s} {'AUROC':>6s} {'cost MAE':>9s} {'cost bias':>9s} {'tokens MAE':>10s}",
    ]
    for m, s in single.items():
        lo, hi = s["resolved_95"]
        line = f"   {m:24s} {s['resolved']:.3f} [{lo:.3f}, {hi:.3f}] {fmt(s['cost'], '.4f', '$'):>8s}  |"
        if m in est:
            e = est[m]
            line += (
                f" {e['brier']:6.3f} {e['ece']:6.3f} {fmt(e['auroc'], '.3f'):>6s}"
                f" {fmt(e['cost_mae'], '.4f', '$'):>9s} {e['cost_bias']:+9.4f}"
                f" {fmt(e['tokens_mae'], '.1f'):>10s}"
            )
        else:
            line += " baseline, no estimates"
        lines.append(line)
    head = "".join(f" ${b:<5}" for b in report["budgets"])
    lines.append(
        f"   {'policy':12s}{head}  | resolve rate at a mean recorded cost per task within the budget"
    )
    for name, ys in (
        ("static mix", report["static_mix"]),
        ("utility", report["utility"]["resolved_at_budget"]),
        ("oracle", report["oracle"]["resolved_at_budget"]),
    ):
        lines.append(f"   {name:12s}" + "".join(f" {y:.3f} " for y in ys))
    lines.append(
        "   utility minus best single model [95% interval], decision value V (plug-in) | cross-validated"
    )
    for v, g in report["utility"]["gain"].items():
        a, b = g["plug_in"], g["cross_validated"]
        lines.append(
            f"   V=${float(v):<6g} {a['mean']:+.3f} [{a['interval_95'][0]:+.3f}, {a['interval_95'][1]:+.3f}]"
            f"  | {b['mean']:+.3f} [{b['interval_95'][0]:+.3f}, {b['interval_95'][1]:+.3f}]"
        )
    lines.append(
        f"   {'budget given':12s}{head}  | applied to estimated cost; above b: tasks whose recorded cost exceeds it"
    )
    local, glob = report["local"], report["global"]
    for name, part, key, spec in (
        ("local", local, "resolved", " {:.3f} "),
        ("  mean cost", local, "cost", " {:.4f}"),
        ("  above b", local, "over_budget", " {:.3f} "),
        ("  no fit", local, "no_fit", " {:.3f} "),
        ("global", glob, "resolved", " {:.3f} "),
        ("  mean cost", glob, "cost", " {:.4f}"),
    ):
        lines.append(f"   {name:12s}" + "".join(spec.format(y) for y in part[key]))
    if any(glob["infeasible_batches"]):
        lines.append(
            "   global: batches whose cheapest assignment exceeds the budget took the cheapest estimate: "
            + " ".join(str(n) for n in glob["infeasible_batches"])
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", required=True)
    ap.add_argument(
        "--estimates",
        required=True,
        help="JSONL of {id, candidate, success_probability, estimated_cost}",
    )
    ap.add_argument(
        "--candidates",
        default="",
        help="comma-separated candidates to route between (default: every candidate with estimates)",
    )
    ap.add_argument(
        "--baselines",
        default="",
        help="comma-separated extra attempts, such as a recorded MoM target, for the hull and the baseline",
    )
    ap.add_argument("--quantiles", type=int, default=QUANTILES)
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--values", default=",".join(f"{v:g}" for v in VALUES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dp-batch", type=int, default=DP_BATCH)
    ap.add_argument("--dp-resolution", type=int, default=DP_RESOLUTION)
    ap.add_argument("--json", help="also write the report as JSON")
    a = ap.parse_args(argv)
    table = CandidateTable(
        read_jsonl(a.rows),
        read_estimates(a.estimates),
        [m for m in a.candidates.split(",") if m] or None,
        [m for m in a.baselines.split(",") if m],
    )
    report = evaluate(
        table,
        budgets=[float(b) for b in a.budgets.split(",")],
        values=[float(v) for v in a.values.split(",")],
        folds=a.folds,
        bootstrap=a.bootstrap,
        seed=a.seed,
        quantiles=a.quantiles,
        batch=a.dp_batch,
        resolution=a.dp_resolution,
    )
    print(render(report))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
