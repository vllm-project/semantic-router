"""Counterfactual evaluation of routing policies on recorded per-task outcomes.

Each row is one attempt of a cheap model on a task, with the recorded outcome
and cost of every other model that ran the same task. A policy decides, from a
score the caller supplies (for example a judge's answer on the cheap attempt),
whether to keep the cheap attempt or pay for another one. Because every
outcome is already recorded, a policy is replayed without calling any model.

Policies, for one cheap model C and one strong model S:
  static     C or S on every task
  mix        the static hull: mixing C and S per task at random, no signal
  escalate   C runs, and when score > t S runs as well and its result is kept
  retry      C runs, and when score > t another C attempt runs (--retry names it)
  boundary   C or S is chosen before any attempt, from a per-task score
  oracle     escalate exactly the attempts S rescues (an upper bound)

The report gives the resolve rate each policy reaches at a budget per task
(sweeping its threshold), utility V * resolved - cost against the best single
model with thresholds chosen by task-grouped cross-validation, the same with
thresholds chosen from the first n tasks of a stream, and the score's AUROC
for the cheap attempt failing, overall and within task.

Usage:
  python policy_eval.py --rows rows.jsonl --cheap MODEL --strong MODEL \
      [--scores scores.jsonl] [--boundary-scores task_scores.jsonl] [--retry MODEL] \
      [--baselines MODEL,MODEL] [--stream 25,50] [--json out.json]
Rows: {"id", "task", "attempts": {model: {"ok": 0..1, "cost": dollars}}}.
Scores: {"id", "score"} per row id. Boundary scores: {"id", "score"} per task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

BUDGETS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
VALUES = (1.0, 2.0, 5.0)
QUANTILES = 81
# An attempt counts as solved for AUROC when its outcome is at least this.
SOLVED = 0.5


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_scores(path):
    """Map id to score from a JSONL file of {id, score}, rejecting a repeated id."""
    scores = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                if r["id"] in scores:
                    raise ValueError(f"{path}:{n}: duplicate id {r['id']!r}")
                scores[r["id"]] = r["score"]
    return scores


class Table:
    """Rows of one (cheap, strong) pair as arrays, with optional scores."""

    def __init__(
        self, rows, cheap, strong, scores=None, boundary=None, retry=None, baselines=()
    ):
        self.singles = list(dict.fromkeys([cheap, strong, *baselines]))
        needed = self.singles + ([retry] if retry else [])
        rows = [r for r in rows if all(m in r["attempts"] for m in needed)]
        if scores is not None:
            rows = [r for r in rows if r["id"] in scores]
        if boundary is not None:
            rows = [r for r in rows if r["task"] in boundary]
        if not rows:
            raise ValueError("no rows carry every requested model and score")
        self.rows, self.cheap, self.strong, self.retry = rows, cheap, strong, retry
        self.task = np.array([r["task"] for r in rows])
        self.ok = {
            m: np.array([float(r["attempts"][m]["ok"]) for r in rows]) for m in needed
        }
        self.cost = {
            m: np.array([float(r["attempts"][m]["cost"]) for r in rows]) for m in needed
        }
        self.score = (
            None if scores is None else np.array([float(scores[r["id"]]) for r in rows])
        )
        self.boundary = (
            None
            if boundary is None
            else np.array([float(boundary[r["task"]]) for r in rows])
        )

    def subset(self, mask):
        sub = object.__new__(Table)
        sub.rows = [r for r, keep in zip(self.rows, mask, strict=True) if keep]
        sub.cheap, sub.strong, sub.retry, sub.singles = (
            self.cheap,
            self.strong,
            self.retry,
            self.singles,
        )
        sub.task = self.task[mask]
        sub.ok = {m: v[mask] for m, v in self.ok.items()}
        sub.cost = {m: v[mask] for m, v in self.cost.items()}
        sub.score = None if self.score is None else self.score[mask]
        sub.boundary = None if self.boundary is None else self.boundary[mask]
        return sub


def policy_names(table):
    names = []
    if table.score is not None:
        names.append("escalate")
        if table.retry:
            names.append("retry")
    if table.boundary is not None:
        names.append("boundary")
    names.append("oracle")
    return names


def policy_score(table, name):
    if name in ("escalate", "retry"):
        return table.score
    if name == "boundary":
        return table.boundary
    return (1 - table.ok[table.cheap]) * table.ok[table.strong]


def run(table, name, threshold):
    """Per-row (ok, cost) of a policy at one threshold."""
    c, s = table.cheap, table.strong
    pick = policy_score(table, name) > threshold
    if name == "boundary":
        return (
            np.where(pick, table.ok[s], table.ok[c]),
            np.where(pick, table.cost[s], table.cost[c]),
        )
    other = table.retry if name == "retry" else s
    return (
        np.where(pick, table.ok[other], table.ok[c]),
        table.cost[c] + pick * table.cost[other],
    )


def thresholds(table, name, quantiles=QUANTILES):
    """-inf, the score's quantiles, +inf: every policy from never to always switching."""
    qs = np.unique(np.quantile(policy_score(table, name), np.linspace(0, 1, quantiles)))
    return np.r_[-np.inf, qs, np.inf]


def points(table, name, quantiles=QUANTILES):
    out = []
    for t in thresholds(table, name, quantiles):
        ok, cost = run(table, name, t)
        out.append((cost.mean(), ok.mean(), t))
    return out


def frontier(pts, budget):
    return max([y for x, y, _ in pts if x <= budget] + [0.0])


def static_mix(table, budget):
    """Resolve rate of the upper concave hull of the single models at a budget."""
    pts = sorted((table.cost[m].mean(), table.ok[m].mean()) for m in table.singles)
    hull = []
    for p in pts:
        while len(hull) > 1 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= (
            p[1] - hull[-2][1]
        ) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    inside = [y for x, y in hull if x <= budget]
    interpolated = (
        float(np.interp(budget, [h[0] for h in hull], [h[1] for h in hull]))
        if budget >= hull[0][0]
        else 0.0
    )
    return max([*inside, interpolated])


def best_threshold(table, name, value, quantiles=QUANTILES):
    return max(points(table, name, quantiles), key=lambda p: value * p[1] - p[0])[2]


def best_single_utility(table, value):
    return max(value * table.ok[m].mean() - table.cost[m].mean() for m in table.singles)


def single_utilities(table, value):
    return {m: value * table.ok[m] - table.cost[m] for m in table.singles}


def cross_validated_gain(table, name, value, folds=5, quantiles=QUANTILES):
    """Per-row utility gain over the fold's best single model, thresholds chosen out of fold.

    Folds are tasks[i::folds] over the sorted task names, so every attempt of a
    task sits in one fold. The mean of the returned array is the reported gain.
    """
    tasks = sorted(set(table.task.tolist()))
    gain = np.zeros(len(table.rows))
    for i in range(folds):
        held = np.isin(table.task, tasks[i::folds])
        train, test = table.subset(~held), table.subset(held)
        if not len(test.rows):
            continue
        ok, cost = run(test, name, best_threshold(train, name, value, quantiles))
        singles = single_utilities(test, value)
        best = max(singles, key=lambda m: singles[m].mean())
        gain[held] = value * ok - cost - singles[best]
    return gain


def task_bootstrap(per_row, task, draws=1000, seed=0):
    """95% interval of the per-row mean, resampling whole tasks."""
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for i, t in enumerate(task.tolist()):
        by_task[t].append(i)
    groups = [np.array(v) for v in by_task.values()]
    means = []
    for _ in range(draws):
        idx = np.concatenate(
            [groups[j] for j in rng.integers(0, len(groups), len(groups))]
        )
        means.append(per_row[idx].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stream_gain(table, name, value, n, draws=100, seed=0, quantiles=QUANTILES):
    """Mean gain when the threshold is chosen from the first n tasks of a random stream.

    Both models' outcomes are assumed known on those n tasks, which a deployment
    only has where it explores, so this is an upper bound on what an outcome
    loop learns from n tasks. Tasks are permuted in the order they first appear
    in the rows.
    """
    tasks = list(dict.fromkeys(table.task.tolist()))
    if n >= len(tasks):
        return None
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(draws):
        order = rng.permutation(len(tasks))
        first = np.isin(table.task, [tasks[i] for i in order[:n]])
        cal, rest = table.subset(first), table.subset(~first)
        ok, cost = run(rest, name, best_threshold(cal, name, value, quantiles))
        gains.append(value * ok.mean() - cost.mean() - best_single_utility(rest, value))
    return float(np.mean(gains))


def auroc(positive, negative):
    """P(score of a positive > score of a negative), ties counted half."""
    if not len(positive) or not len(negative):
        return None
    diff = positive[:, None] - negative[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def judge_auroc(table):
    """AUROC of the score for the cheap attempt failing, overall and within task.

    Within task compares only failed and solved attempts of the same task, so a
    score that only knows which tasks are hard gets 0.5 there.
    """
    fail = table.ok[table.cheap] < SOLVED
    overall = auroc(table.score[fail], table.score[~fail])
    wins = ties = pairs = 0
    for t in set(table.task.tolist()):
        m = table.task == t
        pos, neg = table.score[m & fail], table.score[m & ~fail]
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            wins += (diff > 0).sum()
            ties += (diff == 0).sum()
            pairs += diff.size
    within = float((wins + 0.5 * ties) / pairs) if pairs else None
    return overall, within, int(pairs)


def evaluate(
    table,
    budgets=BUDGETS,
    values=VALUES,
    folds=5,
    stream=(),
    draws=100,
    bootstrap=1000,
    seed=0,
    quantiles=QUANTILES,
):
    c, s = table.cheap, table.strong
    report = {
        "cheap": c,
        "strong": s,
        "rows": len(table.rows),
        "tasks": len(set(table.task.tolist())),
        "single": {
            m: {
                "resolved": float(table.ok[m].mean()),
                "resolved_95": task_bootstrap(table.ok[m], table.task, bootstrap, seed),
                "cost": float(table.cost[m].mean()),
            }
            for m in table.singles
        },
        "budgets": list(budgets),
        "static_mix": [static_mix(table, b) for b in budgets],
        "policies": {},
    }
    if table.score is not None:
        overall, within, pairs = judge_auroc(table)
        report["score_auroc"] = {
            "overall": overall,
            "within_task": within,
            "within_task_pairs": pairs,
        }
    for name in policy_names(table):
        pts = points(table, name, quantiles)
        entry = {"resolved_at_budget": [frontier(pts, b) for b in budgets]}
        if name != "oracle":
            entry["cv_gain"] = {}
            for v in values:
                per_row = cross_validated_gain(table, name, v, folds, quantiles)
                entry["cv_gain"][str(v)] = {
                    "mean": float(per_row.mean()),
                    "interval_95": task_bootstrap(per_row, table.task, bootstrap, seed),
                }
            entry["stream_gain"] = {
                f"V={v:g} n={n}": stream_gain(table, name, v, n, draws, seed, quantiles)
                for v in values
                for n in stream
            }
        report["policies"][name] = entry
    return report


def render(report):
    c, s, single = report["cheap"], report["strong"], report["single"]
    lines = [
        f"== cheap {c}, strong {s}: {report['rows']} attempts over {report['tasks']} tasks",
        "   "
        + " | ".join(
            f"{m} alone {single[m]['resolved']:.3f} [{single[m]['resolved_95'][0]:.3f}, {single[m]['resolved_95'][1]:.3f}] @ ${single[m]['cost']:.4f}"
            for m in single
        ),
    ]
    if "score_auroc" in report:
        a = report["score_auroc"]
        within = (
            "n/a"
            if a["within_task"] is None
            else f"{a['within_task']:.3f} over {a['within_task_pairs']} pairs"
        )
        lines.append(
            f"   score AUROC for the cheap attempt failing: overall {a['overall']:.3f}, within task {within}"
        )
    head = "".join(f" ${b:<5}" for b in report["budgets"])
    lines.append(
        f"   {'policy':12s}{head}  | utility minus best single model, cross-validated [95% interval]"
    )
    lines.append(
        f"   {'static mix':12s}" + "".join(f" {y:.3f} " for y in report["static_mix"])
    )
    for name, entry in report["policies"].items():
        line = f"   {name:12s}" + "".join(
            f" {y:.3f} " for y in entry["resolved_at_budget"]
        )
        if "cv_gain" in entry:
            line += "  | " + "  ".join(
                f"V=${float(v):g} {g['mean']:+.3f} [{g['interval_95'][0]:+.3f}, {g['interval_95'][1]:+.3f}]"
                for v, g in entry["cv_gain"].items()
            )
        lines.append(line)
    stream_rows = [
        (n, e["stream_gain"])
        for n, e in report["policies"].items()
        if e.get("stream_gain")
    ]
    if stream_rows:
        lines.append(
            "   threshold from the first n tasks of a stream (mean gain over draws):"
        )
        for name, cells in stream_rows:
            lines.append(
                f"   {name:12s}"
                + "  ".join(f"{k} {v:+.3f}" for k, v in cells.items() if v is not None)
            )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--strong", required=True)
    ap.add_argument(
        "--scores", help="JSONL of {id, score} per row, for escalate and retry"
    )
    ap.add_argument(
        "--boundary-scores", help="JSONL of {id, score} per task, for boundary"
    )
    ap.add_argument("--retry", help="attempt name of a second cheap attempt, for retry")
    ap.add_argument(
        "--baselines",
        default="",
        help="comma-separated extra single models for the hull and the baseline",
    )
    ap.add_argument(
        "--quantiles",
        type=int,
        default=QUANTILES,
        help="score quantiles swept as thresholds",
    )
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--values", default=",".join(f"{v:g}" for v in VALUES))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--stream",
        default="",
        help="comma-separated n for stream calibration, e.g. 25,50",
    )
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the report as JSON")
    a = ap.parse_args(argv)
    scores = read_scores(a.scores) if a.scores else None
    boundary = read_scores(a.boundary_scores) if a.boundary_scores else None
    baselines = [m for m in a.baselines.split(",") if m]
    table = Table(
        read_jsonl(a.rows), a.cheap, a.strong, scores, boundary, a.retry, baselines
    )
    report = evaluate(
        table,
        budgets=[float(b) for b in a.budgets.split(",")],
        values=[float(v) for v in a.values.split(",")],
        folds=a.folds,
        stream=[int(n) for n in a.stream.split(",") if n],
        draws=a.draws,
        bootstrap=a.bootstrap,
        seed=a.seed,
        quantiles=a.quantiles,
    )
    print(render(report))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
