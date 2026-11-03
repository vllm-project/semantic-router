"""Paid-exploration replay for end-of-attempt escalation.

A cheap attempt is already finished. A score says whether to restart the task
on another model. The strong model's outcome is recorded for evaluation, but
the threshold may use it only after this replay has paid to escalate that
task. Exploration is that payment: a seeded draw escalates even when the
score is below the threshold, and the cost of both attempts is kept.

The threshold starts at never-escalate. Every row of a task is decided with
the threshold from before that task. After the task, the threshold is refit
on the escalated rows only.

Hindsight fits one threshold with every outcome visible and applies it with
exploration off. That number is in-sample. It is the comparison point for the
paid stream, not a deployment result.

Usage:
  python explore.py --rows rows.jsonl --scores scores.jsonl \\
      --cheap MODEL --strong MODEL [--retry MODEL] [--epsilon 0.1] \\
      [--orders 50] [--seed 0] [--values 1,2,4] [--json out.json]
Rows: {"id", "task", "attempts": {model: {"ok": 0..1, "cost": dollars}}}.
Scores: {"id", "score"} per row. A higher score asks for a restart.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

NEVER = float("inf")
ALWAYS = float("-inf")
VALUES = (1.0, 2.0, 4.0)


def read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                rows.append((line_no, json.loads(line)))
    return rows


def load_scores(path):
    """One score per id. Conflicting duplicates name the file and line."""
    scores = {}
    for line_no, record in read_jsonl(path):
        row_id = record["id"]
        score = float(record["score"])
        if row_id in scores and scores[row_id] != score:
            raise ValueError(
                f"{path}:{line_no}: duplicate id {row_id!r} with scores "
                f"{scores[row_id]} and {score}"
            )
        scores[row_id] = score
    return scores


def load_rows(path):
    rows = []
    seen = set()
    for line_no, record in read_jsonl(path):
        row_id = record["id"]
        if row_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate id {row_id!r}")
        seen.add(row_id)
        rows.append(record)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def group_rows(rows):
    """Tasks in first-seen order, each with its rows in input order."""
    order = []
    grouped = {}
    for row in rows:
        task = row["task"]
        if task not in grouped:
            order.append(task)
            grouped[task] = []
        grouped[task].append(row)
    return order, grouped


def _mean(values):
    return sum(values) / len(values)


def realize(row, escalated, cheap, arm):
    """Outcome and dollars. An escalation pays for the cheap attempt too."""
    try:
        first = row["attempts"][cheap]
        cheap_ok = float(first["ok"])
        cheap_cost = float(first["cost"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"row {row.get('id')!r} is missing attempt {cheap}") from exc
    if not escalated:
        return cheap_ok, cheap_cost
    try:
        other = row["attempts"][arm]
        return float(other["ok"]), cheap_cost + float(other["cost"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"row {row.get('id')!r} is missing attempt {arm}") from exc


def _utility(rows, scores, cheap, arm, value, threshold):
    outcomes = [realize(row, scores[row["id"]] > threshold, cheap, arm) for row in rows]
    ok, cost = zip(*outcomes, strict=True)
    return value * _mean(ok) - _mean(cost)


def fit_threshold(rows, scores, cheap, arm, value):
    """Threshold with the best utility on these rows. Ties keep the higher one.

    An empty observation set stays at never-escalate. Candidates are
    never-escalate, each observed score, and always-escalate. A row escalates
    when its score is strictly above the threshold.
    """
    if not rows:
        return NEVER
    best_t = NEVER
    best_u = _utility(rows, scores, cheap, arm, value, NEVER)
    unique = sorted({scores[row["id"]] for row in rows}, reverse=True)
    for threshold in [*unique, ALWAYS]:
        utility = _utility(rows, scores, cheap, arm, value, threshold)
        if utility > best_u:
            best_u = utility
            best_t = threshold
    return best_t


def best_single_utility(rows, value, cheap, strong):
    """Utility of running one model on every row. The retry column is not a candidate."""
    utilities = []
    for model in (cheap, strong):
        ok, cost = [], []
        for row in rows:
            attempt = row["attempts"][model]
            ok.append(float(attempt["ok"]))
            cost.append(float(attempt["cost"]))
        utilities.append(value * _mean(ok) - _mean(cost))
    return max(utilities)


class Replay:
    """Per-row result of one arm, in task-order."""

    def __init__(self, ok, cost, escalated, thresholds):
        self.ok = ok
        self.cost = cost
        self.escalated = escalated
        self.thresholds = thresholds

    def utility(self, value):
        return value * _mean(self.ok) - _mean(self.cost)

    def gain(self, rows, value, cheap, strong):
        return self.utility(value) - best_single_utility(rows, value, cheap, strong)


def flatten(grouped, task_order):
    if len(task_order) != len(set(task_order)) or set(task_order) != set(grouped):
        raise ValueError("task order must list each task once")
    rows = []
    for task in task_order:
        rows.extend(grouped[task])
    return rows


def replay(grouped, task_order, scores, cheap, arm, value, explore_bits):
    """Decide each task with the previous threshold, then refit on escalations."""
    rows = flatten(grouped, task_order)
    if len(explore_bits) != len(rows):
        raise ValueError(
            f"explore bits ({len(explore_bits)}) do not match rows ({len(rows)})"
        )
    missing = [row["id"] for row in rows if row["id"] not in scores]
    if missing:
        raise ValueError(f"missing scores for {missing[:3]}")
    threshold = NEVER
    observed = []
    ok, cost, escalated, thresholds = [], [], [], []
    cursor = 0
    for task in task_order:
        newly = []
        for row in grouped[task]:
            take = explore_bits[cursor] or scores[row["id"]] > threshold
            cursor += 1
            outcome, spent = realize(row, take, cheap, arm)
            ok.append(outcome)
            cost.append(spent)
            escalated.append(take)
            thresholds.append(threshold)
            if take:
                newly.append(row)
        observed.extend(newly)
        threshold = fit_threshold(observed, scores, cheap, arm, value)
    return Replay(ok, cost, escalated, thresholds)


def apply_threshold(rows, scores, cheap, arm, threshold):
    """Apply one threshold with exploration off. Used by the in-sample hindsight."""
    ok, cost, escalated = [], [], []
    for row in rows:
        take = scores[row["id"]] > threshold
        outcome, spent = realize(row, take, cheap, arm)
        ok.append(outcome)
        cost.append(spent)
        escalated.append(take)
    return Replay(ok, cost, escalated, [threshold] * len(rows))


def hindsight_gain(rows, scores, cheap, arm, value, strong):
    """In-sample gain: the threshold sees every outcome, then exploration is off."""
    threshold = fit_threshold(rows, scores, cheap, arm, value)
    played = apply_threshold(rows, scores, cheap, arm, threshold)
    return played.gain(rows, value, cheap, strong)


def retry_arm(rows, cheap, retry):
    """`<cheap>#retry` when every row has it. An explicit name that is absent raises."""
    name = f"{cheap}#retry" if retry is None else retry
    present = all(name in row["attempts"] for row in rows)
    if present:
        return name
    if retry is not None:
        raise ValueError(f"retry attempt {retry!r} is missing from some rows")
    return None


def _draw(grouped, first_seen, orders, seed, epsilon):
    rng = random.Random(seed)
    drawn = []
    for _ in range(orders):
        order = list(first_seen)
        rng.shuffle(order)
        bits = [rng.random() < epsilon for _ in flatten(grouped, order)]
        drawn.append((order, bits))
    return drawn


def stream_gains(
    rows,
    scores,
    cheap,
    strong,
    value,
    epsilon,
    orders,
    seed,
    retry=None,
    explore_bits=None,
    task_order=None,
):
    """Mean gain over task orders. Pass explore_bits to replay one fixed order."""
    first_seen, grouped = group_rows(rows)
    if explore_bits is not None:
        passes = [(task_order if task_order is not None else first_seen, explore_bits)]
    else:
        if orders < 1:
            raise ValueError("orders must be positive")
        passes = _draw(grouped, first_seen, orders, seed, epsilon)
    explore_gains = []
    retry_gains = []
    for order, bits in passes:
        played = replay(grouped, order, scores, cheap, strong, value, bits)
        explore_gains.append(played.gain(rows, value, cheap, strong))
        if retry is not None:
            retried = replay(grouped, order, scores, cheap, retry, value, bits)
            retry_gains.append(retried.gain(rows, value, cheap, strong))
    report = {"explore_gain_mean": _mean(explore_gains)}
    if retry is not None:
        report["retry_explore_gain_mean"] = _mean(retry_gains)
    return report


def evaluate(
    rows,
    scores,
    cheap,
    strong,
    values=VALUES,
    epsilon=0.1,
    orders=50,
    seed=0,
    retry=None,
):
    arm = retry_arm(rows, cheap, retry)
    by_value = {}
    for value in values:
        streamed = stream_gains(
            rows, scores, cheap, strong, value, epsilon, orders, seed, arm
        )
        entry = {
            "explore_gain_mean": streamed["explore_gain_mean"],
            "hindsight_gain": hindsight_gain(
                rows, scores, cheap, strong, value, strong
            ),
        }
        if arm is not None:
            entry["retry_explore_gain_mean"] = streamed["retry_explore_gain_mean"]
            entry["retry_hindsight_gain"] = hindsight_gain(
                rows, scores, cheap, arm, value, strong
            )
        by_value[f"{value:g}"] = entry
    return {
        "cheap": cheap,
        "strong": strong,
        "retry": arm,
        "rows": len(rows),
        "tasks": len({row["task"] for row in rows}),
        "epsilon": epsilon,
        "orders": orders,
        "seed": seed,
        "hindsight": "in-sample",
        "values": by_value,
    }


def render(report):
    lines = [
        (
            f"== cheap {report['cheap']}, strong {report['strong']}: "
            f"{report['rows']} rows over {report['tasks']} tasks, "
            f"epsilon {report['epsilon']:g}, {report['orders']} orders"
        ),
        "   gain versus the best single model; hindsight is in-sample",
    ]
    if report["retry"] is None:
        lines.append("   retry arm omitted: no second cheap attempt on every row")
    else:
        lines.append(f"   retry arm: {report['retry']}")
    for value, entry in report["values"].items():
        line = (
            f"   V=${value}  explore {entry['explore_gain_mean']:+.4f}  "
            f"hindsight {entry['hindsight_gain']:+.4f}"
        )
        if "retry_explore_gain_mean" in entry:
            line += (
                f"  retry explore {entry['retry_explore_gain_mean']:+.4f}  "
                f"retry hindsight {entry['retry_hindsight_gain']:+.4f}"
            )
        lines.append(line)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--cheap", required=True)
    parser.add_argument("--strong", required=True)
    parser.add_argument(
        "--retry",
        help="second cheap attempt; default <cheap>#retry when that column exists",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--orders", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--values", default=",".join(f"{value:g}" for value in VALUES))
    parser.add_argument("--json", help="write the report as JSON")
    args = parser.parse_args(argv)
    rows = load_rows(args.rows)
    scores = load_scores(args.scores)
    report = evaluate(
        rows,
        scores,
        args.cheap,
        args.strong,
        values=[float(value) for value in args.values.split(",") if value],
        epsilon=args.epsilon,
        orders=args.orders,
        seed=args.seed,
        retry=args.retry,
    )
    print(render(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
