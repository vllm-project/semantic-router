#!/usr/bin/env python3
"""Do the heads of an OR rule miss the same texts? (issue #4669, measured on the toxicchat join)

Reads the router_signal_ab.py ``join`` output of one arm for the suite's jailbreak and safety
files and joins the two heads by text. For the rule ``jailbreak OR safety`` a *miss* is a
rule-positive text (either gold positive) that neither head fires on.

Within each gold cell (jailbreak and unsafe, unsafe only, jailbreak only) the misses expected
if the heads failed independently are ``n * P(jailbreak silent) * P(safety silent)``. The gated
ratio is observed over expected misses in the cell where both gold labels are positive, the one
cell where each head has a reason to fire; a pooled ratio over all rule-positive cells is also
reported. The report gives the Mantel-Haenszel odds ratio of the two heads being silent
(conditional on the gold cell, over all texts, 0.5 added to the counts of a cell that has an empty
one) and 95% intervals from a bootstrap that resamples whole groups. Given a second arm, the same
draws give the difference between arms. ``--b-match-fpr`` sets the second arm's thresholds so each
head fires on as many gold-negative texts as the first arm's does (thresholds are fixed, not
re-estimated in each draw).

    python3 tools/guard_miss_correlation.py --suite text --a vela2/preds --a-thresholds 0.95,0.9 \\
        [--b vela1/preds (--b-thresholds 0.5,0.5 | --b-match-fpr)] --out guard.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JAILBREAK_FILE = "toxicchat/jailbreak/hold-toxicchat"
SAFETY_FILE = "toxicchat/safety/hold-toxicchat"
POSITIVE = {"jailbreak": "jailbreak", "safety": "unsafe"}
CELLS = ("both", "safety_only", "jailbreak_only", "neither")


@dataclass
class Table:
    """One row per text: gold of each head, whether each head fires, and the resampling group."""

    jb_gold: np.ndarray
    s_gold: np.ndarray
    jb_fires: np.ndarray
    s_fires: np.ndarray
    group: np.ndarray
    jb_score: np.ndarray
    s_score: np.ndarray

    def cell_index(self) -> np.ndarray:
        """0 both positive, 1 safety only, 2 jailbreak only, 3 neither."""
        return np.where(
            self.jb_gold & self.s_gold,
            0,
            np.where(self.s_gold, 1, np.where(self.jb_gold, 2, 3)),
        )


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_table(suite: str, preds: str, thresholds: tuple[float, float]) -> Table:
    """Join one arm's jailbreak and safety predictions by text; fire a head at its threshold.

    Raises:
        SystemExit: A prediction file is missing, a score lacks the head's positive label, or
            the two files share no text.
    """
    heads = {}
    for task, stem in (("jailbreak", JAILBREAK_FILE), ("safety", SAFETY_FILE)):
        rows = {r["id"]: r for r in read_jsonl(Path(suite) / f"{stem}.jsonl")}
        pred_path = Path(preds) / task / f"{Path(stem).name}.pred.jsonl"
        if not pred_path.exists():
            raise SystemExit(f"missing {pred_path}")
        by_text = {}
        for entry in read_jsonl(pred_path):
            row = rows[entry["id"]]
            if POSITIVE[task] not in entry["scores"]:
                raise SystemExit(
                    f"{pred_path}: no {POSITIVE[task]!r} score in {sorted(entry['scores'])}"
                )
            by_text[row["text"]] = (
                row["label"] == POSITIVE[task],
                entry["scores"][POSITIVE[task]],
                row["group"],
            )
        heads[task] = by_text
    texts = sorted(set(heads["jailbreak"]) & set(heads["safety"]))
    if not texts:
        raise SystemExit("the jailbreak and safety predictions share no text")
    jb, sf = heads["jailbreak"], heads["safety"]
    jb_score = np.array([jb[t][1] for t in texts])
    s_score = np.array([sf[t][1] for t in texts])
    return Table(
        jb_gold=np.array([jb[t][0] for t in texts]),
        s_gold=np.array([sf[t][0] for t in texts]),
        jb_fires=jb_score >= thresholds[0],
        s_fires=s_score >= thresholds[1],
        group=np.array([jb[t][2] for t in texts]),
        jb_score=jb_score,
        s_score=s_score,
    )


def matched_thresholds(reference: Table, other: Table) -> tuple[float, float]:
    """The thresholds that make ``other`` fire on no more gold-negative texts than ``reference`` does.

    Each head is matched alone: the share of its gold-negative texts it fires on (its false
    positive rate) is held at the reference's, by taking the lowest threshold that stays within it.
    """
    out = []
    for gold, ref_fires, score in (
        (reference.jb_gold, reference.jb_fires, other.jb_score),
        (reference.s_gold, reference.s_fires, other.s_score),
    ):
        negatives = np.sort(score[~gold])[::-1]
        allowed = int(ref_fires[~gold].mean() * len(negatives))
        out.append(
            0.0
            if allowed >= len(negatives)
            else float(np.nextafter(negatives[allowed], np.inf))
        )
    return out[0], out[1]


def refire(table: Table, thresholds: tuple[float, float]) -> Table:
    """The same table with each head fired at the given thresholds."""
    return Table(
        table.jb_gold,
        table.s_gold,
        table.jb_score >= thresholds[0],
        table.s_score >= thresholds[1],
        table.group,
        table.jb_score,
        table.s_score,
    )


def miss_counts(
    table: Table, rows: np.ndarray | None = None
) -> dict[str, tuple[int, float, int]]:
    """{cell: (observed misses, expected misses, texts)} over rule-positive cells."""
    rows = np.arange(len(table.group)) if rows is None else rows
    cell = table.cell_index()[rows]
    jb_silent, s_silent = ~table.jb_fires[rows], ~table.s_fires[rows]
    out = {}
    for index, name in enumerate(CELLS[:3]):
        pick = cell == index
        n = int(pick.sum())
        if n == 0:
            continue
        observed = int((jb_silent & s_silent & pick).sum())
        out[name] = (observed, n * jb_silent[pick].mean() * s_silent[pick].mean(), n)
    return out


def pooled_ratio(counts: dict[str, tuple[int, float, int]]) -> float:
    """Observed over expected misses across rule-positive cells (nan when none are expected)."""
    expected = sum(e for _, e, _ in counts.values())
    return (
        sum(o for o, _, _ in counts.values()) / expected if expected > 0 else math.nan
    )


def cell_ratio(counts: dict[str, tuple[int, float, int]], cell: str) -> float:
    """Observed over expected misses in one cell (nan when the cell is empty or none are expected)."""
    if cell not in counts or counts[cell][1] <= 0:
        return math.nan
    return counts[cell][0] / counts[cell][1]


def mantel_haenszel(table: Table, rows: np.ndarray | None = None) -> float:
    """Odds ratio of (jailbreak silent, safety silent) over the gold cells.

    A cell with an empty count has 0.5 added to each of its four counts.
    """
    rows = np.arange(len(table.group)) if rows is None else rows
    cell = table.cell_index()[rows]
    jb_silent, s_silent = ~table.jb_fires[rows], ~table.s_fires[rows]
    num = den = 0.0
    for index in range(len(CELLS)):
        pick = cell == index
        n = pick.sum()
        if n == 0:
            continue
        a = (jb_silent & s_silent & pick).sum()
        b = (jb_silent & ~s_silent & pick).sum()
        c = (~jb_silent & s_silent & pick).sum()
        d = (~jb_silent & ~s_silent & pick).sum()
        if min(a, b, c, d) == 0:
            a, b, c, d, n = a + 0.5, b + 0.5, c + 0.5, d + 0.5, n + 2
        num += a * d / n
        den += b * c / n
    if den == 0:
        return math.inf if num > 0 else math.nan
    return num / den


def poisson_upper_tail(k: int, mean: float) -> float:
    """P(X >= k) for X ~ Poisson(mean)."""
    if k <= 0:
        return 1.0
    term, below = math.exp(-mean), 0.0
    for i in range(k):
        below += term
        term *= mean / (i + 1)
    return max(0.0, 1.0 - below)


def smallest_detectable_ratio(expected: float, alpha: float = 0.025) -> float:
    """The smallest observed to expected ratio a Poisson count would show significantly above 1."""
    if expected <= 0:
        return math.inf
    k = 1
    while poisson_upper_tail(k, expected) >= alpha:
        k += 1
    return k / expected


def bootstrap(tables: list[Table], rounds: int, seed: int) -> dict[str, np.ndarray]:
    """Per-draw ratios and odds ratio for each arm, resampling whole groups with shared draws.

    All arms must hold the same texts in the same order.
    """
    first = tables[0]
    members = collections.defaultdict(list)
    for index, group in enumerate(first.group):
        members[group].append(index)
    groups = list(members.values())
    rng = np.random.default_rng(seed)
    ratios = np.full((len(tables), rounds), np.nan)
    pooled = np.full((len(tables), rounds), np.nan)
    odds = np.full((len(tables), rounds), np.nan)
    for r in range(rounds):
        pick = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        for a, table in enumerate(tables):
            counts = miss_counts(table, rows)
            ratios[a, r] = cell_ratio(counts, "both")
            pooled[a, r] = pooled_ratio(counts)
            odds[a, r] = mantel_haenszel(table, rows)
    return {"ratio": ratios, "pooled": pooled, "odds": odds}


def interval(values: np.ndarray) -> list[float | None]:
    finite = values[np.isfinite(values)]
    if len(finite) < len(values) / 2:
        return [None, None]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def summarize(tables: dict[str, Table], rounds: int, seed: int) -> dict:
    """The report: power facts first, then each arm's numbers, then the paired difference."""
    names = list(tables)
    draws = bootstrap([tables[n] for n in names], rounds, seed)
    report: dict = {"rounds": rounds, "seed": seed, "arms": {}}
    for a, name in enumerate(names):
        table = tables[name]
        counts = miss_counts(table)
        expected = sum(e for _, e, _ in counts.values())
        both_expected = counts["both"][1] if "both" in counts else 0.0
        report["arms"][name] = {
            "texts": len(table.group),
            "rule_positives": int(sum(n for _, _, n in counts.values())),
            "cells": {
                k: {"observed": o, "expected": round(e, 3), "texts": n}
                for k, (o, e, n) in counts.items()
            },
            "observed_misses": int(sum(o for o, _, _ in counts.values())),
            "expected_misses": round(expected, 3),
            "smallest_detectable_ratio": smallest_detectable_ratio(both_expected),
            "ratio": cell_ratio(counts, "both"),
            "ratio_ci": interval(draws["ratio"][a]),
            "pooled_ratio": pooled_ratio(counts),
            "pooled_ratio_ci": interval(draws["pooled"][a]),
            "odds_ratio": mantel_haenszel(table),
            "odds_ratio_ci": interval(draws["odds"][a]),
        }
    if len(names) == 2:
        diff = draws["ratio"][0] - draws["ratio"][1]
        report["difference_ratio"] = {
            "a_minus_b": report["arms"][names[0]]["ratio"]
            - report["arms"][names[1]]["ratio"],
            "ci": interval(diff),
        }
    return report


def verdict(report: dict, arm: str) -> str:
    """The #4669 gate on one arm: pass, fail or underpowered."""
    ci = report["arms"][arm]["ratio_ci"]
    ratio = report["arms"][arm]["ratio"]
    diff = report.get("difference_ratio", {}).get("ci", [None, None])
    if ci[0] is None or ci[1] is None or not math.isfinite(ratio):
        return "underpowered"
    if ci[1] < 1.25:
        return "fail"
    if ci[0] > 1 and ratio >= 1.25 and diff[0] is not None and diff[0] > 0:
        return "pass"
    return "underpowered"


def thresholds(value: str) -> tuple[float, float]:
    jailbreak, safety = (float(v) for v in value.split(","))
    return jailbreak, safety


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--suite", required=True, help="the suite's text/ directory")
    p.add_argument("--a", required=True, help="join output of the arm that is gated")
    p.add_argument(
        "--a-thresholds", type=thresholds, required=True, help="jailbreak,safety"
    )
    p.add_argument("--b", help="join output of the comparison arm (vela1)")
    p.add_argument(
        "--b-thresholds", type=thresholds, help="jailbreak,safety, set by hand"
    )
    p.add_argument(
        "--b-match-fpr",
        action="store_true",
        help="set --b's thresholds so each head fires on as many gold-negative texts as --a's",
    )
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.b and bool(args.b_thresholds) == args.b_match_fpr:
        p.error("--b takes either --b-thresholds or --b-match-fpr")
    if not args.b and (args.b_thresholds or args.b_match_fpr):
        p.error("--b-thresholds and --b-match-fpr need --b")
    tables = {"a": build_table(args.suite, args.a, args.a_thresholds)}
    if args.b:
        tables["b"] = build_table(args.suite, args.b, (0.5, 0.5))
        tables["b"] = refire(
            tables["b"],
            args.b_thresholds or matched_thresholds(tables["a"], tables["b"]),
        )
        if not np.array_equal(tables["a"].group, tables["b"].group):
            raise SystemExit(
                "the arms do not hold the same texts; join them on the same rows"
            )
    report = summarize(tables, args.rounds, args.seed)
    report["thresholds"] = {
        name: [
            float(t.jb_score[t.jb_fires].min()) if t.jb_fires.any() else None,
            float(t.s_score[t.s_fires].min()) if t.s_fires.any() else None,
        ]
        for name, t in tables.items()
    }
    report["verdict"] = verdict(report, "a")
    Path(args.out).write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "arms"}, indent=2, default=str
        )
    )
    for name, arm in report["arms"].items():
        print(name, {k: arm[k] for k in ("rule_positives", "observed_misses", "expected_misses",
                                         "smallest_detectable_ratio", "ratio", "ratio_ci", "pooled_ratio", "odds_ratio")})  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
