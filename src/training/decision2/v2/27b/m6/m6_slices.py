"""Milestone 6 development slices for the ~27B track (host CPU; never a release or formal score).

Inputs are ``kernel_readout slices`` outputs (``<slice>.probs.jsonl``: raw T = 1 probabilities aligned with each row's
option keys) and the SELECT-format row files they were read from.

``pn1``: the mlx-diag-style Noul guard (preregistration gate G5) on PN1 dev. The rule is the 9B M7 / M8 rule,
reused from ``v2/9b/lux9b/m7_rules.py`` (summaries, the noisy-construction list and the paired group bootstrap):
yes = P(true) > 0.5; **hop** = yes-rate on ``pn-hop`` (true paraphrases, all eight languages); **clean gold-no** =
yes-rate on gold-no rows of ``pn-near`` / ``pn-name`` / ``pn-twin`` outside the constructions PN1-r2 dropped for label
noise. The guard passes when hop Δ ≥ −0.03 and clean gold-no Δ ≤ 0 against the reference (point estimates).
Gold, family, language and group come from the PN1 SELECT rows (label = index into the ``false`` / ``true`` options).

``breadth``: IB DEV accuracy by family (argmax of the probabilities against the label); B_dev = macro accuracy over
the families with at least 50 rows; candidate − reference paired group bootstrap (2,000 draws, seed 20261001).
Gate G6 passes when the lower bound is > 0. Families listed with ``--in-distribution`` are reported separately too.

    python3 -m v2.27b.m6.m6_slices pn1 --rows PN1.jsonl --candidate NAME=PROBS --reference NAME=PROBS --output OUT
    python3 -m v2.27b.m6.m6_slices breadth --rows IB.jsonl --candidate NAME=PROBS --reference NAME=PROBS \
        [--in-distribution w2c --in-distribution isarc] --output OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "9b"))
from lux9b import m7_rules  # noqa: E402

SCHEMA = "decision2-27b-m6-slices/1"
ROLE = "development slice; never a release, formal or Index score"
HOP_SLACK = m7_rules.HOP_SLACK
BREADTH_MIN_ROWS = 50
BREADTH_REPS = 2000
BREADTH_SEED = 20261001


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_probs(path: Path, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Probabilities by row id; every row needs one record with the row's option keys."""
    records = {r["id"]: r for r in read_jsonl(path)}
    out = {}
    for row in rows:
        record = records.pop(row["id"], None)
        if record is None:
            raise ValueError(f"{path}: no prediction for {row['id']}")
        keys = [option["key"] for option in row["options"]]
        if record["keys"] != keys or len(record["probabilities"]) != len(keys):
            raise ValueError(f"{path}: {row['id']} keys differ from the row's options")
        out[row["id"]] = [float(p) for p in record["probabilities"]]
    if records:
        raise ValueError(f"{path}: {len(records)} predictions are not in the rows")
    return out


def pn1_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        keys = [option["key"] for option in row["options"]]
        if row["task_type"] != "noul" or sorted(keys) != ["false", "true"]:
            raise ValueError(f"{row['id']} is not a false / true Noul row")
        out.append(
            {
                "id": row["id"],
                "group": row["group_id"],
                "language": row["language"],
                "family": row["family"],
                "gold": keys[row["label"]] == "true",
            }
        )
    return out


def yes_of(
    rows: list[dict[str, Any]], probs: dict[str, list[float]]
) -> dict[str, bool]:
    out = {}
    for row in rows:
        keys = [option["key"] for option in row["options"]]
        out[row["id"]] = probs[row["id"]][keys.index("true")] > 0.5
    return out


def pn1_report(
    rows: list[dict[str, Any]],
    candidate: tuple[str, dict[str, list[float]]],
    reference: tuple[str, dict[str, list[float]]],
) -> dict[str, Any]:
    gold = pn1_rows(rows)
    yes_c, yes_r = yes_of(rows, candidate[1]), yes_of(rows, reference[1])
    c, r = m7_rules.pn1_summary(gold, yes_c), m7_rules.pn1_summary(gold, yes_r)
    delta = {
        k: c[k]["yes"] - r[k]["yes"] for k in ("hop", "clean_no", "all8", "pawsx6")
    }
    reasons = []
    if delta["hop"] < -HOP_SLACK:
        reasons.append(f"hop yes {delta['hop']:+.4f} < -{HOP_SLACK}")
    if delta["clean_no"] > 0:
        reasons.append(f"clean gold-no yes {delta['clean_no']:+.4f} > 0")
    return {
        "schema": SCHEMA,
        "role": ROLE,
        "slice": "pn1-dev",
        "rule": "9B M7 / M8 PN1 guard (lux9b.m7_rules): hop delta >= -0.03 and clean gold-no delta <= 0",
        "candidate": candidate[0],
        "reference": reference[0],
        "candidate_summary": c,
        "reference_summary": r,
        "delta": delta,
        "delta_ci95": m7_rules.pn1_compare(gold, yes_r, yes_c),
        "noisy_constructions_excluded": sorted(
            f"{f}/{lang}" for f, lang in m7_rules.PN1_NOISY
        ),
        "pass": not reasons,
        "reasons": reasons,
    }


def correct_of(
    rows: list[dict[str, Any]], probs: dict[str, list[float]]
) -> dict[str, bool]:
    out = {}
    for row in rows:
        p = probs[row["id"]]
        out[row["id"]] = max(range(len(p)), key=lambda i: (p[i], -i)) == row["label"]
    return out


def family_accuracy(
    rows: list[dict[str, Any]], correct: dict[str, bool]
) -> dict[str, dict[str, Any]]:
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        tally[row["family"]][0] += correct[row["id"]]
        tally[row["family"]][1] += 1
    return {
        f: {"correct": c, "n": n, "accuracy": c / n}
        for f, (c, n) in sorted(tally.items())
    }


def breadth_report(
    rows: list[dict[str, Any]],
    candidate: tuple[str, dict[str, list[float]]],
    reference: tuple[str, dict[str, list[float]]],
    in_distribution: list[str],
    reps: int = BREADTH_REPS,
    seed: int = BREADTH_SEED,
) -> dict[str, Any]:
    ok_c, ok_r = correct_of(rows, candidate[1]), correct_of(rows, reference[1])
    acc_c, acc_r = family_accuracy(rows, ok_c), family_accuracy(rows, ok_r)
    eligible = sorted(f for f, v in acc_c.items() if v["n"] >= BREADTH_MIN_ROWS)
    if not eligible:
        raise ValueError(f"no IB DEV family has {BREADTH_MIN_ROWS} rows")

    def macro(acc: dict[str, dict[str, Any]]) -> float:
        return sum(acc[f]["accuracy"] for f in eligible) / len(eligible)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["family"] in eligible:
            by_group[row["group_id"]].append(row)
    names = sorted(by_group)
    stats = {}
    for g in names:
        cell: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for row in by_group[g]:
            cell[row["family"]][0] += ok_c[row["id"]]
            cell[row["family"]][1] += ok_r[row["id"]]
            cell[row["family"]][2] += 1
        stats[g] = dict(cell)
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        tot: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for g in rng.choices(names, k=len(names)):
            for fam, (c, r, n) in stats[g].items():
                tot[fam][0] += c
                tot[fam][1] += r
                tot[fam][2] += n
        if all(tot[f][2] for f in eligible):
            draws.append(
                sum(tot[f][0] / tot[f][2] - tot[f][1] / tot[f][2] for f in eligible)
                / len(eligible)
            )
    draws.sort()
    n = len(draws)
    ci = [draws[int(0.025 * (n - 1))], draws[int(round(0.975 * (n - 1)))]]
    delta = macro(acc_c) - macro(acc_r)
    return {
        "schema": SCHEMA,
        "role": ROLE,
        "slice": "ib-dev",
        "rule": f"B_dev = macro accuracy over IB DEV families with >= {BREADTH_MIN_ROWS} rows;"
        " pass when the paired group-bootstrap lower bound of candidate - reference is > 0",
        "candidate": candidate[0],
        "reference": reference[0],
        "eligible_families": eligible,
        "in_distribution_families": sorted(in_distribution),
        "B_dev": {"candidate": macro(acc_c), "reference": macro(acc_r)},
        "delta": delta,
        "delta_ci95": ci,
        "bootstrap": {"replicates": reps, "kept": n, "seed": seed, "unit": "group_id"},
        "by_family": {
            f: {
                "candidate": acc_c[f],
                "reference": acc_r[f],
                "delta": acc_c[f]["accuracy"] - acc_r[f]["accuracy"],
                "in_distribution": f in in_distribution,
            }
            for f in acc_c
        },
        "pass": ci[0] > 0,
    }


def named(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not name or not path:
        raise SystemExit(f"bad {spec!r}; use NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("pn1", "breadth"):
        p = sub.add_parser(mode)
        p.add_argument(
            "--rows", type=Path, required=True, help="SELECT-format rows of the slice"
        )
        p.add_argument("--candidate", required=True, help="NAME=PROBS")
        p.add_argument("--reference", required=True, help="NAME=PROBS")
        p.add_argument("--output", type=Path, required=True)
        if mode == "breadth":
            p.add_argument("--in-distribution", action="append", default=[])
    args = parser.parse_args(argv)
    rows = read_jsonl(args.rows)
    (cname, cpath), (rname, rpath) = named(args.candidate), named(args.reference)
    candidate, reference = (cname, read_probs(cpath, rows)), (
        rname,
        read_probs(rpath, rows),
    )
    if args.mode == "pn1":
        report = pn1_report(rows, candidate, reference)
    else:
        report = breadth_report(rows, candidate, reference, args.in_distribution)
    report["inputs_sha256"] = {
        "rows": sha_file(args.rows),
        "candidate": sha_file(cpath),
        "reference": sha_file(rpath),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=1, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "slice": report["slice"],
                "candidate": cname,
                "reference": rname,
                "delta": report["delta"],
                "delta_ci95": report["delta_ci95"],
                "pass": report["pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
