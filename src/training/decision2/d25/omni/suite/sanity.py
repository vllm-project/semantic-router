"""Sanity checks of a built suite: baseline predictors and shape checks against the board's notes.

Predictors, scored with ``suite/score.py``: ``uniform`` (expected accuracy of a uniform guess, which must
give skill 0 exactly), ``oracle`` (gold key, skill 100), ``always_first`` and ``always_last`` (answer
position bias of the build) and ``random`` (one seeded uniform draw per row, skill near 0). Shape
checks: row counts, option and image-count ranges from the board's construction notes, and the
score-lattice fingerprint (``lattice``).

    python -m d25.omni.suite.sanity --rows rows.jsonl.gz [--out sanity.json]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from d25.omni.suite import lattice, score
from d25.omni.suite.rows import stable_rng

# Board construction notes (bench_info.tech) as option / image ranges; Winoground and Hateful Memes are binary.
NOTES = {
    "CV-Bench": {"options": (2, 6), "images": (1, 1)},
    "BLINK": {"options": (2, 4), "images": (1, 4)},
    "RealWorldQA": {"options": (2, 6), "images": (1, 1)},
    "CharXiv": {"options": (4, 4), "images": (1, 1)},
    "InfographicVQA": {"options": (2, 6), "images": (1, 1)},
    "Mind2Web": {"options": (4, 4), "images": (1, 1)},
    "Winoground": {"options": (2, 2), "images": (1, 1)},
    "KIE (CORD+FUNSD)": {"options": (2, 6), "images": (1, 1)},
    "Moderation (Hateful Memes)": {"options": (2, 2), "images": (1, 1)},
    "R-Bench-M": {"options": (2, 6), "images": (1, 1)},
    "MMMU-Pro vision": {"options": (2, 10), "images": (1, 1)},
}


def _answer(question, key):
    keys = list(question["criteria"])
    return {
        "type": "choice",
        "choice": key,
        "probabilities": {k: float(k == key) for k in keys},
    }


def predictor_answers(rows, name: str) -> dict:
    out = {}
    for row in rows:
        answers = {}
        for qid, q in row["questions"].items():
            keys = list(q["criteria"])
            if name == "oracle":
                key = row["expected"][qid]
            elif name == "always_first":
                key = keys[0]
            elif name == "always_last":
                key = keys[-1]
            elif name == "random":
                key = stable_rng("sanity-random", row["id"], qid).choice(keys)
            else:
                raise ValueError(name)
            answers[qid] = _answer(q, key)
        out[row["id"]] = answers
    return out


def uniform_expected(rows) -> dict[str, float]:
    """Skill of the expected accuracy of a uniform guess, per benchmark (exactly 0 by construction)."""
    acc, chance, n = defaultdict(float), defaultdict(float), Counter()
    for row in rows:
        b = score.benchmark_of(row)
        p = score.row_chance(row)
        acc[b] += p
        chance[b] += p
        n[b] += 1
    return {b: score.skill(acc[b] / n[b], chance[b] / n[b]) for b in n}


def run(rows: list[dict]) -> dict:
    by_b = defaultdict(list)
    for r in rows:
        by_b[score.benchmark_of(r)].append(r)
    report = {"benchmarks": {}, "predictors": {}}
    report["predictors"]["uniform"] = uniform_expected(rows)
    for name in ("oracle", "always_first", "always_last", "random"):
        scored = score.score_suite(rows, predictor_answers(rows, name))
        report["predictors"][name] = {
            b: v["skill"] for b, v in scored["benchmarks"].items()
        } | {"public": scored["public"]}
    values = lattice.board_values()
    failures = []
    for b, rs in by_b.items():
        options = Counter(len(r["questions"]["q1"]["criteria"]) for r in rs)
        images = Counter(len(r["images"]) for r in rs)
        lo, hi = NOTES[b]["options"]
        ilo, ihi = NOTES[b]["images"]
        chance_sum = sum(score.row_chance(r) for r in rs)
        misses = lattice.check(len(rs), chance_sum, values.get(b, []))
        checks = {
            "rows_match_board": len(rs) == score.PUBLIC_ROWS[b],
            "options_in_notes": all(lo <= k <= hi for k in options),
            "images_in_notes": all(ilo <= k <= ihi for k in images),
            "lattice_consistent": not misses,
            "oracle_100": math.isclose(report["predictors"]["oracle"][b], 100.0),
            "uniform_0": abs(report["predictors"]["uniform"][b]) < 1e-9,
        }
        failures += [f"{b}: {k}" for k, ok in checks.items() if not ok]
        report["benchmarks"][b] = {
            "rows": len(rs),
            "board_rows": score.PUBLIC_ROWS[b],
            "chance_sum": round(chance_sum, 4),
            "options": dict(sorted(options.items())),
            "images": dict(sorted(images.items())),
            "gold_position": dict(
                sorted(
                    Counter(
                        list(r["questions"]["q1"]["criteria"]).index(
                            r["expected"]["q1"]
                        )
                        for r in rs
                    ).items()
                )
            ),
            "lattice_misses": len(misses),
            "checks": checks,
        }
    report["failures"] = failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = run(list(score.read_jsonl(args.rows)))
    text = json.dumps(report, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
