"""Paired bootstrap of the Index difference between two IX1 runs over the same 0.2.1 panel (private output).

    python3 -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base base/results.jsonl --new new/results.jsonl \
        --external <private frontier-gap JSON> --replicates 2000 --seed S --workers 24 --out boot.json

Run from the private index021 directory with the port's PYTHONPATH (as score.sh). Unit: the scoring case
(``group_id``) within each benchmark, resampled with replacement independently per benchmark; both runs are
scored on the same resampled cases (paired), each copy under a fresh case and run ID, with the port's own
``score_rows`` (native metrics, chance-skill, the board's area weights). A replicate is therefore the whole
headline recomputed, so the area weighting is propagated rather than approximated; the per-benchmark variance
sum Σ w_b² var_b (board weights from the private file) is reported as a cross-check. Before any replicate the
identity resample (every case once, renamed) must reproduce both runs' headline exactly. The output holds
Index values and stays private.

``--exclude NAME ...`` drops those benchmarks' rows before scoring (both the observed headline and every
replicate), so the headline is the port's balanced skill over the remaining benchmarks, e.g. a transfer-only
delta without the benchmarks whose families or formats a model trained on.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing
import random
import statistics
import time
from pathlib import Path
from typing import Any

_STATE: dict[str, Any] = {}


def _cases(rows: list[dict]) -> dict[int, list[list[dict]]]:
    by_benchmark: dict[int, dict[str, list[dict]]] = collections.defaultdict(dict)
    for row in rows:
        e = row["_evaluation"]
        by_benchmark[e["catalog_id"]].setdefault(e["group_id"], []).append(row)
    return {n: list(groups.values()) for n, groups in sorted(by_benchmark.items())}


def _resample(
    cases: dict[int, list[list[dict]]],
    results: tuple[dict, dict],
    picks: dict[int, list[int]],
) -> tuple[list[dict], dict, dict]:
    rows: list[dict] = []
    base, new = {}, {}
    for number, groups in cases.items():
        for j, index in enumerate(picks[number]):
            for row in groups[index]:
                e = row["_evaluation"]
                run_id = f"{e['run_id']}#b{j}"
                rows.append(
                    {
                        **row,
                        "_evaluation": {
                            **e,
                            "group_id": f"{e['group_id']}#b{j}",
                            "run_id": run_id,
                        },
                    }
                )
                for source, target in zip(results, (base, new)):
                    if e["run_id"] in source:
                        target[run_id] = {**source[e["run_id"]], "run_id": run_id}
    return rows, base, new


def _summary(scored: dict) -> dict[str, Any]:
    return {
        "balanced_skill": scored["scores"]["balanced_skill"],
        "areas": {a["id"]: 100 * a["skill"] for a in scored["areas"]},
        "benchmarks": {n: 100 * v["skill"] for n, v in scored["benchmarks"].items()},
    }


def _score_pair(picks: dict[int, list[int]]) -> tuple[dict, dict]:
    from external_index021.score import score_rows

    rows, base, new = _resample(_STATE["cases"], _STATE["results"], picks)
    return _summary(score_rows(rows, base)), _summary(score_rows(rows, new))


def _replicate(index: int) -> tuple[int, dict, dict]:
    rng = random.Random(f"{_STATE['seed']}:{index}")
    picks = {
        n: [rng.randrange(len(groups)) for _ in groups]
        for n, groups in _STATE["cases"].items()
    }
    base, new = _score_pair(picks)
    return index, base, new


def _interval(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    k = len(ordered)

    def q(p: float) -> float:
        x = p * (k - 1)
        lo = math.floor(x)
        return ordered[lo] + (ordered[min(lo + 1, k - 1)] - ordered[lo]) * (x - lo)

    return {
        "se": statistics.stdev(values),
        "ci95": [round(q(0.025), 4), round(q(0.975), 4)],
        "p_le_0": sum(v <= 0 for v in values) / k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", default="20261002")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()

    from decision_index.scoring.report import load_results
    from external_index021.score import score_rows, selected_rows, verified_suite

    from v2.eval.ix1.compare import NAMES

    started = time.time()
    rows, _ = selected_rows(verified_suite(args.suite_dir))
    unknown = set(args.exclude) - set(NAMES.values())
    if unknown:
        raise SystemExit(f"unknown benchmarks {sorted(unknown)}")
    dropped = {n for n, name in NAMES.items() if name in args.exclude}
    rows = [r for r in rows if r["_evaluation"]["catalog_id"] not in dropped]
    results = (load_results(args.base), load_results(args.new))
    cases = _cases(rows)
    observed = tuple(_summary(score_rows(rows, r)) for r in results)
    _STATE.update(cases=cases, results=results, seed=args.seed)
    identity = _score_pair({n: list(range(len(g))) for n, g in cases.items()})
    for got, want in zip(identity, observed):
        if abs(got["balanced_skill"] - want["balanced_skill"]) > 1e-9:
            raise SystemExit("the identity resample does not reproduce the headline")
    weights = json.loads(args.external.read_text())["benchmark_index_weight"]
    weight = {str(n): weights[name] for n, name in NAMES.items() if n not in dropped}

    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        draws = sorted(pool.imap_unordered(_replicate, range(args.replicates), 4))
    base = [d[1] for d in draws]
    new = [d[2] for d in draws]

    def delta(get) -> list[float]:
        return [get(b) - get(a) for a, b in zip(base, new)]

    head = delta(lambda s: s["balanced_skill"])
    obs_delta = observed[1]["balanced_skill"] - observed[0]["balanced_skill"]
    areas = {}
    for area in observed[0]["areas"]:
        d = delta(lambda s, a=area: s["areas"][a])
        areas[area] = {
            "base": round(observed[0]["areas"][area], 3),
            "new": round(observed[1]["areas"][area], 3),
            "delta": round(observed[1]["areas"][area] - observed[0]["areas"][area], 3),
            **_interval(d),
        }
    benchmarks = {}
    variance_sum = 0.0
    for number in weight:
        d = delta(lambda s, n=number: s["benchmarks"][n])
        stats = _interval(d)
        variance_sum += (weight[number] * stats["se"]) ** 2
        benchmarks[NAMES[int(number)]] = {
            "base": round(observed[0]["benchmarks"][number], 2),
            "new": round(observed[1]["benchmarks"][number], 2),
            "delta": round(
                observed[1]["benchmarks"][number] - observed[0]["benchmarks"][number], 2
            ),
            "weight": weight[number],
            "cases": len(cases[int(number)]),
            **stats,
        }
    report = {
        "schema": "ix1-paired-boot/1",
        "label": "independent provisional 0.2.1 reproduction (private)",
        "unit": "scoring case (group_id) within benchmark, paired, with replacement",
        "replicates": args.replicates,
        "seed": args.seed,
        "excluded": sorted(args.exclude),
        "inputs_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in (("base", args.base), ("new", args.new))
        },
        "cases": sum(len(g) for g in cases.values()),
        "rows": len(rows),
        "headline": {
            "base": round(observed[0]["balanced_skill"], 3),
            "new": round(observed[1]["balanced_skill"], 3),
            "delta": round(obs_delta, 3),
            **_interval(head),
            "bootstrap_mean_delta": round(statistics.fmean(head), 4),
            "se_from_benchmark_variances": math.sqrt(variance_sum),
        },
        "areas": areas,
        "benchmarks": benchmarks,
        "seconds": round(time.time() - started, 1),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"replicates": args.replicates, "seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
