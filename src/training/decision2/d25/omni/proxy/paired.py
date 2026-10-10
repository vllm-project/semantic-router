"""Paired Vision-board estimate of a candidate against a reference entrant measured on the same rows.

    python -m d25.omni.proxy.paired --board vision.json --reference-name "Perplexity Decider v1.1 (27B)" \
        --rows SUITE/rows.jsonl.gz --ours OURS/public/results.jsonl --ref REF/public/results.jsonl \
        --proxy BLINK=ROWS,OURS,REF --proxy "Moderation (Hateful Memes)"=ROWS,OURS,REF --out paired.json

Estimate = the reference's official Full + 0.5 dPublic + 0.5 dPrivate, where

- dPublic = sum over the measured public benchmarks of w_b (skill_ours - skill_ref) / 9.75 (floored skills,
  as the board aggregates them); a benchmark missing from the local suite contributes 0 with an
  uncertainty equal to the spread of that benchmark's skill among the board's top entrants;
- dPrivate = mean over the 9 private sets of the estimated private skill difference: from a proxy on
  fresh items when one is given (slope 1), otherwise from the measured public difference times the
  board's cross-entrant slope of private on public skill for that benchmark.

The standard error combines a paired bootstrap over rows (both models resampled together, per benchmark),
the slope uncertainty of each public-to-private map, and the missing-benchmark allowance.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from d25.omni.suite import score

TOP_K_SPREAD = 6


def per_row(rows_path: str, results_path: str) -> dict[str, list[tuple[float, float]]]:
    rows = list(score.read_jsonl(rows_path))
    answers = score.load_answers(results_path)
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        out[score.benchmark_of(row)].append(
            (
                float(score.row_correct(row, answers.get(row["id"]))),
                score.row_chance(row),
            )
        )
    return out


def skill_of(items: list[tuple[float, float]]) -> float:
    acc = sum(c for c, _ in items) / len(items)
    chance = sum(p for _, p in items) / len(items)
    return score.skill(acc, chance)


def paired_skills(a: list, b: list, rng: random.Random | None) -> tuple[float, float]:
    if rng is None:
        return skill_of(a), skill_of(b)
    idx = [rng.randrange(len(a)) for _ in range(len(a))]
    return skill_of([a[i] for i in idx]), skill_of([b[i] for i in idx])


def floor(x: float) -> float:
    return max(0.0, x)


def slope(board: dict, benchmark: str) -> tuple[float, float]:
    """Least-squares slope (and its standard error) of private on public skill across entrants."""
    pts = []
    for entrant in board["entrants"]:
        cell = entrant["bench"].get(benchmark) or {}
        pub, priv = cell.get("pub"), cell.get("private")
        if pub is not None and priv is not None:
            pts.append((floor(pub), floor(priv)))
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    beta = sum((x - mx) * (y - my) for x, y in pts) / sxx
    resid = sum((y - my - beta * (x - mx)) ** 2 for x, y in pts) / max(1, n - 2)
    return beta, math.sqrt(resid / sxx)


def spread(board: dict, benchmark: str, key: str) -> float:
    vals = sorted(
        (
            floor((e["bench"].get(benchmark) or {}).get(key) or 0.0)
            for e in board["entrants"]
        ),
        reverse=True,
    )
    top = vals[:TOP_K_SPREAD]
    mean = sum(top) / len(top)
    return math.sqrt(sum((v - mean) ** 2 for v in top) / max(1, len(top) - 1))


def estimate(
    public: dict, proxies: dict, board: dict, rng: random.Random | None
) -> dict:
    d_pub, d_priv, per = 0.0, 0.0, {}
    for benchmark in score.BENCHMARKS:
        w = score.WEIGHTS[benchmark]
        cell = {}
        if benchmark in public:
            s_ours, s_ref = paired_skills(*public[benchmark], rng)
            d = floor(s_ours) - floor(s_ref)
            d_pub += w * d / 9.75
            cell.update(public_ours=s_ours, public_ref=s_ref, d_public=d)
        if benchmark in score.PRIVATE_SETS:
            if benchmark in proxies:
                p_ours, p_ref = paired_skills(*proxies[benchmark], rng)
                dp = floor(p_ours) - floor(p_ref)
                cell.update(
                    proxy_ours=p_ours,
                    proxy_ref=p_ref,
                    d_private=dp,
                    private_from="proxy",
                )
            elif "d_public" in cell:
                beta, _ = slope(board, benchmark)
                dp = beta * cell["d_public"]
                cell.update(d_private=dp, private_from=f"public x {beta:.3f}")
            else:
                dp = 0.0
                cell.update(d_private=0.0, private_from="missing")
            d_priv += dp / len(score.PRIVATE_SETS)
        per[benchmark] = cell
    return {
        "d_public": d_pub,
        "d_private": d_priv,
        "d_full": 0.5 * d_pub + 0.5 * d_priv,
        "benchmarks": per,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--board", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument(
        "--proxy", action="append", default=[], help="BENCHMARK=ROWS,OURS,REF"
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    board = json.loads(Path(args.board).read_text())
    ref = next(e for e in board["entrants"] if e["name"] == args.reference_name)
    ours_rows, ref_rows = per_row(args.rows, args.ours), per_row(args.rows, args.ref)
    public = {b: (ours_rows[b], ref_rows[b]) for b in ours_rows}
    proxies = {}
    for spec in args.proxy:
        benchmark, paths = spec.split("=", 1)
        rows, ours, refp = paths.split(",")
        o, r = per_row(rows, ours), per_row(rows, refp)
        proxies[benchmark] = (
            [x for v in o.values() for x in v],
            [x for v in r.values() for x in v],
        )
    point = estimate(public, proxies, board, None)
    rng = random.Random(args.seed)
    draws = [
        estimate(public, proxies, board, rng)["d_full"] for _ in range(args.bootstrap)
    ]
    mean = sum(draws) / len(draws)
    se_boot = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
    var_slope = var_missing = 0.0
    for benchmark, cell in point["benchmarks"].items():
        if cell.get("private_from", "").startswith("public x"):
            _, beta_se = slope(board, benchmark)
            var_slope += (
                0.5 * beta_se * cell["d_public"] / len(score.PRIVATE_SETS)
            ) ** 2
        if "d_public" not in cell:
            var_missing += (
                0.5 * score.WEIGHTS[benchmark] * spread(board, benchmark, "pub") / 9.75
            ) ** 2
            if (
                benchmark in score.PRIVATE_SETS
                and cell.get("private_from") == "missing"
            ):
                var_missing += (
                    0.5 * spread(board, benchmark, "private") / len(score.PRIVATE_SETS)
                ) ** 2
    se = math.sqrt(se_boot**2 + var_slope + var_missing)
    est = ref["full"] + point["d_full"]
    entrants = sorted(board["entrants"], key=lambda e: -e["full"])
    result = {
        "reference": {
            "name": ref["name"],
            "full": ref["full"],
            "pub": ref["pub"],
            "priv": ref["priv"],
        },
        "board_generated": board.get("generated_utc"),
        "estimate_full": est,
        "estimate_public": ref["pub"] + point["d_public"],
        "estimate_private": ref["priv"] + point["d_private"],
        "se": se,
        "se_parts": {
            "bootstrap": se_boot,
            "slope": math.sqrt(var_slope),
            "missing": math.sqrt(var_missing),
        },
        "bound_1se": est - se,
        "bound_90": est - 1.2816 * se,
        "live": {
            f"#{i + 1}": {"name": e["name"], "full": e["full"]}
            for i, e in enumerate(entrants[:3])
        },
        **point,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "estimate_full",
                    "se",
                    "bound_1se",
                    "bound_90",
                    "d_public",
                    "d_private",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
