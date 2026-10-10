"""Latency and cross-device answer comparison of Decision Index result files (stdlib only).

``latency``: the board's gate statistics over one kit run (``results.jsonl[.gz]``), one request at a time,
warm-up rows excluded (their run ids come from the sample's ``.json`` design file):

    python compare.py latency --results run/results.jsonl --design latency-760.jsonl.gz.json [--gate-ms 1000]

``answers``: two runs of the same rows (e.g. RTX PRO 6000 vs MI325X): argmax agreement, noul flips at 0.5,
probability drift, and whether every flip is a near-tie in the reference run:

    python compare.py answers --results cuda/results.jsonl --reference rocm/results.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATE_MS = 1000.0


def read_results(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = {}
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[row["run_id"]] = row
    return rows


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency(
    results: dict[str, dict], warmup: set[str], gate_ms: float = GATE_MS
) -> dict:
    timed = [r for rid, r in results.items() if rid not in warmup]
    ok = [r for r in timed if r["status"] == "ok"]
    values = [r["total_wall_ms"] for r in ok]
    per = defaultdict(list)
    for r in ok:
        per[str(r.get("catalog_id"))].append(r["total_wall_ms"])
    stats = {
        "timed_rows": len(timed),
        "ok": len(ok),
        "not_timed": {
            s: sum(r["status"] == s for r in timed)
            for s in ("unsupported", "error", "abstained")
        },
        "median_ms": round(statistics.median(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
        "p80_ms": round(percentile(values, 80), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "max_ms": round(max(values), 1),
        "per_catalog_median_ms": {
            k: round(statistics.median(v), 1)
            for k, v in sorted(per.items(), key=lambda x: int(x[0]))
            if len(v) >= 5
        },
        "gate_ms": gate_ms,
    }
    stats["gate_pass"] = all(
        stats[k] < gate_ms for k in ("median_ms", "mean_ms", "p80_ms")
    )
    return stats


def answer_values(answer: dict) -> tuple[str, list[str], list[float]]:
    if answer["type"] == "noul":
        p = float(answer["noul"])
        return "noul", ["false", "true"], [1 - p, p]
    keys = list(answer["probabilities"])
    return "choice", keys, [float(answer["probabilities"][k]) for k in keys]


def answers(
    results: dict[str, dict], reference: dict[str, dict], near_tie: float = 0.02
) -> dict:
    common = [rid for rid in reference if rid in results]
    stats = {
        "rows_reference": len(reference),
        "rows_common": len(common),
        "status_mismatch": 0,
        "questions": 0,
        "argmax_changes": 0,
        "noul_flips": 0,
        "flips_near_tie": 0,
        "max_abs_dp": 0.0,
        "mean_abs_dp": 0.0,
    }
    diffs, flips = [], []
    for rid in common:
        a, b = results[rid], reference[rid]
        if a["status"] != b["status"]:
            stats["status_mismatch"] += 1
            continue
        if a["status"] != "ok":
            continue
        for key, ref in b["response"]["answers"].items():
            kind, keys, want = answer_values(ref)
            _, other_keys, have = answer_values(a["response"]["answers"][key])
            if other_keys != keys:
                stats["status_mismatch"] += 1
                continue
            stats["questions"] += 1
            diff = max(abs(x - y) for x, y in zip(want, have))
            diffs.append(diff)
            top = sorted(want, reverse=True)
            margin = top[0] - (top[1] if len(top) > 1 else 0.0)
            if kind == "noul":
                flipped = (want[1] >= 0.5) != (have[1] >= 0.5)
                margin = abs(want[1] - 0.5) * 2
                stats["noul_flips"] += flipped
            else:
                flipped = max(range(len(want)), key=want.__getitem__) != max(
                    range(len(have)), key=have.__getitem__
                )
                stats["argmax_changes"] += flipped
            if flipped:
                stats["flips_near_tie"] += margin <= near_tie
                flips.append(
                    {
                        "run_id": rid,
                        "question": key,
                        "type": kind,
                        "reference_margin": round(margin, 5),
                        "abs_dp": round(diff, 5),
                    }
                )
    if diffs:
        stats["max_abs_dp"] = round(max(diffs), 6)
        stats["mean_abs_dp"] = round(statistics.fmean(diffs), 6)
        stats["p99_abs_dp"] = round(percentile(diffs, 99), 6)
    stats["near_tie_margin"] = near_tie
    stats["all_flips_near_ties"] = (
        stats["flips_near_tie"] == stats["argmax_changes"] + stats["noul_flips"]
    )
    stats["flips"] = flips[:50]
    return stats


def tail(results: dict[str, dict], warmup: set[str], top: int = 15) -> dict:
    """The slowest timed requests with their size, and time per thousand input tokens across all of them."""
    rows = []
    for rid, r in results.items():
        if rid in warmup or r["status"] != "ok":
            continue
        tokens = r["response"].get("usage", {}).get("input_tokens") or 0
        rows.append(
            {
                "run_id": rid,
                "catalog_id": r.get("catalog_id"),
                "questions": len(r["response"]["answers"]),
                "input_tokens": tokens,
                "ms": round(r["total_wall_ms"], 1),
                "ms_per_1k_tokens": (
                    round(1000 * r["total_wall_ms"] / tokens, 2) if tokens else None
                ),
            }
        )
    rows.sort(key=lambda x: -x["ms"])
    total_ms = sum(x["ms"] for x in rows)
    heavy = rows[: max(1, len(rows) // 20)]
    rates = sorted(x["ms_per_1k_tokens"] for x in rows if x["ms_per_1k_tokens"])
    return {
        "requests": len(rows),
        "slowest": rows[:top],
        "share_of_time_in_slowest_5pct": (
            round(sum(x["ms"] for x in heavy) / total_ms, 3) if total_ms else None
        ),
        "ms_per_1k_tokens": (
            {
                "median": round(statistics.median(rates), 2),
                "p95": round(percentile(rates, 95), 2),
            }
            if rates
            else None
        ),
        "tokens": {
            "median": statistics.median([x["input_tokens"] for x in rows]),
            "max": max(x["input_tokens"] for x in rows),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lat = sub.add_parser("latency")
    lat.add_argument("--results", required=True)
    lat.add_argument("--design", help="sample design JSON with warmup_run_ids")
    lat.add_argument("--gate-ms", type=float, default=GATE_MS)
    tl = sub.add_parser("tail")
    tl.add_argument("--results", required=True)
    tl.add_argument("--design", help="sample design JSON with warmup_run_ids")
    tl.add_argument("--top", type=int, default=15)
    ans = sub.add_parser("answers")
    ans.add_argument("--results", required=True)
    ans.add_argument("--reference", required=True)
    ans.add_argument("--near-tie", type=float, default=0.02)
    for p in (lat, tl, ans):
        p.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "latency":
        warm = (
            set(json.loads(Path(args.design).read_text())["warmup_run_ids"])
            if args.design
            else set()
        )
        out = latency(read_results(args.results), warm, args.gate_ms)
    elif args.cmd == "tail":
        warm = (
            set(json.loads(Path(args.design).read_text())["warmup_run_ids"])
            if args.design
            else set()
        )
        out = tail(read_results(args.results), warm, args.top)
    else:
        out = answers(
            read_results(args.results), read_results(args.reference), args.near_tie
        )
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
