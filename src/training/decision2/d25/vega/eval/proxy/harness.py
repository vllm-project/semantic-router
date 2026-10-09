"""Anchor harness checks against the board's public per-benchmark skills.

    python -m d25.vega.eval.proxy.harness sample --out /data/d25/vega/proxy/public-sample/sample.jsonl.gz
    python -m d25.vega.eval.proxy.harness check --anchor pplx --results RUN_DIR [--out check.json]

Sample: 40 complete cases per public index benchmark (salted hash order, seed 20261009), kit edition 0.3
rows with exclusions and subsets applied. Check: kit-chance skills on the sample vs the board; admission
rule pre-registered in ws-proxy STATUS.md (sample index within +-3.0 points; no benchmark off by > 25).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from d25.vega.eval.proxy.common import AREAS, rank, write_jsonl

PER_BENCH = 40
BOARD = Path(__file__).with_name("board_v03.json")


def board_entry(engine_id: str) -> dict:
    data = json.loads(BOARD.read_text())
    return next(m for m in data["models"] if m["engine"] == engine_id)


def build_sample(out: str, suite_dir: str = "/data/d25/shared/index-suite-0.3"):
    from decision_index.suite.io import Suite

    index_ids = {n for ids in AREAS.values() for n in ids}
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in Suite(suite_dir, "0.3").rows(apply_exclusions=True):
        n = r["_evaluation"]["catalog_id"]
        if n in index_ids:
            groups[n][r["_evaluation"]["group_id"]].append(r)
    rows = []
    for n, g in sorted(groups.items()):
        keep = sorted(g, key=lambda gid: rank(f"public-sample:{n}", gid))[:PER_BENCH]
        for gid in keep:
            rows += g[gid]
    write_jsonl(out, rows)
    print(json.dumps({"rows": len(rows), "benchmarks": len(groups)}))


def check(anchor: str, results_dir: str, sample: str, out: str | None):
    from d25.vega.eval.proxy.anchors import BOARD as IDS
    from d25.vega.eval.proxy.common import read_jsonl
    from d25.vega.eval.proxy.score import load_results, public_sample_index

    rows = collections.defaultdict(list)
    for r in read_jsonl(sample):
        rows[r["_evaluation"]["catalog_id"]].append(r)
    res = load_results([results_dir])
    got = public_sample_index(rows, res)
    board = board_entry(IDS.get(anchor, anchor))
    diffs, limits = {}, {}
    for n, v in got["benchmarks"].items():
        bb = board["benchmarks"].get(str(n), {})
        b = bb.get("skill")
        if b is None:
            continue
        diffs[n] = round(100 * (v["skill"] - b), 1)
        # noise allowance (amended 00:50, see STATUS.md): 3 sampling SEs of the skill on the sampled cases
        p, c = min(max(bb.get("raw") or 0.5, 0.05), 0.95), v.get("chance") or 0.0
        se = (p * (1 - p) / max(1, v["cases"])) ** 0.5 / max(1e-6, 1 - c)
        limits[n] = round(max(25.0, 300 * se), 1)
    big = {n: d for n, d in diffs.items() if abs(d) > limits[n]}
    report = {
        "anchor": anchor,
        "board_engine": board["engine"],
        "sample_index": round(got["index"], 2),
        "board_public": board["public"],
        "delta_index": round(got["index"] - board["public"], 2),
        "benchmark_deltas": diffs,
        "gross_limits": limits,
        "gross": big,
        "unanswered": {
            n: v["requests"] - v["answered"]
            for n, v in got["benchmarks"].items()
            if v["requests"] != v["answered"]
        },
        "admit": abs(got["index"] - board["public"]) <= 3.0 and not big,
    }
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument(
        "--out", default="/data/d25/vega/proxy/public-sample/sample.jsonl.gz"
    )
    c = sub.add_parser("check")
    c.add_argument("--anchor", required=True)
    c.add_argument("--results", required=True)
    c.add_argument(
        "--sample", default="/data/d25/vega/proxy/public-sample/sample.jsonl.gz"
    )
    c.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        build_sample(a.out)
    else:
        check(a.anchor, a.results, a.sample, a.out)


if __name__ == "__main__":
    main()
