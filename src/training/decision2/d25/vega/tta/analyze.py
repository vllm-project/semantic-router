"""Paired ON - OFF deltas of the d3 permutation-average arm, with paired bootstrap SEs, from tta_job outputs.

    python -m d25.vega.tta.analyze --runs <local copy of the work dataset's runs/> --proxy-build <pv1> \
        --public-rows latency-760.jsonl.gz parity-600.jsonl.gz [--vision-rows rows.jsonl.gz] --out measured.json

Both arms are scored by the same code on the same items, and every bootstrap replicate resamples the items once and
scores both arms on that resample:
  O-proxy   question groups within each task, ``score.score_o_task`` (top-1 accuracy, chance-corrected skill);
  S-proxy   case groups within benchmark x track, ``score.score_benchmark`` with proxy chance (its Monte Carlo F1
            chance fixed at the full-sample value) and the board's same-skill weights;
  public    case groups within benchmark x track of the stratified public sample (latency-760 + parity-600, the
            rows of index benchmarks), ``score.public_sample_index`` (kit metrics, kit chance, area weights);
  vision    rows of the vision proxy, accuracy over its questions.
Runs: ``tta-proxy-*/proxy``, ``tta-par/parity-600``, ``tta-lat/kit-760-{off,on}``, ``tta-vision*/vision``.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import statistics
import sys
import time
from pathlib import Path

from d25.vega.eval.proxy import score as S
from d25.vega.eval.proxy.common import AREAS, S_BENCHMARKS, read_jsonl
from d25.vega.tta.tta_run import read as read_rows

G: dict = {}


def load_arm(dirs: list[Path]) -> dict:
    out = {}
    for d in dirs:
        if d.exists():
            out.update(S.load_results([d]))
    for record in out.values():
        record.setdefault("total_wall_ms", 0.0)  # the kit's report scorer reads it
    return out


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def groups_by_stratum(rows: list[dict], key) -> list[tuple]:
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[key(r)][r["_evaluation"]["group_id"]].append(r)
    return [
        (k, list(v.values())) for k, v in sorted(by.items(), key=lambda kv: str(kv[0]))
    ]


def resample(strata: list[tuple], rng: random.Random) -> tuple[list[dict], dict]:
    rows, idmap = [], {}
    for _, groups in strata:
        seen = collections.Counter()
        for _ in range(len(groups)):
            gi = rng.randrange(len(groups))
            j = seen[gi]
            seen[gi] += 1
            for r in groups[gi]:
                e = dict(r["_evaluation"])
                rid = e["run_id"]
                if j:
                    e["run_id"], e["group_id"] = f"{rid}#{j}", f"{e['group_id']}#{j}"
                idmap[e["run_id"]] = rid
                rows.append({**r, "_evaluation": e})
    return rows, idmap


def remap(results: dict, idmap: dict) -> dict:
    return {new: results[old] for new, old in idmap.items() if old in results}


# ------------------------------------------------------------------ O-proxy


def o_hits(rows: list[dict], results: dict) -> list[float]:
    hits = []
    for r in rows:
        x = results.get(r["_evaluation"]["run_id"], {})
        if x.get("status") != "ok":
            hits.append(0.0)
            continue
        q, gold, a = (
            r["questions"]["q"],
            r["expected"]["q"],
            x["response"]["answers"]["q"],
        )
        hits.append(
            float((a["noul"] >= 0.5 if q["type"] == "noul" else a["choice"]) == gold)
        )
    return hits


def o_bootstrap(o_rows: dict, off: dict, on: dict, B: int, seed: int) -> dict:
    tasks = []
    for t, rows in sorted(o_rows.items()):
        index = collections.defaultdict(list)
        for i, r in enumerate(rows):
            index[r["_evaluation"]["group_id"]].append(i)
        inv = [
            1
            / (
                2
                if r["questions"]["q"]["type"] == "noul"
                else len(r["questions"]["q"]["criteria"])
            )
            for r in rows
        ]
        tasks.append(
            (t, list(index.values()), o_hits(rows, off), o_hits(rows, on), inv)
        )

    def index_of(sel):
        out = {"off": [], "on": []}
        for _, groups, h_off, h_on, inv in tasks:
            idx = [i for g in sel(groups) for i in g]
            c = statistics.fmean(inv[i] for i in idx)
            for arm, h in (("off", h_off), ("on", h_on)):
                acc = statistics.fmean(h[i] for i in idx)
                out[arm].append(S.skill(acc, c))
        return {arm: 100 * statistics.fmean(v) for arm, v in out.items()}, out

    point, per = index_of(lambda groups: groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(B):
        rep, _ = index_of(
            lambda groups: [groups[rng.randrange(len(groups))] for _ in groups]
        )
        deltas.append(rep["on"] - rep["off"])
    flips = {"to_right": 0, "to_wrong": 0}
    for _, _, h_off, h_on, _ in tasks:
        for a, b in zip(h_off, h_on):
            flips["to_right"] += a < b
            flips["to_wrong"] += a > b
    return {
        "off": round(point["off"], 3),
        "on": round(point["on"], 3),
        "delta": round(point["on"] - point["off"], 3),
        "se": round(sd(deltas), 3),
        "B": B,
        "questions": sum(len(t[2]) for t in tasks),
        "answered": {
            arm: sum(
                res.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"
                for rr in o_rows.values()
                for r in rr
            )
            for arm, res in (("off", off), ("on", on))
        },
        "flips": flips,
        "task_delta": {
            t[0]: round(100 * (b - a), 2)
            for t, a, b in zip(tasks, per["off"], per["on"])
        },
    }


# ------------------------------------------------------------------ S-proxy and public sample


def _memo_mc_f1():
    original = S._mc_f1
    cache = {}

    def mc_f1(rows, kind, positive=None):
        key = (
            rows[0]["_evaluation"]["catalog_id"],
            kind,
            positive,
            tuple(sorted({str(r["_evaluation"].get("track")) for r in rows})),
        )
        if key not in cache:
            cache[key] = original(rows, kind, positive)
        return cache[key]

    S._mc_f1 = mc_f1


def s_index(rows: list[dict], results: dict) -> float:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["_evaluation"]["catalog_id"]].append(r)
    per = {
        n: S.score_benchmark(n, rr, results, "proxy")
        for n, rr in sorted(by.items())
        if n in S_BENCHMARKS
    }
    return S.s_aggregate(per)


def p_index(rows: list[dict], results: dict) -> float:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["_evaluation"]["catalog_id"]].append(r)
    return S.public_sample_index(dict(by), results)["index"]


def _replicate(b: int) -> float:
    rows, idmap = resample(G["strata"], random.Random(G["seed"] * 100003 + b))
    f = G["index"]
    return f(rows, remap(G["on"], idmap)) - f(rows, remap(G["off"], idmap))


def paired(
    name: str,
    rows: list[dict],
    off: dict,
    on: dict,
    index,
    B: int,
    seed: int,
    workers: int,
) -> dict:
    t0 = time.time()
    point = {"off": index(rows, off), "on": index(rows, on)}
    strata = groups_by_stratum(
        rows,
        lambda r: (r["_evaluation"]["catalog_id"], str(r["_evaluation"].get("track"))),
    )
    G.update(strata=strata, seed=seed, index=index, off=off, on=on)
    with mp.get_context("fork").Pool(workers) as pool:
        deltas = pool.map(_replicate, range(B), chunksize=max(1, B // (4 * workers)))
    print(f"{name}: point {point} B {B} in {time.time() - t0:.0f}s", flush=True)
    return {
        "off": round(point["off"], 3),
        "on": round(point["on"], 3),
        "delta": round(point["on"] - point["off"], 3),
        "se": round(sd(deltas), 3),
        "B": B,
        "requests": len(rows),
        "answered": {
            arm: sum(
                res.get(r["_evaluation"]["run_id"], {}).get("status") == "ok"
                for r in rows
            )
            for arm, res in (("off", off), ("on", on))
        },
    }


def choice_flips(rows: list[dict], off: dict, on: dict) -> dict:
    changed = total = 0
    for r in rows:
        a, b = off.get(r["_evaluation"]["run_id"], {}), on.get(
            r["_evaluation"]["run_id"], {}
        )
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        for k, q in r["questions"].items():
            if q["type"] == "choice" and len(q["criteria"]) >= 2:
                total += 1
                changed += (
                    a["response"]["answers"][k]["choice"]
                    != b["response"]["answers"][k]["choice"]
                )
    return {"choice_questions": total, "argmax_changed": changed}


# ------------------------------------------------------------------ vision


def vision(rows: list[dict], off: dict, on: dict, B: int, seed: int) -> dict:
    def hits(results):
        out = []
        for r in rows:
            x = results.get(r["_evaluation"]["run_id"], {})
            for k, q in r["questions"].items():
                if x.get("status") != "ok":
                    out.append(0.0)
                    continue
                a = x["response"]["answers"][k]
                out.append(
                    float(
                        (a["noul"] >= 0.5 if q["type"] == "noul" else a["choice"])
                        == r["expected"][k]
                    )
                )
        return out

    h_off, h_on = hits(off), hits(on)
    rng = random.Random(seed)
    n = len(h_off)
    deltas = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(100 * statistics.fmean(h_on[i] - h_off[i] for i in idx))
    return {
        "off_accuracy": round(100 * statistics.fmean(h_off), 3),
        "on_accuracy": round(100 * statistics.fmean(h_on), 3),
        "delta": round(100 * (statistics.fmean(h_on) - statistics.fmean(h_off)), 3),
        "se": round(sd(deltas), 3),
        "B": B,
        "questions": n,
        **choice_flips(rows, off, on),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--proxy-build", required=True, type=Path)
    ap.add_argument("--public-rows", nargs="+", type=Path, default=[])
    ap.add_argument("--vision-rows", type=Path)
    ap.add_argument("--B-o", type=int, default=1000)
    ap.add_argument("--B-s", type=int, default=200)
    ap.add_argument("--B-p", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    _memo_mc_f1()
    report: dict = {"method": __doc__.split("\n\n")[1]}
    proxy_dirs = sorted(args.runs.glob("tta-proxy-*/proxy"))
    if proxy_dirs:
        off, on = (load_arm([d / arm for d in proxy_dirs]) for arm in ("off", "on"))
        s_rows, o_rows = S.load_build(args.proxy_build)
        report["dO_proxy"] = o_bootstrap(o_rows, off, on, args.B_o, args.seed)
        rows = [r for n, rr in s_rows.items() if n in S_BENCHMARKS for r in rr]
        report["dS_proxy"] = paired(
            "S-proxy", rows, off, on, s_index, args.B_s, args.seed + 1, args.workers
        )
        report["dS_proxy"].update(choice_flips(rows, off, on))
        report["proxy_shards"] = len(proxy_dirs)
    if args.public_rows:
        index_ids = {n for ids in AREAS.values() for n in ids}
        rows, seen = [], set()
        for path in args.public_rows:
            for r in read_jsonl(path):
                if (
                    r["_evaluation"]["catalog_id"] in index_ids
                    and r["_evaluation"]["run_id"] not in seen
                ):
                    seen.add(r["_evaluation"]["run_id"])
                    rows.append(r)
        dirs = {
            arm: [
                args.runs / "tta-par" / "parity-600" / arm,
                args.runs / "tta-lat" / f"kit-760-{arm}",
            ]
            for arm in ("off", "on")
        }
        off, on = load_arm(dirs["off"]), load_arm(dirs["on"])
        report["d_public_sample"] = paired(
            "public", rows, off, on, p_index, args.B_p, args.seed + 2, args.workers
        )
        report["d_public_sample"].update(choice_flips(rows, off, on))
    for arm in ("off", "on"):
        path = args.runs / "tta-lat" / f"latency-{arm}.json"
        if path.exists():
            report.setdefault("latency", {})[arm] = {
                k: v
                for k, v in json.loads(path.read_text()).items()
                if k
                in (
                    "timed_rows",
                    "ok",
                    "median_ms",
                    "mean_ms",
                    "p80_ms",
                    "p95_ms",
                    "max_ms",
                    "gate_pass",
                )
            }
    parity = args.runs / "tta-par" / "text-parity-rev.json"
    if parity.exists():
        report["parity_off_vs_v3.0.2"] = {
            k: v
            for k, v in json.loads(parity.read_text()).items()
            if k
            in (
                "requests",
                "questions",
                "identical_responses",
                "answer_changes",
                "max_abs_dp",
                "runtime_sha256",
                "pass",
            )
        }
    vision_dirs = sorted(args.runs.glob("tta-vision*/vision"))
    if vision_dirs and args.vision_rows:
        off, on = (load_arm([d / arm for d in vision_dirs]) for arm in ("off", "on"))
        report["vision"] = vision(
            read_rows(args.vision_rows), off, on, args.B_o, args.seed + 3
        )
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "method"}, indent=1)[:4000]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
