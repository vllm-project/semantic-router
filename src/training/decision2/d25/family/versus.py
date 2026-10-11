"""Paired comparison of a family candidate with the released checkpoint of its size, on both boards.

    python -m d25.family.versus --cand-full C/full/step-005135 --ref-full R/full/step-005135 \
        --cand-vision C/vision/step-005135 --ref-vision R/vision/step-005135 --out versus.json

Text: delta Full_hat (candidate - release) on the paired-gate calibration path (conservative T8 and naive) from
both models' public kit results and O-proxy probabilities (``ckpt_eval --what full`` outputs). SE = paired
bootstrap: the same public case groups (within benchmark x track) and the same O-proxy question groups are
resampled for both models, as in ``d25.vega.eval.proxy.paired_arms``.

Vision: delta of the paired vision estimate of ``d25.omni.proxy.paired`` with the release as the reference
(public skills on the same rows; private sets from the proxies or the board's public-to-private slopes).
SE = paired row bootstrap + slope uncertainty. The missing-benchmark allowance of the board estimate is
reported separately and left out of the comparison SE: both checkpoints miss the same benchmark.

Verdict per board: ``better`` when delta - SE > 0, ``worse`` when delta + SE < 0, else ``within noise``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

VISION_BOARD = "/data/d25/omni/suite/vision-live-20261009T2353Z.json"
VISION_ROWS = "/data/d25/omni/suite/vision-0.3.1b/rows.jsonl.gz"
VISION_PROXIES = {
    "BLINK": ("blink-proxy-1500", "blink-proxy"),
    "Moderation (Hateful Memes)": ("moderation-proxy", "moderation-proxy"),
    "CV-Bench": ("cvbench-proxy", "cvbench-proxy"),
    "CharXiv": ("charxiv-proxy", "charxiv-proxy"),
    "InfographicVQA": ("infovqa-proxy", "infovqa-proxy"),
    "KIE (CORD+FUNSD)": ("kie-proxy", "kie-proxy"),
    "Mind2Web": ("mind2web-proxy", "mind2web-proxy"),
}


def verdict(delta: float, se: float) -> str:
    if delta - se > 0:
        return "better"
    if delta + se < 0:
        return "worse"
    return "within noise"


G: dict = {}


def _replicate(b: int) -> dict:
    from d25.vega.eval.proxy.paired_arms import o_from
    from d25.vega.eval.proxy.paired_boot import kit_score, resample

    rng = random.Random(G["seed"] * 7919 + b)
    sample, idmap = resample(G["strata"], rng)
    picks = {
        t: [rng.randrange(len(g)) for _ in g]
        for t, g in G["models"]["ref"]["tab"].items()
    }
    rep = {}
    for name, m in G["models"].items():
        mapped = {k: m["res"][v] for k, v in idmap.items() if v in m["res"]}
        idx, bench = kit_score(G["edition"], sample, mapped)
        rep[name] = (idx, bench, o_from(m["tab"], picks)[0])
    return rep


def text(a) -> dict:
    sys.path.insert(0, a.kit)
    from decision_index.scoring.report import load_results
    from decision_index.suite.io import Suite

    from d25.vega.eval.proxy.calibrate_v2 import OUR_T
    from d25.vega.eval.proxy.gate import DEFAULT, _fhat
    from d25.vega.eval.proxy.paired_arms import o_from, o_tables
    from d25.vega.eval.proxy.paired_boot import (
        kit_score,
        load_ours_proxy,
        sd,
        strata_of,
    )
    from d25.vega.eval.proxy.score import load_build

    cal = json.loads(Path(a.calibration or DEFAULT).read_text())
    _, build_rows = load_build(a.proxy_build)
    suite = Suite(a.suite, "0.3")
    rows = list(suite.rows(apply_exclusions=True))
    strata = strata_of(rows)
    models = {}
    for name, d in (("cand", Path(a.cand_full)), ("ref", Path(a.ref_full))):
        status = json.loads((d / "public" / "status.json").read_text())
        if status.get("event") != "complete":
            raise SystemExit(f"{d}: public run not complete")
        res = load_results(str(d / "public" / "results.jsonl"))
        tab = o_tables(
            build_rows,
            load_ours_proxy(a.proxy_build, d / "proxy" / "rows" / "probs.jsonl"),
        )
        idx, bench = kit_score(suite.edition, rows, res)
        models[name] = {
            "res": res,
            "tab": tab,
            "public": idx,
            "bench": bench,
            "O": o_from(tab)[0],
        }
    G.update(strata=strata, models=models, edition=suite.edition, seed=a.seed)
    with mp.get_context("fork").Pool(a.workers) as pool:
        reps = list(pool.imap_unordered(_replicate, range(a.B_public)))
    out = {
        "public": {k: round(m["public"], 3) for k, m in models.items()},
        "O_proxy": {k: round(m["O"], 3) for k, m in models.items()},
        "replicates": a.B_public,
    }
    for label, T in (("conservative_T8", tuple(OUR_T)), ("naive", ())):
        f = lambda p, b, o: _fhat(cal, p, b, o, T)[0]  # noqa: E731
        c, r = models["cand"], models["ref"]
        d0 = f(c["public"], c["bench"], c["O"]) - f(r["public"], r["bench"], r["O"])
        se = sd([f(*rep["cand"]) - f(*rep["ref"]) for rep in reps])
        out[label] = {
            "d_full": round(d0, 3),
            "se": round(se, 3),
            "verdict": verdict(d0, se),
        }
    return out


def vision(a) -> dict:
    from d25.omni.proxy.paired import estimate, per_row, slope, spread
    from d25.omni.suite import score

    board = json.loads(Path(a.vision_board).read_text())
    cand, ref = Path(a.cand_vision), Path(a.ref_vision)
    c_rows = per_row(a.vision_rows, str(cand / "public" / "results.jsonl"))
    r_rows = per_row(a.vision_rows, str(ref / "public" / "results.jsonl"))
    public = {b: (c_rows[b], r_rows[b]) for b in c_rows}
    proxies = {}
    for bench, (rows_dir, sub) in VISION_PROXIES.items():
        rows = f"{a.proxy_root}/{rows_dir}/rows.jsonl.gz"
        pair = [per_row(rows, str(d / sub / "results.jsonl")) for d in (cand, ref)]
        proxies[bench] = tuple([x for v in p.values() for x in v] for p in pair)
    point = estimate(public, proxies, board, None)
    rng = random.Random(a.seed)
    draws = [estimate(public, proxies, board, rng)["d_full"] for _ in range(a.B)]
    mean = sum(draws) / len(draws)
    se_boot = math.sqrt(sum((d - mean) ** 2 for d in draws) / (len(draws) - 1))
    var_slope = var_missing = 0.0
    for bench, cell in point["benchmarks"].items():
        if cell.get("private_from", "").startswith("public x"):
            _, beta_se = slope(board, bench)
            var_slope += (
                0.5 * beta_se * cell["d_public"] / len(score.PRIVATE_SETS)
            ) ** 2
        if "d_public" not in cell:
            var_missing += (
                0.5 * score.WEIGHTS[bench] * spread(board, bench, "pub") / 9.75
            ) ** 2
    se = math.sqrt(se_boot**2 + var_slope)
    return {
        "d_full": round(point["d_full"], 3),
        "d_public": round(point["d_public"], 3),
        "d_private": round(point["d_private"], 3),
        "se": round(se, 3),
        "se_parts": {
            "bootstrap": round(se_boot, 3),
            "slope": round(math.sqrt(var_slope), 3),
            "missing_not_included": round(math.sqrt(var_missing), 3),
        },
        "verdict": verdict(point["d_full"], se),
        "benchmarks": {
            b: {k: round(v, 4) if isinstance(v, float) else v for k, v in cell.items()}
            for b, cell in point["benchmarks"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cand-full", required=True)
    parser.add_argument("--ref-full", required=True)
    parser.add_argument("--cand-vision", required=True)
    parser.add_argument("--ref-vision", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--proxy-build", default="/data/d25/omni/family/proxy/pv1")
    parser.add_argument("--kit", default="/data/d25/shared/decision-index-kit")
    parser.add_argument("--suite", default="/data/d25/shared/index-suite-0.3")
    parser.add_argument("--calibration")
    parser.add_argument("--vision-board", default=VISION_BOARD)
    parser.add_argument("--vision-rows", default=VISION_ROWS)
    parser.add_argument("--proxy-root", default="/data/d25/omni/proxy")
    parser.add_argument("--B", type=int, default=1000)
    parser.add_argument("--B-public", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261011)
    a = parser.parse_args()
    out = {
        "candidate": {"full": a.cand_full, "vision": a.cand_vision},
        "release": {"full": a.ref_full, "vision": a.ref_vision},
        "vision": vision(a),
    }
    out["text"] = text(a)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                "text": out["text"]["conservative_T8"],
                "text_naive": out["text"]["naive"],
                "vision": {k: out["vision"][k] for k in ("d_full", "se", "verdict")},
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
