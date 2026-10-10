"""Paired scoring of training candidates against the locked release (directive v4 candidate swap rule).

Every finished ckpt_eval output on this node (``<results>/<arm>/<step>/proxy/scores.json``) is paired with the
release on the same resampled O-proxy question groups (within task) and, when both have public kit results on
this node, on the same resampled public case groups (linked cases kept whole, within benchmark x track).
Delta Full_hat (candidate - release) follows the paired-gate calibration path (conservative T8 and naive).
Proxy-only candidates are screened on the O part alone (public and S assumed equal to the release's), so a
screen pass only means "run --what full and re-score"; it is never a swap by itself.

Swap rule (directive v4): delta Full >= 0.5 beyond 1 SE (delta - SE >= 0.5) on a full result, and no vision
loss (checked by the Omni lead).

    python -m d25.vega.eval.proxy.paired_arms --proxy-build PV1 --ref-probs PROBS --results RESULTS --out DIR \
        [--ref-public RESULTS.jsonl --kit KIT --suite SUITE] [--loop-min 10 --hours 12]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import EQ, OUR_T
from d25.vega.eval.proxy.gate import DEFAULT, _fhat
from d25.vega.eval.proxy.paired_boot import (
    kit_score,
    load_ours_proxy,
    o_score,
    resample,
    sd,
    strata_of,
)
from d25.vega.eval.proxy.score import skill

SWAP = 0.5
REF = ("w1-t2-nc", "step-005135")
G: dict = {}


def row_hits(rows, res):
    """Per-row (hit, 1/k) exactly as score.score_o_task counts them."""
    out = []
    for r in rows:
        q, gold = r["questions"]["q"], r["expected"]["q"]
        k = 2 if q["type"] == "noul" else len(q["criteria"])
        x = res.get(r["_evaluation"]["run_id"], {})
        hit = 0.0
        if x.get("status") == "ok":
            a = x["response"]["answers"]["q"]
            pred = a["noul"] >= 0.5 if q["type"] == "noul" else a["choice"]
            hit = float(pred == gold)
        out.append((hit, 1 / k))
    return out


def o_tables(o_rows, res):
    """task -> list of groups -> (sum hits, sum 1/k, n rows)."""
    tab = {}
    for t, rr in sorted(o_rows.items()):
        groups = [g for _, gs in strata_of(rr) for g in gs]
        tab[t] = []
        for g in groups:
            h = row_hits(g, res)
            tab[t].append((sum(x for x, _ in h), sum(c for _, c in h), len(h)))
    return tab


def o_from(tab, picks=None):
    per = {}
    for t, groups in tab.items():
        idx = picks[t] if picks else range(len(groups))
        hs = sum(groups[i][0] for i in idx)
        cs = sum(groups[i][1] for i in idx)
        n = sum(groups[i][2] for i in idx)
        per[t] = skill(hs / n, cs / n)
    return 100 * statistics.mean(per.values()), per


def o_paired(tab_c, tab_r, B, seed):
    rng = random.Random(seed)
    d = []
    for _ in range(B):
        picks = {t: [rng.randrange(len(g)) for _ in g] for t, g in tab_r.items()}
        d.append(o_from(tab_c, picks)[0] - o_from(tab_r, picks)[0])
    return d


def _pub_rep(b):
    rng = random.Random(G["seed"] * 7919 + b)
    rows, idmap = resample(G["strata"], rng)
    out = {}
    for m in ("cand", "ref"):
        res = G["res"][m]
        idx, bench = kit_score(
            G["edition"], rows, {k: res[v] for k, v in idmap.items() if v in res}
        )
        out[m] = (idx, bench)
    return out


def score_candidate(a, d: Path, cal, build_rows, ref):
    from decision_index.scoring.report import load_results

    tab_c = o_tables(
        build_rows, load_ours_proxy(a.proxy_build, d / "proxy" / "rows" / "probs.jsonl")
    )
    o_c, per_c = o_from(tab_c)
    stored = json.loads((d / "proxy" / "scores.json").read_text())["O_proxy"]
    o_r = ref["O"]
    do = o_paired(tab_c, ref["tab"], a.B, a.seed)
    slope = cal["final"]["O_coef"][1]
    c_o = 0.3 * EQ["sB"] / EQ["sO"] * slope
    rec = {
        "dir": str(d),
        "O_proxy": round(o_c, 3),
        "O_check": (
            "ok" if abs(o_c - stored) < 1e-6 else f"mismatch vs stored {stored:.4f}"
        ),
        "ref_O_proxy": round(o_r, 3),
        "dO": round(o_c - o_r, 3),
        "dO_se": round(sd(do), 3),
        "dO_tasks": {t: round(100 * (per_c[t] - ref["per"][t]), 1) for t in per_c},
        "dFull_O_only": round(c_o * (o_c - o_r), 3),
        "dFull_O_only_se": round(c_o * sd(do), 3),
    }
    pub = d / "public" / "results.jsonl"
    status = d / "public" / "status.json"
    st = json.loads(status.read_text()) if status.exists() else {}
    complete = st.get("event") == "complete" or st.get("complete") is True
    if pub.exists() and complete and ref.get("pub"):
        res_c = load_results(str(pub))
        idx_c, bench_c = kit_score(G["edition"], G["rows"], res_c)
        idx_r, bench_r = ref["pub"]
        G["res"] = {"cand": res_c, "ref": ref["pub_res"]}
        import multiprocessing as mp

        with mp.get_context("fork").Pool(a.workers) as pool:
            reps = list(pool.imap_unordered(_pub_rep, range(a.B_public)))
        rng = random.Random(a.seed + 1)
        picks = [
            {t: [rng.randrange(len(g)) for _ in g] for t, g in ref["tab"].items()}
            for _ in reps
        ]
        for name, T in (("conservative_T8", tuple(OUR_T)), ("naive", ())):
            f = lambda pi, bi, oi: _fhat(cal, pi, bi, oi, T)[0]
            d0 = f(idx_c, bench_c, o_c) - f(idx_r, bench_r, o_r)
            ds = [
                f(r["cand"][0], r["cand"][1], o_from(tab_c, p)[0])
                - f(r["ref"][0], r["ref"][1], o_from(ref["tab"], p)[0])
                for r, p in zip(reps, picks)
            ]
            se = sd(ds)
            rec[name] = {
                "dFull": round(d0, 3),
                "se": round(se, 3),
                "lower_1se": round(d0 - se, 3),
                "swap_rule_pass": d0 - se >= SWAP,
            }
        rec["public"] = round(idx_c, 2)
        rec["ref_public"] = round(idx_r, 2)
        rec["d_public"] = round(idx_c - idx_r, 2)
        rec["verdict"] = (
            "SWAP-RULE PASS (check vision)"
            if rec["conservative_T8"]["swap_rule_pass"]
            else "no swap"
        )
    else:
        lo = rec["dFull_O_only"] - rec["dFull_O_only_se"]
        rec["verdict"] = (
            "O screen pass: run --what full and re-score"
            if lo >= SWAP
            else "no swap (O screen)"
        )
        if pub.exists() and complete:
            rec["public_unpaired"] = json.loads(
                (d / "public" / "index.json").read_text()
            ).get("index")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-build", required=True)
    ap.add_argument("--ref-probs", required=True)
    ap.add_argument("--ref-public", default="")
    ap.add_argument("--kit", default="/data/d25/shared/decision-index-kit")
    ap.add_argument("--suite", default="/data/d25/shared/index-suite-0.3")
    ap.add_argument("--results", default="/data/d25/vega/results")
    ap.add_argument("--calibration", default=str(DEFAULT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--B-public", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--loop-min", type=float, default=0)
    ap.add_argument("--hours", type=float, default=12)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.kit)
    from d25.vega.eval.proxy.score import load_build

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path(a.calibration).read_text())
    _, build_rows = load_build(a.proxy_build)
    tab_r = o_tables(build_rows, load_ours_proxy(a.proxy_build, a.ref_probs))
    o_r, per_r = o_from(tab_r)
    ref = {"tab": tab_r, "O": o_r, "per": per_r}
    print(f"release O_proxy {o_r:.4f}", flush=True)
    if a.ref_public:
        from decision_index.scoring.report import load_results
        from decision_index.suite.io import Suite

        suite = Suite(a.suite, "0.3")
        G["rows"] = list(suite.rows(apply_exclusions=True))
        G["edition"] = suite.edition
        G["strata"] = strata_of(G["rows"])
        G["seed"] = a.seed
        ref["pub_res"] = load_results(a.ref_public)
        ref["pub"] = kit_score(G["edition"], G["rows"], ref["pub_res"])
        print(f"release public {ref['pub'][0]}", flush=True)
    t_end = time.time() + 3600 * a.hours
    while True:
        for sc in sorted(Path(a.results).glob("*/*/proxy/scores.json")):
            d = sc.parent.parent
            arm, step = d.parent.name, d.name
            if (
                (arm, step) == REF
                or arm.startswith("v0-")
                or not (d / "proxy" / "rows" / "probs.jsonl").exists()
            ):
                continue
            f = out / f"{arm}__{step}.json"
            pub_done = (d / "public" / "status.json").exists()
            if f.exists():
                old = json.loads(f.read_text())
                if (
                    old.get("scored_mtime", 0) >= sc.stat().st_mtime
                    and old.get("had_public") == pub_done
                ):
                    continue
            try:
                rec = score_candidate(a, d, cal, build_rows, ref)
            except (
                Exception
            ) as e:  # keep the loop alive; the error is recorded for the successor
                rec = {"dir": str(d), "error": f"{type(e).__name__}: {e}"[:500]}
            rec.update(
                arm=arm,
                step=step,
                scored_mtime=sc.stat().st_mtime,
                had_public=pub_done,
                calibration=Path(a.calibration).name,
                scored=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            f.write_text(json.dumps(rec, indent=1))
            print(
                json.dumps(
                    {
                        k: rec.get(k)
                        for k in (
                            "arm",
                            "step",
                            "O_proxy",
                            "dO",
                            "dO_se",
                            "dFull_O_only",
                            "dFull_O_only_se",
                            "verdict",
                            "error",
                        )
                    }
                ),
                flush=True,
            )
        if not a.loop_min or time.time() > t_end:
            break
        time.sleep(60 * a.loop_min)


if __name__ == "__main__":
    main()
