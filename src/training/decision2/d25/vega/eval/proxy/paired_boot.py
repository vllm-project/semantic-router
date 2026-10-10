"""Paired item bootstrap of the paired gate (ours vs pplx-decider-v1.1-27b) + calibration uncertainty.

Every replicate resamples, with replacement and identically for both models,
  public: case groups (``_evaluation.group_id``, i.e. linked cases stay together) within each
          (benchmark, track) stratum of the kit 0.3 suite, rescored with the kit's own 0.3 scorer;
  O-proxy: question groups within each O task (pv1 build), rescored with ``score.score_o_task``;
and draws one calibration refit: S maps refitted on a bootstrap of the board rows, O1 slope/intercept
refitted on a bootstrap of the anchors. Delta = Full_hat(ours) - Full_hat(pplx) per replicate is
computed with (i) the fixed pv2 calibration (item SE), (ii) the replicate's calibration draw (total SE),
and the point data with each calibration draw (calibration SE). Resumable: one file per replicate.

    python -m d25.vega.eval.proxy.paired_boot --kit KIT --suite SUITE --ours-public RESULTS \
        --pplx-public RESULTS --proxy-build PV1 --ours-proxy-probs PROBS --pplx-proxy-results R \
        --out DIR [--B 400 --workers 6]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import (
    OUR_T,
    PRIVATE,
    fit_maps,
    load_board,
    ols,
    t90,
)
from d25.vega.eval.proxy.common import read_jsonl
from d25.vega.eval.proxy.gate import DEFAULT, PPLX, _fhat

G: dict = {}


class _Rows:
    def __init__(self, edition, rows):
        self.edition, self._rows = edition, rows

    def rows(self, apply_exclusions=False):
        return iter(self._rows)


def kit_score(edition, rows, results):
    """Kit 0.3 scoring path (pipeline.score_run_v02 without file output): index + per-benchmark skill."""
    from decision_index.scoring import added, index02
    from decision_index.scoring.report import benchmark_summary

    spec = index02.spec(edition["id"])
    added_ids = {int(n) for n in spec["added"]}
    base, extra = [], collections.defaultdict(list)
    for r in rows:
        n = r["_evaluation"]["catalog_id"]
        (extra[n] if n in added_ids else base).append(r)
    metrics = {
        int(n): (m["name"], m["key"]) for n, m in spec.get("metrics", {}).items()
    }
    suite = _Rows(edition, rows)
    summary = benchmark_summary(
        suite, results, "boot", None, rows=base, metrics=metrics
    )
    reports = {n: added.report(n, rr, results) for n, rr in sorted(extra.items())}
    index = index02.index_entry(suite, results, summary, reports, spec)
    bench = {k: v["skill"] for k, v in index["benchmarks"].items() if v.get("in_index")}
    return index["index"], bench


def o_score(o_rows, results):
    from d25.vega.eval.proxy.score import score_o_task

    per = {t: score_o_task(rr, results)["skill"] for t, rr in o_rows.items()}
    return 100 * statistics.mean(per.values()), per


def load_ours_proxy(build, probs_path):
    from d25.vega.common import decision_format as df

    probs = {r["id"]: r for r in read_jsonl(probs_path)}
    out = {}
    for f in sorted((Path(build) / "o").glob("*.jsonl.gz")):
        for row in read_jsonl(f):
            rid = row["_evaluation"]["run_id"]
            answers, status = {}, "ok"
            for k, q in row["questions"].items():
                p = probs.get(f"{rid}\t{k}")
                if p is None or p["status"] != "ok":
                    status = "unsupported" if p is not None else "error"
                    break
                answers[k] = df.to_answer(q, p["probs"])
            out[rid] = {
                "status": status,
                **({"response": {"answers": answers}} if status == "ok" else {}),
            }
    return out


def resample(strata, rng):
    """strata: list of (key, [group rows lists]). Returns new rows with suffixed ids and the id map."""
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
                    e["run_id"] = f"{rid}#{j}"
                    e["group_id"] = f"{e['group_id']}#{j}"
                idmap[e["run_id"]] = rid
                rows.append({**r, "_evaluation": e})
    return rows, idmap


def strata_of(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        e = r["_evaluation"]
        by[e["catalog_id"]][e["group_id"]].append(r)
    out = collections.defaultdict(list)
    for n, groups in by.items():
        for g, rr in groups.items():
            out[(n, rr[0]["_evaluation"]["track"])].append(rr)
    return sorted(out.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))


def replicate(b):
    f = G["out"] / "boot" / f"b{b:04d}.json"
    if f.exists():
        return b
    rng = random.Random(G["seed"] * 100003 + b)
    rows, idmap = resample(G["pub_strata"], rng)
    rec = {"b": b}
    for m in ("ours", "pplx"):
        res = G["pub_res"][m]
        mapped = {k: res[v] for k, v in idmap.items() if v in res}
        idx, bench = kit_score(G["edition"], rows, mapped)
        rec[m] = {"public": idx, "bench": bench}
    orows = {}
    oid = {}
    for t, groups in G["o_strata"]:
        sub = collections.Counter()
        new = []
        for _ in range(len(groups)):
            gi = rng.randrange(len(groups))
            j = sub[gi]
            sub[gi] += 1
            for r in groups[gi]:
                e = dict(r["_evaluation"])
                if j:
                    e["run_id"] = f"{e['run_id']}#{j}"
                oid[e["run_id"]] = r["_evaluation"]["run_id"]
                new.append({**r, "_evaluation": e})
        orows[t] = new
    for m in ("ours", "pplx"):
        res = G["o_res"][m]
        mapped = {k: res[v] for k, v in oid.items() if v in res}
        rec[m]["O_proxy"], rec[m]["O_tasks"] = o_score(orows, mapped)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.rename(f)
    return b


def cal_draws(cal, B, seed):
    board = load_board()
    pts = cal["anchor_points"]
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        rows = [board[rng.randrange(len(board))] for _ in board]
        a = [pts[rng.randrange(len(pts))] for _ in pts]
        while len({p["name"] for p in a}) < 3:
            a = [pts[rng.randrange(len(pts))] for _ in pts]
        draws.append(
            {
                "maps_T": {
                    str(k): v for k, v in fit_maps(rows, "SA+", tuple(OUR_T)).items()
                },
                "maps": {str(k): v for k, v in fit_maps(rows, "SA+", ()).items()},
                "O_coef": ols([[p["O_proxy"]] for p in a], [p["O"] for p in a]),
            }
        )
    return draws


def delta(cal, ours, pplx, T, draw=None):
    maps = None
    o_coef = None
    if draw:
        maps = draw["maps_T"] if T else draw["maps"]
        o_coef = draw["O_coef"]
    calx = cal
    if draw and not T:
        calx = dict(cal, final=dict(cal["final"], maps=maps))
        maps = None
    fo = _fhat(calx, ours["public"], ours["bench"], ours["O_proxy"], T, maps, o_coef)[0]
    fp = _fhat(calx, pplx["public"], pplx["bench"], pplx["O_proxy"], T, maps, o_coef)[0]
    return fo - fp


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else float("nan")


# O-proxy tasks whose source family is covered by a corpus that is in our M2T-v5 mixture but not in pplx v1.1's
# data manifest (both share the same tasksource list). Rationale per task in vega/evals/proxy-calibration.md §9.
SUSPECT = {
    "primary": (
        "atis",  # intents: CLINC150, BANKING77 (indist), Decision 1.0 intents (A7)
        "climate_fever",  # claim verification: HoVer (IB2)
        "halueval_qa",  # faithfulness: SQuAD2-faithfulness (IB4), quote check (HS1)
        "summedits",
        "felm",
        "rewardbench",  # response quality judging: human ordinal Score (A6g/A6h, A7q OASST1)
        "mtbench_human",
        "include",  # knowledge MCQ: aqua_rat, ARC, CommonsenseQA, BoolQ (knowledge part)
        "truthfulqa_mc1",
        "popqa_mc",
        "popqa_verify",
    ),
}
SUSPECT["broad"] = SUSPECT["primary"] + (
    "sib200",  # multilingual Score/Noul (A7q/k/s/r, v1 ko/ja)
    "belebele",
    "legal_citizenship",  # policy packets / unmet condition (HS1), verifiable rules (v1)
    "legal_issue",
    "legal_rules",
    "legal_mc",
    "newsgroups",  # topic classification (IB1 snips/sms, A7 typed)
)


def with_o_subset(rec, drop):
    """Pair with our O advantage replaced by the mean advantage on the tasks not in ``drop``."""
    if not drop:
        return rec["ours"], rec["pplx"]
    o, p = rec["ours"]["O_tasks"], rec["pplx"]["O_tasks"]
    keep = [t for t in o if t not in drop]
    adv = 100 * statistics.mean(o[t] - p[t] for t in keep)
    return dict(rec["ours"], O_proxy=rec["pplx"]["O_proxy"] + adv), rec["pplx"]


def allowance(cal, rec, tasks):
    """Pre-registered A = O1 slope x 0.3 x sB/sO x sum over suspect tasks of max(advantage, 0) / n_tasks."""
    from d25.vega.eval.proxy.calibrate_v2 import EQ

    o, p = rec["ours"]["O_tasks"], rec["pplx"]["O_tasks"]
    adv = sum(max(0.0, 100 * (o[t] - p[t])) for t in tasks)
    return cal["final"]["O_coef"][1] * 0.3 * EQ["sB"] / EQ["sO"] * adv / len(o)


def summarize(
    cal, point, recs, draws, s_anchor, anchor_bias, n_err, drop=(), A=0.0, s_loo=None
):
    out = {}
    pp = dict(zip(("ours", "pplx"), with_o_subset(point, drop)))
    for name, T in (("conservative_T8", tuple(OUR_T)), ("naive", ())):
        d0 = delta(cal, pp["ours"], pp["pplx"], T)
        est = PPLX["full"] + d0
        pairs = [(r["b"], with_o_subset(r, drop)) for r in recs]
        item = [delta(cal, o, p, T) for _, (o, p) in pairs]
        tot = [delta(cal, o, p, T, draws[b % len(draws)]) for b, (o, p) in pairs]
        calo = [delta(cal, pp["ours"], pp["pplx"], T, dr) for dr in draws]
        se = sd(tot)
        q = sorted(tot)
        k = t90(n_err - 1)
        out[name] = {
            "estimate": round(est, 2),
            "delta_point": round(d0, 3),
            "paired_se": round(se, 3),
            "se_item": round(sd(item), 3),
            "se_calibration": round(sd(calo), 3),
            "boot_mean_shift": round(statistics.mean(tot) - d0, 3),
            "boot_p05_estimate": round(PPLX["full"] + q[int(0.05 * (len(q) - 1))], 2),
            "push_lower": round(est - se, 2),
            "s_anchor": s_anchor[name],
            "anchor_bias": anchor_bias[name],
            "t90_df": n_err - 1,
            "flip_lower_without_A": round(
                est - k * math.sqrt(se**2 + s_anchor[name] ** 2), 2
            ),
        }
        if s_loo:
            out[name]["s_anchor_loo"] = s_loo[name]
            out[name]["flip_lower_without_A_loo"] = round(
                est - k * math.sqrt(se**2 + s_loo[name] ** 2), 2
            )
        if A:
            out[name]["A"] = round(A, 3)
            out[name]["flip_lower_with_A"] = round(
                out[name]["flip_lower_without_A"] - A, 2
            )
            if s_loo:
                out[name]["flip_lower_with_A_loo"] = round(
                    out[name]["flip_lower_without_A_loo"] - A, 2
                )
    return out


def anchor_dispersion(cal, pplx_point, T):
    from d25.vega.eval.proxy.gate import _board_bench

    pb, _ = _board_bench(PPLX["engine"])
    o_pplx = next(a["O_proxy"] for a in cal["anchor_points"] if a["name"] == "pplx")
    f_p = _fhat(cal, PPLX["public"], pb, o_pplx, T)[0]
    errs = {}
    for a in cal["anchor_points"]:
        if a["name"] == "pplx":
            continue
        ab, _ = _board_bench(a["engine"])
        errs[a["name"]] = (
            PPLX["full"]
            + _fhat(cal, a["public"], ab, a["O_proxy"], T)[0]
            - f_p
            - a["full"]
        )
    return errs


def anchor_dispersion_loo(cal, T):
    """Paired errors with the anchor left out of both calibration parts (its board row for the S maps, its point
    for O1); pplx stays in every fit because it is the reference of the paired estimate.
    """
    from d25.vega.eval.proxy.gate import _board_bench

    board = load_board()
    pb, _ = _board_bench(PPLX["engine"])
    pts = cal["anchor_points"]
    o_pplx = next(a["O_proxy"] for a in pts if a["name"] == "pplx")
    errs = {}
    for a in pts:
        if a["name"] == "pplx":
            continue
        rows = [r for r in board if r["engine"] != a["engine"]]
        rest = [p for p in pts if p["name"] != a["name"]]
        final = dict(
            cal["final"],
            maps={str(k): v for k, v in fit_maps(rows, "SA+", ()).items()},
            maps_T={str(k): v for k, v in fit_maps(rows, "SA+", tuple(OUR_T)).items()},
            O_coef=ols([[p["O_proxy"]] for p in rest], [p["O"] for p in rest]),
        )
        cal_i = dict(cal, final=final)
        ab, _ = _board_bench(a["engine"])
        f_p = _fhat(cal_i, PPLX["public"], pb, o_pplx, T)[0]
        errs[a["name"]] = (
            PPLX["full"]
            + _fhat(cal_i, a["public"], ab, a["O_proxy"], T)[0]
            - f_p
            - a["full"]
        )
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--ours-public", required=True)
    ap.add_argument("--pplx-public", required=True)
    ap.add_argument("--proxy-build", required=True)
    ap.add_argument("--ours-proxy-probs", required=True)
    ap.add_argument("--pplx-proxy-results", required=True)
    ap.add_argument("--calibration", default=str(DEFAULT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument(
        "--tag",
        default="",
        help="suffix of cal_draws/summary files (e.g. another calibration)",
    )
    a = ap.parse_args(argv)
    sys.path.insert(0, a.kit)
    from decision_index.scoring.report import load_results
    from decision_index.suite.io import Suite

    out = Path(a.out)
    (out / "boot").mkdir(parents=True, exist_ok=True)
    cal = json.loads(Path(a.calibration).read_text())
    if cal["final"]["O_model"] != "O1":
        raise SystemExit(
            "the paired gate is pre-registered with the O1 map; pass an O1 calibration"
        )
    t0 = time.time()
    suite = Suite(a.suite, "0.3")
    rows = list(suite.rows(apply_exclusions=True))
    pub_res = {"ours": load_results(a.ours_public), "pplx": load_results(a.pplx_public)}
    from d25.vega.eval.proxy.score import load_build, load_results as proxy_results

    _, o_rows = load_build(a.proxy_build)
    o_res = {
        "ours": load_ours_proxy(a.proxy_build, a.ours_proxy_probs),
        "pplx": proxy_results([a.pplx_proxy_results]),
    }
    print(f"loaded {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
    pfile = out / "point.json"
    if pfile.exists():
        point = json.loads(pfile.read_text())
    else:
        point = {}
        for m in ("ours", "pplx"):
            t1 = time.time()
            idx, bench = kit_score(suite.edition, rows, pub_res[m])
            op, ot = o_score(o_rows, o_res[m])
            point[m] = {"public": idx, "bench": bench, "O_proxy": op, "O_tasks": ot}
            print(
                f"point {m}: public {idx} O_proxy {op:.3f} ({time.time() - t1:.0f}s)",
                flush=True,
            )
        pfile.write_text(json.dumps(point, indent=1))
    dfile = out / f"cal_draws{a.tag}.json"
    if dfile.exists():
        draws = json.loads(dfile.read_text())
    else:
        draws = cal_draws(cal, a.B, a.seed + 7)
        dfile.write_text(json.dumps(draws))
    if not a.summarize_only:
        G.update(
            out=out,
            seed=a.seed,
            edition=suite.edition,
            pub_strata=strata_of(rows),
            o_strata=[
                (t, [g for _, gs in strata_of(rr) for g in gs])
                for t, rr in sorted(o_rows.items())
            ],
            pub_res=pub_res,
            o_res=o_res,
        )
        import multiprocessing as mp

        todo = [b for b in range(a.B) if not (out / "boot" / f"b{b:04d}.json").exists()]
        print(f"{len(todo)} replicates to run", flush=True)
        with mp.get_context("fork").Pool(a.workers) as pool:
            for i, b in enumerate(pool.imap_unordered(replicate, todo)):
                if i % 10 == 0:
                    print(
                        f"done {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )
    recs = [json.loads(p.read_text()) for p in sorted((out / "boot").glob("b*.json"))]
    for r in recs:
        for m in ("ours", "pplx"):
            r[m]["bench"] = {k: v for k, v in r[m]["bench"].items()}
    errs = {
        n: anchor_dispersion(cal, point["pplx"], T)
        for n, T in (("conservative_T8", tuple(OUR_T)), ("naive", ()))
    }
    s_anchor = {n: round(sd(list(e.values())), 3) for n, e in errs.items()}
    bias = {n: round(statistics.mean(e.values()), 3) for n, e in errs.items()}
    n_err = len(errs["naive"])
    errs_loo = {
        n: anchor_dispersion_loo(cal, T)
        for n, T in (("conservative_T8", tuple(OUR_T)), ("naive", ()))
    }
    s_loo = {n: round(sd(list(e.values())), 3) for n, e in errs_loo.items()}
    kw = dict(s_loo=s_loo)
    summ = {
        "all_tasks": summarize(cal, point, recs, draws, s_anchor, bias, n_err, **kw)
    }
    for s, tasks in SUSPECT.items():
        A = allowance(cal, point, tasks)
        summ[f"with_A_{s}"] = summarize(
            cal, point, recs, draws, s_anchor, bias, n_err, A=A, **kw
        )
        summ[f"excluding_{s}"] = summarize(
            cal, point, recs, draws, s_anchor, bias, n_err, drop=tasks, **kw
        )
    summ["o_task_advantage"] = {
        t: round(100 * (point["ours"]["O_tasks"][t] - point["pplx"]["O_tasks"][t]), 2)
        for t in sorted(point["ours"]["O_tasks"])
    }
    o_diff = [r["ours"]["O_proxy"] - r["pplx"]["O_proxy"] for r in recs]
    p_diff = [r["ours"]["public"] - r["pplx"]["public"] for r in recs]
    res = {
        "B": len(recs),
        "calibration_draws": len(draws),
        "point": {m: {k: point[m][k] for k in ("public", "O_proxy")} for m in point},
        "d_public": {
            "point": round(point["ours"]["public"] - point["pplx"]["public"], 3),
            "se": round(sd(p_diff), 3),
        },
        "d_O_proxy": {
            "point": round(point["ours"]["O_proxy"] - point["pplx"]["O_proxy"], 3),
            "se": round(sd(o_diff), 3),
        },
        "anchor_errors": {
            n: {k: round(v, 2) for k, v in e.items()} for n, e in errs.items()
        },
        "anchor_errors_loo": {
            n: {k: round(v, 2) for k, v in e.items()} for n, e in errs_loo.items()
        },
        "anchor_bias_loo": {
            n: round(statistics.mean(e.values()), 3) for n, e in errs_loo.items()
        },
        "paired": summ,
    }
    res["calibration"] = a.calibration
    (out / f"summary{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
