"""Second-stage bias covariate X3 = top-1 agreement with pplx on the O-proxy (proxy-calibration.md §13).

python -m d25.vega.eval.proxy.agreement answers --proxy-build PV1 --out DIR NAME=SOURCE [NAME=SOURCE ...]
    SOURCE = a results file or glob (kit result records, e.g. results.shard*.jsonl or an HF anchor's
    results.jsonl), or probs:<path> for our checkpoints' probs.jsonl(.gz)
python -m d25.vega.eval.proxy.agreement fit --answers DIR --summary SUMMARY [--exclude a,b] --out OUT.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

from d25.vega.eval.proxy.bias_model import T90, T95, tq


def top1(q, res):
    if not res or res.get("status") != "ok":
        return None
    a = res["response"]["answers"]["q"]
    return (
        ("true" if a["noul"] >= 0.5 else "false")
        if q["type"] == "noul"
        else a["choice"]
    )


def cmd_answers(a):
    from d25.vega.eval.proxy.paired_boot import load_ours_proxy
    from d25.vega.eval.proxy.score import load_build, load_results

    _, o_rows = load_build(a.proxy_build)
    rows = [r for rr in o_rows.values() for r in rr]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.sources:
        name, src = spec.split("=", 1)
        if src.startswith("probs:"):
            res = load_ours_proxy(a.proxy_build, src[len("probs:") :])
        else:
            res = load_results(sorted(glob.glob(src)))
        ans = {
            r["_evaluation"]["run_id"]: top1(
                r["questions"]["q"], res.get(r["_evaluation"]["run_id"])
            )
            for r in rows
        }
        (out / f"{name}.json").write_text(json.dumps(ans))
        print(
            name,
            len(ans),
            "answered",
            sum(v is not None for v in ans.values()),
            flush=True,
        )


def ols1(x, y):
    mx, my = st.mean(x), st.mean(y)
    b = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sum((u - mx) ** 2 for u in x)
    return my - b * mx, b


def rmse(v):
    return math.sqrt(st.mean(t * t for t in v))


def cmd_fit(a):
    d = Path(a.answers)
    pp = json.loads((d / "pplx.json").read_text())
    summ = json.loads(Path(a.summary).read_text())
    excl = {x for x in a.exclude.split(",") if x}
    errs = {
        k: v
        for k, v in summ["anchor_errors_loo"]["conservative_T8"].items()
        if k not in excl
    }

    def agree(name):
        ans = json.loads((d / f"{name}.json").read_text())
        return 100 * st.mean(
            float(ans.get(k) is not None and ans.get(k) == v)
            for k, v in pp.items()
            if v is not None
        )

    x3 = {n: agree(n) for n in errs}
    d3 = agree("d3")
    names = sorted(errs)
    loo0 = [st.mean(errs[j] for j in names if j != i) - errs[i] for i in names]
    loo4 = []
    for i in names:
        rest = [j for j in names if j != i]
        c0, c3 = ols1([x3[j] for j in rest], [errs[j] for j in rest])
        loo4.append(c0 + c3 * x3[i] - errs[i])
    c0, c3 = ols1([x3[n] for n in names], [errs[n] for n in names])
    res = {
        "summary": Path(a.summary).name,
        "excluded": sorted(excl),
        "n": len(names),
        "X3": {n: round(x3[n], 2) for n in names},
        "d3_X3": round(d3, 2),
        "loo_rmse": {"M0": round(rmse(loo0), 3), "M4": round(rmse(loo4), 3)},
        "M4_coef": [round(c0, 4), round(c3, 4)],
        "corr_X3_e": round(
            st.correlation([x3[n] for n in names], [errs[n] for n in names]), 3
        ),
    }
    adopted = "M4" if res["loo_rmse"]["M4"] < res["loo_rmse"]["M0"] else "M0"
    bias = (c0 + c3 * d3) if adopted == "M4" else st.mean(errs.values())
    s, p = res["loo_rmse"][adopted], (2 if adopted == "M4" else 1)
    cons = summ["paired"]["all_tasks"]["conservative_T8"]
    e, se = cons["estimate"], cons["paired_se"]
    sig, df, adj = math.sqrt(se**2 + s**2), len(names) - p, e - bias
    res["adopted"] = adopted
    res["d3_bias_M4"] = round(c0 + c3 * d3, 3)
    res["d3"] = {
        "estimate": e,
        "paired_se": se,
        "bias": round(bias, 3),
        "adjusted_estimate": round(adj, 2),
        "sigma": round(sig, 3),
        "df": df,
        "interval80": [
            round(adj - tq(T90, df) * sig, 2),
            round(adj + tq(T90, df) * sig, 2),
        ],
        "interval90": [
            round(adj - tq(T95, df) * sig, 2),
            round(adj + tq(T95, df) * sig, 2),
        ],
        "p_above_62.75_normal_approx": round(
            0.5 * (1 + math.erf((adj - 62.75) / (sig * math.sqrt(2)))), 3
        ),
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("answers")
    p.add_argument("--proxy-build", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("sources", nargs="+")
    f = sub.add_parser("fit")
    f.add_argument("--answers", required=True)
    f.add_argument("--summary", required=True)
    f.add_argument("--exclude", default="")
    f.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cmd_answers(a) if a.cmd == "answers" else cmd_fit(a)


if __name__ == "__main__":
    main()
