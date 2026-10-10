"""Covariate model of the paired estimator's bias (pre-registered in vega/evals/proxy-calibration.md §11).

Target: the anchors' leave-one-out paired errors e_i (conservative T8 path) from a paired_boot summary. Covariates:
X1 "T8 excess" = 100 x mean over T8 of pub_b - (alpha_b + beta_b g), g = public index without T8, maps fitted on
the board population (Full >= 40); X2 "O-proxy excess" = O_proxy - (a + b public) fitted on the anchors. Models
M0 (constant), M1 (X1), M2 (X2), M3 (X1 + X2); leave-one-anchor-out with every covariate fit redone without the
held-out anchor; nested selection; a covariate model is adopted only if its nested LOO RMSE beats M0's LOO RMSE.

    python -m d25.vega.eval.proxy.bias_model --calibration CAL --summary SUMMARY --point POINT [--exclude a,b] --out OUT
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import (
    OUR_T,
    index_without,
    load_board,
    ols,
    pred,
    rmse,
)

T8 = tuple(OUR_T)
MODELS = {"M0": (), "M1": ("X1",), "M2": ("X2",), "M3": ("X1", "X2")}
T90 = {
    1: 3.078,
    2: 1.886,
    3: 1.638,
    4: 1.533,
    5: 1.476,
    6: 1.440,
    7: 1.415,
    8: 1.397,
    9: 1.383,
    10: 1.372,
    11: 1.363,
    12: 1.356,
    13: 1.350,
    14: 1.345,
    15: 1.341,
    16: 1.337,
    17: 1.333,
    18: 1.330,
    19: 1.328,
    20: 1.325,
    21: 1.323,
    22: 1.321,
    23: 1.319,
    24: 1.318,
    25: 1.316,
    26: 1.315,
    27: 1.314,
    28: 1.313,
    29: 1.311,
    30: 1.310,
}
T95 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    21: 1.721,
    22: 1.717,
    23: 1.714,
    24: 1.711,
    25: 1.708,
    26: 1.706,
    27: 1.703,
    28: 1.701,
    29: 1.699,
    30: 1.697,
}


def tq(table, df):
    return table[min(max(df, 1), 30)]


def norm_bench(bench):
    out = {}
    for k, v in bench.items():
        x = v["skill"] if isinstance(v, dict) else v
        if x is not None:
            out[int(k)] = x / 100 if x > 1.0001 else x
    return out


class Covariates:
    def __init__(self, board, anchors, d3):
        self.board = {r["engine"]: r for r in board}
        self.pop = [r for r in board if r["full"] >= 40]
        self.anchors = anchors  # list of anchor_points (incl. pplx)
        self.d3 = d3
        self.cache = {}

    def fits(self, excluded: frozenset):
        if excluded not in self.cache:
            eng = {a["engine"] for a in self.anchors if a["name"] in excluded}
            rows = [r for r in self.pop if r["engine"] not in eng]
            maps = {}
            for b in T8:
                xs = [r for r in rows if b in r["pub"]]
                maps[b] = ols(
                    [[index_without(r["pub"], T8)] for r in xs],
                    [r["pub"][b] for r in xs],
                )
            fa = [a for a in self.anchors if a["name"] not in excluded]
            o = ols([[a["public"]] for a in fa], [a["O_proxy"] for a in fa])
            self.cache[excluded] = (maps, o)
        return self.cache[excluded]

    def x(self, name, excluded: frozenset):
        maps, o = self.fits(excluded)
        if name == "d3":
            pub, public, o_proxy = (
                self.d3["bench"],
                self.d3["public"],
                self.d3["O_proxy"],
            )
        else:
            a = next(a for a in self.anchors if a["name"] == name)
            pub, public, o_proxy = (
                self.board[a["engine"]]["pub"],
                a["public"],
                a["O_proxy"],
            )
        g = index_without(pub, T8)
        x1 = 100 * st.mean(pub[b] - pred(maps[b], [g]) for b in T8)
        x2 = o_proxy - pred(o, [public])
        return {"X1": x1, "X2": x2}


def fit_predict(model, train, test_x):
    feats = MODELS[model]
    if not feats:
        return st.mean(e for _, e in train)
    c = ols([[x[f] for f in feats] for x, _ in train], [e for _, e in train])
    return pred(c, [test_x[f] for f in feats])


def loo(cov, names, errs, model, outer=frozenset()):
    """LOO errors of one model over ``names`` (all excluding ``outer``); covariate fits exclude outer + held-out."""
    out = {}
    for i in names:
        ex = outer | {i}
        train = [(cov.x(j, ex), errs[j]) for j in names if j != i]
        out[i] = fit_predict(model, train, cov.x(i, ex)) - errs[i]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--point", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cal = json.loads(Path(a.calibration).read_text())
    summ = json.loads(Path(a.summary).read_text())
    point = json.loads(Path(a.point).read_text())["ours"]
    excl = {x for x in a.exclude.split(",") if x}
    anchors = [p for p in cal["anchor_points"] if p["name"] not in excl]
    errs = {
        k: v
        for k, v in summ["anchor_errors_loo"]["conservative_T8"].items()
        if k not in excl
    }
    names = sorted(errs)
    d3 = {
        "bench": norm_bench(point["bench"]),
        "public": point["public"],
        "O_proxy": point["O_proxy"],
    }
    cov = Covariates(load_board(), anchors, d3)
    res = {
        "calibration": Path(a.calibration).name,
        "summary": Path(a.summary).name,
        "excluded": sorted(excl),
        "n": len(names),
    }
    full_x = {n: cov.x(n, frozenset()) for n in names}
    res["covariates"] = {
        n: {k: round(v, 3) for k, v in full_x[n].items()} | {"e": errs[n]}
        for n in names
    }
    d3x = cov.x("d3", frozenset())
    res["d3_covariates"] = {k: round(v, 3) for k, v in d3x.items()}
    loo_err = {m: loo(cov, names, errs, m) for m in MODELS}
    res["loo_rmse"] = {m: round(rmse(list(v.values())), 3) for m, v in loo_err.items()}
    # nested: selection inside each outer fold
    nested, picks = {}, {}
    for i in names:
        rest = [j for j in names if j != i]
        inner = {
            m: rmse(list(loo(cov, rest, errs, m, outer=frozenset({i})).values()))
            for m in MODELS
        }
        best = min(inner, key=inner.get)
        picks[i] = best
        train = [(cov.x(j, frozenset({i})), errs[j]) for j in rest]
        nested[i] = fit_predict(best, train, cov.x(i, frozenset({i}))) - errs[i]
    res["nested_rmse"] = round(rmse(list(nested.values())), 3)
    res["nested_picks"] = picks
    selected = min(res["loo_rmse"], key=res["loo_rmse"].get)
    adopted = (
        selected
        if (selected != "M0" and res["nested_rmse"] < res["loo_rmse"]["M0"])
        else "M0"
    )
    res["selected_full_sample"] = selected
    res["adopted"] = adopted
    s = res["nested_rmse"] if adopted != "M0" else res["loo_rmse"]["M0"]
    train_all = [(full_x[n], errs[n]) for n in names]
    res["coefficients"] = {
        m: (
            [round(st.mean(errs.values()), 4)]
            if not MODELS[m]
            else [
                round(c, 4)
                for c in ols(
                    [[x[f] for f in MODELS[m]] for x, _ in train_all],
                    [e for _, e in train_all],
                )
            ]
        )
        for m in MODELS
    }
    res["d3_bias_by_model"] = {
        m: round(fit_predict(m, train_all, d3x), 3) for m in MODELS
    }
    cons = summ["paired"]["all_tasks"]["conservative_T8"]
    e, se = cons["estimate"], cons["paired_se"]
    bias = res["d3_bias_by_model"][adopted]
    p = 1 + len(MODELS[adopted])
    df = len(names) - p
    sig = math.sqrt(se**2 + s**2)
    adj = e - bias
    A = summ["paired"].get("with_A_primary", {}).get("conservative_T8", {}).get("A")
    res["d3"] = {
        "estimate": e,
        "paired_se": se,
        "adopted_model": adopted,
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
        "one_sided90_lower": round(adj - tq(T90, df) * sig, 2),
        "A_primary": A,
        "one_sided90_lower_minus_A": round(adj - tq(T90, df) * sig - (A or 0), 2),
        "p_above_62.75_normal_approx": round(
            0.5 * (1 + math.erf((adj - 62.75) / (sig * math.sqrt(2)))), 3
        ),
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "n",
                    "excluded",
                    "d3_covariates",
                    "loo_rmse",
                    "nested_rmse",
                    "selected_full_sample",
                    "adopted",
                    "coefficients",
                    "d3_bias_by_model",
                    "d3",
                )
            },
            indent=1,
        )
    )
    for n in names:
        print(n, res["covariates"][n])


if __name__ == "__main__":
    main()
