"""Calibration v2 (pre-registered in ws-proxy STATUS.md, 2026-10-10 11:15). Pure Python (no numpy).

S part, fitted on the whole board (115 rows, per-benchmark public and private skills):
  S0  S = a + b public                                (fit on population rows, Full >= 40)
  SA  private_b = a_b + c_b pub_b                     (per benchmark, all other board rows)
  SA+ private_b = a_b + c_b pub_b + d_b public
  SAT SA+ with the pub_b term dropped for the in-distribution set T and the index recomputed without T
      (used for our arms; its error is measured by applying it to every population row with that T)
  S_hat = board S weights over the clipped per-benchmark predictions.
O part, on anchors (O_proxy): O1 O_proxy; O2 + public; O3 + (S_proxy - S_proxy_clean); O4 task-reweighted
  O_proxy (tasks with per-task Pearson >= 0.5 vs board O on the fold's anchors).
Every selection is nested inside an outer leave-one-out; margin = mean(d) + t(0.90, n-1) sd(d) sqrt(1+1/n)
on the anchors' nested-LOO over-predictions d of Full_hat.

    python -m d25.vega.eval.proxy.calibrate_v2 --anchors DIR --out calibration_pv2.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

from d25.vega.eval.proxy.common import AREA_WEIGHTS, AREAS, GOLD, s_weights

HERE = Path(__file__).parent
EQ = {
    "mB": 23.4973,
    "sB": 16.539081,
    "mS": 24.268626,
    "sS": 16.350451,
    "mO": 23.682619,
    "sO": 12.286109,
}
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
    30: 1.310,
    40: 1.303,
    60: 1.296,
}
OUR_T = [
    1,
    4,
    5,
    28,
    29,
    30,
    62,
    64,
]  # M2 in-distribution train splits: BFCL, BANKING77, CLINC150, WinoGrande, HellaSwag, GSM8K, When2Call, New Yorker
ANCHOR_BOARD = {
    "pplx": "pplx-decider-v1.1-27b",
    "torchcast": "torchcast-decision-27b",
    "kev": "kev-27b",
    "quyet": "quyet-1.0-large",
    "vega2": "decision-2.0-vega-27b",
    "jebadiah": "jebadiah-27b",
    "clef": "clef",
    "lux2": "decision-2.0-lux-9b",
    "nox2": "decision-2.0-nox-4b",
    "kev9": "kev-9b-v2",
    "jade": "jade",
    "blink": "blink-v0.3-26b-a4b-nvfp4",
    "rune": "rune-26b-a4b-v3",
    "deck31b": "deck31b",
    "decider-gemma4-31b": "decider-chat-gemma4-31b",
    "decider-qwen36-27b": "decider-chat-qwen3.6-27b",
    "eikos": "eikos-27b-fp8",
    "clef-flash": "clef-flash",
    "xor-26b": "xor-26b-a4b",
    "xor12": "xor-1.2",
    "decider-4b": "decider-4b",
    "decisio-qwen": "decisio-0.4.0-qwen3.6-35b-a3b",
    "decisio-gemma12": "decisio-0.4.0-gemma-4-12b",
    "ajev": "ajev-lora5",
    "jiwo-4b": "jiwo-4b",
}
FAMILY = {
    "pplx": "qwen3.8 full FT",
    "kev": "qwen3.8 full FT",
    "clef": "qwen3.8 full FT",
    "torchcast": "LoRA on full FT",
    "jebadiah": "qwen3.8 LoRA",
    "jade": "qwen3.8 LoRA",
    "vega2": "decision 2.0",
    "lux2": "decision 2.0",
    "nox2": "decision 2.0",
    "quyet": "gemma-4",
    "kev9": "qwen3.5-9B full FT",
    "blink": "gemma-4 MoE full FT",
    "rune": "gemma-4 MoE full FT",
    "deck31b": "gemma-4 stock + readout",
    "decider-gemma4-31b": "gemma-4 stock + readout",
    "decider-qwen36-27b": "qwen3.6 stock + readout",
    "eikos": "qwen3.8 LoRA",
    "clef-flash": "qwen3.5-9B full FT",
    "xor-26b": "gemma-4 MoE LoRA",
    "xor12": "qwen3.6 MoE LoRA",
    "decider-4b": "qwen3.5-4B full FT",
    "decisio-qwen": "qwen3.6 stock + readout",
    "decisio-gemma12": "gemma-4 stock + readout",
    "ajev": "gemma-4 LoRA",
    "jiwo-4b": "qwen3.5-4B full FT",
}
PRIVATE = sorted(s_weights())
INDEX_IDS = sorted({n for ids in AREAS.values() for n in ids})


def t90(df):
    return (
        T90.get(df) or T90[max(k for k in T90 if k <= df)] if df >= 1 else float("nan")
    )


def eq_s(s):
    return EQ["mB"] + EQ["sB"] * (s - EQ["mS"]) / EQ["sS"]


def eq_o(o):
    return EQ["mB"] + EQ["sB"] * (o - EQ["mO"]) / EQ["sO"]


def full(pub, s, o):
    return 0.2 * pub + 0.5 * eq_s(s) + 0.3 * eq_o(o)


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def ols(X, y, ridge=1e-9):
    """X: rows of covariates (no intercept). Returns [intercept, coefs...]."""
    Z = [[1.0] + list(x) for x in X]
    k = len(Z[0])
    A = [
        [
            sum(z[i] * z[j] for z in Z) + (ridge if i == j and i else 0.0)
            for j in range(k)
        ]
        for i in range(k)
    ]
    b = [sum(z[i] * yy for z, yy in zip(Z, y)) for i in range(k)]
    return solve(A, b)


def pred(c, x):
    return c[0] + sum(ci * xi for ci, xi in zip(c[1:], x))


def rmse(e):
    return math.sqrt(sum(x * x for x in e) / len(e)) if e else float("nan")


def pearson(a, b):
    try:
        return st.correlation(a, b)
    except Exception:  # noqa: BLE001
        return float("nan")


def spearman(a, b):
    ra = {i: r for r, i in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {i: r for r, i in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    return pearson([ra[i] for i in range(len(a))], [rb[i] for i in range(len(b))])


def index_without(pub_skill: dict, drop=()) -> float:
    tot, wsum = 0.0, 0.0
    for a, ids in AREAS.items():
        keep = [n for n in ids if n not in drop and pub_skill.get(n) is not None]
        if not keep:
            continue
        g = sum(GOLD.get(n, 1.0) for n in keep)
        tot += AREA_WEIGHTS[a] * sum(GOLD.get(n, 1.0) * pub_skill[n] for n in keep) / g
        wsum += AREA_WEIGHTS[a]
    return 100 * tot / wsum


# ------------------------------------------------------------------------------------------- board S maps


def load_board():
    data = json.loads((HERE / "board_v03.json").read_text())
    rows = []
    for m in data["models"]:
        if m["S"] is None or m["full"] is None:
            continue
        pub = {
            int(k): v["skill"]
            for k, v in m["benchmarks"].items()
            if v.get("skill") is not None
        }
        prv = {
            int(k): v["private_skill"]
            for k, v in m["benchmarks"].items()
            if v.get("private_skill") is not None
        }
        if len(pub) < 37 or len(prv) < 34:
            continue
        rows.append(
            {
                "engine": m["engine"],
                "public": m["public"],
                "S": m["S"],
                "O": m["O"],
                "full": m["full"],
                "pub": pub,
                "prv": prv,
            }
        )
    return rows


def fit_maps(train, kind, T=()):
    maps = {}
    for b in PRIVATE:
        X, y = [], []
        for r in train:
            if b not in r["prv"]:
                continue
            g = index_without(r["pub"], T) if T else r["public"]
            if kind == "SA":
                x = [r["pub"][b]]
            elif b in T:
                x = [g]
            else:
                x = [r["pub"][b], g]
            X.append(x)
            y.append(r["prv"][b])
        maps[b] = ols(X, y)
    return maps


def s_from_maps(maps, kind, pub: dict, public: float, T=()):
    w = s_weights()
    g = index_without(pub, T) if T else public
    num = 0.0
    for b in PRIVATE:
        x = [pub[b]] if kind == "SA" else [g] if b in T else [pub[b], g]
        num += w[b] * min(1.0, max(0.0, pred(maps[b], x)))
    return 100 * num / sum(w.values())


def s_predict(kind, train, row, T=(), pop=None):
    if kind == "S0":
        pop = [r for r in (pop or train) if r["full"] >= 40]
        c = ols([[r["public"]] for r in pop], [r["S"] for r in pop])
        return pred(c, [row["public"]])
    kk = "SA" if kind == "SA" else "SA+"
    return s_from_maps(fit_maps(train, kk, T), kk, row["pub"], row["public"], T)


def loo_s(board, kind, eval_rows, T=()):
    out = {}
    for r in eval_rows:
        train = [x for x in board if x["engine"] != r["engine"]]
        out[r["engine"]] = s_predict(kind, train, r, T) - r["S"]
    return out


def nested_s(board, cands, eval_rows, T=()):
    """Outer LOO over eval rows; inner selection by LOO RMSE over the other eval rows."""
    out, picks = {}, {}
    for r in eval_rows:
        rest = [x for x in board if x["engine"] != r["engine"]]
        inner_eval = [x for x in eval_rows if x["engine"] != r["engine"]]
        scores = {k: rmse(list(loo_s(rest, k, inner_eval, T).values())) for k in cands}
        best = min(scores, key=scores.get)
        picks[r["engine"]] = best
        out[r["engine"]] = s_predict(best, rest, r, T) - r["S"]
    return out, picks


# ------------------------------------------------------------------------------------------- O on anchors


def o_features(kind, a, tasks=None):
    if kind == "O1":
        return [a["O_proxy"]]
    if kind == "O2":
        return [a["O_proxy"], a["public"]]
    if kind == "O3":
        return [a["O_proxy"], a["S_proxy"] - a["S_proxy_clean"]]
    return [100 * st.mean(a["o_task"][t] for t in tasks)]


def o_select_tasks(train):
    keep = []
    for t in sorted(train[0]["o_task"]):
        r = pearson([a["o_task"][t] for a in train], [a["O"] for a in train])
        if r == r and r >= 0.5:
            keep.append(t)
    return keep or sorted(train[0]["o_task"])


def o_predict(kind, train, a):
    tasks = o_select_tasks(train) if kind == "O4" else None
    c = ols([o_features(kind, x, tasks) for x in train], [x["O"] for x in train])
    return pred(c, o_features(kind, a, tasks)), c, tasks


def loo_o(anchors, kind):
    return {
        a["name"]: o_predict(kind, [x for x in anchors if x["name"] != a["name"]], a)[0]
        - a["O"]
        for a in anchors
    }


def nested_o(anchors, cands):
    out, picks = {}, {}
    for a in anchors:
        rest = [x for x in anchors if x["name"] != a["name"]]
        scores = {k: rmse(list(loo_o(rest, k).values())) for k in cands}
        best = min(scores, key=scores.get)
        picks[a["name"]] = best
        out[a["name"]] = o_predict(best, rest, a)[0] - a["O"]
    return out, picks


# ------------------------------------------------------------------------------------------- main


def load_anchors(adir: Path, board):
    by = {r["engine"]: r for r in board}
    out = []
    for name, eng in ANCHOR_BOARD.items():
        f = adir / name / "pv1" / "scores.json"
        if not f.exists() or eng not in by:
            continue
        s = json.loads(f.read_text())
        if s["coverage"]["o_answered"] < s["coverage"]["o_questions"] * 0.99:
            continue
        b = by[eng]
        has_s = s["coverage"]["s_answered"] > 0.9 * s["coverage"]["s_requests"]
        out.append(
            {
                "name": name,
                "engine": eng,
                "family": FAMILY.get(name, "other"),
                "public": b["public"],
                "S": b["S"],
                "O": b["O"],
                "full": b["full"],
                "O_proxy": s["O_proxy"],
                "S_proxy": s["S_proxy"] if has_s else None,
                "S_proxy_clean": s["S_proxy_clean"] if has_s else None,
                "o_task": {t: v["skill"] for t, v in s["o"].items()},
                "row": b,
                "s_bench": (
                    {k: v["skill"] for k, v in s["s"].items()} if has_s else None
                ),
            }
        )
    return out


def margin(d):
    n = len(d)
    return (
        st.mean(d) + t90(n - 1) * st.stdev(d) * math.sqrt(1 + 1 / n)
        if n > 2
        else float("nan")
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="/data/d25/vega/proxy/anchors")
    ap.add_argument("--out", default=str(HERE / "calibration_pv2.json"))
    a = ap.parse_args(argv)
    board = load_board()
    pop = [r for r in board if r["full"] >= 40]
    anchors = load_anchors(Path(a.anchors), board)
    s_cands = ["S0", "SA", "SA+"]
    res = {
        "version": "pv2-calibration",
        "pre_registration": "ws-proxy STATUS.md 2026-10-10 11:15",
        "n_board": len(board),
        "n_population": len(pop),
        "anchors": [x["name"] for x in anchors],
        "our_T": OUR_T,
    }
    # S: population LOO per candidate, nested selection, SAT with our T
    s_loo = {k: loo_s(board, k, pop) for k in s_cands}
    res["S_population_loo_rmse"] = {k: rmse(list(v.values())) for k, v in s_loo.items()}
    s_nested, s_picks = nested_s(board, s_cands, pop)
    res["S_population_nested_rmse"] = rmse(list(s_nested.values()))
    res["S_nested_picks"] = dict(sorted(collections.Counter(s_picks.values()).items()))
    s_choice = min(res["S_population_loo_rmse"], key=res["S_population_loo_rmse"].get)
    res["S_choice"] = s_choice
    sat = (
        loo_s(board, "SA+", pop, T=OUR_T)
        if s_choice != "S0"
        else loo_s(board, "S0", pop)
    )
    res["SAT_population_loo_rmse"] = rmse(list(sat.values()))
    res["SAT_population_bias"] = st.mean(sat.values())
    res["S_anchor_loo"] = {
        k: {
            x["name"]: round(s_loo_k, 3)
            for x in anchors
            for s_loo_k in [loo_s(board, k, [x["row"]])[x["engine"]]]
        }
        for k in s_cands
    }
    # SB: S-proxy restricted to anchor-validated benchmarks (nested; anchors with S-proxy only), reported
    sa = [x for x in anchors if x.get("s_bench")]
    if len(sa) >= 5:
        w = s_weights()
        sb_err = {}
        for x in sa:
            rest = [y for y in sa if y["name"] != x["name"]]
            keep = [
                b
                for b in PRIVATE
                if str(b) in x["s_bench"]
                and pearson(
                    [y["s_bench"][str(b)] for y in rest],
                    [y["row"]["prv"][b] for y in rest],
                )
                >= 0.8
            ]
            sub = lambda y: (
                100
                * sum(w[b] * y["s_bench"][str(b)] for b in keep)
                / sum(w[b] for b in keep)
                if keep
                else y["S_proxy"]
            )
            for k, feats in (
                ("SB", lambda y: [sub(y)]),
                ("SB+pub", lambda y: [sub(y), y["public"]]),
            ):
                c = ols([feats(y) for y in rest], [y["S"] for y in rest])
                sb_err.setdefault(k, {})[x["name"]] = pred(c, feats(x)) - x["S"]
        res["SB_anchor_nested_rmse"] = {
            k: rmse(list(v.values())) for k, v in sb_err.items()
        }
        res["S0_SA+_anchor_rmse_same_anchors"] = {
            k: rmse(
                [
                    res_s
                    for res_s in [loo_s(board, k, [y["row"]])[y["engine"]] for y in sa]
                ]
            )
            for k in ("S0", "SA+")
        }
    # O: anchor candidates and nested selection
    o_cands = ["O1", "O2", "O4"] + (
        ["O3"] if all(x["S_proxy"] is not None for x in anchors) else []
    )
    res["O_anchor_loo_rmse"] = {
        k: rmse(list(loo_o(anchors, k).values())) for k in o_cands
    }
    o_nested, o_picks = nested_o(anchors, o_cands)
    res["O_nested_rmse"] = rmse(list(o_nested.values()))
    res["O_nested_picks"] = o_picks
    o_choice = min(res["O_anchor_loo_rmse"], key=res["O_anchor_loo_rmse"].get)
    res["O_choice"] = o_choice
    # Full on anchors: nested S (population selection without the anchor) + nested O
    d, rows = {}, []
    for x in anchors:
        rest_board = [r for r in board if r["engine"] != x["engine"]]
        rest_pop = [r for r in rest_board if r["full"] >= 40]
        inner = {
            k: rmse(list(loo_s(rest_board, k, rest_pop).values())) for k in s_cands
        }
        sk = min(inner, key=inner.get)
        s_hat = s_predict(sk, rest_board, x["row"])
        rest_a = [y for y in anchors if y["name"] != x["name"]]
        inner_o = {
            k: rmse(list(loo_o(rest_a, k).values()))
            for k in o_cands
            if k != "O3" or all(y["S_proxy"] is not None for y in rest_a)
        }
        ok = min(inner_o, key=inner_o.get)
        o_hat = o_predict(ok, rest_a, x)[0]
        f_hat = full(x["public"], s_hat, o_hat)
        d[x["name"]] = f_hat - x["full"]
        rows.append(
            {
                "anchor": x["name"],
                "family": x["family"],
                "full": x["full"],
                "full_hat": round(f_hat, 2),
                "d": round(f_hat - x["full"], 2),
                "S": x["S"],
                "S_hat": round(s_hat, 2),
                "S_model": sk,
                "O": x["O"],
                "O_hat": round(o_hat, 2),
                "O_model": ok,
            }
        )
    dv = list(d.values())
    res["full_anchor_nested"] = rows
    res["full_loo_rmse"] = rmse(dv)
    res["margin"] = margin(dv)
    res["margin_empirical_p90"] = sorted(dv)[
        min(len(dv) - 1, math.ceil(0.9 * len(dv)) - 1)
    ]
    sigma_s = res["S_population_nested_rmse"]
    so = [o_nested[x["name"]] for x in anchors]
    ss = [s_nested.get(x["engine"], 0.0) for x in anchors]
    rho = pearson(ss, so)
    cs, co = 0.5 * EQ["sB"] / EQ["sS"], 0.3 * EQ["sB"] / EQ["sO"]
    sig = math.sqrt(
        (cs * sigma_s) ** 2
        + (co * res["O_nested_rmse"]) ** 2
        + 2 * cs * co * (rho if rho == rho else 0) * sigma_s * res["O_nested_rmse"]
    )
    res["margin_decomposition"] = st.mean(dv) + t90(len(dv) - 1) * sig * math.sqrt(
        1 + 1 / len(dv)
    )
    res["decomposition"] = {
        "sigma_S_population": sigma_s,
        "sigma_O_anchors": res["O_nested_rmse"],
        "rho_anchor": rho,
        "sigma_full": sig,
    }
    res["spearman_full_hat_vs_full"] = spearman(
        [r["full_hat"] for r in rows], [r["full"] for r in rows]
    )
    res["residuals_by_family"] = {
        f: [r["d"] for r in rows if r["family"] == f]
        for f in sorted({r["family"] for r in rows})
    }
    # final maps on all data for gate.py
    res["final"] = {
        "S_model": s_choice,
        "S0": ols([[r["public"]] for r in pop], [r["S"] for r in pop]),
        "maps": {
            str(b): c
            for b, c in fit_maps(board, "SA" if s_choice == "SA" else "SA+").items()
        },
        "maps_T": {str(b): c for b, c in fit_maps(board, "SA+", OUR_T).items()},
        "O_model": o_choice,
    }
    res["anchor_points"] = [
        {
            "name": x["name"],
            "engine": x["engine"],
            "public": x["public"],
            "S": x["S"],
            "O": x["O"],
            "full": x["full"],
            "O_proxy": x["O_proxy"],
        }
        for x in anchors
    ]
    oc, coef, tasks = o_predict(o_choice, anchors, anchors[0])
    res["final"]["O_coef"] = coef
    res["final"]["O_tasks"] = tasks
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(
        json.dumps(
            {
                k: res[k]
                for k in (
                    "anchors",
                    "S_population_loo_rmse",
                    "S_population_nested_rmse",
                    "SAT_population_loo_rmse",
                    "SAT_population_bias",
                    "S_choice",
                    "O_anchor_loo_rmse",
                    "O_nested_rmse",
                    "O_choice",
                    "full_loo_rmse",
                    "margin",
                    "margin_empirical_p90",
                    "margin_decomposition",
                )
            },
            indent=1,
        )
    )
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
