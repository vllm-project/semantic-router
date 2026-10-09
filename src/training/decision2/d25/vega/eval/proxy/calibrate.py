"""Calibrate proxy scores against open board anchors (pre-registered in ws-proxy STATUS.md).

Models (OLS, equal anchor weights): S = a + b S_proxy (M_S1), S = a + b S_proxy + c public (M_S2),
O = a + b O_proxy (M_O1), O = a + b O_proxy + c public (M_O2); public-only baselines M_S0, M_O0 and
Full = a + b public (M_F0). Per part, M2 replaces M1 only if its LOO RMSE is >= 10% lower; the proxy model
is used only if its LOO RMSE beats the public-only baseline. Full_hat = 0.2 public + 0.5 eqS + 0.3 eqO with
both maps refitted without the held-out anchor. Over-prediction d_i = Full_hat(-i) - Full_i;
margin = max(mean(d) + t(0.90, n-1) sd(d) sqrt(1 + 1/n), empirical 90th percentile of d).

    python -m d25.vega.eval.proxy.calibrate --anchors /data/d25/vega/proxy/anchors --out calibration_pv1.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from d25.vega.eval.proxy.anchors import BOARD
from d25.vega.eval.proxy.harness import board_entry

EQ = {
    "mB": 23.4973,
    "sB": 16.539081,
    "mS": 24.268626,
    "sS": 16.350451,
    "mO": 23.682619,
    "sO": 12.286109,
}
T90 = {
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
}
FAMILY = {
    "pplx": "qwen3.8-fullft",
    "kev": "qwen3.8-fullft",
    "clef": "qwen3.8-fullft",
    "torchcast": "qwen3.8-lora-on-ft",
    "jebadiah": "qwen3.8-lora",
    "jade": "qwen3.8-lora",
    "eikos": "qwen3.8-lora",
    "vega2": "decision2.0",
    "lux2": "decision2.0",
    "nox2": "decision2.0",
    "quyet": "gemma4-31b",
    "deck31b": "gemma4-31b",
}


def eq_s(s):
    return EQ["mB"] + EQ["sB"] * (s - EQ["mS"]) / EQ["sS"]


def eq_o(o):
    return EQ["mB"] + EQ["sB"] * (o - EQ["mO"]) / EQ["sO"]


def full(public, s, o):
    return 0.2 * public + 0.5 * eq_s(s) + 0.3 * eq_o(o)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def predict(coef, X):
    return np.column_stack([np.ones(len(X)), X]) @ coef


def loo(X, y):
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        out[i] = predict(ols(X[m], y[m]), X[i : i + 1])[0]
    return out


def rmse(e):
    return float(math.sqrt(np.mean(np.square(e))))


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def margin_of(d):
    n = len(d)
    t = T90.get(n - 1, 1.282)
    para = (
        float(np.mean(d) + t * np.std(d, ddof=1) * math.sqrt(1 + 1 / n))
        if n > 2
        else float("nan")
    )
    emp = float(np.quantile(d, 0.9, method="higher"))
    return max(para, emp), para, emp


def load_anchor_scores(root: Path, admitted_only: bool = True):
    rows = []
    for d in sorted(root.iterdir()):
        sc, ck = d / "pv1" / "scores.json", d / "public-sample" / "check.json"
        if not sc.exists() or d.name not in BOARD:
            continue
        check = json.loads(ck.read_text()) if ck.exists() else {}
        if admitted_only and not check.get("admit", False):
            continue
        s = json.loads(sc.read_text())
        b = board_entry(BOARD[d.name])
        rows.append(
            {
                "anchor": d.name,
                "family": FAMILY.get(d.name, "other"),
                "public": b["public"],
                "S": b["S"],
                "O": b["O"],
                "full": b["full"],
                "S_proxy": s["S_proxy"],
                "S_proxy_clean": s["S_proxy_clean"],
                "S_proxy_no_apibank": s.get("S_proxy_no_apibank"),
                "O_proxy": s["O_proxy"],
                "O_families": s["O_families"],
                "unanswered": {
                    "s": s["coverage"]["s_requests"] - s["coverage"]["s_answered"],
                    "o": s["coverage"]["o_questions"] - s["coverage"]["o_answered"],
                },
                "s_bench": {k: v["skill"] for k, v in s["s"].items()},
                "o_task": {k: v["skill"] for k, v in s["o"].items()},
                "board_private": {
                    k: v.get("private_skill") for k, v in b["benchmarks"].items()
                },
                "check": {
                    k: check.get(k) for k in ("sample_index", "delta_index", "admit")
                },
            }
        )
    return rows


def calibrate(rows, s_key="S_proxy"):
    pub = np.array([r["public"] for r in rows])
    S = np.array([r["S"] for r in rows])
    O = np.array([r["O"] for r in rows])
    F = np.array([r["full"] for r in rows])
    sp = np.array([r[s_key] for r in rows])
    op = np.array([r["O_proxy"] for r in rows])
    cand = {
        "S1": (sp[:, None], S),
        "S2": (np.column_stack([sp, pub]), S),
        "S0": (pub[:, None], S),
        "O1": (op[:, None], O),
        "O2": (np.column_stack([op, pub]), O),
        "O0": (pub[:, None], O),
    }
    fits = {}
    for k, (X, y) in cand.items():
        pred = loo(X, y)
        fits[k] = {
            "coef": ols(X, y).tolist(),
            "loo_pred": pred.tolist(),
            "loo_rmse": rmse(pred - y),
            "r2": float(
                1
                - np.sum((predict(ols(X, y), X) - y) ** 2) / np.sum((y - y.mean()) ** 2)
            ),
        }
    choose = {}
    for part in ("S", "O"):
        m = (
            f"{part}2"
            if fits[f"{part}2"]["loo_rmse"] <= 0.9 * fits[f"{part}1"]["loo_rmse"]
            else f"{part}1"
        )
        if fits[m]["loo_rmse"] >= fits[f"{part}0"]["loo_rmse"]:
            m = f"{part}0"
        choose[part] = m
    s_hat = np.array(fits[choose["S"]]["loo_pred"])
    o_hat = np.array(fits[choose["O"]]["loo_pred"])
    f_hat = full(pub, s_hat, o_hat)
    d = f_hat - F
    f0 = loo(pub[:, None], F)
    margin, para, emp = margin_of(d)
    margin0, _, _ = margin_of(f0 - F)
    return {
        "n_anchors": len(rows),
        "s_key": s_key,
        "chosen": choose,
        "fits": {
            k: {kk: vv for kk, vv in v.items() if kk != "loo_pred"}
            for k, v in fits.items()
        },
        "full_hat_loo": {r["anchor"]: round(float(x), 3) for r, x in zip(rows, f_hat)},
        "over_prediction": {r["anchor"]: round(float(x), 3) for r, x in zip(rows, d)},
        "full_loo_rmse": rmse(d),
        "full_loo_mae": float(np.mean(np.abs(d))),
        "margin": margin,
        "margin_parametric": para,
        "margin_empirical": emp,
        "baseline_full_from_public": {
            "coef": ols(pub[:, None], F).tolist(),
            "loo_rmse": rmse(f0 - F),
            "margin": margin0,
        },
        "spearman": {
            "S_proxy_vs_S": spearman(sp, S),
            "O_proxy_vs_O": spearman(op, O),
            "Full_hat_vs_Full": spearman(f_hat, F),
            "public_vs_Full": spearman(pub, F),
        },
        "residuals_by_family": {
            fam: [round(float(x), 2) for r, x in zip(rows, d) if r["family"] == fam]
            for fam in sorted({r["family"] for r in rows})
        },
    }


def per_benchmark(rows):
    out = {}
    keys = sorted({k for r in rows for k in r["s_bench"]}, key=int)
    for k in keys:
        pairs = [
            (r["s_bench"][k], r["board_private"].get(k))
            for r in rows
            if r["board_private"].get(k) is not None and k in r["s_bench"]
        ]
        if (
            len(pairs) >= 4
            and statistics.pstdev([p for p, _ in pairs]) > 0
            and statistics.pstdev([q for _, q in pairs]) > 0
        ):
            a, b = np.array(pairs).T
            out[k] = {
                "pearson": round(float(np.corrcoef(a, b)[0, 1]), 3),
                "spearman": round(spearman(a, b), 3),
                "mean_proxy": round(100 * float(a.mean()), 1),
                "mean_private": round(100 * float(b.mean()), 1),
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="/data/d25/vega/proxy/anchors")
    ap.add_argument("--out", default="calibration_pv1.json")
    ap.add_argument("--include-unadmitted", action="store_true")
    a = ap.parse_args(argv)
    rows = load_anchor_scores(Path(a.anchors), admitted_only=not a.include_unadmitted)
    result = {
        "proxy": "pv1",
        "anchors": [
            {
                k: v
                for k, v in r.items()
                if k not in ("s_bench", "o_task", "board_private")
            }
            for r in rows
        ],
        "primary": calibrate(rows, "S_proxy"),
        "sensitivity_clean": calibrate(rows, "S_proxy_clean"),
        "sensitivity_no_apibank": calibrate(rows, "S_proxy_no_apibank"),
        "per_benchmark": per_benchmark(rows),
        "o_tasks": {
            t: [round(r["o_task"].get(t, float("nan")), 3) for r in rows]
            for t in sorted({t for r in rows for t in r["o_task"]})
        },
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    p = result["primary"]
    print(
        json.dumps(
            {
                "n": p["n_anchors"],
                "chosen": p["chosen"],
                "full_loo_rmse": p["full_loo_rmse"],
                "margin": p["margin"],
                "baseline_margin": p["baseline_full_from_public"]["margin"],
                "spearman": p["spearman"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
