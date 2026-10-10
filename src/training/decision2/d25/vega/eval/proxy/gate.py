"""Release gate estimate: Full_hat from the local public index (+ per-benchmark skills) and proxy scores.

    python -m d25.vega.eval.proxy.gate --result /data/d25/vega/results/<arm>/<step>/result.json
    python -m d25.vega.eval.proxy.gate --public 62.1 --o-proxy 66.0 [--s-proxy 61 --s-proxy-clean 60]

Calibration v2 (``calibration_pv2.json``, see ``calibrate_v2.py``): S_hat from board-fitted per-benchmark
maps with the in-distribution set T of our arms excluded from the public evidence (SAT; without
per-benchmark input it falls back to S0 = a + b public); O_hat from the O-proxy map chosen on anchors.
Full_hat = 0.2 public + 0.5 eqS + 0.3 eqO. ``margin`` is the larger of the pre-registered anchor bound and
the bound with SAT's population error (our S path), so the decision flags are conservative.
The legacy v1 file (``calibration_pv1.json``) is still accepted (S0 / O map from ``fits``).
"""

from __future__ import annotations

import argparse
import statistics
import json
import math
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import (
    EQ,
    OUR_T,
    PRIVATE,
    eq_o,
    eq_s,
    full,
    index_without,
    pred,
    t90,
)

HERE = Path(__file__).parent
DEFAULT = (
    HERE / "calibration_pv2.json"
    if (HERE / "calibration_pv2.json").exists()
    else HERE / "calibration_pv1.json"
)


def board_rank(value: float) -> dict:
    data = json.loads((HERE / "board_v03.json").read_text())
    entries = {m["engine"]: m["full"] for m in data["models"] if m["full"] is not None}
    entries["__candidate__"] = value
    ordered = sorted(entries.items(), key=lambda kv: -kv[1])
    rank, start, ranks = 1, None, {}
    for i, (k, v) in enumerate(ordered):
        if start is None or start - v > data["tie_band"]:
            rank, start = i + 1, v
        ranks[k] = rank
    pos = [k for k, _ in ordered].index("__candidate__") + 1
    return {
        "position": pos,
        "tie_rank": ranks["__candidate__"],
        "nearest_above": [(k, round(v, 2)) for k, v in ordered[: pos - 1]][-3:],
    }


def _s_v2(final, public, bench, T):
    from d25.vega.eval.proxy.common import s_weights as weights

    w = weights()
    pub = {int(k): (v["skill"] if isinstance(v, dict) else v) for k, v in bench.items()}
    pub = {k: (x / 100 if x is not None and x > 1.0001 else x) for k, x in pub.items()}
    if any(pub.get(b) is None for b in PRIVATE) or len(pub) < 37:
        return None
    maps = final["maps_T"] if T else final["maps"]
    g = index_without(pub, T) if T else public
    num = sum(
        w[b] * min(1.0, max(0.0, pred(maps[str(b)], [g] if b in T else [pub[b], g])))
        for b in PRIVATE
    )
    return 100 * num / sum(w.values())


def gate(
    public,
    s_proxy,
    o_proxy,
    calibration=DEFAULT,
    s_proxy_clean=None,
    public_bench=None,
    T=tuple(OUR_T),
):
    cal = json.loads(Path(calibration).read_text())
    if "final" not in cal:  # legacy v1
        fit = cal["primary"]
        c = lambda m: fit["fits"][m]["coef"]
        s_hat = (
            c("S0")[0] + c("S0")[1] * public
            if fit["chosen"]["S"] == "S0"
            else c(fit["chosen"]["S"])[0] + c(fit["chosen"]["S"])[1] * s_proxy
        )
        o_hat = c(fit["chosen"]["O"])[0] + c(fit["chosen"]["O"])[1] * o_proxy
        margin, notes = fit["margin"], ["calibration v1"]
    else:
        fin = cal["final"]
        s_hat = _s_v2(fin, public, public_bench, tuple(T)) if public_bench else None
        notes = []
        if s_hat is None:
            s_hat = pred(fin["S0"], [public])
            notes.append("S from S0 (no complete per-benchmark public input)")
        om = fin["O_model"]
        x = {
            "O1": [o_proxy],
            "O2": [o_proxy, public],
            "O3": [o_proxy, (s_proxy or 0) - (s_proxy_clean or 0)],
        }.get(om)
        if x is None:
            raise ValueError(
                f"O model {om} needs per-task O skills; use calibrate_v2 output with O1/O2/O3"
            )
        o_hat = pred(fin["O_coef"], x)
        n = len(cal["anchors"])
        cs, co = 0.5 * EQ["sB"] / EQ["sS"], 0.3 * EQ["sB"] / EQ["sO"]
        d = cal["decomposition"]
        sig = math.sqrt(
            (cs * cal["SAT_population_loo_rmse"]) ** 2
            + (co * d["sigma_O_anchors"]) ** 2
            + 2
            * cs
            * co
            * (d["rho_anchor"] if d["rho_anchor"] == d["rho_anchor"] else 0)
            * cal["SAT_population_loo_rmse"]
            * d["sigma_O_anchors"]
        )
        bias = sum(r["d"] for r in cal["full_anchor_nested"]) / n
        margin_sat = bias + t90(n - 1) * sig * math.sqrt(1 + 1 / n)
        margin = max(cal["margin"], margin_sat)
        notes.append(
            f"margin = max(anchor bound {cal['margin']:.2f}, SAT bound {margin_sat:.2f})"
        )
    f_hat = full(public, s_hat, o_hat)
    board = json.loads((HERE / "board_v03.json").read_text())["models"]
    top = sorted((m["full"] for m in board if m["full"] is not None), reverse=True)
    models = {
        "S": (
            "S0"
            if any("S0" in n for n in notes) or "final" not in cal
            else ("SAT" if T else cal["final"]["S_model"])
        ),
        "O": (
            cal["final"]["O_model"] if "final" in cal else cal["primary"]["chosen"]["O"]
        ),
    }
    return {
        "inputs": {
            "public_local": public,
            "S_proxy": s_proxy,
            "S_proxy_clean": s_proxy_clean,
            "O_proxy": o_proxy,
            "per_benchmark": bool(public_bench),
            "T": list(T),
        },
        "models": models,
        "S_hat": round(s_hat, 2),
        "O_hat": round(o_hat, 2),
        "eqS": round(eq_s(s_hat), 2),
        "eqO": round(eq_o(o_hat), 2),
        "Full_hat": round(f_hat, 2),
        "margin": round(margin, 2),
        "Full_lower_90": round(f_hat - margin, 2),
        "board": board_rank(f_hat),
        "board_at_lower_bound": board_rank(f_hat - margin),
        "decisions": {
            f"top3_private_push (lower > #2 {top[1]:.2f})": f_hat - margin > top[1],
            f"number1_public (lower > #1 + 0.9 = {top[0] + 0.9:.2f})": f_hat - margin
            > top[0] + 0.9,
        },
        "calibration": {"file": str(calibration), "notes": notes},
    }


PPLX = {
    "full": 62.75,
    "public": 62.28,
    "engine": "pplx-decider-v1.1-27b",
}  # official Full; local kit 0.3 public (ws-eval)


def _board_bench(engine):
    m = next(
        x
        for x in json.loads((HERE / "board_v03.json").read_text())["models"]
        if x["engine"] == engine
    )
    return {k: v["skill"] for k, v in m["benchmarks"].items()}, m


def _fhat(cal, public, bench, o_proxy, T, maps=None, o_coef=None):
    fin = dict(cal["final"], **({"maps_T": maps} if maps else {}))
    s = _s_v2(fin, public, bench, tuple(T))
    o = pred(o_coef or fin["O_coef"], [o_proxy])
    return full(public, s, o), s, o


def paired(ours: dict, calibration=DEFAULT, T=tuple(OUR_T), live=None, jackknife=True):
    """ours = {public, per_benchmark, O_proxy, per_task?}. Pre-registered in ws-proxy STATUS.md (13:05)."""
    from d25.vega.eval.proxy.calibrate_v2 import fit_maps, load_board, ols

    cal = json.loads(Path(calibration).read_text())
    if cal["final"]["O_model"] != "O1":
        raise ValueError("paired gate v1 expects the O1 map")
    pb, pm = _board_bench(PPLX["engine"])
    o_pplx = next(a["O_proxy"] for a in cal["anchor_points"] if a["name"] == "pplx")
    f_o, s_o, oo = _fhat(cal, ours["public"], ours["per_benchmark"], ours["O_proxy"], T)
    f_p, s_p, op = _fhat(cal, PPLX["public"], pb, o_pplx, T)
    est = PPLX["full"] + f_o - f_p
    # (a) measurement: binomial SE per benchmark (board request counts) and per O task, both models, independent
    var_a = 0.0
    for bench, pub_i, model in (
        (ours["per_benchmark"], ours["public"], "ours"),
        (pb, PPLX["public"], "pplx"),
    ):
        base = _fhat(cal, pub_i, bench, 50.0, T)[0]
        for k, v in bench.items():
            x = v["skill"] if isinstance(v, dict) else v
            x = x / 100 if x is not None and x > 1.0001 else x
            n = max(1, pm["benchmarks"].get(str(k), {}).get("requests") or 100)
            se = math.sqrt(max(x * (1 - x), 0.01) / n)
            b2 = dict(bench)
            b2[k] = min(1.0, x + se)
            var_a += (_fhat(cal, pub_i, b2, 50.0, T)[0] - base) ** 2
    se_o = 1.0  # O_proxy points per model: sqrt(mean p(1-p)/n_t)/(1-1/k)/sqrt(22) ~ 1.0 (22 tasks x ~246 items)
    var_a += 2 * (0.3 * EQ["sB"] / EQ["sO"] * cal["final"]["O_coef"][1] * se_o) ** 2
    # (b) calibration: jackknife of the difference over refits
    var_b, notes = 0.0, []
    if jackknife:
        board = load_board()
        pop = [r for r in board if r["full"] >= 40]
        diffs = []
        for r in pop:
            maps = {
                str(b): c
                for b, c in fit_maps(
                    [x for x in board if x["engine"] != r["engine"]], "SA+", T
                ).items()
            }
            diffs.append(
                _fhat(cal, ours["public"], ours["per_benchmark"], 50.0, T, maps)[0]
                - _fhat(cal, PPLX["public"], pb, 50.0, T, maps)[0]
            )
        n = len(diffs)
        var_b += (n - 1) / n * sum((d - st_mean(diffs)) ** 2 for d in diffs)
        pts = cal["anchor_points"]
        od = []
        for a in pts:
            c = ols(
                [[p["O_proxy"]] for p in pts if p["name"] != a["name"]],
                [p["O"] for p in pts if p["name"] != a["name"]],
            )
            od.append(0.3 * EQ["sB"] / EQ["sO"] * c[1] * (ours["O_proxy"] - o_pplx))
        var_b += (len(od) - 1) / len(od) * sum((d - st_mean(od)) ** 2 for d in od)
    se = math.sqrt(var_a + var_b)
    # anchor dispersion of the paired estimator (same path for every anchor)
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
    s_anchor = statistics.stdev(errs.values())
    live = live or {}
    out = {
        "estimate": round(est, 2),
        "Full_hat_ours": round(f_o, 2),
        "Full_hat_pplx": round(f_p, 2),
        "S_hat": [round(s_o, 2), round(s_p, 2)],
        "O_hat": [round(oo, 2), round(op, 2)],
        "paired_se": round(se, 2),
        "se_measurement": round(math.sqrt(var_a), 2),
        "se_calibration": round(math.sqrt(var_b), 2),
        "anchor_errors": {k: round(v, 2) for k, v in errs.items()},
        "s_anchor": round(s_anchor, 2),
        "anchor_bias": round(st_mean(errs.values()), 2),
        "push_lower": round(est - se, 2),
        "flip_lower_without_A": round(
            est - t90(len(errs) - 1) * math.sqrt(se**2 + s_anchor**2), 2
        ),
    }
    if "third" in live:
        out["push"] = est - se > live["third"]
    out["flip"] = "not evaluated (allowance A pending)"
    return out


def st_mean(x):
    x = list(x)
    return sum(x) / len(x)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result",
        help="ckpt_eval result.json (public index, per-benchmark skills, proxy scores)",
    )
    ap.add_argument("--public", type=float)
    ap.add_argument("--s-proxy", type=float)
    ap.add_argument("--s-proxy-clean", type=float)
    ap.add_argument("--o-proxy", type=float)
    ap.add_argument("--calibration", default=str(DEFAULT))
    ap.add_argument(
        "--paired",
        action="store_true",
        help="paired comparison with pplx v1.1 (needs --result)",
    )
    ap.add_argument("--live-third", type=float, help="live board #3 Full (push rule)")
    a = ap.parse_args(argv)
    if a.paired:
        r = json.loads(Path(a.result).read_text())
        ours = {
            "public": r["public"]["index"],
            "per_benchmark": r["public"]["per_benchmark"],
            "O_proxy": r["proxy"]["O_proxy"],
        }
        print(
            json.dumps(
                paired(
                    ours,
                    a.calibration,
                    live={"third": a.live_third} if a.live_third else None,
                ),
                indent=1,
            )
        )
        return
    bench = None
    if a.result:
        r = json.loads(Path(a.result).read_text())
        pub = r.get("public") or {}
        a.public = a.public if a.public is not None else pub.get("index")
        bench = pub.get("per_benchmark")
        a.s_proxy, a.s_proxy_clean, a.o_proxy = (
            r["proxy"]["S_proxy"],
            r["proxy"].get("S_proxy_clean"),
            r["proxy"]["O_proxy"],
        )
    print(
        json.dumps(
            gate(a.public, a.s_proxy, a.o_proxy, a.calibration, a.s_proxy_clean, bench),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
