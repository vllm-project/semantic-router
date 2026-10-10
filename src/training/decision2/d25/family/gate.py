"""Paired text-board estimate for a family model against a same-size reference.

    python -m d25.family.gate --result <ckpt_eval full result.json> --ref nox2 [--ref-o-proxy 50.05]

Same estimator as d3's paired gate (``d25.vega.eval.proxy.gate.paired``), with the reference chosen per size
instead of pplx-decider-v1.1-27b: estimate = official Full(ref) + Full_hat(ours) − Full_hat(ref), both
Full_hat values from Vega's calibration v2 (public per-benchmark skills + O_proxy), the reference's public
skills from the board and its O_proxy from the calibration anchors (or ``--ref-o-proxy`` for a reference
measured on the same frozen proxies outside the anchor set). SE = binomial measurement error of both models
+ jackknife of the calibration maps; ``anchor_errors`` shows how the same path predicts every other anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from d25.vega.eval.proxy import gate as G


def reference(name: str, cal: dict, o_proxy: float | None) -> dict:
    from d25.family.anchors import EXTRA

    anchor = next((a for a in cal["anchor_points"] if a["name"] == name), None)
    engine = anchor["engine"] if anchor else EXTRA[name][2] if name in EXTRA else name
    bench, model = G._board_bench(engine)
    full = anchor["full"] if anchor else model["full"]
    public = anchor["public"] if anchor else model["public"]
    o = o_proxy if o_proxy is not None else (anchor or {}).get("O_proxy")
    if o is None:
        raise SystemExit(f"{name}: no O_proxy in the anchors; pass --ref-o-proxy")
    return {
        "name": name,
        "engine": engine,
        "full": full,
        "public": public,
        "bench": bench,
        "O_proxy": o,
        "board": model,
    }


def paired(ours: dict, ref: dict, cal: dict, T=tuple(G.OUR_T), jackknife=True) -> dict:
    from d25.vega.eval.proxy.calibrate_v2 import fit_maps, load_board

    f_o, s_o, o_o = G._fhat(
        cal, ours["public"], ours["per_benchmark"], ours["O_proxy"], T
    )
    f_r, s_r, o_r = G._fhat(cal, ref["public"], ref["bench"], ref["O_proxy"], T)
    est = ref["full"] + f_o - f_r
    var_a = 0.0
    for bench, pub in (
        (ours["per_benchmark"], ours["public"]),
        (ref["bench"], ref["public"]),
    ):
        base = G._fhat(cal, pub, bench, 50.0, T)[0]
        for k, v in bench.items():
            x = v["skill"] if isinstance(v, dict) else v
            x = x / 100 if x is not None and x > 1.0001 else x
            n = max(
                1, ref["board"]["benchmarks"].get(str(k), {}).get("requests") or 100
            )
            se = math.sqrt(max(x * (1 - x), 0.01) / n)
            b2 = dict(bench)
            b2[k] = min(1.0, x + se)
            var_a += (G._fhat(cal, pub, b2, 50.0, T)[0] - base) ** 2
    var_a += 2 * (0.3 * G.EQ["sB"] / G.EQ["sO"] * cal["final"]["O_coef"][1] * 1.0) ** 2
    var_b = 0.0
    if jackknife:
        board = load_board()
        diffs = []
        for r in [r for r in board if r["full"] >= 40]:
            maps = {
                str(b): c
                for b, c in fit_maps(
                    [x for x in board if x["engine"] != r["engine"]], "SA+", T
                ).items()
            }
            diffs.append(
                G._fhat(cal, ours["public"], ours["per_benchmark"], 50.0, T, maps)[0]
                - G._fhat(cal, ref["public"], ref["bench"], 50.0, T, maps)[0]
            )
        n = len(diffs)
        var_b = (n - 1) / n * sum((d - statistics.mean(diffs)) ** 2 for d in diffs)
    se = math.sqrt(var_a + var_b)
    errs = {}
    for a in cal["anchor_points"]:
        if a["name"] == ref["name"]:
            continue
        ab, _ = G._board_bench(a["engine"])
        errs[a["name"]] = (
            ref["full"]
            + G._fhat(cal, a["public"], ab, a["O_proxy"], T)[0]
            - f_r
            - a["full"]
        )
    return {
        "reference": ref["name"],
        "estimate": round(est, 2),
        "Full_hat_ours": round(f_o, 2),
        "Full_hat_ref": round(f_r, 2),
        "S_hat": [round(s_o, 2), round(s_r, 2)],
        "O_hat": [round(o_o, 2), round(o_r, 2)],
        "paired_se": round(se, 2),
        "lower_1se": round(est - se, 2),
        "anchor_errors": {k: round(v, 2) for k, v in errs.items()},
        "anchor_rmse": round(
            math.sqrt(statistics.mean(e * e for e in errs.values())), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--result", required=True, help="ckpt_eval result.json with public + proxy"
    )
    parser.add_argument(
        "--ref",
        required=True,
        help="anchor name (lux2, nox2, kev9, pplx) or board engine id",
    )
    parser.add_argument("--ref-o-proxy", type=float)
    parser.add_argument("--calibration", default=str(G.DEFAULT))
    parser.add_argument("--out")
    args = parser.parse_args()
    cal = json.loads(Path(args.calibration).read_text())
    result = json.loads(Path(args.result).read_text())
    ours = {
        "public": result["public"]["index"],
        "per_benchmark": {
            k: v["skill"] for k, v in result["public"]["per_benchmark"].items()
        },
        "O_proxy": result["proxy"]["O_proxy"],
    }
    out = paired(ours, reference(args.ref, cal, args.ref_o_proxy), cal)
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
