"""Release gate estimate: Full_hat from the local public index and calibrated proxy scores.

    python -m d25.vega.eval.proxy.gate --public 62.1 --scores RUN/pv1/scores.json
    python -m d25.vega.eval.proxy.gate --public 62.1 --s-proxy 58.3 --o-proxy 61.0

Inputs: the locally measured public index (ws-eval, exact kit reproduction) and either the proxy
``scores.json`` written by ``score``/``run_proxy`` or the two proxy aggregates. The maps, margin and
fallbacks come from the calibration file (``calibration_pv1.json`` next to this module by default).
Output: S_hat, O_hat, their equated values, Full_hat, the margin, and the rank Full_hat would take on the
board snapshot (0.9-point tie bands as the board computes them), plus the top-3 / #1 decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
EQ = {
    "mB": 23.4973,
    "sB": 16.539081,
    "mS": 24.268626,
    "sS": 16.350451,
    "mO": 23.682619,
    "sO": 12.286109,
}


def eq_s(s):
    return EQ["mB"] + EQ["sB"] * (s - EQ["mS"]) / EQ["sS"]


def eq_o(o):
    return EQ["mB"] + EQ["sB"] * (o - EQ["mO"]) / EQ["sO"]


def full(public, s, o):
    return 0.2 * public + 0.5 * eq_s(s) + 0.3 * eq_o(o)


def apply(fit: dict, model: str, proxy: float, public: float) -> float:
    c = fit["fits"][model]["coef"]
    if model.endswith("2"):
        return c[0] + c[1] * proxy + c[2] * public
    if model.endswith("0"):
        return c[0] + c[1] * public
    return c[0] + c[1] * proxy


def board_rank(value: float, exclude: str | None = None) -> dict:
    data = json.loads((HERE / "board_v03.json").read_text())
    entries = {
        m["engine"]: m["full"]
        for m in data["models"]
        if m["full"] is not None and m["engine"] != exclude
    }
    entries["__candidate__"] = value
    ordered = sorted(entries.items(), key=lambda kv: -kv[1])
    rank, start_value, out = 1, None, {}
    for i, (k, v) in enumerate(ordered):
        if start_value is None or start_value - v > data["tie_band"]:
            rank, start_value = i + 1, v
        out[k] = rank
    pos = [k for k, _ in ordered].index("__candidate__") + 1
    above = [(k, round(v, 2)) for k, v in ordered[: pos - 1]][-3:]
    return {"position": pos, "tie_rank": out["__candidate__"], "nearest_above": above}


def gate(
    public: float,
    s_proxy: float,
    o_proxy: float,
    calibration: Path,
    s_proxy_clean: float | None = None,
) -> dict:
    cal = json.loads(calibration.read_text())
    fit = cal["primary"]
    s_hat = apply(fit, fit["chosen"]["S"], s_proxy, public)
    o_hat = apply(fit, fit["chosen"]["O"], o_proxy, public)
    f_hat = full(public, s_hat, o_hat)
    margin = fit["margin"]
    board = json.loads((HERE / "board_v03.json").read_text())["models"]
    top = sorted((m["full"] for m in board if m["full"] is not None), reverse=True)
    out = {
        "inputs": {"public_local": public, "S_proxy": s_proxy, "O_proxy": o_proxy},
        "models": fit["chosen"],
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
            "top3_private_push (Full_hat - margin > #2 = %.2f)" % top[1]: f_hat - margin
            > top[1],
            "number1_public (Full_hat - margin > #1 + 0.9 = %.2f)"
            % (top[0] + 0.9): f_hat
            - margin
            > top[0] + 0.9,
        },
        "calibration": {
            "file": str(calibration),
            "n_anchors": fit["n_anchors"],
            "full_loo_rmse": round(fit["full_loo_rmse"], 2),
        },
    }
    if s_proxy_clean is not None and "sensitivity_clean" in cal:
        sc = cal["sensitivity_clean"]
        s2 = apply(sc, sc["chosen"]["S"], s_proxy_clean, public)
        o2 = apply(sc, sc["chosen"]["O"], o_proxy, public)
        out["sensitivity_clean"] = {
            "S_hat": round(s2, 2),
            "Full_hat": round(full(public, s2, o2), 2),
            "margin": round(sc["margin"], 2),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--public",
        type=float,
        required=True,
        help="local public index (kit, edition 0.3)",
    )
    ap.add_argument("--scores", help="scores.json from score / run_proxy")
    ap.add_argument("--s-proxy", type=float)
    ap.add_argument("--o-proxy", type=float)
    ap.add_argument("--calibration", default=str(HERE / "calibration_pv1.json"))
    a = ap.parse_args(argv)
    clean = None
    if a.scores:
        sc = json.loads(Path(a.scores).read_text())
        s, o, clean = sc["S_proxy"], sc["O_proxy"], sc.get("S_proxy_clean")
    else:
        s, o = a.s_proxy, a.o_proxy
    print(json.dumps(gate(a.public, s, o, Path(a.calibration), clean), indent=1))


if __name__ == "__main__":
    main()
