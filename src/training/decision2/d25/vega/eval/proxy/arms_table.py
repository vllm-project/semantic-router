"""Rescore stored ckpt_eval results with the current gate and write the arms table (markdown).

    python -m d25.vega.eval.proxy.arms_table --results DIR_OF_result_jsons --out arms.md

Rows with a measured public index get the full gate (per-benchmark SAT S_hat when available). Proxy-only rows
get the public index they would need for the 90% lower bound to clear #2 (top-3) and #1 + 0.9, using the
S0 map for S (SAT needs per-benchmark public skills) and the calibrated O map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from d25.vega.eval.proxy.calibrate_v2 import full, pred
from d25.vega.eval.proxy.gate import DEFAULT, gate

ANCHORS = {
    "pplx (board 62.75)": (62.28, 60.75, 59.03, 66.70),
    "kev (58.78)": (56.69, 60.66, 60.06, 62.40),
    "vega2 (55.88)": (56.98, 61.61, 60.37, 61.37),
}


def public_needed(o_proxy, cal, target):
    fin = cal["final"]

    def f(p):
        x = {"O1": [o_proxy], "O2": [o_proxy, p]}[fin["O_model"]]
        return full(p, pred(fin["S0"], [p]), pred(fin["O_coef"], x))

    k0, k1 = f(0.0), f(100.0) - f(0.0)
    return (target - k0) / (k1 / 100.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=str(DEFAULT))
    a = ap.parse_args(argv)
    cal = json.loads(Path(a.calibration).read_text())
    board = json.loads((Path(__file__).with_name("board_v03.json")).read_text())[
        "models"
    ]
    top = sorted((m["full"] for m in board if m["full"] is not None), reverse=True)
    lines = [
        "| arm | step | public | S_proxy | S_proxy_clean | O_proxy | S_hat | O_hat | Full_hat | lower 90% | public needed (top-3 / #1) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = []
    for f in sorted(Path(a.results).glob("*.json")):
        r = json.loads(f.read_text())
        p, pub = r.get("proxy") or {}, r.get("public") or {}
        if not p:
            continue
        rows.append(
            (
                r.get("arm") or f.stem,
                r.get("step"),
                pub.get("index"),
                p["S_proxy"],
                p.get("S_proxy_clean"),
                p["O_proxy"],
                pub.get("per_benchmark"),
            )
        )
    for name, (pubi, s, sc, o) in ANCHORS.items():
        rows.append((f"anchor {name}", "", pubi, s, sc, o, None))
    for arm, step, pubi, s, sc, o, bench in rows:
        need = f"{public_needed(o, cal, top[1] + cal['margin']):.1f} / {public_needed(o, cal, top[0] + 0.9 + cal['margin']):.1f}"
        if pubi is not None:
            g = gate(pubi, s, o, a.calibration, sc, bench)
            cells = [
                f"{pubi:.2f}",
                f"{s:.2f}",
                f"{sc:.2f}",
                f"{o:.2f}",
                f"{g['S_hat']:.2f}",
                f"{g['O_hat']:.2f}",
                f"{g['Full_hat']:.2f}",
                f"{g['Full_lower_90']:.2f}",
            ]
        else:
            cells = ["-", f"{s:.2f}", f"{sc:.2f}", f"{o:.2f}", "-", "-", "-", "-"]
        lines.append(
            f"| {arm} | {step if step is not None else ''} | "
            + " | ".join(cells)
            + f" | {need} |"
        )
    head = (
        f"Gate: `{Path(a.calibration).name}` (S {cal['final']['S_model']} / SAT for our arms, O {cal['final']['O_model']}; "
        f"margin {cal['margin']:.2f}; anchors {len(cal['anchors'])}). Top-3 needs lower bound > {top[1]:.2f}; untied #1 > {top[0] + 0.9:.2f}.\n\n"
    )
    Path(a.out).write_text(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
