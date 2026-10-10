"""Predicted ON latency from a measured OFF latency run and the forward-pass shapes of ``prepare_check.py``.

A relative-error least-squares model of the measured per-request wall time (timed rows only) on the request's
forward passes, ``ms = a + b * passes + c * padded_ktokens + d * sum(padded * longest) / 1e6`` (the last term
stands in for attention cost growing with length), restricted to non-negative coefficients (a feature whose
coefficient comes out negative is dropped and the fit repeated), fitted on OFF and applied to the ON shapes of
the same requests as a correction of each measured time. A second, model-free estimate scales each request's
measured time by its ON / OFF padded tokens (no fixed cost is saved, so it errs high for short requests).
Neither replaces a measured ON run.

    python -m d25.vega.tta.latency_model --check prepare-check.json --results kit-760/results.jsonl.gz \
        --design latency-760.jsonl.gz.json --out latency-model.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from d25.vega.release.compare import percentile, read_results


def features(passes: list[dict]) -> list[float]:
    padded = sum(p["padded"] for p in passes)
    quadratic = sum(p["padded"] * p["padded"] / p["sequences"] for p in passes) / 1e6
    return [1.0, float(len(passes)), padded / 1000.0, quadratic]


def stats(values: list[float]) -> dict:
    return {
        "median_ms": round(statistics.median(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
        "p80_ms": round(percentile(values, 80), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "max_ms": round(max(values), 1),
    }


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", required=True, type=Path)
    ap.add_argument("--results", required=True)
    ap.add_argument("--design", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    shapes = json.loads(args.check.read_text())["requests"]
    design = json.loads(args.design.read_text())
    warm = set(design["warmup_run_ids"])
    results = read_results(args.results)
    rows = [
        (rid, r["total_wall_ms"], shapes[rid])
        for rid, r in results.items()
        if rid not in warm and r["status"] == "ok" and rid in shapes
    ]
    X = np.array([features(s["off"]) for _, _, s in rows])
    X_on = np.array([features(s["on"]) for _, _, s in rows])
    y = np.array([ms for _, ms, _ in rows])
    keep = list(range(X.shape[1]))
    while True:
        w = 1.0 / y
        sub, *_ = np.linalg.lstsq(X[:, keep] * w[:, None], y * w, rcond=None)
        if (sub >= 0).all():
            break
        keep.pop(int(np.argmin(sub)))
    coef = np.zeros(X.shape[1])
    coef[keep] = sub
    fit_off = X @ coef
    on = X_on @ coef
    # Requests the flag does not change (no choice question with 2+ options) keep their measured time.
    unchanged = np.array([s["on"] == s["off"] for _, _, s in rows])
    on_model = np.where(unchanged, y, y + (on - fit_off))
    ratio = np.array(
        [
            ms * sum(p["padded"] for p in s["on"]) / sum(p["padded"] for p in s["off"])
            for _, ms, s in rows
        ]
    )
    r2 = 1 - float(((y - fit_off) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    report = {
        "timed_rows": len(rows),
        "model": "ms = a + b*passes + c*padded_ktokens + d*sum(padded*longest)/1e6, relative-error least squares on "
        "OFF, non-negative coefficients; ON = measured OFF + model(ON) - model(OFF) per request",
        "coefficients": dict(
            zip(
                ("a_ms", "b_ms_per_pass", "c_ms_per_kpadded", "d_quadratic"),
                (round(float(c), 3) for c in coef),
            )
        ),
        "r2_off": round(r2, 4),
        "off_measured": stats(list(y)),
        "off_fitted": stats(list(fit_off)),
        "on_predicted_model": stats(list(on_model)),
        "on_predicted_ratio": stats(list(ratio)),
        "requests_changed_by_flag": int((~unchanged).sum()),
        "on_over_off_padded": round(
            sum(sum(p["padded"] for p in s["on"]) for _, _, s in rows)
            / sum(sum(p["padded"] for p in s["off"]) for _, _, s in rows),
            4,
        ),
        "rule": "median, mean and p80 < 1000 ms",
    }
    for key in ("on_predicted_model", "on_predicted_ratio"):
        report[key]["rule_pass"] = all(
            report[key][k] < 1000 for k in ("median_ms", "mean_ms", "p80_ms")
        )
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
