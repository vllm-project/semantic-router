"""Apply a factor screen's frozen decision rules to its development readout.

Inputs: the ``dev_readout`` report (arms plus ``<arm>-minus-<control>``
comparisons), each arm's CAL calibration report and BEST SELECT metrics. For
each treatment it reports Δproxy against the control, the seed floor
σ_seed = |control − seed arm| / √2 per metric, retention floors (typed
Choice/Noul/Score −3.0 points, H −1.5 points, CAL Brier +0.010, invalid answers)
and the verdict (confirmed / suggestive / negative). The formal rule compares
the best arm with the same-runtime 1.0 control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def cal_brier(report: dict[str, Any]) -> float:
    """Overall CAL Brier after the fitted per-type temperatures."""
    return report["overall"]["after"]["brier"]


def type_accuracy(arm: dict[str, Any]) -> dict[str, float]:
    return {k: v["correct"] / v["n"] for k, v in arm["by_type"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--seed-arm", required=True)
    parser.add_argument("--baseline", required=True, help="same-runtime 1.0 arm name")
    parser.add_argument("--treatment", action="append", required=True)
    parser.add_argument("--calibration", action="append", default=[], help="arm=path")
    parser.add_argument(
        "--select", action="append", default=[], help="arm=BEST metrics path"
    )
    parser.add_argument("--formal-margin", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.readout.read_text())
    arms, comparisons = report["arms"], report["comparisons"]
    cal = {
        k: cal_brier(json.loads(Path(v).read_text()))
        for k, v in (spec.split("=", 1) for spec in args.calibration)
    }
    select = {
        k: json.loads(Path(v).read_text())
        for k, v in (spec.split("=", 1) for spec in args.select)
    }
    control, seed = arms[args.control], arms[args.seed_arm]
    sigma = {
        m: abs(control[m] - seed[m]) / math.sqrt(2)
        for m in ("proxy", "T", "H", "proxy_mean_H")
    }
    rows = {}
    for name in args.treatment:
        arm = arms[name]
        interval = comparisons[f"{name}-minus-{args.control}"]["delta_b_minus_a"]
        delta = arm["proxy"] - control["proxy"]
        types_c, types_t = type_accuracy(control), type_accuracy(arm)
        floors = {
            "types_within_3_points": all(
                (types_t[k] - types_c[k]) * 100 >= -3.0 for k in types_c
            ),
            "H_within_1_5_points": (arm["H"] - control["H"]) * 100 >= -1.5,
            "cal_brier_within_0_010": name not in cal
            or args.control not in cal
            or cal[name] - cal[args.control] <= 0.010,
            "invalid_not_increased": sum(v["invalid"] for v in arm["by_type"].values())
            <= sum(v["invalid"] for v in control["by_type"].values()),
        }
        effect = interval["proxy"]["lower95"] > 0 and abs(delta) > 2 * sigma["proxy"]
        if delta < 1.0 or not all(floors.values()):
            verdict = "negative"
        else:
            verdict = "confirmed" if effect else "suggestive"
        rows[name] = {
            "proxy": arm["proxy"],
            "delta_proxy": delta,
            "interval_proxy": interval["proxy"],
            "delta_T": arm["T"] - control["T"],
            "delta_H": arm["H"] - control["H"],
            "delta_proxy_mean_H": arm["proxy_mean_H"] - control["proxy_mean_H"],
            "type_accuracy_delta_points": {
                k: round((types_t[k] - types_c[k]) * 100, 2) for k in types_c
            },
            "score_predicted_levels": arm["score_predicted_levels"],
            "cal_brier": cal.get(name),
            "select_best": select.get(name),
            "retention_floors": floors,
            "effect_rule_lower95_and_2sigma": effect,
            "verdict": verdict,
        }
    candidates = [args.control, *args.treatment]
    best = max(candidates, key=lambda n: arms[n]["proxy"])
    base = arms[args.baseline]
    formal_key = f"{best}-minus-{args.baseline}"
    formal_interval = comparisons.get(formal_key, {}).get("delta_b_minus_a", {})
    formal = {
        "best_arm": best,
        "best_proxy": arms[best]["proxy"],
        "baseline_proxy": base["proxy"],
        "threshold": base["proxy"] + args.formal_margin,
        "interval_vs_baseline": formal_interval.get("proxy"),
        "all_valid": sum(v["invalid"] for v in arms[best]["by_type"].values()) == 0,
    }
    formal["qualifies"] = bool(
        formal["best_proxy"] >= formal["threshold"]
        and formal["interval_vs_baseline"]
        and formal["interval_vs_baseline"]["lower95"] > 0
        and formal["all_valid"]
    )
    out = {
        "schema_version": "dec-factor-summary/1",
        "control": args.control,
        "seed_arm": args.seed_arm,
        "sigma_seed": sigma,
        "control_proxy": control["proxy"],
        "seed_arm_proxy": seed["proxy"],
        "treatments": rows,
        "formal_rule": formal,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sigma_seed": sigma, "formal": formal}, sort_keys=True))
    for name, row in rows.items():
        print(
            name,
            round(row["proxy"], 3),
            round(row["delta_proxy"], 3),
            row["interval_proxy"],
            row["verdict"],
            row["retention_floors"],
        )


if __name__ == "__main__":
    main()
