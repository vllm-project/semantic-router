"""Development readout summary and the preregistered Milestone 1 decision rule.

Inputs are unchanged ``benchmark.score`` (typed DEV) and ``transfer.score``
(CSS pilot) reports. ``P_dev = 100*sqrt(T_dev*H_pilot)`` with ``T_dev`` the DEV
family-macro accuracy and ``H_pilot`` the CSS-pilot median task macro-F1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

TYPES = ("choice", "noul", "score")


def summarize(dev: dict[str, Any], css: dict[str, Any]) -> dict[str, Any]:
    pilot = css["roles"]["pilot"]
    tasks = {
        name: task["macro_f1_all"]
        for name, task in css["tasks"].items()
        if task.get("role") == "pilot"
    }
    t_dev = dev["macro_family_accuracy"]
    h_pilot = pilot["median_task_macro_f1_all"]
    return {
        "T_dev": t_dev,
        "H_pilot": h_pilot,
        "H_pilot_mean": statistics.fmean(tasks.values()),
        "P_dev": 100 * math.sqrt(t_dev * h_pilot),
        "dev_correct": dev["overall"]["correct_n"],
        "dev_invalid": dev["overall"]["invalid_or_missing_n"],
        "by_type": {
            kind: {
                "correct": dev["by_type"][kind]["correct_n"],
                "n": dev["by_type"][kind]["n"],
                "accuracy": dev["by_type"][kind]["correct_n"]
                / dev["by_type"][kind]["n"],
                "brier": dev["by_type"][kind].get("brier"),
            }
            for kind in TYPES
        },
        "by_family": {k: v.get("accuracy_all") for k, v in dev["by_family"].items()},
        "css_task_macro_f1": tasks,
        "css_invalid": pilot["items"] - pilot["valid_items"],
        "dev_brier": dev["overall"].get("brier"),
    }


def decide(
    control: dict[str, Any], challenger: dict[str, Any], sigma_seed: float | None
) -> dict[str, Any]:
    margin = max(2.0, 2 * sigma_seed) if sigma_seed is not None else 2.0
    type_drops = {
        kind: 100
        * (
            control["by_type"][kind]["accuracy"]
            - challenger["by_type"][kind]["accuracy"]
        )
        for kind in TYPES
    }
    checks = {
        "p_dev_margin": challenger["P_dev"] >= control["P_dev"] + margin,
        "no_type_drop_over_3": all(drop <= 3.0 for drop in type_drops.values()),
        "h_pilot_drop_at_most_0.015": control["H_pilot"] - challenger["H_pilot"]
        <= 0.015,
        "invalid_not_increased": challenger["dev_invalid"] + challenger["css_invalid"]
        <= control["dev_invalid"] + control["css_invalid"],
    }
    return {
        "delta_P_dev": challenger["P_dev"] - control["P_dev"],
        "required_margin": margin,
        "type_drop_points": type_drops,
        "checks": checks,
        "displaces_control": all(checks.values()),
    }


def load(path: str) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True, help="NAME=DEV_SCORE,CSS_SCORE"
    )
    parser.add_argument("--control", default="C0")
    parser.add_argument(
        "--seed-replicate", help="Arm name whose |dP| to control defines sigma_seed"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arms, sources = {}, {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        dev_path, css_path = paths.split(",")
        (dev, dev_sha), (css, css_sha) = load(dev_path), load(css_path)
        arms[name] = summarize(dev, css)
        sources[name] = {"dev_score_sha256": dev_sha, "css_score_sha256": css_sha}
    control = arms[args.control]
    sigma = None
    if args.seed_replicate:
        sigma = abs(arms[args.seed_replicate]["P_dev"] - control["P_dev"]) / math.sqrt(
            2
        )
    decisions = {
        name: decide(control, arm, sigma)
        for name, arm in arms.items()
        if name not in (args.control, args.seed_replicate)
    }
    result = {
        "arms": arms,
        "sources": sources,
        "sigma_seed": sigma,
        "decisions": decisions,
    }
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
