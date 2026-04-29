#!/usr/bin/env python3
"""Calibrate a B200 serving-power profile from summary text files.

Expected inputs are the plain-text summaries emitted by summarize_power.py:
  - summary_idle_5min.txt
  - summary_loaded_70b_8k_b1_5min.txt
  - summary_run_70b_8k_b1.txt

The script prints a workload-specific recommendation for the simulator's
ManualProfile power terms:
  - true_idle_w      : fully unloaded device idle (informational)
  - power_idle_w     : loaded-but-not-serving baseline used by the simulator
  - power_nominal_w  : active ceiling (defaults to run p95)
  - power_logistic_x0: midpoint fitted from the run mean at b=1

Usage:
  python3 scripts/calibrate_b200_power.py \
      --idle /path/to/summary_idle_5min.txt \
      --loaded /path/to/summary_loaded_70b_8k_b1_5min.txt \
      --run /path/to/summary_run_70b_8k_b1.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_summary(path: Path) -> dict[str, float | str]:
    data: dict[str, float | str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value
    return data


def solve_logistic_x0(power_idle_w: float, power_nominal_w: float, power_at_one_w: float) -> float:
    power_at_one_w = max(power_idle_w + 1e-6, min(power_nominal_w - 1e-6, power_at_one_w))
    frac = (power_at_one_w - power_idle_w) / (power_nominal_w - power_idle_w)
    return math.log((1.0 / frac) - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle", required=True, type=Path, help="Fully idle summary file.")
    parser.add_argument("--loaded", required=True, type=Path, help="Model-loaded summary file.")
    parser.add_argument("--run", required=True, type=Path, help="Active inference summary file.")
    parser.add_argument(
        "--nominal-source",
        choices=("p95", "p50", "mean", "max"),
        default="p95",
        help="Which run statistic to treat as the active ceiling.",
    )
    parser.add_argument(
        "--anchor-source",
        choices=("mean", "p50", "p95"),
        default="mean",
        help="Which run statistic to treat as the b=1 anchor.",
    )
    args = parser.parse_args()

    idle = parse_summary(args.idle)
    loaded = parse_summary(args.loaded)
    run = parse_summary(args.run)

    true_idle_w = float(idle["per_gpu_mean_w"])
    serving_floor_w = float(loaded["per_gpu_mean_w"])
    nominal_key = f"per_gpu_{args.nominal_source}_w"
    anchor_key = f"per_gpu_{args.anchor_source}_w"
    nominal_w = float(run[nominal_key])
    anchor_w = float(run[anchor_key])
    x0 = solve_logistic_x0(serving_floor_w, nominal_w, anchor_w)

    print("Suggested B200 calibration")
    print("==========================")
    print(f"true_idle_w       : {true_idle_w:.2f}")
    print(f"power_idle_w      : {serving_floor_w:.2f}   # model loaded, no inference")
    print(f"power_nominal_w   : {nominal_w:.2f}   # from run {args.nominal_source}")
    print(f"power_logistic_k  : 1.00")
    print(f"power_logistic_x0 : {x0:.4f}   # fitted from run {args.anchor_source}")
    print()
    print("Source summaries")
    print("---------------")
    print(f"idle   : {args.idle}")
    print(f"loaded : {args.loaded}")
    print(f"run    : {args.run}")
    print()
    print("Recommended simulator mode")
    print("--------------------------")
    print("Set `FLEETSIM_B200_POWER_MODE=measured_b1` and copy the numbers into")
    print("`scripts/profiles.py` if you want this run to become the default measured profile.")


if __name__ == "__main__":
    main()
