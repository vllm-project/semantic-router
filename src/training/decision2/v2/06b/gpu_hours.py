"""GPU-hours as wall-clock occupancy per GPU (overlapping runs on one GPU count once).

    python3 -m v2.06b.gpu_hours --timing 'LOGDIR/m5-*.timing.json' [--timing ...] \
        [--gpu-time RUN/GPU-TIME.json ...] --output hours.json

`--timing` globs read `run_container.sh` timing files, `--gpu-time` reads the eval runner's
`GPU-TIME.json`. Intervals are merged per GPU; the report lists each run with its GPU and
interval, the merged occupancy per GPU and the total.
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def stamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def merged_seconds(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    start = None
    for low, high in sorted(intervals):
        if end is None or low > end:
            if end is not None:
                total += end - start
            start, end = low, high
        else:
            end = max(end, high)
    if end is not None:
        total += end - start
    return total


def occupancy(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_gpu: dict[int, list[tuple[float, float]]] = {}
    for run in runs:
        by_gpu.setdefault(run["gpu"], []).append(
            (stamp(run["start_utc"]), stamp(run["end_utc"]))
        )
    hours = {gpu: merged_seconds(v) / 3600 for gpu, v in sorted(by_gpu.items())}
    return {
        "per_gpu_hours": {str(g): round(h, 4) for g, h in hours.items()},
        "total_gpu_hours": round(sum(hours.values()), 4),
        "summed_run_hours": round(
            sum(stamp(r["end_utc"]) - stamp(r["start_utc"]) for r in runs) / 3600, 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timing", action="append", default=[])
    parser.add_argument("--gpu-time", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for pattern in args.timing:
        for path in sorted(glob.glob(pattern)):
            t = json.loads(Path(path).read_text())
            runs.append(
                {
                    "run": t["run"],
                    "gpu": int(t["gpu"]),
                    "start_utc": t["start_utc"],
                    "end_utc": t["end_utc"],
                    "exit": t["exit"],
                }
            )
    for path in args.gpu_time:
        t = json.loads(Path(path).read_text())
        runs.append(
            {
                "run": str(Path(path).parent.name),
                "gpu": int(t["gpu"]),
                "start_utc": t["start_utc"],
                "end_utc": t["end_utc"],
                "exit": t["exit_code"],
            }
        )
    report = {"runs": sorted(runs, key=lambda r: r["start_utc"]), **occupancy(runs)}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))


if __name__ == "__main__":
    main()
