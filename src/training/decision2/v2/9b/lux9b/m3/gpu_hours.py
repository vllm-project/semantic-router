"""Sum wall-clock GPU time of 9B Milestone 3 run directories (one GPU per run).

python3 gpu_hours.py [/data/dev2/runs/9b/m3] [/data/dev2/runs/9b/formal-m3 ...]
"""

import datetime as dt
import json
import sys
from pathlib import Path


def parse(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.read_text().strip(), "%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    roots = [Path(p) for p in sys.argv[1:]] or [Path("/data/dev2/runs/9b/m3")]
    rows = []
    for root in roots:
        for run in sorted(p for p in root.iterdir() if (p / "start-utc.txt").exists()):
            end = run / "end-utc.txt"
            seconds = (
                (parse(end) - parse(run / "start-utc.txt")).total_seconds()
                if end.exists()
                else None
            )
            code = (
                (run / "exit-code.txt").read_text().strip()
                if (run / "exit-code.txt").exists()
                else "running"
            )
            gpu = (
                (run / "gpu.txt").read_text().strip()
                if (run / "gpu.txt").exists()
                else None
            )
            rows.append(
                {
                    "run": f"{root.name}/{run.name}",
                    "gpu": gpu,
                    "seconds": seconds,
                    "exit": code,
                }
            )
        for time_file in sorted(root.glob("*/GPU-TIME.json")):
            record = json.loads(time_file.read_text())
            seconds = record.get("wall_seconds") or record.get("seconds")
            rows.append(
                {
                    "run": f"{root.name}/{time_file.parent.name}",
                    "gpu": record.get("gpu"),
                    "seconds": seconds,
                    "exit": record.get("exit_code"),
                }
            )
    total = sum(r["seconds"] or 0 for r in rows)
    print(
        json.dumps(
            {"runs": rows, "gpu_seconds": total, "gpu_hours": round(total / 3600, 4)},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
