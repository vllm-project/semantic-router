"""GPU-hours of decoder M5 jobs on node B: wall-clock x 1 GPU from every GPU launch receipt under m5/.

usage: python3 m5-gpuh.py [root]   (default /data/dev2/runs/dec/m5)
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/dev2/runs/dec/m5")
jobs = []
for path in sorted(root.rglob("*.launch.json")):
    r = json.loads(path.read_text())
    if r.get("gpu") is None:
        continue
    start = datetime.fromisoformat(r["start_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(r["end_utc"].replace("Z", "+00:00"))
    jobs.append(
        {
            "job": r["job"],
            "gpu": r["gpu"],
            "wall_seconds": (end - start).total_seconds(),
            "exit": r["exit_status"],
        }
    )
by_group = defaultdict(float)
for j in jobs:
    name = j["job"]
    key = (
        name.rsplit("-", 1)[0]
        if name.split("-")[-1] in ("zero", "one", "gate", "full", "cal", "dev")
        else name
    )
    by_group[key] += j["wall_seconds"]
total = sum(j["wall_seconds"] for j in jobs)
print(
    json.dumps(
        {
            "jobs": jobs,
            "gpu_hours_total": round(total / 3600, 4),
            "gpu_hours_by_group": {
                k: round(v / 3600, 4) for k, v in sorted(by_group.items())
            },
        },
        indent=1,
    )
)
