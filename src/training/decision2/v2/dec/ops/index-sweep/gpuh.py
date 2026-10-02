"""GPU-hours of the Index sweep on this node (wall clock x GPUs, every interval; node side, stdlib).

    python3 gpuh.py            -> one JSON line: per item GPU-hours and the node total

Counts, for every IX1 name starting with IS-: the parity gate (ref and kit passes), each shard's start / end
intervals (a shard still running counts up to now; a stopped shard up to its end marker), and the voided or stopped
partial runs under ix1/void/IS-*; plus the 9B formal panels this sweep ran on node A GPU3 (formal9b.sh holds the
GPU from its "GPU3 -> track=9b-m9" line to its "formal NAME exit" line).
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import time
from pathlib import Path

R = Path("/data/dev2/private/eval/index021/ix1")
FORMAL_LOGS = Path("/data/dev2/runs/9b/formal-m9/logs")


def interval(work: Path) -> float:
    total = 0.0
    for start in sorted(work.glob("start_epoch*")):
        end = work / start.name.replace("start", "end")
        s = float(start.read_text())
        e = float(end.read_text()) if end.exists() else time.time()
        total += max(0.0, e - s)
    return total / 3600


def utc(line: str) -> dt.datetime:
    return dt.datetime.fromisoformat(line.split()[0].replace("Z", "+00:00"))


def main() -> None:
    items: dict[str, float] = {}
    for parity in sorted(R.glob("parity/IS-*")):
        items[f"parity {parity.name}"] = sum(
            interval(parity / p) for p in ("ref", "kit") if (parity / p).is_dir()
        )
    for root, label in ((R / "runs", "run"), (R / "void", "void")):
        for run in sorted(root.glob("IS-*")):
            hours = sum(interval(Path(w)) for w in glob.glob(f"{run}/shard-*"))
            if hours:
                items[f"{label} {run.name}"] = hours
    for log in sorted(FORMAL_LOGS.glob("isweep-formal-*.log")):
        lines = log.read_text().splitlines()
        start = [l for l in lines if "-> track=9b-m9" in l]
        end = [l for l in lines if " exit " in l and l.split()[1] == "formal"]
        if start and end:
            items[f"formal {log.stem[len('isweep-formal-'):]}"] = (
                utc(end[-1]) - utc(start[0])
            ).total_seconds() / 3600
    print(
        json.dumps(
            {
                "items": {k: round(v, 3) for k, v in items.items()},
                "total": round(sum(items.values()), 2),
            }
        )
    )


if __name__ == "__main__":
    main()
