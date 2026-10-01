"""Decoder M11 GPU-hours on this node from the launch receipts (wall clock x one GPU per container).

usage: m11_gpuh.py total | arm <ARM> | seed <ARM> <i> | table   [--root /data/dev2/runs/dec/m11]

Every GPU job writes ``<out>.launch.json`` (m11-launch.sh); CPU jobs have ``gpu: null`` and are not counted. A job
still running has no receipt yet; ``table`` adds running containers' elapsed time from ``running/*.start`` files when
present (none are written by the chains, so the stop rules see finished jobs only).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def receipts(root: Path) -> list[dict]:
    found = []
    for path in sorted(root.rglob("*.launch.json")):
        if "inputs" in path.relative_to(root).parts:
            continue  # inputs copied from other milestones carry their own receipts
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not data.get("gpu"):
            continue
        start = datetime.fromisoformat(data["start_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(data["end_utc"].replace("Z", "+00:00"))
        found.append(
            {
                "job": data["job"],
                "gpu": data["gpu"],
                "hours": max(0.0, (end - start).total_seconds() / 3600),
                "exit": data["exit_status"],
                "path": str(path),
            }
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("total", "arm", "seed", "table"))
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("/data/dev2/runs/dec/m11"))
    args = parser.parse_args()
    rows = receipts(args.root)
    if args.mode == "total":
        print(round(sum(r["hours"] for r in rows), 4))
    elif args.mode == "arm":
        prefix = f"m11-{args.rest[0]}-"
        print(round(sum(r["hours"] for r in rows if r["job"].startswith(prefix)), 4))
    elif args.mode == "seed":
        prefix = f"m11-{args.rest[0]}-s{args.rest[1]}-"
        print(round(sum(r["hours"] for r in rows if r["job"].startswith(prefix)), 4))
    else:
        by_group: dict[str, float] = {}
        for r in rows:
            parts = r["job"].split("-")
            group = "-".join(parts[:3]) if r["job"].startswith("m11-") else parts[0]
            by_group[group] = by_group.get(group, 0.0) + r["hours"]
        print(
            json.dumps(
                {
                    "total": round(sum(by_group.values()), 4),
                    "by_group": {k: round(v, 4) for k, v in sorted(by_group.items())},
                    "jobs": len(rows),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
