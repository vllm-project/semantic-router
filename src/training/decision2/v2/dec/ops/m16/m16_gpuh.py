"""Decoder M16 GPU-hours on this node from the launch receipts (wall clock x one GPU per container).

usage: m16_gpuh.py total | table   [--root /data/dev2/runs/dec/m16] [--node A|B]

Every GPU job writes ``<out>.launch.json`` (m16-launch.sh); CPU jobs (interpolation builds, compares) have
``gpu: null`` and are not counted. Only receipts of jobs that ran on this node count (``--node``, default
``$M16_NODE``); receipts under ``inputs/`` (M14's reference readouts) are skipped. ``table`` groups by job kind and
point (``lines-<point>-<panel>`` -> ``<point>``).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

PANELS = (
    "css-pilot",
    "ht-dev2",
    "score5t-dev",
    "hs1-dev",
    "pn1-dev",
    "m10-probes",
    "ib-dev",
    "mlxdev",
    "dev",
)


def receipts(root: Path, node: str | None = None) -> list[dict]:
    found = []
    for path in sorted(root.rglob("*.launch.json")):
        if "inputs" in path.relative_to(root).parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not data.get("gpu"):
            continue
        if node and not data["gpu"].startswith(f"node {node.upper()} GPU"):
            continue
        start = datetime.fromisoformat(data["start_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(data["end_utc"].replace("Z", "+00:00"))
        found.append(
            {
                "job": data["job"],
                "gpu": data["gpu"],
                "hours": max(0.0, (end - start).total_seconds() / 3600),
                "exit": data["exit_status"],
            }
        )
    return found


def group(job: str) -> str:
    if job.startswith("lines-"):
        rest = job[len("lines-") :]
        for panel in PANELS:
            if rest.endswith("-" + panel):
                return rest[: -len(panel) - 1]
        return rest
    return job.split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("total", "table"))
    parser.add_argument("--root", type=Path, default=Path("/data/dev2/runs/dec/m16"))
    parser.add_argument("--node", default=os.environ.get("M16_NODE"))
    args = parser.parse_args()
    rows = receipts(args.root, args.node)
    if args.mode == "total":
        print(round(sum(r["hours"] for r in rows), 4))
        return
    by_group: dict[str, float] = {}
    for r in rows:
        by_group[group(r["job"])] = by_group.get(group(r["job"]), 0.0) + r["hours"]
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
