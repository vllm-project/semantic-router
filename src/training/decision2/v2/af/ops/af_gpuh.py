"""Arm factory GPU-hours on this node from the launch receipts (wall clock x one GPU per container; M10's gpuh.py).

usage: af_gpuh.py total | prefix <job-prefix> | table   [--root /data/dev2/runs/af] [--running]

Every GPU job writes ``<out>.launch.json`` (af-launch.sh); CPU jobs have ``gpu: null`` and are not counted. A job
still running has no receipt yet; ``--running`` adds the elapsed time of running ``af-*`` containers that hold a GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def receipts(root: Path) -> list[dict]:
    found = []
    for path in sorted(root.rglob("*.launch.json")):
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
                "hours": max(0.0, (end - start).total_seconds() / 3600),
            }
        )
    return found


def running() -> list[dict]:
    try:
        names = subprocess.run(
            ["docker", "ps", "--filter", "name=^af-", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    now = datetime.now(timezone.utc)
    for name in names:
        fmt = (
            "{{.State.StartedAt}}|{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}"
        )
        out = subprocess.run(
            ["docker", "inspect", "-f", fmt, name], capture_output=True, text=True
        ).stdout
        started, _, devices = out.strip().partition("|")
        try:
            start = datetime.fromisoformat(started[:26].rstrip("Z") + "+00:00")
        except ValueError:
            continue
        if "renderD" in devices:
            rows.append(
                {
                    "job": name.removeprefix("af-"),
                    "hours": (now - start).total_seconds() / 3600,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("total", "prefix", "table"))
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("/data/dev2/runs/af"))
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    rows = receipts(args.root) + (running() if args.running else [])
    if args.mode == "total":
        print(round(sum(r["hours"] for r in rows), 4))
    elif args.mode == "prefix":
        print(
            round(sum(r["hours"] for r in rows if r["job"].startswith(args.rest[0])), 4)
        )
    else:
        groups: dict[str, float] = {}
        for r in rows:
            key = r["job"].rsplit("-", 1)[0]
            groups[key] = groups.get(key, 0.0) + r["hours"]
        total = round(sum(groups.values()), 4)
        print(
            json.dumps(
                {
                    "total": total,
                    "by_job": {k: round(v, 4) for k, v in sorted(groups.items())},
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
