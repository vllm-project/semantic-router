"""9B M10 GPU-hours on this node from the launch receipts (wall clock x one GPU per container).

usage: gpuh.py total | arm <ARM> | table   [--root /data/dev2/runs/9b/m10] [--running]

Every GPU job writes ``<out>.launch.json`` (m10/launch.sh); CPU jobs have ``gpu: null`` and are not counted. A job
still running has no receipt yet; ``--running`` adds the elapsed time of running ``m10-*`` containers (docker ps).
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
                "exit": data["exit_status"],
            }
        )
    return found


def running() -> list[dict]:
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "name=^m10-", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    now = datetime.now(timezone.utc)
    for name in out:
        started = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.StartedAt}}", name],
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            start = datetime.fromisoformat(started[:26].rstrip("Z") + "+00:00")
        except ValueError:
            continue
        devices = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}",
                name,
            ],
            capture_output=True,
            text=True,
        ).stdout
        if "renderD" in devices:
            rows.append(
                {
                    "job": name.removeprefix("m10-"),
                    "hours": (now - start).total_seconds() / 3600,
                    "exit": None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("total", "arm", "table"))
    parser.add_argument("rest", nargs="*")
    parser.add_argument("--root", type=Path, default=Path("/data/dev2/runs/9b/m10"))
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()
    rows = receipts(args.root) + (running() if args.running else [])
    if args.mode == "total":
        print(round(sum(r["hours"] for r in rows), 4))
    elif args.mode == "arm":
        prefix = f"m10-{args.rest[0]}-"
        print(round(sum(r["hours"] for r in rows if r["job"].startswith(prefix)), 4))
    else:
        by_group: dict[str, float] = {}
        for r in rows:
            parts = r["job"].split("-")
            group = "-".join(parts[:2]) if r["job"].startswith("m10-") else parts[0]
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
