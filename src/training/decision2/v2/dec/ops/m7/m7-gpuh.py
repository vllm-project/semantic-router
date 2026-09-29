"""GPU-hours of decoder M7 training-side jobs on this node (wall-clock x 1 GPU per GPU job), from launch receipts.

Every `*.launch.json` with a GPU under /data/dev2/runs/dec/m7/{arms,soup} is one job (line and
readout jobs elsewhere are not arm costs); running `dec-m7-*` GPU containers add their elapsed time (a running
container of another directory whose name matches an arm counts too, conservatively). Jobs are attributed to arms by name (`m7-<ARM>-s<i>...`, postrun
`m7-arms-full-m7-<ARM>-s<i>-...`).


usage: python3 m7-gpuh.py arm <ARM>      -> prints the arm's GPU-h (labels included)
       python3 m7-gpuh.py seed <ARM> <i> -> prints that seed's GPU-h
       python3 m7-gpuh.py write a|b      -> writes /data/dev2/runs/dec/m7/gpuh-node-<a|b>.json
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

M = Path("/data/dev2/runs/dec/m7")
ARM = re.compile(r"m7-(?:arms-full-m7-)?(N7H|N7P|N7C|S7H|S7P|S7C)-s([123])")
LABELS: dict = {}


def utc(text):
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def jobs():
    out = {}
    receipts = [p for d in ("arms", "soup") for p in (M / d).rglob("*.launch.json")]
    for path in receipts:
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if r.get("gpu"):
            secs = (utc(r["end_utc"]) - utc(r["start_utc"])).total_seconds()
            out[r["job"]] = {
                "seconds": secs,
                "gpu": r["gpu"],
                "exit": r["exit_status"],
                "receipt": str(path),
            }
    try:
        names = subprocess.run(
            ["docker", "ps", "--filter", "name=dec-m7-", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        names = []
    now = datetime.now(timezone.utc)
    for name in names:
        job = name.removeprefix("dec-")
        if job in out:
            continue
        info = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.StartedAt}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}",
                name,
            ],
            capture_output=True,
            text=True,
        ).stdout.split()
        if info and any("renderD" in d for d in info[1:]):
            out[job] = {
                "seconds": (now - utc(info[0])).total_seconds(),
                "gpu": " ".join(d for d in info[1:] if "renderD" in d),
                "exit": None,
                "running": True,
            }
    return out


def attribute(all_jobs):
    arms, seeds = {}, {}
    for job, info in all_jobs.items():
        hit = ARM.search(job)
        if hit:
            arm, seed = hit.groups()
            arms[arm] = arms.get(arm, 0) + info["seconds"]
            seeds[f"{arm}-s{seed}"] = seeds.get(f"{arm}-s{seed}", 0) + info["seconds"]
        for prefix, owners in LABELS.items():
            if job == prefix:
                for arm in owners:
                    arms[arm] = arms.get(arm, 0) + info["seconds"]
    return arms, seeds


def main():
    all_jobs = jobs()
    arms, seeds = attribute(all_jobs)
    if sys.argv[1] == "arm":
        print(f"{arms.get(sys.argv[2], 0) / 3600:.4f}")
    elif sys.argv[1] == "seed":
        print(f"{seeds.get(f'{sys.argv[2]}-s{sys.argv[3]}', 0) / 3600:.4f}")
    elif sys.argv[1] == "write":
        node = sys.argv[2]
        doc = {
            "node": node,
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_gpu_h": round(
                sum(j["seconds"] for j in all_jobs.values()) / 3600, 4
            ),
            "arms_gpu_h_for_caps": {
                k: round(v / 3600, 4) for k, v in sorted(arms.items())
            },
            "seeds_gpu_h": {k: round(v / 3600, 4) for k, v in sorted(seeds.items())},
            "jobs": dict(sorted(all_jobs.items())),
        }
        out = M / f"gpuh-node-{node}.json"
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(out)
    else:
        sys.exit(f"unknown command {sys.argv[1]}")


if __name__ == "__main__":
    main()
