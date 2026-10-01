"""GPU-hours of decoder M9 jobs on node A (wall-clock x 1 GPU per GPU job), from launch receipts.

Every `*.launch.json` with a GPU under /data/dev2/runs/dec/m9 is one job (training, postruns, soups, early reads,
lines, diagnostics); every runner `GPU-TIME.json` under m9/formal adds its wall seconds; running `dec-m9-*` GPU
containers add their elapsed time. Co-tenant jobs are counted in full (conservative). Training jobs are attributed
to arms by name (`m9-<ARM>-s<i>...`, postrun `m9-arms-full-m9-<ARM>-s<i>-...`).

usage: python3 m9_gpuh.py total          -> prints the milestone's GPU-h
       python3 m9_gpuh.py arm <ARM>      -> prints the arm's training GPU-h
       python3 m9_gpuh.py seed <ARM> <i> -> prints that seed's GPU-h
       python3 m9_gpuh.py write          -> writes /data/dev2/runs/dec/m9/gpuh-node-a.json
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

M = Path(os.environ.get("M9_ROOT", "/data/dev2/runs/dec/m9"))
ARM = re.compile(r"m9-(?:arms-full-m9-)?(H9|C9)-s([123])")


def utc(text):
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def jobs():
    out = {}
    for path in M.rglob("*.launch.json"):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if r.get("gpu"):
            out[r["job"]] = {
                "seconds": (utc(r["end_utc"]) - utc(r["start_utc"])).total_seconds(),
                "gpu": r["gpu"],
                "exit": r["exit_status"],
                "receipt": str(path),
            }
    for path in (
        (M / "formal").rglob("GPU-TIME.json") if (M / "formal").is_dir() else ()
    ):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        out[f"runner:{path.parent.relative_to(M)}"] = {
            "seconds": float(r.get("wall_seconds") or 0),
            "gpu": r.get("gpu"),
            "exit": r.get("exit_code"),
            "receipt": str(path),
        }
    try:
        names = subprocess.run(
            ["docker", "ps", "--filter", "name=dec-m9-", "--format", "{{.Names}}"],
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
    return arms, seeds


def main():
    all_jobs = jobs()
    arms, seeds = attribute(all_jobs)
    cmd = sys.argv[1]
    if cmd == "total":
        print(f"{sum(j['seconds'] for j in all_jobs.values()) / 3600:.4f}")
    elif cmd == "arm":
        print(f"{arms.get(sys.argv[2], 0) / 3600:.4f}")
    elif cmd == "seed":
        print(f"{seeds.get(f'{sys.argv[2]}-s{sys.argv[3]}', 0) / 3600:.4f}")
    elif cmd == "write":
        doc = {
            "node": "a",
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
        out = M / "gpuh-node-a.json"
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        tmp.replace(out)
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
