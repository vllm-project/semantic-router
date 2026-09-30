#!/usr/bin/env bash
# ~27B M4 status (node host): one JSON line per M4 arm-seed directory on this node (stage receipts and
# GPU-hours, the running container and its last update, mean seconds per update so far, checkpoints, BEST,
# COMPLETE, projected full-attempt GPU-hours), then the node's ~27B GPU leases.
# With --loop SECONDS it also rewrites /data/dev2/runs/27b/m4-logs/heartbeat-<node>.json every SECONDS until
# no M4 container has run on this node for two rounds (the heartbeat other workers can read).
# Usage: m4-status.sh NODE [--loop SECONDS]
set -euo pipefail
NODE=${1:?NODE a|b}
LOOP=0
[ "${2:-}" = --loop ] && LOOP=${3:?SECONDS}
case "$NODE" in a) GPUS=(2 3 4) ;; b) GPUS=(5 6 7) ;; *) echo "NODE is a or b" >&2; exit 2 ;; esac
R=/data/dev2/runs/27b
status() {
  python3 - "$R" "$NODE" "${GPUS[@]}" <<'EOF'
import glob, json, math, os, subprocess, sys, time
from datetime import datetime, timezone
root, node, gpus = sys.argv[1], sys.argv[2], sys.argv[3:]
planned = {"a20": 3561, "ar": 4023}
out = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "node": node, "arms": [], "leases": {}}
running = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.RunningFor}}"], capture_output=True,
                         text=True).stdout.split("\n")
for arm_dir in sorted(glob.glob(f"{root}/M4-*-s[12]")):
    arm = os.path.basename(arm_dir)
    receipts = {}
    for p in sorted(glob.glob(f"{arm_dir}/receipts/*.json")):
        r = json.load(open(p))
        receipts[os.path.basename(p)[:-5]] = {"gpu_hours": r["gpu_hours"], "exit": r["exit_code"],
                                               "watchdog": r["watchdog_fired"]}
    info = {"arm": arm, "receipts": receipts, "gpu_hours_closed": round(sum(v["gpu_hours"] for v in receipts.values()), 3)}
    names = [line.split("\t")[0] for line in running if line.startswith(f"d2-27b-{arm}-")]
    if names:
        name = names[0]
        started = subprocess.run(["docker", "inspect", "-f", "{{.State.StartedAt}}", name], capture_output=True,
                                 text=True).stdout.strip()
        start = datetime.fromisoformat(started[:26].rstrip("Z") + "+00:00") if started else None
        elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 3600 if start else 0.0
        logs = subprocess.run(["docker", "logs", "--tail", "400", name], capture_output=True, text=True)
        steps = []
        for line in (logs.stdout + logs.stderr).replace("\r", "\n").split("\n"):
            if line.startswith('{"event": "train"'):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                steps.append((event["step"], event["seconds"]))
        info["running"] = {"container": name, "hours": round(elapsed, 3)}
        if steps:
            info["running"]["step"] = steps[-1][0]
            recent = [s for _, s in steps[-200:]]
            info["running"]["recent_seconds_per_update"] = round(sum(recent) / len(recent), 2)
    ckpts = sorted(os.path.basename(p) for p in glob.glob(f"{arm_dir}/full/*/checkpoint-*") if not p.endswith(".pending"))
    info["checkpoints"] = len(ckpts)
    info["latest"] = ckpts[-1] if ckpts else None
    for key in ("BEST", "COMPLETE"):
        hits = glob.glob(f"{arm_dir}/full/*/{key}.json")
        info[key.lower()] = json.load(open(hits[0])) if hits else None
    info["gpu_hours_total"] = round(info["gpu_hours_closed"] + info.get("running", {}).get("hours", 0.0), 3)
    mix = "ar" if arm.startswith("M4-Ar-") else "a20"
    run = info.get("running", {})
    if run.get("step") and "full" in run.get("container", ""):
        rate = run["hours"] / run["step"]
        info["projected_full_gpu_hours"] = round(rate * planned[mix], 2)
    out["arms"].append(info)
for gpu in gpus:
    try:
        text = open(f"/data/dev2/leases/gpu{gpu}.lock/owner").read()
        lease = json.loads(text)
        out["leases"][gpu] = {k: lease.get(k) for k in ("track", "status", "container", "purpose")}
    except (OSError, ValueError):
        out["leases"][gpu] = "unreadable"
out["m4_containers"] = sum(1 for a in out["arms"] if "running" in a)
print(json.dumps(out, sort_keys=True))
EOF
}
if [ "$LOOP" = 0 ]; then
  status
  exit 0
fi
idle=0
mkdir -p "$R/m4-logs"
while [ "$idle" -lt 2 ]; do
  line=$(status)
  printf '%s\n' "$line" > "$R/m4-logs/heartbeat-$NODE.json.pending"
  mv "$R/m4-logs/heartbeat-$NODE.json.pending" "$R/m4-logs/heartbeat-$NODE.json"
  printf '%s\n' "$line" >> "$R/m4-logs/heartbeat-$NODE.log"
  case "$line" in *'"m4_containers": 0'*) idle=$((idle + 1)) ;; *) idle=0 ;; esac
  sleep "$LOOP"
done
