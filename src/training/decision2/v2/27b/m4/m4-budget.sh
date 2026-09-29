#!/usr/bin/env bash
# ~27B M4 GPU-hour budget (run on the workstation; reads both nodes): every M4 launch receipt
# (decision2-27b-launch-receipt/1) and eval-runner GPU-TIME.json (dev2-gpu-time/1) against the 70 GPU-hour
# milestone cap (prereg "Budget": wall-clock x GPUs, preflights and evaluation included). Read on each node:
#   arm-seeds M4-{A20,A20r,Ar}-s{1,2} and the cross-node probe M4-xnode-A20-s1: <dir>/receipts/*.json on the node
#     that trained them; a node-B copy of a node-A arm-seed (RELAY.json, receipts kept as receipts.node-a/) is
#     skipped, so relayed receipts never count twice;
#   soups / finalists M4-{A20,A20r,Ar}-soup: launch receipts in receipts/ directories and every GPU-TIME.json
#     under the soup directory (run_finalist.sh's GPU-HOURS.json rule: readout CAL fit + collection, CAL698 fit,
#     formal smoke + collection; the soup itself is a CPU container);
#   mlx-diag: /data/dev2/runs/27b/m4-mlx/*/GPU-TIME.json (collections and smokes; scoring on node A is CPU).
# A receipt path seen on both nodes counts once if the two copies hash equal, else the script fails. Running M4
# containers (d2-27b-M4-*, dev2-27b-gpu*) have no receipt yet: their hours so far are listed, with the node
# heartbeat's projected full attempt for a training run (m4-status.sh). M4_LEFT = 70 - receipts - running work at
# max(projection, hours so far): the value m4-tail.sh's GPU stages take.
# Usage: m4-budget.sh [--json]
set -euo pipefail
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
R=/data/dev2/runs/27b
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
IFS= read -r -d '' SCAN <<'EOF' || true
import glob, hashlib, json, os, re, subprocess, sys
from datetime import datetime, timezone
root, node = sys.argv[1:]
items, relayed, running = [], [], []
ARM = re.compile(r"M4-(A20|A20r|Ar)-s[12]|M4-xnode-A20-s1")
SOUP = re.compile(r"M4-(A20|A20r|Ar)-soup")

def add(path, kind, owner):
    try:
        record = json.load(open(path))
    except (OSError, ValueError):
        return
    if not isinstance(record, dict) or (
        record.get("schema_version") != "decision2-27b-launch-receipt/1"
        and record.get("schema") != "dev2-gpu-time/1"
    ):
        return
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    items.append({"node": node, "kind": kind, "owner": owner, "file": os.path.relpath(path, root),
                  "gpu_hours": float(record["gpu_hours"]), "sha256": sha})

for folder in sorted(glob.glob(f"{root}/M4-*")):
    name = os.path.basename(folder)
    if not os.path.isdir(folder):
        continue
    if ARM.fullmatch(name):
        if os.path.exists(f"{folder}/RELAY.json"):
            relayed.append(name)
            continue
        for path in sorted(glob.glob(f"{folder}/receipts/*.json")):
            add(path, "probe" if "xnode" in name else "arm-seed", name)
    elif SOUP.fullmatch(name):
        for sub, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if d != "triton-cache")
            for f in sorted(files):
                if f.endswith(".json") and (f == "GPU-TIME.json" or os.path.basename(sub) == "receipts"):
                    add(os.path.join(sub, f), "finalist", name)
for path in sorted(glob.glob(f"{root}/m4-mlx/*/GPU-TIME.json")):
    add(path, "mlx-diag", os.path.basename(os.path.dirname(path)))

projection = {}
try:
    beat = json.load(open(f"{root}/m4-logs/heartbeat-{node}.json"))
    for arm in beat.get("arms", []):
        run = arm.get("running") or {}
        if run.get("container") and arm.get("projected_full_gpu_hours") is not None:
            projection[run["container"]] = arm["projected_full_gpu_hours"]
except (OSError, ValueError):
    pass
names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
for container in names.stdout.split():
    if not container.startswith(("d2-27b-M4-", "dev2-27b-gpu")):
        continue
    started = subprocess.run(["docker", "inspect", "-f", "{{.State.StartedAt}}", container],
                             capture_output=True, text=True).stdout.strip()
    start = datetime.fromisoformat(started[:26].rstrip("Z") + "+00:00") if started else None
    hours = (datetime.now(timezone.utc) - start).total_seconds() / 3600 if start else 0.0
    running.append({"node": node, "container": container, "hours": hours,
                    "projected_full_gpu_hours": projection.get(container)})
print(json.dumps({"node": node, "items": items, "relayed": relayed, "running": running}))
EOF
scan_a=$(on_a "python3 - '$R' a" <<< "$SCAN")
scan_b=$(on_b "python3 - '$R' b" <<< "$SCAN")
python3 - "$scan_a" "$scan_b" "${1:-}" <<'EOF'
import json, sys
from collections import defaultdict
CAP = 70.0
scans, as_json = [json.loads(sys.argv[1]), json.loads(sys.argv[2])], sys.argv[3] == "--json"
seen, items = {}, []
for scan in scans:
    for item in scan["items"]:
        key = item["file"]
        if key in seen:
            if seen[key]["sha256"] != item["sha256"]:
                raise SystemExit(f"{key} differs between node {seen[key]['node']} and node {item['node']}")
            continue
        seen[key] = item
        items.append(item)
owners = defaultdict(float)
for item in items:
    owners[(item["kind"], item["owner"], item["node"])] += item["gpu_hours"]
total = sum(i["gpu_hours"] for i in items)
running = [r for scan in scans for r in scan["running"]]
in_flight = sum(r["hours"] for r in running)
projected = sum(max(r["projected_full_gpu_hours"] or 0.0, r["hours"]) for r in running)
result = {
    "cap": CAP,
    "receipts_gpu_hours": total,
    "running_gpu_hours_so_far": in_flight,
    "running_projected_gpu_hours": projected,
    "left_now": CAP - total - in_flight,
    "M4_LEFT": CAP - total - projected,
    "items": items,
    "by_owner": [{"kind": k, "owner": o, "node": n, "gpu_hours": h} for (k, o, n), h in sorted(owners.items())],
    "relayed_copies_skipped": {s["node"]: s["relayed"] for s in scans},
    "running": running,
}
if as_json:
    print(json.dumps(result, indent=1, sort_keys=True))
    raise SystemExit(0)
for item in items:
    print(f"  node {item['node']}  {item['kind']:9s} {item['owner']:22s} {item['file']:70s} {item['gpu_hours']:8.3f}")
for entry in result["by_owner"]:
    print(f"{entry['kind']:9s} {entry['owner']:22s} node {entry['node']}  {entry['gpu_hours']:8.3f} GPU-h")
for r in running:
    projection = r["projected_full_gpu_hours"]
    print(f"running   {r['container']:40s} node {r['node']}  {r['hours']:6.2f} h so far"
          + (f", projected full attempt {projection:.2f}" if projection is not None else ""))
for node, arms in result["relayed_copies_skipped"].items():
    if arms:
        print(f"relayed copies on node {node} not counted there: {', '.join(arms)}")
print(f"receipts {total:.3f} GPU-h; running so far {in_flight:.3f}; cap {CAP:.0f}; left now {result['left_now']:.3f}")
print(f"M4_LEFT={result['M4_LEFT']:.2f}")
EOF
