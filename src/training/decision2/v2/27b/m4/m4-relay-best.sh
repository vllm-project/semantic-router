#!/usr/bin/env bash
# ~27B M4 relay of one node-A arm-seed to node B (run on the workstation, since the nodes do not reach each other;
# gzip streams, the link runs at about 0.13-0.22 MB/s). Needs full/RUN/COMPLETE.json with a frozen BEST and a
# finished driver on node A (last driver.log line "arm ARM stages ... complete", no d2-27b-ARM- container). Only
# what the node-B soup and readout (run_finalist.sh best_of, lora_soup.py; the soup readout reads nothing of a
# member) and the records need goes to the same paths on node B:
#   full/RUN_DIR; full/RUN/{BEST,COMPLETE,LATEST,provenance}.json; full/RUN/select-step-*-metrics.json;
#   full/RUN/<BEST>/ without trainer_state.pt; driver.log; triton-cache.copy.json and .post.json (not the cache);
#   receipts/* stored as receipts.node-a/* (below).
# Idempotent: a node-B file with the same SHA-256 is kept, a differing one is never overwritten (exit 1). New files
# land in node B's ARM.relay-pending/ and move into place only once their SHA-256 matches. Then the per-file
# SHA-256 list of node B's ARM directory must equal node A's, with no other file but ARM/RELAY.json, which
# records the list (m4-tail.sh soup rehashes it before a soup). One summary line.
# GPU-hours: node A's receipts/ stay the only counted copy of this arm-seed's training. On node B they are kept as
# receipts.node-a/, a name no accounting reads: run_finalist.sh GPU-HOURS.json counts launch receipts only in
# directories named receipts under the soup's own directory (members are never under it), m4-status.sh reads
# ARM/receipts/ (node B's heartbeat shows 0 GPU-hours for a relayed arm), and m4-budget.sh skips an arm directory
# with RELAY.json. So node A's hours are neither double counted nor merged into a soup's GPU-HOURS.json.
# Usage: m4-relay-best.sh ARM   (M4-A20-s2, M4-Ar-s1 or M4-Ar-s2)
set -euo pipefail
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
ARM=${1:?ARM}
case "$ARM" in M4-A20-s2 | M4-Ar-s1 | M4-Ar-s2) ;; *) echo "$ARM does not train on node A" >&2; exit 2 ;; esac
D=/data/dev2/runs/27b/$ARM
P=$D.relay-pending
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }
start=$(date +%s)

source_list=$(on_a "python3 - '$D' '$ARM'" <<'EOF'
import glob, hashlib, json, os, subprocess, sys
root, arm = sys.argv[1:]
lines = [l for l in open(f"{root}/driver.log", encoding="utf-8").read().splitlines() if l.strip()]
if not (lines and lines[-1].startswith(f"arm {arm} stages ") and lines[-1].endswith(" complete")):
    raise SystemExit(f"{arm}: the driver has not finished (last driver.log line: {lines[-1] if lines else ''!r})")
running = subprocess.run(["docker", "ps", "-q", "--filter", f"name=d2-27b-{arm}-"], capture_output=True, text=True)
if running.returncode or running.stdout.strip():
    raise SystemExit(f"{arm}: a container is still running or docker ps failed")
run = open(f"{root}/full/RUN_DIR", encoding="utf-8").read().strip()
full = f"{root}/full/{run}"
best = json.load(open(f"{full}/BEST.json"))["checkpoint"]
complete = json.load(open(f"{full}/COMPLETE.json"))
if complete.get("status") != "complete" or complete.get("best") != best:
    raise SystemExit(f"{full} is not complete with a frozen BEST")
names = ["full/RUN_DIR", "driver.log", "triton-cache.copy.json", "triton-cache.post.json"]
names += [f"full/{run}/{n}.json" for n in ("BEST", "COMPLETE", "LATEST", "provenance")]
names += [os.path.relpath(p, root) for p in glob.glob(f"{full}/select-step-*-metrics.json")]
names += [os.path.relpath(p, root) for p in glob.glob(f"{root}/receipts/*")]
for folder, _, files in os.walk(f"{full}/{best}"):
    names += [os.path.relpath(os.path.join(folder, f), root) for f in files if f != "trainer_state.pt"]
out = {}
for name in sorted(set(names)):
    path = os.path.join(root, name)
    if os.path.islink(path) or not os.path.isfile(path):
        raise SystemExit(f"{path} is missing or not a regular file")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    out[name] = [digest.hexdigest(), os.path.getsize(path)]
print(json.dumps({"run_dir": run, "best": best, "files": out}))
EOF
)
plan=$(python3 - "$source_list" <<'EOF'
import json, re, sys
source = json.loads(sys.argv[1])
target = {}
for name, (sha, size) in source["files"].items():
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", name) or ".." in name.split("/"):
        raise SystemExit(f"unexpected file name {name!r}")
    dest = "receipts.node-a/" + name[len("receipts/"):] if name.startswith("receipts/") else name
    target[dest] = {"source": name, "sha256": sha, "bytes": size}
print(json.dumps({"run_dir": source["run_dir"], "best": source["best"], "files": target}))
EOF
)
state=$(on_b "python3 - '$D' '$P'" <<'EOF'
import hashlib, json, os, shutil, sys
root, pending = sys.argv[1:]
if os.path.isdir(pending):
    shutil.rmtree(pending)
present = {}
if os.path.isdir(root):
    for folder, _, files in os.walk(root):
        for f in files:
            path = os.path.join(folder, f)
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            present[os.path.relpath(path, root)] = digest.hexdigest()
print(json.dumps(present))
EOF
)
todo=$(python3 - "$plan" "$state" <<'EOF'
import json, sys
plan, present = json.loads(sys.argv[1]), json.loads(sys.argv[2])
files = plan["files"]
extra = sorted(set(present) - set(files) - {"RELAY.json"})
differ = sorted(n for n in files if n in present and present[n] != files[n]["sha256"])
if extra or differ:
    raise SystemExit(f"node B already holds other bytes: differing {differ}, unexpected {extra}; nothing overwritten")
print(" ".join(files[n]["source"] for n in sorted(files) if n not in present))
EOF
)
read -r -a missing <<< "$todo"
if [ ${#missing[@]} -gt 0 ]; then
  on_b "mkdir -p '$P'"
  on_a "cd '$D' && tar --transform 's,^receipts/,receipts.node-a/,' -czf - -- ${missing[*]}" | on_b "tar -C '$P' -xzf -"
  on_b "python3 - '$P' '$D' '$plan'" <<'EOF'
import hashlib, json, os, shutil, sys
pending, root, plan = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
for folder, _, files in os.walk(pending):
    for f in files:
        path = os.path.join(folder, f)
        name = os.path.relpath(path, pending)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if name not in plan["files"] or digest.hexdigest() != plan["files"][name]["sha256"]:
            raise SystemExit(f"relayed {name} does not match node A")
        target = os.path.join(root, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.link(path, target)
        os.unlink(path)
shutil.rmtree(pending)
EOF
fi
final=$(on_b "python3 - '$D' '$plan'" <<'EOF'
import hashlib, json, os, sys
from datetime import datetime, timezone
root, plan = sys.argv[1], json.loads(sys.argv[2])
expected = {n: f["sha256"] for n, f in plan["files"].items()}
present = {}
for folder, _, files in os.walk(root):
    for f in files:
        path = os.path.join(folder, f)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        present[os.path.relpath(path, root)] = digest.hexdigest()
relay = present.pop("RELAY.json", None)
if present != expected:
    raise SystemExit("node B's SHA-256 list differs from node A's after the relay")
path = os.path.join(root, "RELAY.json")
if relay is None:
    record = {
        "schema": "decision2-27b-m4-relay/1",
        "arm": os.path.basename(root),
        "source": "node A (same path), relayed through the workstation",
        "run_dir": plan["run_dir"],
        "best": plan["best"],
        "files_sha256": expected,
        "bytes": sum(f["bytes"] for f in plan["files"].values()),
        "renamed": {"receipts/": "receipts.node-a/"},
        "not_relayed": ["trainer_state.pt", "the other checkpoints", "triton-cache/", "admit/", "onestep/",
                        "pipeline/", "select predictions", "train-metrics.jsonl"],
        "gpu_hours": "counted from node A's receipts/ only; receipts.node-a/ is a copy that no accounting reads",
        "relayed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
        stream.write("\n")
elif json.load(open(path))["files_sha256"] != expected:
    raise SystemExit("RELAY.json records another file list")
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
print(json.dumps({"files": len(expected), "relay_sha256": digest}))
EOF
)
python3 - "$ARM" "$plan" "$final" "$todo" "$(($(date +%s) - start))" <<'EOF'
import json, sys
arm, plan, final, todo, seconds = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4].split(), sys.argv[5]
files = plan["files"].values()
total = sum(f["bytes"] for f in files)
sent = sum(f["bytes"] for f in files if f["source"] in todo)
print(f"{arm}: node A -> node B, BEST {plan['best']}, {final['files']} files ({total / 1e6:.1f} MB): "
      f"{len(todo)} sent ({sent / 1e6:.1f} MB), {final['files'] - len(todo)} already equal; SHA-256 lists "
      f"equal; RELAY.json {final['relay_sha256'][:12]}; {seconds} s")
EOF
