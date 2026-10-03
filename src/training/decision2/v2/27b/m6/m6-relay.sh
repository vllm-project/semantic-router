#!/usr/bin/env bash
# ~27B M6 on node A (host side): wait for one arm-seed to finish (run_lora_arm.sh writes full/RUN_DIR after
# COMPLETE.json), then hardlink its BEST checkpoint into /data/dev2/xfer/27b-m6/relay/NAME/ with its SHA-256 list,
# BEST.json, COMPLETE.json and the arm-seed's receipts; BUDGET-nodeA.json sums node A's M6 GPU-hour receipts (the node B
# chain's budget check; BUDGET-noded.json on node D, RELAY_NODE=d; node E / F RELAY_NODE=e / f). Node B pulls it with
# `m6-tail.sh pull NAME` over the M6 node link (node A) or `m6-tail.sh pull-d|pull-e|pull-f NAME` over node B's
# transfer key. If the driver exits without a finished run, RELAY-FAILED.txt records it and nothing is relayed (no rerun),
# except for a seed stopped by its GPU-hour cap (the last full-stage receipt has watchdog_fired): the M8 / M9 prereg stop
# rule keeps its last BEST.json save, which is relayed with CAPPED.json in place of COMPLETE.json.
# Usage: m6-relay.sh NAME DRIVER_PID     (NAME an M6 / M7 / M8 / M9 arm-seed, e.g. M6-IBX-s2, M7-IB14ML-s1, M8-IB124-s4,
#        M9-IB2-lrh-s6; DRIVER_PID the pid the arm launcher printed). Detached.
set -euo pipefail
echo "m6 relay $*: start $(date -u +%FT%TZ)"
NAME=${1:?NAME} PID=${2:?DRIVER_PID}
[[ "$NAME" =~ ^M[6-9]-[A-Za-z0-9]+(-lrh)?-s[1-6]$ ]] ||
  { echo "NAME is an M6 / M7 / M8 / M9 arm-seed (M6-IBX-s2, M9-IB2-lrh-s6, ...)" >&2; exit 2; }
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "DRIVER_PID must be a pid" >&2; exit 2; }
RUN=${M6_RELAY_RUNS:-/data/dev2/runs/27b}/$NAME RELAY=${M6_RELAY_XFER:-/data/dev2/xfer/27b-m6/relay}/$NAME
NODE=${RELAY_NODE:-a}
case "$NODE" in a | c | d | e | f) ;; *) echo "RELAY_NODE is a, c, d, e or f" >&2; exit 2 ;; esac
[ -d "$RUN" ] || { echo "no arm-seed run $RUN" >&2; exit 2; }
[ ! -e "$RELAY" ] || { echo "$RELAY exists: refusing to overwrite" >&2; exit 66; }
mkdir -p "$(dirname "$RELAY")"
capped=""
while [ ! -f "$RUN/full/RUN_DIR" ]; do
  if ! kill -0 "$PID" 2> /dev/null; then
    sleep "${M6_RELAY_GRACE:-10}"
    [ -f "$RUN/full/RUN_DIR" ] && break
    capped=$(python3 - "$RUN" <<'EOF'
import glob, json, os, sys
run = sys.argv[1]
receipts = sorted(glob.glob(f"{run}/receipts/full*.json"), key=os.path.getmtime)
last = json.load(open(receipts[-1])) if receipts else {}
for d in sorted(glob.glob(f"{run}/full/run*"), key=os.path.getmtime, reverse=True):
    best = os.path.join(d, "BEST.json")
    if last.get("watchdog_fired") and os.path.isfile(best):
        ckpt = json.load(open(best))["checkpoint"]
        if os.path.isdir(os.path.join(d, ckpt)) and not os.path.exists(os.path.join(d, ckpt + ".pending")):
            print(os.path.basename(d), ckpt, os.path.basename(receipts[-1]))
        break
EOF
)
    [ -n "$capped" ] && break
    mkdir -p "$RELAY"
    { echo "$(date -u +%FT%TZ) $NAME: driver $PID ended without a finished run"; tail -n 20 "$RUN/driver.log"; } \
      > "$RELAY/RELAY-FAILED.txt"
    echo "no relay: $(head -n 1 "$RELAY/RELAY-FAILED.txt")"
    exit 3
  fi
  sleep 300
done
if [ -n "$capped" ]; then
  read -r rundir best receipt <<< "$capped"
  run=$RUN/full/$rundir
  echo "$(date -u +%FT%TZ) $NAME: stopped by its cap ($receipt); relaying its last BEST save $best (prereg stop rule)"
else
  run=$RUN/full/$(cat "$RUN/full/RUN_DIR")
  python3 - "$run" <<'EOF'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
best = json.loads((run / "BEST.json").read_text())["checkpoint"]
complete = json.loads((run / "COMPLETE.json").read_text())
if complete.get("status") != "complete" or complete.get("best") != best:
    raise SystemExit(f"{run} is not complete with a frozen BEST")
EOF
  best=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$run/BEST.json")
fi
mkdir -p "$RELAY.part"
cp -al "$run/$best" "$RELAY.part/checkpoint"
cp -p "$run/BEST.json" "$RELAY.part/"
if [ -n "$capped" ]; then
  python3 - "$RELAY.part/CAPPED.json" "$NAME" "$best" "$RUN/receipts/$receipt" <<'EOF'
import json, sys
from datetime import datetime, timezone
out, name, best, receipt = sys.argv[1:]
r = json.load(open(receipt))
with open(out, "x") as f:
    json.dump({"status": "capped", "arm_seed": name, "best": best, "receipt": receipt,
               "watchdog_fired": r.get("watchdog_fired"), "gpu_hours": r.get("gpu_hours"),
               "rule": "a seed stopped by its cap keeps its last BEST.json save (M8 / M9 prereg stop rules)",
               "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, f, indent=1)
EOF
else
  cp -p "$run/COMPLETE.json" "$RELAY.part/"
fi
cp -rp "$RUN/receipts" "$RELAY.part/receipts"
python3 - "$RELAY.part/BUDGET-node$NODE.json" "$NODE" <<'EOF'
import glob, json, os, sys
from datetime import datetime, timezone
seen, items = set(), []
paths = glob.glob("/data/dev2/runs/27b/M6-*/**/*.json", recursive=True)
paths += glob.glob("/data/dev2/runs/27b/m6/**/*.json", recursive=True)
for path in paths:
    real = os.path.realpath(path)
    if real in seen or "/relay/" in path or "/triton-cache" in path:
        continue
    seen.add(real)
    try:
        record = json.load(open(path))
    except Exception:
        continue
    if not isinstance(record, dict):
        continue
    if os.path.basename(path) == "GPU-TIME.json" or (
        os.path.basename(os.path.dirname(path)) == "receipts" and "gpu_hours" in record
    ):
        items.append([path, float(record.get("gpu_hours", 0))])
total = round(sum(h for _, h in items), 4)
with open(sys.argv[1], "x") as out:
    json.dump({"node": sys.argv[2], "gpu_hours": total, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "items": sorted(items)}, out, indent=1)
print(f"node {sys.argv[2]} M6 receipts {total} GPU-h ({len(items)} items)")
EOF
(cd "$RELAY.part/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$RELAY.part/SHA256SUMS.tmp"
mv "$RELAY.part/SHA256SUMS.tmp" "$RELAY.part/SHA256SUMS"
mv -T "$RELAY.part" "$RELAY"
echo "$(date -u +%FT%TZ) relay ready: $RELAY ($best, $(wc -l < "$RELAY/SHA256SUMS") files)"
