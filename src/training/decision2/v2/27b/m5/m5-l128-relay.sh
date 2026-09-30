#!/usr/bin/env bash
# ~27B M5 branch B1 on node A (host side): wait for one L128 arm-seed to finish (run_lora_arm.sh writes full/RUN_DIR
# after COMPLETE.json), then hardlink its BEST checkpoint into the relay directory /data/dev2/xfer/27b-m5/relay/NAME/
# with its SHA-256 list, BEST.json, COMPLETE.json and the arm-seed's receipts, as m5-lane.sh does for the FF seeds;
# node B pulls it with m5-pull.sh. BUDGET-nodeA.json there sums node A's M5 GPU-hour receipts (the chain's budget
# check). If the driver exits without a finished run, RELAY-FAILED.txt records it and nothing is relayed (no rerun).
# Usage: m5-l128-relay.sh NAME DRIVER_PID     (NAME M5-L128-s2; DRIVER_PID the pid m5-l128.sh printed). Detached.
set -euo pipefail
echo "m5 l128 relay $*: start $(date -u +%FT%TZ)"
NAME=${1:?NAME} PID=${2:?DRIVER_PID}
case "$NAME" in M5-L128-s1 | M5-L128-s2) ;; *) echo "NAME is M5-L128-s1 or M5-L128-s2" >&2; exit 2 ;; esac
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "DRIVER_PID must be a pid" >&2; exit 2; }
RUN=/data/dev2/runs/27b/$NAME RELAY=/data/dev2/xfer/27b-m5/relay/$NAME
[ -d "$RUN" ] || { echo "no arm-seed run $RUN" >&2; exit 2; }
[ ! -e "$RELAY" ] || { echo "$RELAY exists: refusing to overwrite" >&2; exit 66; }
while [ ! -f "$RUN/full/RUN_DIR" ]; do
  if ! kill -0 "$PID" 2> /dev/null; then
    sleep 10
    [ -f "$RUN/full/RUN_DIR" ] && break
    mkdir -p "$RELAY"
    { echo "$(date -u +%FT%TZ) $NAME: driver $PID ended without a finished run"; tail -n 20 "$RUN/driver.log"; } \
      > "$RELAY/RELAY-FAILED.txt"
    echo "no relay: $(head -n 1 "$RELAY/RELAY-FAILED.txt")"
    exit 3
  fi
  sleep 300
done
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
mkdir -p "$RELAY.part"
cp -al "$run/$best" "$RELAY.part/checkpoint"
cp -p "$run/BEST.json" "$run/COMPLETE.json" "$RELAY.part/"
cp -rp "$RUN/receipts" "$RELAY.part/receipts"
python3 - "$RELAY.part/BUDGET-nodeA.json" <<'EOF'
import glob, json, os, sys
from datetime import datetime, timezone
seen, items = set(), []
paths = glob.glob("/data/dev2/runs/27b/m5/**/*.json", recursive=True)
paths += glob.glob("/data/dev2/runs/27b/M5-L128-s*/**/*.json", recursive=True)
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
    json.dump({"node": "a", "gpu_hours": total, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "items": sorted(items)}, out, indent=1)
print(f"node A M5 receipts {total} GPU-h ({len(items)} items)")
EOF
(cd "$RELAY.part/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$RELAY.part/SHA256SUMS.tmp"
mv "$RELAY.part/SHA256SUMS.tmp" "$RELAY.part/SHA256SUMS"
mv -T "$RELAY.part" "$RELAY"
echo "$(date -u +%FT%TZ) relay ready: $RELAY ($best, $(wc -l < "$RELAY/SHA256SUMS") files)"
