#!/usr/bin/env bash
# ~27B M6 on node A (host side): wait for one arm-seed to finish (run_lora_arm.sh writes full/RUN_DIR after
# COMPLETE.json), then hardlink its BEST checkpoint into /data/dev2/xfer/27b-m6/relay/NAME/ with its SHA-256 list,
# BEST.json, COMPLETE.json and the arm-seed's receipts; BUDGET-nodeA.json sums node A's M6 GPU-hour receipts (the node B
# chain's budget check). Node B pulls it with `m6-tail.sh pull NAME` over the M6 node link. If the driver exits
# without a finished run, RELAY-FAILED.txt records it and nothing is relayed (no rerun).
# Usage: m6-relay.sh NAME DRIVER_PID     (NAME e.g. M6-IBX-s2; DRIVER_PID the pid m6-arm.sh printed). Detached.
set -euo pipefail
echo "m6 relay $*: start $(date -u +%FT%TZ)"
NAME=${1:?NAME} PID=${2:?DRIVER_PID}
[[ "$NAME" =~ ^M6-[A-Z0-9]+-s[12]$ ]] || { echo "NAME is an M6 arm-seed (M6-IBX-s2, ...)" >&2; exit 2; }
[[ "$PID" =~ ^[0-9]+$ ]] || { echo "DRIVER_PID must be a pid" >&2; exit 2; }
RUN=/data/dev2/runs/27b/$NAME RELAY=/data/dev2/xfer/27b-m6/relay/$NAME
[ -d "$RUN" ] || { echo "no arm-seed run $RUN" >&2; exit 2; }
[ ! -e "$RELAY" ] || { echo "$RELAY exists: refusing to overwrite" >&2; exit 66; }
mkdir -p "$(dirname "$RELAY")"
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
    json.dump({"node": "a", "gpu_hours": total, "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "items": sorted(items)}, out, indent=1)
print(f"node A M6 receipts {total} GPU-h ({len(items)} items)")
EOF
(cd "$RELAY.part/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$RELAY.part/SHA256SUMS.tmp"
mv "$RELAY.part/SHA256SUMS.tmp" "$RELAY.part/SHA256SUMS"
mv -T "$RELAY.part" "$RELAY"
echo "$(date -u +%FT%TZ) relay ready: $RELAY ($best, $(wc -l < "$RELAY/SHA256SUMS") files)"
