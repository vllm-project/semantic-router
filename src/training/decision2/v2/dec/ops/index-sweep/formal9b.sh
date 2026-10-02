#!/usr/bin/env bash
# Index sweep: the formal post-key panel of one 9B M9 candidate on the 9B formal path (v2/9b/lux9b/m9/formal.sh:
# CAL698 fit, smoke, typed FINAL + CSS15 + public 231 at 16,384 tokens, mlx-diag, seal, report, the 9B track's
# comparisons) on node A GPU3, a GPU this sweep holds (M9_FORMAL_GPUS=3). It is run once per candidate that ships
# (Index-first integrity item 3, the release parity source and the card's JevArena values). Node side, detached:
#   1. waits (<= 3 h) until GPU3 is idle and its owner file names this sweep (track=eval-ix1);
#   2. hands GPU3 to the formal path: owner track=9b-m9 (purpose names the sweep and NAME), the old file kept;
#   3. formal.sh 3 NAME CHECKPOINT SOURCE LABEL > formal-m9/logs/isweep-formal-NAME.log;
#   4. GPU3 owner -> track=eval-ix1 status=released.
# usage: formal9b.sh NAME   (NAME: K-a12IB | L9IB; SOURCE = the Lux 1.0 package, as both soups' M9 readouts used)
set -uo pipefail
NAME=${1:?NAME}
OWN=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
case $NAME in
  K-a12IB) CKPT=/data/dev2/runs/9b/m9/soup/K-a12IB/build/K-a12IB SOURCE=/data/decision20-20260926/models/Decision-1.0-Lux-9B ;;
  L9IB) CKPT=/data/dev2/runs/9b/m9/soup/L9IB/build/L9IB-soup SOURCE=/data/decision20-20260926/models/Decision-1.0-Lux-9B ;;
  *) echo "unknown NAME $NAME" >&2; exit 2 ;;
esac
F=/data/dev2/runs/9b/formal-m9
LOG=$F/logs/isweep-formal-$NAME.log
L=/data/dev2/leases/gpu3.lock
mkdir -p "$F/logs" "$L"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
idle3() {
  rocm-smi --showuse --showmeminfo vram --json | python3 -c '
import json, sys
c = json.load(sys.stdin)["card3"]
sys.exit(0 if float(c["GPU use (%)"]) <= 5 and int(c["VRAM Total Used Memory (B)"]) <= 2 * 2**30 else 1)'
}
t=0
until idle3 && grep -qx 'track=eval-ix1' "$L/owner" 2>/dev/null; do
  (( t == 0 )) && log "waiting for GPU3 (idle, owner track=eval-ix1)"
  (( t < 10800 )) || { log "GPU3 not available after 3 h; stop"; exit 1; }
  sleep 30
  t=$((t + 30))
done
cp -p "$L/owner" "$L/owner.prev-$(date -u +%Y%m%dT%H%M%SZ)"
printf 'track=9b-m9\npurpose=Index sweep: formal post-key panel of the 9B M9 candidate %s on the 9B formal path (GPU3 is the sweep'"'"'s)\nstart_utc=%s\nexpected_end_utc=%s\n' \
  "$NAME" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date -u -d '+2 hours' +%Y-%m-%dT%H:%M:%SZ)" > "$L/owner"
log "GPU3 -> track=9b-m9; formal $NAME from $OWN"
M9_FORMAL_GPUS=3 M9_NODE=a bash "$OWN/v2/9b/lux9b/m9/formal.sh" 3 "$NAME" "$CKPT" "$SOURCE" "9B M9 $NAME" >> "$LOG" 2>&1
rc=$?
log "formal $NAME exit $rc"
printf 'track=eval-ix1\nstatus=released (Index sweep: formal of %s done, exit %s)\nlast_job_end_utc=%s\n' "$NAME" "$rc" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$L/owner"
exit "$rc"
