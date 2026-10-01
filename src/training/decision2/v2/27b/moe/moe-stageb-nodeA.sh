#!/usr/bin/env bash
# 27B MoE Stage B, node A side (host; detached). Waits until both Gemma seeds have finished (COMPLETE.json) or ended
# without it, relays their BEST checkpoints and node A's receipt total to node B (moe-tail.sh relay), then waits for
# node B's mlx-diag collection of the soup (X/mlx/NAME.PUSHED on node B; NAME.SKIP ends the wait) and scores and pairs
# it on node A, where the mlx-diag gold lives (moe-tail.sh mlx-score). A seed that ends without COMPLETE.json is
# recorded (relay/STAGEB-ABSENT), never rerun; there is then no soup.
# Usage: moe-stageb-nodeA.sh MIRROR [NAME]
set -euo pipefail
MIR=$1 NAME=${2:-MOE-Git-soup}
S=/data/dev2/src/$MIR/src/training/decision2
R=/data/dev2/runs/27b-moe
KEY=/data/dev2/tmp/27b-moe-xfer
STAGE=/data/dev2/tmp/27b-moe-relay
TAIL=$S/v2/27b/moe/moe-tail.sh
ARMS=(MOE-Git-s1 MOE-Git-s2)
PEER=$(cat "$KEY/peer")
RS=(rsync -a --mkpath -e "ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes")
log() { echo "$(date -u +%FT%TZ) $*"; }
driver_ended() { [ -d "$R/$1" ] && flock -n "$R/$1/.driver.lock" true; }
[ -f "$TAIL" ] || { log "missing mirror $MIR"; exit 2; }
while :; do
  pending=0 failed=()
  for arm in "${ARMS[@]}"; do
    if [ -f "$R/$arm/full/run/COMPLETE.json" ]; then
      continue
    elif driver_ended "$arm"; then
      failed+=("$arm")
    else
      pending=1
    fi
  done
  [ "$pending" = 0 ] && break
  sleep 300
done
mkdir -p "$STAGE"
if [ "${#failed[@]}" -gt 0 ]; then
  log "ended without COMPLETE.json: ${failed[*]} (recorded, never rerun): no soup"
  echo "${failed[*]}" > "$STAGE/STAGEB-ABSENT"
  "${RS[@]}" "$STAGE/STAGEB-ABSENT" "root@$PEER:relay/STAGEB-ABSENT"
  exit 0
fi
log "both seeds finished; relaying the BEST checkpoints"
bash "$TAIL" relay "$MIR" "${ARMS[@]}"
log "waiting for node B's mlx-diag collection of $NAME"
mkdir -p /data/dev2/tmp/27b-moe-mlx
while :; do
  if "${RS[@]}" "root@$PEER:mlx/$NAME.SKIP" /data/dev2/tmp/27b-moe-mlx/ 2> /dev/null; then
    log "node B makes no mlx-diag collection: $(cat "/data/dev2/tmp/27b-moe-mlx/$NAME.SKIP")"
    exit 0
  fi
  "${RS[@]}" "root@$PEER:mlx/$NAME.PUSHED" /data/dev2/tmp/27b-moe-mlx/ 2> /dev/null && break
  sleep 300
done
log "mlx-diag collection pushed; scoring and pairing on node A"
bash "$TAIL" mlx-score "$MIR" "$NAME"
log "stage B node A done"
