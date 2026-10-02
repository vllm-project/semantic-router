#!/usr/bin/env bash
# Arm factory measurement pipeline (prereg / amendment 2 "Measured"), node side, detached. First GPUS go to the IX1
# harness: each arm-factory lease once its chain or merge has released it (absent leases too; another track's lease
# is never taken) becomes track=eval-ix1 idle. Then for each NAME in order: wait (<= 8 h) for soup/NAME/DONE (a FAILED
# soup is skipped), stage it unless staged (af-stage.sh: BF16 release copy, restage, checks), and start M10's
# ixchain.sh for it in the background (parity gate, greedy shards, scoring, family delta, the two paired bootstraps);
# the next NAME starts once every shard of this one has ended, so scoring and bootstraps overlap the next run.
# References: 4B IS-4b-LHA10SDML-bf16 (node C's run, or ix1/af/refs elsewhere); 9B K-a13IB-bf16 (node C's run, or
# ix1/m10/refs on node A).
#
# AF_LEASE_AFTER_SOUPS=1: take the leases only once every NAME's soup is built (when its LoRA merges use these GPUs).
#
# usage: AF_NODE=a|c|f af-measure.sh <mirror-dir> <PANEL> "<GPUS>" <SHARDS> <NAME>...
set -u
SRC=$1 PANEL=$2 GPUS=$3 SHARDS=$4
shift 4
NODE=${AF_NODE:?set AF_NODE}
case $NODE in a) SIZE=9b ;; *) SIZE=4b ;; esac
M=/data/dev2/runs/af/$SIZE
MIR=/data/dev2/src/$SRC
S=$MIR/src/training/decision2
OPS=$S/v2/af/ops
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/af
mkdir -p "$M/logs" "$R/logs"
log() { echo "$(date -u +%FT%TZ) measure $*" | tee -a "$M/OPERATIONS.log"; }
case $SIZE in
  4b) TAG=r13d42143 BIG=4B ref=$R/af/refs/IS-4b-LHA10SDML-bf16; [ "$NODE" = c ] && ref=$R/runs/IS-4b-LHA10SDML-bf16 ;;
  9b) TAG=re51f9881 BIG=9B ref=$R/m10/refs/K-a13IB-bf16; [ "$NODE" = c ] && ref=$R/runs/K-a13IB-bf16 ;;
esac
[ -f "$ref/merged/results.jsonl" ] || { log "no reference run $ref"; exit 1; }
if [ "${AF_LEASE_AFTER_SOUPS:-0}" = 1 ]; then  # soups that merge LoRA seeds on these GPUs finish first
  for NAME in "$@"; do
    n=0
    until [ -f "$M/soup/$NAME/DONE" ] || [ -f "$M/soup/$NAME/FAILED" ] || [ $n -gt 480 ]; do
      [ $((n % 30)) = 0 ] && log "$NAME: waits for its soup before the leases are taken"
      n=$((n + 1))
      sleep 60
    done
  done
fi
for g in $GPUS; do
  d=/data/dev2/leases/gpu$g.lock n=0
  mkdir -p "$d"
  until [ ! -s "$d/owner" ] || grep -qx 'track=eval-ix1' "$d/owner" \
    || { grep -q '^track=arm-factory' "$d/owner" && grep -qE '^status=(released|idle)' "$d/owner"; }; do
    grep -q '^track=arm-factory' "$d/owner" || { log "GPU$g is another track's; not used"; continue 2; }
    [ $((n % 30)) = 0 ] && log "GPU$g: waits for its arm-factory chain to release it"
    n=$((n + 1))
    [ $n -gt 480 ] && { log "GPU$g stayed busy; not used"; continue 2; }
    sleep 60
  done
  if [ -s "$d/owner" ] && ! grep -qx 'track=eval-ix1' "$d/owner"; then
    mv "$d/owner" "$d/owner.prev-af-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  [ -s "$d/owner" ] || printf 'track=eval-ix1\nstatus=idle\npurpose=IX1 arm factory private Index runs (COORDINATION 2026-10-02 22:00)\nstart_utc=%s\n' \
    "$(date -u +%FT%TZ)" > "$d/owner"
done
chains=()
for NAME in "$@"; do
  n=0
  until [ -f "$M/soup/$NAME/DONE" ] || [ -f "$M/soup/$NAME/FAILED" ]; do
    [ $((n % 30)) = 0 ] && log "$NAME: waits for its soup"
    n=$((n + 1))
    [ $n -gt 480 ] && break
    sleep 60
  done
  [ -f "$M/soup/$NAME/DONE" ] || { log "$NAME: no soup; not measured"; continue; }
  if [ ! -f "$MD/AF-$NAME-bf16-$TAG/MODEL_MANIFEST.json" ]; then
    AF_NODE=$NODE bash "$OPS/af-stage.sh" "$SRC" "$NAME" >> "$M/logs/stage-$NAME.log" 2>&1 \
      || { log "$NAME: staging FAILED (see logs/stage-$NAME.log)"; continue; }
  fi
  m=AF-$NAME-bf16
  log "$m: Index chain on $PANEL over GPUs '$GPUS' ($SHARDS shards, greedy)"
  M10_SHARDS=$SHARDS bash "$S/v2/9b/lux9b/m10/ixchain.sh" "$MIR" "$PANEL" "$GPUS" "$m=$BIG=$ref" \
    > "$R/logs/af-chain-$m.out" 2>&1 &
  chains+=("$!")
  until ! kill -0 "${chains[-1]}" 2> /dev/null || [ -f "$R/runs/$m/merged/receipt.json" ] \
    || python3 - "$R/runs/$m" "$SHARDS" << 'EOF'
import os, sys
run, n = sys.argv[1], int(sys.argv[2])
sys.exit(0 if all(os.path.exists(f"{run}/shard-{k}/end_epoch") for k in range(n)) else 1)
EOF
  do sleep 60; done
  log "$m: shards ended (scoring and bootstraps continue)"
done
for p in "${chains[@]}"; do wait "$p"; done
log "all Index chains finished: $*"
