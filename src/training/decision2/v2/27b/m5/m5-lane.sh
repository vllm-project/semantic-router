#!/usr/bin/env bash
# ~27B M5 training lane (host side): full-parameter arm-seeds, one after another on one lane's three GPUs, through
# v2/27b/m4b/run_ff_arm.sh (M4b's FSDP2 driver and recipe, unchanged) with Milestone 5's launch3 allocation.
# Usage: m5-lane.sh NODE GPUS MIRROR_SHA ARM:SEED[,ARM:SEED...]
#   NODE   a | b (launch3 allocation m5-a / m5-b, track 27b)
#   GPUS   the lane's three GPUs: 5,6,7 on node B or 2,3,4 on node A
#   ARM    FF20 (a20), FF20H (a20h) or FF20X (cross-node probe: FF20-s1's exact one-step on node A, onestep only)
# Caps (preregistration): FF20 full attempt 13.5 / arm-seed 14.1 GPU-h; FF20H 17.5 / 18.1. Each arm-seed stops at its
# driver's first failed stage (no rerun). /data/dev2/runs/27b/m5/STOP-<ARM> (branch rule B1, budget rule G1) makes the
# lane skip every later arm-seed of that arm; a node A lane hardlinks each finished BEST checkpoint into the relay
# directory /data/dev2/xfer/27b-m5/relay/<ARM>-<SEED>/ with a SHA-256 list, for node B to pull.
set -uo pipefail
echo "m5 lane $*: start $(date -u +%FT%TZ)"
NODE=${1:?NODE} GPUS=${2:?GPUS} SHA=${3:?MIRROR_SHA} PLAN=${4:?ARM:SEED list}
case "$NODE" in a | b) ;; *) echo "NODE must be a or b" >&2; exit 2 ;; esac
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
CODE=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$CODE/v2/27b/m4b/run_ff_arm.sh" ] || { echo "missing mirror $SHA" >&2; exit 2; }
DATA=/data/dev2/private/27b/m5-data/mixtures-m5-1
ROOT=/data/dev2/runs/27b/m5
mkdir -p "$ROOT"
export DEV2_27B_LAUNCH_ALLOC=m5-$NODE FF_ROOT=$ROOT FF_LABEL=m5 FF_PREFIX=d2-27b-m5 GPUS
export SEED_CACHE=/data/dev2/runs/27b/m4-train-cache-T0
export SEED_CACHE_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
export SELECT_FILE=/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl

check_sha() {  # PATH SHA256
  local got
  got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "$1 is $got, expected $2" >&2; return 1; }
  echo "verified $1 $got"
}
check_sha "$SELECT_FILE" 32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6 || exit 2
IFS=, read -r -a items <<< "$PLAN"
for item in "${items[@]}"; do
  ARM=${item%%:*} SEED=${item#*:}
  if [ -e "$ROOT/STOP-$ARM" ]; then
    echo "$(date -u +%FT%TZ) $ARM-$SEED skipped: $(cat "$ROOT/STOP-$ARM")"
    continue
  fi
  case "$ARM" in
    FF20 | FF20X) TRAIN=$DATA/a20.train.jsonl TSHA=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
      FULL_CAP=13.5 ARM_CAP=14.1 ;;
    FF20H) TRAIN=$DATA/a20h.train.jsonl TSHA=4a9d93f56e5dc4f5715dfe7c310375199484546b51d452dc94b6ca0332306202
      FULL_CAP=17.5 ARM_CAP=18.1 ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
  esac
  STAGES=onestep,reload,full
  [ "$ARM" = FF20X ] && STAGES=onestep
  check_sha "$TRAIN" "$TSHA" || exit 2
  echo "$(date -u +%FT%TZ) arm-seed $ARM-$SEED on node $NODE GPU $GPUS: stages $STAGES, caps $FULL_CAP / $ARM_CAP"
  status=0
  TRAIN_FILE=$TRAIN FULL_CAP=$FULL_CAP ARM_CAP=$ARM_CAP STAGES=$STAGES \
    bash "$CODE/v2/27b/m4b/run_ff_arm.sh" "$ARM" "$SEED" "$SHA" || status=$?
  echo "$(date -u +%FT%TZ) arm-seed $ARM-$SEED exit $status"
  [ "$status" = 0 ] || { echo "lane stopped at $ARM-$SEED"; exit "$status"; }
  if [ "$NODE" = a ] && [ "$ARM" != FF20X ]; then
    run=$ROOT/$ARM-$SEED/full/$(cat "$ROOT/$ARM-$SEED/full/RUN_DIR")
    best=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$run/BEST.json")
    relay=/data/dev2/xfer/27b-m5/relay/$ARM-$SEED
    mkdir -p "$relay"
    cp -al "$run/$best" "$relay/checkpoint"
    cp -p "$run/BEST.json" "$run/COMPLETE.json" "$relay/"
    (cd "$relay/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$relay/SHA256SUMS"
    echo "$(date -u +%FT%TZ) relay ready: $relay ($best, $(wc -l < "$relay/SHA256SUMS") files)"
  fi
done
echo "m5 lane $NODE $GPUS complete: $(date -u +%FT%TZ)"
