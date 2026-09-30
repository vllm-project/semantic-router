#!/usr/bin/env bash
# ~27B M4 training launcher (node host), run from its own mirror: one arm-seed, stages admit, onestep, reload
# and full of run_lora_arm.sh on the M3 template, with the LoRA size of its arm, the frozen mixture (SHA-256
# checked here), the training cache seeded from the frozen T0 (tree hash checked by run_lora_arm.sh) and the
# 13 GPU-hour cumulative cap per arm-seed. The driver is detached (setsid, no inherited stdin or stdout).
# Usage: m4-launch-arm.sh NODE ARM GPU MIXTURE SEED [STAGES]
#   NODE a|b; ARM M4-A20-s1 | M4-A20r-s2 | M4-Ar-s1 ...; MIXTURE a20|ar; SEED 20260926|20260928
#   ARM M4-xnode-A20-s1 (STAGES onestep only) repeats M4-A20-s1's first update on the other node, so its saved
#   adapter can be compared by SHA-256 with the node-B one (cross-node training probe; never trained further).
set -euo pipefail
NODE=$1 ARM=$2 GPU=$3 MIX=$4 SEED=$5 STAGES_ARG=${6:-admit,onestep,reload,full}
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
MX=/data/dev2/private/27b/m4-data/mixtures-m4-1
T0=/data/dev2/runs/27b/m4-train-cache-T0
T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
BASE=/data/decision20-20260926/models/Qwen3.8-27B
REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
case "$NODE:$GPU" in b:5 | b:6 | b:7 | a:2 | a:3 | a:4) ;; *) echo "node $NODE GPU$GPU is outside M4" >&2; exit 2 ;; esac
case "$ARM" in
  M4-A20-s[12]) RANK=8 ALPHA=16 WANT=a20 ;;
  M4-A20r-s[12]) RANK=32 ALPHA=64 WANT=a20 ;;
  M4-Ar-s[12]) RANK=8 ALPHA=16 WANT=ar ;;
  M4-xnode-A20-s1) RANK=8 ALPHA=16 WANT=a20
    [ "$STAGES_ARG" = onestep ] || { echo "the cross-node probe runs onestep only" >&2; exit 2; } ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
case "$ARM:$SEED" in *-s1:20260926 | *-s2:20260928) ;; *) echo "$ARM does not take seed $SEED" >&2; exit 2 ;; esac
case "$MIX" in
  a20) MIX_SHA=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4 SAVE=446 ;;
  ar) MIX_SHA=aadeef1a6a9e7aa8cbd7de743ef84c703c90433ac3ffba856edda24b4ad65e81 SAVE=503 ;;
  *) echo "unknown mixture $MIX" >&2; exit 2 ;;
esac
[ "$MIX" = "$WANT" ] || { echo "$ARM trains on $WANT, not $MIX" >&2; exit 2; }
[ "$(sha256sum < "$MX/$MIX.train.jsonl" | cut -c1-64)" = "$MIX_SHA" ] || { echo "$MIX.train.jsonl changed" >&2; exit 2; }
[ -d "$BASE" ] && [ -d "$T0" ] || { echo "missing base or T0" >&2; exit 2; }
[ ! -e "/data/dev2/runs/27b/$ARM" ] || { echo "/data/dev2/runs/27b/$ARM exists: one attempt per arm-seed" >&2; exit 66; }
export TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE
mkdir -p "/data/dev2/runs/27b/$ARM"
cd "$S"
echo "=== $(date -u +%FT%TZ) $ARM node $NODE GPU$GPU mixture $MIX ($MIX_SHA) seed $SEED rank $RANK alpha $ALPHA" \
  "save-every $SAVE mirror $SRC" >> "/data/dev2/runs/27b/$ARM/driver.log"
TRAIN_FILE=$MX/$MIX.train.jsonl SAVE_EVERY=$SAVE ARM_CAP=13.0 FULL_CAP=13.0 LORA_RANK=$RANK LORA_ALPHA=$ALPHA \
TRAIN_CACHE_FROZEN=$T0 TRAIN_CACHE_SHA=$T0_SHA TRITON_AUTOTUNE_CACHE=1 \
TRAIN_PYTHONPATH=/pipeline:/code:/opt/decision-fla STAGES=$STAGES_ARG \
setsid nohup bash v2/27b/run_lora_arm.sh "$ARM" "$GPU" "$BASE" . "$REV" "$SEED" "$SRC" \
  >> "/data/dev2/runs/27b/$ARM/driver.log" 2>&1 < /dev/null &
echo "launched $ARM on node $NODE GPU$GPU (pid $!)"
