#!/usr/bin/env bash
# ~27B M7 arm-seed launcher (node host), run from its own mirror: m6-arm.sh's recipe (run_lora_arm.sh stages admit,
# onestep, reload, full on L128's template, rank 128 / alpha 256, the frozen T0 training cache, the M6 launcher
# allocation DEV2_27B_ALLOC=m6) with an M7 mixture of m7-data/mixtures-m7-1 (SHA-256 checked here) and a third seed for
# three-member soups. The driver is detached; its PID is printed. One attempt per arm-seed.
# Usage: m7-arm.sh NODE GPU ARM SEED MIXTURE MIXTURE_SHA SAVE_EVERY CAP
#   NODE b (GPU0) | d (GPU0-7); ARM M7-IB14ML | M7-IB124ML; SEED s1 | s2 | s3; MIXTURE a20ib14ml | a20ib124ml;
#   CAP <= 22 GPU-h
set -euo pipefail
NODE=${1:?NODE} GPU=${2:?GPU} ARM=${3:?ARM} SEED=${4:?SEED} MIX=${5:?MIXTURE} MIX_SHA=${6:?MIXTURE_SHA}
SAVE=${7:?SAVE_EVERY} CAP=${8:?CAP}
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
case "$NODE:$GPU" in b:0 | d:[0-7]) ;; *) echo "node $NODE GPU$GPU is outside M7's allocation" >&2; exit 2 ;; esac
case "$ARM:$MIX" in M7-IB14ML:a20ib14ml | M7-IB124ML:a20ib124ml) ;; *) echo "unknown arm / mixture $ARM $MIX" >&2; exit 2 ;; esac
case "$SEED" in s1) SEED_VALUE=20260926 ;; s2) SEED_VALUE=20260928 ;; s3) SEED_VALUE=20261002 ;; *) echo "SEED is s1, s2 or s3" >&2; exit 2 ;; esac
[[ "$MIX_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "bad mixture SHA-256 $MIX_SHA" >&2; exit 2; }
if ! [[ "$SAVE" =~ ^[0-9]+$ ]] || ! python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 22 else 1)" "$CAP"; then
  echo "bad SAVE_EVERY $SAVE or CAP $CAP (at most 22)" >&2
  exit 2
fi
NAME=$ARM-$SEED
MIXFILE=/data/dev2/private/27b/m7-data/mixtures-m7-1/$MIX.train.jsonl
T0=/data/dev2/runs/27b/m4-train-cache-T0 T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
BASE=/data/decision20-20260926/models/Qwen3.8-27B REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
[ "$(sha256sum < "$MIXFILE" | cut -c1-64)" = "$MIX_SHA" ] || { echo "$MIXFILE is not $MIX_SHA" >&2; exit 2; }
[ -d "$BASE" ] && [ -d "$T0" ] || { echo "missing base or T0" >&2; exit 2; }
[ ! -e "/data/dev2/runs/27b/$NAME" ] || { echo "/data/dev2/runs/27b/$NAME exists: one attempt per arm-seed" >&2; exit 66; }
export TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE DEV2_27B_ALLOC=m6
mkdir -p "/data/dev2/runs/27b/$NAME"
cd "$S"
echo "=== $(date -u +%FT%TZ) $NAME node $NODE GPU$GPU $MIX ($MIX_SHA) seed $SEED_VALUE rank 128 alpha 256" \
  "save-every $SAVE cap $CAP mirror $SRC" >> "/data/dev2/runs/27b/$NAME/driver.log"
TRAIN_FILE=$MIXFILE SAVE_EVERY=$SAVE ARM_CAP=$CAP FULL_CAP=$CAP LORA_RANK=128 LORA_ALPHA=256 \
TRAIN_CACHE_FROZEN=$T0 TRAIN_CACHE_SHA=$T0_SHA TRITON_AUTOTUNE_CACHE=1 \
TRAIN_PYTHONPATH=/pipeline:/code:/opt/decision-fla STAGES=admit,onestep,reload,full \
setsid nohup bash v2/27b/run_lora_arm.sh "$NAME" "$GPU" "$BASE" . "$REV" "$SEED_VALUE" "$SRC" \
  >> "/data/dev2/runs/27b/$NAME/driver.log" 2>&1 < /dev/null &
echo "launched $NAME on node $NODE GPU$GPU (pid $!)"
