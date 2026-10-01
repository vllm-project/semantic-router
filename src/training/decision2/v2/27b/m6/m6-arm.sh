#!/usr/bin/env bash
# ~27B M6 arm-seed launcher (node host), run from its own mirror: one arm-seed of run_lora_arm.sh (stages admit,
# onestep, reload, full) on L128's template (rank 128 / alpha 256) with the frozen M6 mixture (SHA-256 checked here),
# the training cache seeded from the frozen T0, SAVE_EVERY and the per-seed cap of the data lock, and the M6 launcher
# allocation (DEV2_27B_ALLOC=m6: node B GPU0 / GPU1 / GPU5, node A GPU2). The driver is detached (setsid, no inherited
# stdin or stdout); its PID is printed. One attempt per arm-seed.
# Usage: m6-arm.sh NODE GPU ARM SEED MIXTURE MIXTURE_SHA SAVE_EVERY CAP
#   NODE a|b|d; ARM M6-IB | M6-IBX | M6-IB2 | M6-IB2PN; SEED s1 | s2; MIXTURE a20ib1 | a20ib1x | <stage-2 name>
#   (a20ib12pn, amendment 3, lives in mixtures-m6pn-1); CAP <= 20 GPU-h, <= 22 for M6-IB2PN (amendment 3)
set -euo pipefail
NODE=${1:?NODE} GPU=${2:?GPU} ARM=${3:?ARM} SEED=${4:?SEED} MIX=${5:?MIXTURE} MIX_SHA=${6:?MIXTURE_SHA}
SAVE=${7:?SAVE_EVERY} CAP=${8:?CAP}
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
case "$NODE:$GPU" in b:0 | b:1 | b:5 | a:2 | d:[0-7]) ;; *) echo "node $NODE GPU$GPU is outside M6's allocation" >&2; exit 2 ;; esac
case "$ARM" in M6-IB | M6-IBX | M6-IB2) MAXCAP=20 ;; M6-IB2PN) MAXCAP=22 ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
case "$SEED" in s1) SEED_VALUE=20260926 ;; s2) SEED_VALUE=20260928 ;; *) echo "SEED is s1 or s2" >&2; exit 2 ;; esac
[[ "$MIX" =~ ^[a-z0-9]+$ ]] && [[ "$MIX_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "bad mixture $MIX / $MIX_SHA" >&2; exit 2; }
if ! [[ "$SAVE" =~ ^[0-9]+$ ]] || ! python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$CAP" "$MAXCAP"; then
  echo "bad SAVE_EVERY $SAVE or CAP $CAP (at most $MAXCAP for $ARM)" >&2
  exit 2
fi
NAME=$ARM-$SEED
case "$MIX" in *pn) MIXDIR=mixtures-m6pn-1 ;; *) MIXDIR=mixtures-m6-1 ;; esac
MIXFILE=/data/dev2/private/27b/m6-data/$MIXDIR/$MIX.train.jsonl
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
