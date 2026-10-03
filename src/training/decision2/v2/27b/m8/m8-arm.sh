#!/usr/bin/env bash
# ~27B M8 arm-seed launcher (node host), run from its own mirror (prereg records/m8-prereg-2026-10-02.md): m7-arm.sh's
# recipe (run_lora_arm.sh stages admit, onestep, reload, full on L128's template, rank 128 / alpha 256, the frozen T0
# training cache, the M6 launcher allocation DEV2_27B_ALLOC=m6) with an existing M6 or M7 mixture (SHA-256 checked
# here) and new seeds, so the next cross-arm soup has more distinct members. The driver is detached; its PID is
# printed. One attempt per arm-seed. M8_RESUME=1 continues an interrupted arm-seed instead (its run directory copied
# from a node that left the pool, COORDINATION 2026-10-03 01:25, or a hung attempt stopped by hand): run_lora_arm.sh's
# exact resume from the latest complete checkpoint (stage full only, attempt 2 or 3: container and receipt full-rN),
# the same mixture, seed and caps.
# Usage: m8-arm.sh NODE GPU ARM SEED MIXTURE MIXTURE_SHA SAVE_EVERY CAP
#   NODE a (GPU1-7; M9) | b (GPU0-7; M9) | d (GPU0-7) | e (GPU0-3, 6-7; never GPU4-5) | f (GPU2-7; never GPU0-1)
#   ARM:MIXTURE  M8-IB:a20ib1 | M8-IB2:a20ib12 | M9-IB:a20ib1 | M9-IB2:a20ib12 (m6-data/mixtures-m6-1; M6-IB / M6-IB2's
#                files); M8-IB14:a20ib14 | M8-IB124:a20ib124 | M8-IB14ML:a20ib14ml | M8-IB124ML:a20ib124ml
#                (m7-data/mixtures-m7-1); M9-IB1ML:a20ib1ml | M9-IB12ML:a20ib12ml (m9-data/mixtures-m9-1, M9 prereg);
#                M9-IB-lrh:a20ib1 | M9-IB2-lrh:a20ib12 (M9 amendment 1: half LR, LoRA 1e-5 / head 5e-5 / backbone 5e-7);
#                M9-IB14ML-lrh:a20ib14ml | M9-IB124ML-lrh:a20ib124ml (M9 amendment 4: the same half LRs)
#   NODE c (GPU1-7; never GPU0) as well; SEED s3 | s4 | s5 | s6; CAP <= 22 GPU-h
set -euo pipefail
NODE=${1:?NODE} GPU=${2:?GPU} ARM=${3:?ARM} SEED=${4:?SEED} MIX=${5:?MIXTURE} MIX_SHA=${6:?MIXTURE_SHA}
SAVE=${7:?SAVE_EVERY} CAP=${8:?CAP}
S=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
case "$NODE:$GPU" in a:[1-7] | b:[0-7] | c:[1-7] | d:[0-7] | e:[0-3] | e:[67] | f:[2-7]) ;; *) echo "node $NODE GPU$GPU is outside M8's allocation" >&2; exit 2 ;; esac
LRS=(2e-5 1e-4 1e-6)
case "$ARM:$MIX" in
  M8-IB:a20ib1 | M8-IB2:a20ib12 | M9-IB:a20ib1 | M9-IB2:a20ib12) DATA=m6-data/mixtures-m6-1 ;;
  M9-IB-lrh:a20ib1 | M9-IB2-lrh:a20ib12) DATA=m6-data/mixtures-m6-1 LRS=(1e-5 5e-5 5e-7) ;;
  M9-IB14ML-lrh:a20ib14ml | M9-IB124ML-lrh:a20ib124ml) DATA=m7-data/mixtures-m7-1 LRS=(1e-5 5e-5 5e-7) ;;
  M8-IB14:a20ib14 | M8-IB124:a20ib124 | M8-IB14ML:a20ib14ml | M8-IB124ML:a20ib124ml) DATA=m7-data/mixtures-m7-1 ;;
  M9-IB1ML:a20ib1ml | M9-IB12ML:a20ib12ml) DATA=m9-data/mixtures-m9-1 ;;
  *) echo "unknown arm / mixture $ARM $MIX" >&2; exit 2 ;;
esac
case "$SEED" in
  s3) SEED_VALUE=20261002 ;; s4) SEED_VALUE=20261003 ;; s5) SEED_VALUE=20261004 ;; s6) SEED_VALUE=20261005 ;;
  *) echo "SEED is s3, s4, s5 or s6" >&2; exit 2 ;;
esac
[[ "$MIX_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "bad mixture SHA-256 $MIX_SHA" >&2; exit 2; }
if ! [[ "$SAVE" =~ ^[0-9]+$ ]] || ! python3 -c "import sys; sys.exit(0 if 0 < float(sys.argv[1]) <= 22 else 1)" "$CAP"; then
  echo "bad SAVE_EVERY $SAVE or CAP $CAP (at most 22)" >&2
  exit 2
fi
NAME=$ARM-$SEED
MIXFILE=/data/dev2/private/27b/$DATA/$MIX.train.jsonl
T0=/data/dev2/runs/27b/m4-train-cache-T0 T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
BASE=/data/decision20-20260926/models/Qwen3.8-27B REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
[ "$(sha256sum < "$MIXFILE" | cut -c1-64)" = "$MIX_SHA" ] || { echo "$MIXFILE is not $MIX_SHA" >&2; exit 2; }
[ -d "$BASE" ] && [ -d "$T0" ] || { echo "missing base or T0" >&2; exit 2; }
RUN=/data/dev2/runs/27b/$NAME STAGES=admit,onestep,reload,full ATTEMPT=1 MODE="seed $SEED_VALUE"
if [ "${M8_RESUME:-0}" = 1 ]; then
  # attempt 2 resumes the copy from the node that left; attempt 3 is run_lora_arm.sh's last recovery (a hung attempt
  # stopped by hand counts as an interruption, like exit 139)
  for ATTEMPT in 2 3 none; do [ ! -e "$RUN/receipts/full-r$ATTEMPT.json" ] && break; done
  [ -f "$RUN/receipts/reload.json" ] && [ ! -e "$RUN/full/run/COMPLETE.json" ] && [ "$ATTEMPT" != none ] ||
    { echo "$RUN is not an interrupted arm-seed (preflights passed, not complete, a recovery left)" >&2; exit 66; }
  latest=$(find "$RUN/full/run" -maxdepth 1 -type d -name 'checkpoint-*' ! -name '*.pending' | sort | tail -1)
  [ -n "$latest" ] || { echo "$RUN has no complete checkpoint to resume from" >&2; exit 66; }
  grep -q " $MIX ($MIX_SHA) seed $SEED_VALUE rank 128 alpha 256 save-every $SAVE cap $CAP " "$RUN/driver.log" ||
    { echo "$RUN/driver.log names another mixture, seed, save interval or cap" >&2; exit 66; }
  STAGES=full MODE="exact resume from $(basename "$latest") as attempt $ATTEMPT, seed $SEED_VALUE"
else
  [ ! -e "$RUN" ] || { echo "$RUN exists: one attempt per arm-seed" >&2; exit 66; }
fi
export TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE DEV2_27B_ALLOC=m6
mkdir -p "$RUN"
cd "$S"
echo "=== $(date -u +%FT%TZ) $NAME node $NODE GPU$GPU $MIX ($MIX_SHA) seed $SEED_VALUE rank 128 alpha 256" \
  "save-every $SAVE cap $CAP mirror $SRC lr ${LRS[*]}${M8_RESUME:+ ($MODE)}" >> "$RUN/driver.log"
LORA_LR=${LRS[0]} HEAD_LR=${LRS[1]} BACKBONE_LR=${LRS[2]} \
TRAIN_FILE=$MIXFILE SAVE_EVERY=$SAVE ARM_CAP=$CAP FULL_CAP=$CAP LORA_RANK=128 LORA_ALPHA=256 \
TRAIN_CACHE_FROZEN=$T0 TRAIN_CACHE_SHA=$T0_SHA TRITON_AUTOTUNE_CACHE=1 \
TRAIN_PYTHONPATH=/pipeline:/code:/opt/decision-fla STAGES=$STAGES ATTEMPT_START=$ATTEMPT \
setsid nohup bash v2/27b/run_lora_arm.sh "$NAME" "$GPU" "$BASE" . "$REV" "$SEED_VALUE" "$SRC" \
  >> "$RUN/driver.log" 2>&1 < /dev/null &
echo "launched $NAME on node $NODE GPU$GPU ($MODE; pid $!)"
