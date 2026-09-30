#!/usr/bin/env bash
# shellcheck disable=SC2153  # SEEDS, START and the other tier tables come from m8s-lib.sh
# One decoder M8-small top-up seed on node B (prereg dec-m8s-prereg-2026-09-30.md, "Arms"): v2.dec.train_dec
# --init decision2 from the released start of the tier, one epoch over the tier's top-up file, one checkpoint at the
# final update. --preflight first runs the zero-step and one-update runs and v2.dec.preflight_dec (the start as
# reference), which must PASS. Stops at the first failed step; writes m8s/status/<run>.DONE | .FAILED.
# usage: m8s-run.sh <gpu> <tier> <D1|D2|C> <seed index 1-3> [--preflight]
set -uo pipefail
. "$(dirname "$0")/m8s-lib.sh"
GPU=$1 T=$2 ARM=$3 I=$4
PF=0
[ "${5:-}" = --preflight ] && PF=1
RUN=$T-$ARM-s$I SEED=${SEEDS[$((I - 1))]}
D=$M/arms/$RUN ST=$M/status
dec_env "$GPU" || exit 2
start=$(incontainer "${START[$T]}")
train=/runs/m8s/data/$T/topup/train.jsonl
teach=()
case $ARM in
  D1) teach=(--teacher "/runs/m8s/teacher/$T-D1/teacher.jsonl" --teacher-kl-weight "$LAMBDA") ;;
  D2) teach=(--teacher "/runs/m8s/teacher/$T-D2/teacher.jsonl" --teacher-kl-weight "$LAMBDA" --teacher-partial) ;;
  C) [ -n "${CKL[$T]}" ] && teach=(--teacher "/runs/m8s/teacher/$T-C/teacher.jsonl" --teacher-kl-weight "${CKL[$T]}") ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
TRAIN=(-m v2.dec.train_dec --init decision2 --model-path "$start" --train "$train" --select /data/select.jsonl
  --cal /data/cal.jsonl --arm "m8s-$RUN" --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05 --epochs 1
  --batching tokens --eval-batch 2 --max-length 8192 --checkpoint-schedule every --save-every 1000000
  --selection matrix-v1 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --max-batch-tokens 32768
  --max-batch-rows 64 --update-rows 64 --seed "$SEED" "${teach[@]}")
fail() { echo "$1; $(date -u +%FT%TZ)" > "$ST/$RUN.FAILED"; log "$RUN FAILED: $1"; lease "$GPU" idle "$RUN failed"; exit 1; }
job() {  # <name> <out dir> <purpose> -- <python args>
  local name=$1 out=$2 purpose=$3 rc
  shift 4
  [ -e "$out.launch.json" ] && { grep -q '"exit_status": 0' "$out.launch.json" && return 0; return 1; }
  lease "$GPU" busy "$purpose"
  bash "$LAUNCH" "m8s-$name" "$SRC" "$out" -- "$@"
  rc=$?
  gpu_seconds "$out.launch.json" "arm:$T-$ARM" "$purpose"
  return $rc
}
[ -f "$ST/$RUN.DONE" ] && { log "$RUN already DONE"; exit 0; }
[ -f "$ST/$RUN.FAILED" ] && { log "$RUN failed earlier; not rerun"; exit 1; }
mkdir -p "$D/pre"
wait_gpu "$GPU" 100
if [ $PF = 1 ]; then
  job "$RUN-zero" "$D/pre/zero" "$RUN preflight zero-step" -- "${TRAIN[@]}" --output /out --zero-step-only \
    || fail "preflight failed (zero-step)"
  job "$RUN-one" "$D/pre/one" "$RUN preflight one-update" -- "${TRAIN[@]}" --output /out --max-steps 1 \
    || fail "preflight failed (one-update)"
  job "$RUN-gate" "$D/gate" "$RUN preflight parity/reload" -- -m v2.dec.preflight_dec --source-path "$start" \
    --select /data/select.jsonl --zero-run "/runs/m8s/arms/$RUN/pre/zero" --one-run "/runs/m8s/arms/$RUN/pre/one" \
    --output /out/preflight-receipt.json || fail "preflight failed (gate job)"
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$D/gate/preflight-receipt.json")
  log "$RUN preflight $status"
  [ "$status" = PASS ] || fail "preflight failed ($status)"
fi
log "$RUN start on GPU$GPU (seed $SEED; ${teach[*]:-gold only})"
job "$RUN-full" "$D/full" "$RUN top-up training" -- "${TRAIN[@]}" --output /out || fail "full run failed"
[ -f "$D/full/COMPLETE.json" ] || fail "full run incomplete"
best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$D/full/BEST.json")
printf 'checkpoint=%s\nseed=%s\nfinished_utc=%s\n' "$D/full/$best" "$SEED" "$(date -u +%FT%TZ)" > "$ST/$RUN.DONE"
lease "$GPU" idle "$RUN done"
log "$RUN DONE ($best; arm GPU-h $(gpuh_item "arm:$T-$ARM"))"
