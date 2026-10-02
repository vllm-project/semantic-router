#!/usr/bin/env bash
# 9B M10: one preregistered seed on one M10 GPU (M9's seed runner): a zero-step run, a one-step run, the preflight_dec gate, and the full
# run only on PASS (no full run for a --zero-only seed such as B0). A failed stage stops the seed; nothing is rerun.
# With M10_PREWARM=1 the seed writes status/prewarm.DONE (status/$M10_PREWARM_MARK if set) after its one-step run (the
# 14:30 pre-warm convention: the other seeds wait for it). Outputs under /data/dev2/runs/9b/m10/arms/{pre,gates,full}/<run>.
#
# usage: M10_NODE=b arm.sh <run> <gpu> <mirror-dir> <start-path-in-container> [--zero-only] -- <train_dec args>
set -uo pipefail

run=$1 gpu=$2 src=$3 start=$4
shift 4
zero_only=0
[[ ${1:-} == --zero-only ]] && { zero_only=1; shift; }
[[ ${1:-} == -- ]] && shift
L=/data/dev2/src/$src/src/training/decision2/v2/9b/lux9b/m10/launch.sh
M=/data/dev2/runs/9b/m10
R=$M/arms
mkdir -p "$R/pre" "$R/gates" "$R/full" "$M/status"
log() { echo "$(date -u +%FT%TZ) $run $*" | tee -a "$R/OPERATIONS.log"; }
common=(--model-path "$start" --select /data/select.jsonl --cal /data/cal.jsonl --output /out --arm "$run" "$@")

bash "$L" "$run-zero" "$src" "$R/pre/$run-zero" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  --zero-step-only || { log "zero-step FAILED"; exit 1; }
log "zero-step done"
if [[ $zero_only == 1 ]]; then
  log "zero-only seed complete"
  exit 0
fi
bash "$L" "$run-one" "$src" "$R/pre/$run-one" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  --max-steps 1 || { log "one-step FAILED"; exit 1; }
log "one-step done"
[[ ${M10_PREWARM:-0} == 1 ]] && {
  date -u +%FT%TZ > "$M/status/${M10_PREWARM_MARK:-prewarm.DONE}"
  log "pre-warm marker written"
}
bash "$L" "$run-gate" "$src" "$R/gates/$run" --gpu "$gpu" -- -m v2.dec.preflight_dec --source-path "$start" \
  --select /data/select.jsonl --zero-run "/runs/m10/arms/pre/$run-zero" --one-run "/runs/m10/arms/pre/$run-one" \
  --output /out/preflight-receipt.json || { log "gate job FAILED"; exit 1; }
status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$R/gates/$run/preflight-receipt.json")
log "preflight $status"
[[ $status == PASS ]] || exit 1
bash "$L" "$run-full" "$src" "$R/full/$run" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  || { log "full run FAILED"; exit 1; }
log "full run complete"
