#!/usr/bin/env bash
# Reasoning track: one preregistered seed on one GPU: a zero-step run, a one-step run, the preflight_dec gate, then
# the full run only on PASS. A failed stage stops the seed with a FAILED marker; nothing is rerun.
# Outputs under /data/dev2/runs/reasoning/arms/{pre,gates,full}/<run>; markers in .../arms/status/<run>.{DONE,FAILED}.
#
# usage: rsn-arm.sh <run> <gpu> <cpus> <mirror-dir> <image 4b|9b> <start-in-container> -- <train_dec args>
set -uo pipefail
run=$1 gpu=$2 cpus=$3 src=$4 image=$5 start=$6
shift 6
[[ ${1:-} == -- ]] && shift
L=/data/dev2/src/$src/src/training/decision2/v2/reasoning/ops/rsn-launch.sh
R=/data/dev2/runs/reasoning/arms
mkdir -p "$R/pre" "$R/gates" "$R/full" "$R/status"
log() { echo "$(date -u +%FT%TZ) $run $*" | tee -a "$R/OPERATIONS.log"; }
fail() { log "$1"; echo "$1" > "$R/status/$run.FAILED"; exit 1; }
[[ -e $R/status/$run.DONE || -e $R/status/$run.FAILED ]] && { log "already ended"; exit 1; }
sel=${RSN_SEL:-/af/4b/inputs/dec/m10/inputs/sel700-cal698}
common=(--model-path "$start" --select "$sel/select.jsonl" --cal "$sel/cal.jsonl" --output /out --arm "$run" "$@")
opts=(--gpu "$gpu" --cpus "$cpus" --image "$image")
bash "$L" "$run-zero" "$src" "$R/pre/$run-zero" "${opts[@]}" -- -m v2.dec.train_dec "${common[@]}" --zero-step-only \
  || fail "zero-step FAILED"
bash "$L" "$run-one" "$src" "$R/pre/$run-one" "${opts[@]}" -- -m v2.dec.train_dec "${common[@]}" --max-steps 1 \
  || fail "one-step FAILED"
bash "$L" "$run-gate" "$src" "$R/gates/$run" "${opts[@]}" -- -m v2.dec.preflight_dec --source-path "$start" \
  --select "$sel/select.jsonl" --zero-run "/rsn/arms/pre/$run-zero" --one-run "/rsn/arms/pre/$run-one" \
  --output /out/preflight-receipt.json || fail "gate job FAILED"
status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$R/gates/$run/preflight-receipt.json")
log "preflight $status"
[[ $status == PASS ]] || fail "preflight $status"
bash "$L" "$run-full" "$src" "$R/full/$run" "${opts[@]}" -- -m v2.dec.train_dec "${common[@]}" || fail "full run FAILED"
log "full run complete"
date -u +%FT%TZ > "$R/status/$run.DONE"
