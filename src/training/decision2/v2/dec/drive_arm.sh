#!/usr/bin/env bash
# One preregistered decoder arm on one allocated GPU: zero-step run, one-step
# run, preflight gate, the full run only on PASS, then CAL temperatures and one
# typed DEV + CSS pilot readout of the frozen BEST (postrun.sh).
#
# usage: DEC_NODE=a|b drive_arm.sh <arm> <gpu> <mirror-dir> <run-root> <start-path> -- <train_dec args>
#   <mirror-dir>  /data/dev2/src/<mirror-dir> (full or subtree mirror)
#   <run-root>    run root relative to /data/dev2/runs/dec (for example m2/arms)
#   <start-path>  model path inside the container (/hf/...)
# The trainer args must include --train; --model-path/--select/--cal/--output/--arm
# are supplied here. A failed stage stops the arm (it is recorded, not rerun).
# DEC_DATA_DIR (optional) replaces the node's default SELECT/CAL directory
# (mounted as /data; it must hold select.jsonl and cal.jsonl).
set -uo pipefail

arm=$1 gpu=$2 src=$3 rel=$4 start=$5
shift 5
[[ ${1:-} == -- ]] && shift
declare -A RENDER
case ${DEC_NODE:?set DEC_NODE=a or b} in
  a)
    RENDER=([5]=/dev/dri/renderD169)
    export DEC_IMAGE=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
    export DEC_DATA=${DEC_DATA_DIR:-/data/decision20-20260926/data/hf-private-decision20-clean-v2}
    ;;
  b)
    # GPU3-4 moved to the decoder track at 2026-09-28 16:30 UTC+8.
    RENDER=([0]=/dev/dri/renderD129 [1]=/dev/dri/renderD137 [2]=/dev/dri/renderD145
      [3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
    export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    export DEC_DATA=${DEC_DATA_DIR:-/data/dev2/runs/dec/data-cleanv2}
    ;;
  *) echo "unknown node $DEC_NODE" >&2; exit 2 ;;
esac
render=${RENDER[$gpu]:-}
[[ -n $render ]] || { echo "GPU $gpu is not allocated to the decoder track on node $DEC_NODE" >&2; exit 2; }
export DEC_RENDER=$render DEC_GPU_LABEL="node ${DEC_NODE^^} GPU$gpu"
S=/data/dev2/src/$src/src/training/decision2/v2/dec
R=/data/dev2/runs/dec/$rel
log() { echo "$(date -u +%FT%TZ) $arm $*" | tee -a "$R/OPERATIONS.log"; }
mkdir -p "$R/pre" "$R/gates" "$R/full"
common=(--model-path "$start" --select /data/select.jsonl --cal /data/cal.jsonl --output /out --arm "$arm" "$@")

bash "$S/launch.sh" "$arm-zero" "$src" "$R/pre/$arm-zero" -- -m v2.dec.train_dec "${common[@]}" --zero-step-only \
  || { log "zero-step FAILED"; exit 1; }
bash "$S/launch.sh" "$arm-one" "$src" "$R/pre/$arm-one" -- -m v2.dec.train_dec "${common[@]}" --max-steps 1 \
  || { log "one-step FAILED"; exit 1; }
bash "$S/launch.sh" "$arm-gate" "$src" "$R/gates/$arm" -- -m v2.dec.preflight_dec --source-path "$start" \
  --select /data/select.jsonl --zero-run "/runs/$rel/pre/$arm-zero" --one-run "/runs/$rel/pre/$arm-one" \
  --output /out/preflight-receipt.json || { log "gate job FAILED"; exit 1; }
status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$R/gates/$arm/preflight-receipt.json")
log "preflight $status"
[[ $status == PASS ]] || exit 1
bash "$S/launch.sh" "$arm-full" "$src" "$R/full/$arm" -- -m v2.dec.train_dec "${common[@]}" \
  || { log "full run FAILED"; exit 1; }
log "full run complete"
bash "$S/postrun.sh" "$src" "$rel/full/$arm" "$start" && log "postrun complete"
