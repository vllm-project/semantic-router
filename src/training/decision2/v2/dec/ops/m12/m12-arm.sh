#!/usr/bin/env bash
# Decoder M12: one preregistered seed on one M12 GPU (drive_arm.sh's sequence without its postrun): a zero-step run,
# a one-step run, the preflight_dec gate, and the full run only on PASS. A failed stage stops the seed; nothing is
# rerun. Outputs under /data/dev2/runs/dec/m12/arms/{pre,gates,full}/<run>. With M12_WARM_MARKER set (the pre-warm
# seed of its node and tier), the marker file is written once the preflight stages have finished, pass or fail, so
# the waiting seeds of that tier start only after this job filled the cache alone.
#
# usage: M12_NODE=e|f M12_CACHE=<tier>-train [M12_WARM_MARKER=<file>] m12-arm.sh <run> <gpu> <mirror-dir> \
#          <start-path-in-container> -- <train_dec args incl. --train>
set -uo pipefail

run=$1 gpu=$2 src=$3 start=$4
shift 4
[[ ${1:-} == -- ]] && shift
L=/data/dev2/src/$src/src/training/decision2/v2/dec/ops/m12/m12-launch.sh
R=/data/dev2/runs/dec/m12/arms
mkdir -p "$R/pre" "$R/gates" "$R/full"
log() { echo "$(date -u +%FT%TZ) $run $*" | tee -a "$R/OPERATIONS.log"; }
warm() { [[ -n ${M12_WARM_MARKER:-} ]] && echo "$run $1 $(date -u +%FT%TZ)" > "$M12_WARM_MARKER"; return 0; }
common=(--model-path "$start" --select /data/select.jsonl --cal /data/cal.jsonl --output /out --arm "$run" "$@")

bash "$L" "$run-zero" "$src" "$R/pre/$run-zero" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  --zero-step-only || { log "zero-step FAILED"; warm failed; exit 1; }
bash "$L" "$run-one" "$src" "$R/pre/$run-one" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  --max-steps 1 || { log "one-step FAILED"; warm failed; exit 1; }
bash "$L" "$run-gate" "$src" "$R/gates/$run" --gpu "$gpu" -- -m v2.dec.preflight_dec --source-path "$start" \
  --select /data/select.jsonl --zero-run "/runs/m12/arms/pre/$run-zero" --one-run "/runs/m12/arms/pre/$run-one" \
  --output /out/preflight-receipt.json || { log "gate job FAILED"; warm failed; exit 1; }
status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$R/gates/$run/preflight-receipt.json")
log "preflight $status"
warm "$status"
[[ $status == PASS ]] || exit 1
bash "$L" "$run-full" "$src" "$R/full/$run" --gpu "$gpu" -- -m v2.dec.train_dec "${common[@]}" \
  || { log "full run FAILED"; exit 1; }
log "full run complete"
