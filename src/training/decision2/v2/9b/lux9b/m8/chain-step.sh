#!/usr/bin/env bash
# usage: chain-step.sh SHA GPU CHAIN STEP REMAINING_MIN -- command...
# One Milestone 8 chain step from the exact mirror of SHA: writes this track's lease entry
# gpuN.lock/owner.9b-m8 as track=9b-m8 status=reserved (never "running", which the job wrapper
# refuses) with the milestone purpose and the step's expected end; the ~27B owner entry
# gpuN.lock/owner is never rewritten. Logs start / end UTC and the exit code to
# /data/dev2/runs/9b/m8/logs/CHAIN.log, runs the command and returns its exit code. A chain file is
# a bash script of such steps; start it with launch.sh.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; chain=$3; name=$4; remaining=$5; shift 5; [ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command for step $name" >&2; exit 2; }
L=$(code_dir "$sha")/v2/9b/lux9b/m8
log=$M8/logs/$chain.log
lock=$LEASES/gpu$gpu.lock
purpose="9B Milestone 8: DEV2.0-27B distillation top-ups + development readouts + formal runs (GPU lent by 27B)"
[ -d "$L" ] || { echo "mirror $L missing" >&2; exit 2; }
mkdir -p "$M8/logs" "$lock"

note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log"; }

lease() {
  printf "track=%s\nstatus=reserved\npurpose=%s\nchain=%s\nstep=%s\nexpected_end_utc=%s\nupdated_utc=%s\n" \
    "$TRACK" "$purpose" "$chain" "$name" "$(date -u -d "+$1 minutes" +%FT%TZ)" "$(date -u +%FT%TZ)" > "$lock/$LEASE_ENTRY"
}

yield_gpu7 "$gpu" "$log"
lease "$remaining"
t0=$(date -u +%FT%TZ)
note "start step=$name sha=$sha gpu=$gpu cmd=$*"
"$@"
status=$?
note "end step=$name start=$t0 end=$(date -u +%FT%TZ) exit=$status"
lease 0
exit $status
