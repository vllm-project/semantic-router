#!/usr/bin/env bash
# usage: chain-step.sh SHA GPU CHAIN STEP REMAINING_MIN -- command...
# One Milestone 6 chain step from the exact mirror of SHA: on GPU7 first waits while the eval
# track's C1 lease entry (gpu7.lock/owner.eval) is active; then rewrites the GPU lease owner as
# track=9b-m6 status=reserved (never "running", which the job wrapper refuses) with the milestone
# purpose and the step's expected end (a foreign owner file is kept once as
# owner.before-9b-m6), logs start / end UTC and the exit code to
# /data/dev2/runs/9b/m6/logs/CHAIN.log, runs the command and returns its exit code. A chain file
# is a bash script of such steps; start it with launch.sh.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; chain=$3; name=$4; remaining=$5; shift 5; [ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command for step $name" >&2; exit 2; }
L=$(code_dir "$sha")/v2/9b/lux9b/m6
log=$M6/logs/$chain.log
lock=$LEASES/gpu$gpu.lock
purpose="9B Milestone 6: successor arms + development readouts + formal runs"
[ -d "$L" ] || { echo "mirror $L missing" >&2; exit 2; }
mkdir -p "$M6/logs" "$lock"

note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log"; }

lease() {
  if [ -f "$lock/owner" ] && ! grep -qx "track=$TRACK" "$lock/owner" && [ ! -e "$lock/owner.before-$TRACK" ]; then
    cp "$lock/owner" "$lock/owner.before-$TRACK"
  fi
  printf "track=%s\nstatus=reserved\npurpose=%s\nchain=%s\nstep=%s\nexpected_end_utc=%s\nupdated_utc=%s\n" \
    "$TRACK" "$purpose" "$chain" "$name" "$(date -u -d "+$1 minutes" +%FT%TZ)" "$(date -u +%FT%TZ)" > "$lock/owner"
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
