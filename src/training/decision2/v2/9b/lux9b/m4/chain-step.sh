#!/usr/bin/env bash
# usage: chain-step.sh SHA GPU CHAIN STEP REMAINING_MIN -- command...
# One hand-launched Milestone 4 chain step (node B GPU0-2, or a node-A GPU whose chain driver
# was stopped; prereg amendment 2), from the exact mirror of SHA: rewrites the GPU lease owner
# as status=reserved (never "running", which the job wrapper refuses) with the milestone purpose
# and the step's expected end, logs start / end UTC and the exit code to
# /data/dev2/runs/9b/m4/logs/CHAIN.log like chain-gpu6.sh / chain-gpu7.sh, runs the command and
# returns its exit code. Launch detached (setsid nohup) with D2_9B_GPUS set to the node's 9B GPUs.
set -uo pipefail
sha=$1; gpu=$2; chain=$3; name=$4; remaining=$5; shift 5; [ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no command for step $name" >&2; exit 2; }
L=/data/dev2/src/$sha-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4
M4=/data/dev2/runs/9b/m4
log=$M4/logs/$chain.log
lock=/data/dev2/leases/gpu$gpu.lock
purpose="9B Milestone 4: full fine-tuning K/P/KN + development readouts + formal runs"
[ -d "$L" ] || { echo "mirror $L missing" >&2; exit 2; }
mkdir -p "$M4/logs" "$lock"

note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log"; }

lease() {
  printf "track=9b-clm\nstatus=reserved\npurpose=%s\nchain=%s\nstep=%s\nexpected_end_utc=%s\nupdated_utc=%s\n" \
    "$purpose" "$chain" "$name" "$(date -u -d "+$1 minutes" +%FT%TZ)" "$(date -u +%FT%TZ)" > "$lock/owner"
}

lease "$remaining"
t0=$(date -u +%FT%TZ)
note "start step=$name sha=$sha gpu=$gpu cmd=$*"
"$@"
status=$?
note "end step=$name start=$t0 end=$(date -u +%FT%TZ) exit=$status"
lease 0
exit $status
