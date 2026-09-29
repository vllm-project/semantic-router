#!/usr/bin/env bash
# usage: chain-gpu6.sh SHA
# Milestone 4 GPU6 chain (node side, from the exact mirror of SHA): arm K seed 20260926 with
# preflights (K-s1), then only if it completed K-s2 (seed 1), then only if that completed K-s3
# (seed 2); a failed K step stops the chain. Each step's start / end UTC and exit code go to
# /data/dev2/runs/9b/m4/logs/chain-gpu6.log; between steps the GPU6 lease owner is rewritten as
# status=reserved (never "running", which the job wrapper refuses) with the milestone purpose and
# a refreshed expected end. Launch detached with D2_9B_GPUS="6 7" exported.
set -uo pipefail
sha=$1
gpu=6
L=/data/dev2/src/$sha-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4
M4=/data/dev2/runs/9b/m4
log=$M4/logs/chain-gpu$gpu.log
lock=/data/dev2/leases/gpu$gpu.lock
purpose="9B Milestone 4: full fine-tuning K/P/KN + development readouts + formal runs"
milestone_end=${D2_M4_EXPECTED_END:-2026-09-29T08:30:00Z}
chain_start=$(date -u +%FT%TZ)
export D2_9B_GPUS="${D2_9B_GPUS:-6 7}"
[ -d "$L" ] || { echo "mirror $L missing" >&2; exit 2; }
mkdir -p "$M4/logs"

note() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log"; }

lease() {
  local remaining=$1 projected end last
  projected=$(date -u -d "+$remaining minutes" +%FT%TZ)
  end=$milestone_end; [[ "$projected" > "$end" ]] && end=$projected
  last=$(grep -E '^(last_run|last_exit)=' "$lock/owner" 2>/dev/null | tr '\n' ' ')
  printf "track=9b-clm\nstatus=reserved\npurpose=%s\nchain=chain-gpu%s\nstart_utc=%s\nexpected_end_utc=%s\nlast=%s\nupdated_utc=%s\n" \
    "$purpose" "$gpu" "$chain_start" "$end" "$last" "$(date -u +%FT%TZ)" > "$lock/owner"
}

step() {
  local name=$1 remaining=$2 status t0; shift 2
  lease "$remaining"
  t0=$(date -u +%FT%TZ)
  note "start step=$name cmd=$*"
  "$@"
  status=$?
  note "end step=$name start=$t0 end=$(date -u +%FT%TZ) exit=$status"
  return $status
}

note "chain start sha=$sha gpu=$gpu"
if step K-s1 570 "$L/arm.sh" "$sha" $gpu K-s1 20260926 m4-k-xl-r2-60m --preflight --kl 1.0 \
  && step K-s2 360 "$L/arm.sh" "$sha" $gpu K-s2 1 m4-k-xl-r2-60m --kl 1.0 \
  && step K-s3 180 "$L/arm.sh" "$sha" $gpu K-s3 2 m4-k-xl-r2-60m --kl 1.0; then
  note "chain done: K chain complete"
else
  note "chain stopped: K chain step failed"
fi
lease 0
