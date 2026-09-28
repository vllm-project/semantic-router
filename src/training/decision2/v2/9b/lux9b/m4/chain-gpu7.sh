#!/usr/bin/env bash
# usage: chain-gpu7.sh SHA
# Milestone 4 GPU7 chain (node side, from the exact mirror of SHA): (a) the same-runtime Lux 1.0
# reference readout (m4/lux-16k); (b) the D line, each step independent (a failure is logged and
# the chain continues): alpha 1/4, 1/3, 2/3 interpolations of the Milestone 3 D soup toward Lux
# (D-a14, D-a13, D-a23) and fresh readouts of DW (D-a12) and the D soup (D-a1); (c) arm P seed
# 20260926 with preflights (P-s1), then whatever its outcome arm KN seed 20260926 with preflights
# (KN-s1), then only if that completed KN-s2 (seed 1). Each step's start / end UTC and exit code
# go to /data/dev2/runs/9b/m4/logs/chain-gpu7.log; between steps the GPU7 lease owner is
# rewritten as status=reserved (never "running", which the job wrapper refuses) with the milestone
# purpose and a refreshed expected end. Launch detached with D2_9B_GPUS="6 7" exported.
set -uo pipefail
sha=$1
gpu=7
L=/data/dev2/src/$sha-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4
M4=/data/dev2/runs/9b/m4
log=$M4/logs/chain-gpu$gpu.log
lock=/data/dev2/leases/gpu$gpu.lock
purpose="9B Milestone 4: full fine-tuning K/P/KN + development readouts + formal runs"
milestone_end=${D2_M4_EXPECTED_END:-2026-09-29T08:30:00Z}
chain_start=$(date -u +%FT%TZ)
DSOUP=/m3/D-soup-build/soup
DW=/m3/DW-build/soup
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
step lux-16k 750 "$L/lux_readout.sh" "$sha" $gpu
step D-a14 730 "$L/interp.sh" "$sha" $gpu D-a14 "$DSOUP" 1 4
step D-a13 700 "$L/interp.sh" "$sha" $gpu D-a13 "$DSOUP" 1 3
step D-a23 670 "$L/interp.sh" "$sha" $gpu D-a23 "$DSOUP" 2 3
step D-a12 640 "$L/readout.sh" "$sha" $gpu D-a12 "$DW"
step D-a1 620 "$L/readout.sh" "$sha" $gpu D-a1 "$DSOUP"
step P-s1 600 "$L/arm.sh" "$sha" $gpu P-s1 20260926 m4-k-xl-r2-60m --preflight --no-teacher
if step KN-s1 390 "$L/arm.sh" "$sha" $gpu KN-s1 20260926 m4-kn-xl-r2-nohum-60m --preflight --kl 1.0 \
  && step KN-s2 180 "$L/arm.sh" "$sha" $gpu KN-s2 1 m4-kn-xl-r2-nohum-60m --kl 1.0; then
  note "chain done: KN chain complete"
else
  note "chain stopped: KN chain step failed"
fi
lease 0
