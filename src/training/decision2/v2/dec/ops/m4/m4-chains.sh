#!/usr/bin/env bash
# Decoder M4 launcher (prereg dec-m4-prereg-2026-09-29.md): five per-GPU chains on node B GPU0-4 and one
# soup watcher per arm. Each chain step waits for its mixture's READY marker, which is written only after
# the data lock naming that mixture's train / teacher hashes is committed. A mkdir lock makes a repeated
# invocation a no-op (the M3 duplicate-launch incident).
# usage: m4-chains.sh <mirror-dir>
set -u
SRC=$1
M=/data/dev2/runs/dec/m4
A=$M/arms C=$M/chains
mkdir -p "$A" "$C" "$M/soup"
mkdir "$C/launch.lock" 2>/dev/null || { echo "M4 chains already launched"; exit 0; }
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m4
D=/data/dev2/src/$SRC/src/training/decision2/v2/dec/drive_arm.sh
NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
BASE="--train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
declare -A MIX=([N4LR]=m4-v2m-ret-r2 [N4LR2]=m4-v2m-ret-r2 [N4LRQ]=m4-v2m-ret-r2-q20 [N4XA]=m4-xl-a7v1-29m [N4XF]=m4-xl-full-29m)
declare -A KL=([N4LR]=1.0 [N4LR2]=2.0 [N4LRQ]=1.0 [N4XA]=1.0 [N4XF]=1.0)
declare -A EXTRA=([N4XF]=--teacher-partial)
declare -A SEED=([1]=20260926 [2]=20260927 [3]=20260928)
step() {  # <gpu> <group> <seed index>
  local gpu=$1 g=$2 s=$3
  local mix=${MIX[$g]}
  printf '%s\n' "until [ -f $M/data/$mix/READY ]; do sleep 60; done"
  printf '%s\n' "printf 'track=dec\\nstatus=busy\\npurpose=decoder M4 m4-$g-s$s (training, about 1 h)\\nstart_utc=%s\\n' \$(date -u +%FT%TZ) > /data/dev2/leases/gpu$gpu.lock/owner"
  printf '%s\n' "DEC_NODE=b DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698 bash $D m4-$g-s$s $gpu $SRC m4/arms $NOX -- --train /runs/m4/data/$mix/train.jsonl --teacher /runs/m4/teacher/$mix/lux-teacher.jsonl --teacher-kl-weight ${KL[$g]} ${EXTRA[$g]:-} $BASE --seed ${SEED[$s]}"
}
chain() {  # <gpu> <group:seed>...
  local gpu=$1
  shift
  {
    echo "set -u"
    for item in "$@"; do step "$gpu" "${item%:*}" "${item#*:}"; done
    printf '%s\n' "printf 'track=dec\\nstatus=idle (decoder M4 chain finished)\\nend_utc=%s\\n' \$(date -u +%FT%TZ) > /data/dev2/leases/gpu$gpu.lock/owner"
  } > "$C/chain-$gpu.sh"
  setsid nohup bash "$C/chain-$gpu.sh" > "$C/chain-$gpu.log" 2>&1 < /dev/null &
}
chain 0 N4LR:1 N4XA:1 N4LRQ:1
chain 1 N4LR:2 N4XA:2 N4LRQ:2
chain 2 N4LR:3 N4XA:3 N4LRQ:3
chain 3 N4LR2:1 N4LR2:3 N4XF:2
chain 4 N4LR2:2 N4XF:1 N4XF:3
for item in N4LR:0 N4LR2:3 N4XA:1 N4XF:4 N4LRQ:2; do
  setsid nohup bash "$OPS/m4-soup.sh" "$SRC" "${item%:*}" "${item#*:}" "$NOX" > "$M/soup/${item%:*}.nohup" 2>&1 < /dev/null &
done
echo "$(date -u +%FT%TZ) M4 chains + soup watchers launched from $SRC (lock chains/launch.lock)" >> "$M/OPERATIONS.log"
