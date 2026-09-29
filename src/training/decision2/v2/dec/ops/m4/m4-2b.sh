#!/usr/bin/env bash
# Decoder M4 optional 2B probe (prereg dec-m4-prereg-2026-09-29.md, last section): the best 4B single-arm
# recipe on own Sol 1.0, three seeds on node B GPU0-2 (each after that GPU's M4 chain has finished), then a
# soup readout against Sol 1.0 and the 2B release candidate S2T (the readout's nox1 / n4lkr slots).
# usage: m4-2b.sh <mirror-dir> <group e.g. S2LR> <mixture> <KL weight> [extra trainer flag]
set -u
SRC=$1 G=$2 MIX=$3 KLW=$4 EXTRA=${5:-}
M=/data/dev2/runs/dec/m4
C=$M/chains
mkdir "$C/2b.lock" 2>/dev/null || { echo "2B probe already launched"; exit 0; }
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m4
D=/data/dev2/src/$SRC/src/training/decision2/v2/dec/drive_arm.sh
SOL=/hf/models--llm-semantic-router--Decision-1.0-Sol-2B/snapshots/ce0c018a28de16d6639b1cd203b761bf643b89e6
BASE="--train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
declare -A SEED=([1]=20260926 [2]=20260927 [3]=20260928)
for s in 1 2 3; do
  gpu=$((s - 1))
  {
    echo "set -u"
    printf '%s\n' "until grep -q 'decoder M4 chain finished' /data/dev2/leases/gpu$gpu.lock/owner; do sleep 60; done"
    printf '%s\n' "printf 'track=dec\\nstatus=busy\\npurpose=decoder M4 2B probe m4-$G-s$s (about 35 min)\\nstart_utc=%s\\n' \$(date -u +%FT%TZ) > /data/dev2/leases/gpu$gpu.lock/owner"
    printf '%s\n' "DEC_NODE=b DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698 bash $D m4-$G-s$s $gpu $SRC m4/arms $SOL -- --train /runs/m4/data/$MIX/train.jsonl --teacher /runs/m4/teacher/$MIX/lux-teacher.jsonl --teacher-kl-weight $KLW $EXTRA $BASE --seed ${SEED[$s]}"
    printf '%s\n' "printf 'track=dec\\nstatus=idle (decoder M4 2B probe finished)\\nend_utc=%s\\n' \$(date -u +%FT%TZ) > /data/dev2/leases/gpu$gpu.lock/owner"
  } > "$C/2b-$gpu.sh"
  setsid nohup bash "$C/2b-$gpu.sh" > "$C/2b-$gpu.log" 2>&1 < /dev/null &
done
M4_CTL=sol1 M4_REF=/data/dev2/runs/dec/m3/soup/S2T setsid nohup bash "$OPS/m4-soup.sh" "$SRC" "$G" 0 "$SOL" > "$M/soup/$G.nohup" 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) M4 2B probe $G ($MIX, KL $KLW) chains + soup watcher launched from $SRC" >> "$M/OPERATIONS.log"
