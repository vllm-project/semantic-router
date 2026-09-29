#!/usr/bin/env bash
# Decoder M5 launcher (prereg dec-m5-prereg-2026-09-29.md): per-GPU chains on node B GPU3 / GPU4. Every chain
# holds its GPU's flock for its whole run, so a later phase's chain queues behind an earlier one on the same
# GPU. Each training step waits for its mixture's READY marker (written only after the data lock naming its
# train / teacher hashes is committed), writes the lease owner, runs drive_arm.sh (zero-step, one-step +
# reload, preflight gate, full run, postrun) and then m5-soup.sh, which builds the arm's soup only on the
# chain that completes its last seed. A mkdir lock per phase makes a repeated invocation a no-op.
#   phase n5n:   GPU3 N5N-s1 -> N5N-s3; GPU4 N5N-s2
#   phase block: GPU3 baselines (MLX-DEV: Nox 1.0, N4XF s1-s3, N4XF soup, pending M5 soups) -> N5B-s2 -> N5BN-s2;
#                GPU4 N5B-s1 -> N5B-s3 -> N5BN-s1 -> N5BN-s3
#   phases n5bn / n5b: the block phase split by arm, so one stopped arm does not hold the other's chain
#                (n5bn: GPU3 baselines -> N5BN-s2, GPU4 N5BN-s1 -> N5BN-s3; n5b: GPU3 N5B-s2, GPU4 N5B-s1 -> N5B-s3)
# usage: m5-chains.sh <mirror-dir> n5n|block|n5bn|n5b
set -u
SRC=$1 PHASE=$2
M=/data/dev2/runs/dec/m5
A=$M/arms C=$M/chains
mkdir -p "$A" "$C" "$M/soup" "$M/logs"
mkdir "$C/launch-$PHASE.lock" 2>/dev/null || { echo "M5 $PHASE chains already launched"; exit 0; }
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m5
D=/data/dev2/src/$SRC/src/training/decision2/v2/dec/drive_arm.sh
NOX=/hf/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
BASE="--train-mode full --backbone-lr 5e-6 --head-lr 5e-5 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
declare -A TRAIN=([N5N]=/runs/m4/data/m4-xl-full-29m/train.jsonl [N5B]=/runs/m5/data/n5b/train.jsonl [N5BN]=/runs/m5/data/n5b/train.jsonl)
declare -A TEACH=([N5N]=/runs/m5/teacher/n5n/teacher.jsonl [N5B]=/runs/m5/teacher/n5b/teacher.jsonl [N5BN]=/runs/m5/teacher/n5bn/teacher.jsonl)
declare -A READY=([N5N]=$M/data/n5n/READY [N5B]=$M/data/n5b/READY [N5BN]=$M/data/n5bn/READY)
declare -A SEED=([1]=20260926 [2]=20260927 [3]=20260928)
lease() {  # <gpu> <purpose> <minutes>
  printf '%s\n' "printf 'track=dec\\nstatus=busy\\npurpose=decoder M5 $2\\nstart_utc=%s\\nexpected_end_utc=%s\\n' \$(date -u +%FT%TZ) \$(date -u -d '+$3 min' +%FT%TZ) > /data/dev2/leases/gpu$1.lock/owner"
}
step() {  # <gpu> <arm> <seed index>
  local gpu=$1 g=$2 s=$3
  printf '%s\n' "until [ -f ${READY[$g]} ]; do sleep 60; done"
  lease "$gpu" "m5-$g-s$s (training, about 1 h)" 62
  printf '%s\n' "echo \"\$(date -u +%FT%TZ) start m5-$g-s$s on GPU$gpu\""
  printf '%s\n' "DEC_NODE=b DEC_DATA_DIR=/data/dev2/runs/dec/m3/data-sel700-cal698 bash $D m5-$g-s$s $gpu $SRC m5/arms $NOX -- --train ${TRAIN[$g]} --teacher ${TEACH[$g]} --teacher-kl-weight 1.0 --teacher-partial $BASE --seed ${SEED[$s]}"
  lease "$gpu" "m5-$g soup check / build" 15
  printf '%s\n' "bash $OPS/m5-soup.sh $SRC $g $gpu"
}
baselines() {  # <gpu>
  local gpu=$1 s b
  printf '%s\n' "until [ -f $M/mlxdev/READY ]; do sleep 60; done"
  lease "$gpu" "MLX-DEV baseline readouts" 60
  printf '%s\n' "bash $OPS/m5-mlx.sh $SRC $gpu nox1 package $NOX"
  for s in 1 2 3; do
    b="\$(python3 -c \"import json;print(json.load(open('/data/dev2/runs/dec/m4/arms/full/m4-N4XF-s$s/BEST.json'))['checkpoint'])\")"
    printf '%s\n' "bash $OPS/m5-mlx.sh $SRC $gpu n4xf-s$s checkpoint /runs/m4/arms/full/m4-N4XF-s$s/$b"
  done
  printf '%s\n' "bash $OPS/m5-mlx.sh $SRC $gpu n4xf-soup checkpoint /runs/m4/soup/N4XF/build/N4XF-soup"
  printf '%s\n' "for g in N5N N5B N5BN; do [ -d $M/soup/\$g/build/\$g-soup ] && [ ! -f $M/mlxdev/readouts/m5-\$g-soup/vs-n4xf-soup.json ] && bash $OPS/m5-mlx.sh $SRC $gpu m5-\$g-soup checkpoint /runs/m5/soup/\$g/build/\$g-soup; done"
}
chain() {  # <gpu> <item>...  (item = ARM:seed or baselines)
  local gpu=$1 f=$C/chain-$PHASE-$1.sh
  shift
  {
    echo "set -u"
    for item in "$@"; do
      if [ "$item" = baselines ]; then baselines "$gpu"; else step "$gpu" "${item%:*}" "${item#*:}"; fi
    done
    printf '%s\n' "printf 'track=dec\\nstatus=idle (decoder M5 $PHASE chain finished)\\nend_utc=%s\\n' \$(date -u +%FT%TZ) > /data/dev2/leases/gpu$gpu.lock/owner"
  } > "$f"
  setsid nohup flock "$C/gpu$gpu.flock" bash "$f" > "$M/logs/chain-$PHASE-$gpu.log" 2>&1 < /dev/null &
}
case $PHASE in
  n5n)
    chain 3 N5N:1 N5N:3
    chain 4 N5N:2
    ;;
  block)
    chain 3 baselines N5B:2 N5BN:2
    chain 4 N5B:1 N5B:3 N5BN:1 N5BN:3
    ;;
  n5bn)
    chain 3 baselines N5BN:2
    chain 4 N5BN:1 N5BN:3
    ;;
  n5b)
    chain 3 N5B:2
    chain 4 N5B:1 N5B:3
    ;;
  *) echo "unknown phase $PHASE" >&2; rmdir "$C/launch-$PHASE.lock"; exit 2 ;;
esac
echo "$(date -u +%FT%TZ) M5 $PHASE chains launched from $SRC (lock chains/launch-$PHASE.lock)" >> "$M/OPERATIONS.log"
