#!/usr/bin/env bash
# ~27B M6 private Index full run on node C or node D (node side; m6-index.sh run starts it detached on every node that
# has shards). The listed shards of IX1's panel-8 with IX1's launch.sh (run --only k) on the listed GPUs of this node:
# the i-th listed shard runs on GPU number i mod n of the list, the shards of one GPU in turn, each with a copy of
# A20r's frozen autotune cache (parity/DEV2.0-27B/cache-frozen, as IX1's M5-L128 diagnostic). Shards that run on the
# other node get the placeholder GPU 9 in launch.sh's list (it touches only the selected shards), as IX1's node C runs
# did. The GPU loops start 300 s apart so at most one 27B shard loads at a time; a loop stops at the first shard that
# does not end with exit code 0 (rerun with launch.sh resume, IX1's procedure). Prints run IDs, statuses and times only.
# Usage: m6-index-run.sh MIRROR_SHA ARM NODE SHARDS GPU...
#   NODE    d (GPU0-7: GPU4-7 shared Index GPUs, GPU0-3 M6's own leases once its seeds ended) or c (GPU1-7; GPU0 is a
#           K8s pod and never used)
#   SHARDS  comma-separated shard indices 0-7 that this node runs, e.g. 0,1,2,3
# M6_INDEX_DRY=1 prints the shard -> GPU plan and exits before touching the node.
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
NODE=${3:?NODE} SHARDS=${4:?SHARDS}
shift 4
case "$NODE" in
  d) ALLOWED='^[0-7]$' ;;
  c) ALLOWED='^[1-7]$' ;;
  *) echo "NODE must be c or d, not $NODE" >&2; exit 2 ;;
esac
[[ "$SHARDS" =~ ^[0-7](,[0-7]){0,7}$ ]] || { echo "SHARDS: comma-separated shard indices 0-7, not '$SHARDS'" >&2; exit 2; }
IFS=, read -r -a SHARD_LIST <<< "$SHARDS"
[ "$(printf '%s\n' "${SHARD_LIST[@]}" | sort -u | wc -l)" = "${#SHARD_LIST[@]}" ] || { echo "SHARDS lists a shard twice" >&2; exit 2; }
GPU_LIST=("$@")
[ ${#GPU_LIST[@]} -gt 0 ] || { echo "no GPU listed" >&2; exit 2; }
for g in "${GPU_LIST[@]}"; do [[ "$g" =~ $ALLOWED ]] || { echo "node $NODE: GPU $g is not allowed" >&2; exit 2; }; done
[ "$(printf '%s\n' "${GPU_LIST[@]}" | sort -u | wc -l)" = "${#GPU_LIST[@]}" ] || { echo "a GPU is listed twice" >&2; exit 2; }
n=${#GPU_LIST[@]}
(( n <= ${#SHARD_LIST[@]} )) || n=${#SHARD_LIST[@]}
SLOT=(9 9 9 9 9 9 9 9)
for i in "${!SHARD_LIST[@]}"; do SLOT[${SHARD_LIST[$i]}]=${GPU_LIST[$((i % n))]}; done
GPUS="${SLOT[*]}"
if [ "${M6_INDEX_DRY:-0}" = 1 ]; then
  for i in "${!SHARD_LIST[@]}"; do echo "shard ${SHARD_LIST[$i]}: node $NODE GPU${GPU_LIST[$((i % n))]}"; done
  echo "launch.sh --gpus \"$GPUS\""
  exit 0
fi
M=/data/dev2/src/$SHA-src_training_decision2
L=$M/src/training/decision2/v2/eval/ix1/launch.sh
R=/data/dev2/private/eval/index021/ix1
CACHE=$R/parity/DEV2.0-27B/cache-frozen
[ -f "$L" ] || { echo "missing mirror $SHA" >&2; exit 2; }
[ -f "$R/parity/$ARM/parity.json" ] || { echo "no parity gate for $ARM" >&2; exit 2; }
loop() {  # i: GPU GPU_LIST[i] runs the listed shards at positions i, i+n, ...
  local i=$1 j k e g=${GPU_LIST[$1]}
  sleep $((300 * i))
  for ((j = i; j < ${#SHARD_LIST[@]}; j += n)); do
    k=${SHARD_LIST[$j]}
    echo "$(date -u +%FT%TZ) $ARM shard $k on node $NODE GPU$g: start"
    (cd "$M/src/training/decision2" && bash "$L" run --src "$M" --model "$ARM" --gpus "$GPUS" --only "$k" \
      --run "$R/runs/$ARM" --rows-dir "$R/panel-8" --cache "$CACHE") ||
      { echo "$(date -u +%FT%TZ) $ARM shard $k: launch failed"; return 1; }
    while [ ! -f "$R/runs/$ARM/shard-$k/end_epoch" ]; do sleep 60; done
    e=$(cat "$R/runs/$ARM/shard-$k/exit_code")
    echo "$(date -u +%FT%TZ) $ARM shard $k on node $NODE GPU$g: exit $e"
    [ "$e" = 0 ] || return 1
  done
}
echo "m6 index run $SHA $ARM on node $NODE: shards $SHARDS on GPU ${GPU_LIST[*]:0:$n}: start $(date -u +%FT%TZ)"
pids=()
for ((i = 0; i < n; i++)); do
  loop "$i" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "m6 index run $ARM on node $NODE complete (status $status): $(date -u +%FT%TZ)"
exit "$status"
