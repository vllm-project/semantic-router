#!/usr/bin/env bash
# ~27B M6 private Index full run on node D (node side; m6-index.sh run starts it detached). The eight shards of IX1's
# panel-8 with IX1's launch.sh (run --only k) on the listed GPUs (default 4 5 6 7): shard k runs on GPU number
# k mod n of the list, the shards of one GPU in turn, each with a copy of A20r's frozen autotune cache
# (parity/DEV2.0-27B/cache-frozen, as IX1's M5-L128 diagnostic). The GPU loops start 300 s apart so at most one 27B
# shard loads at a time; a loop stops at the first shard that does not end with exit code 0 (rerun with launch.sh
# resume, IX1's procedure). Prints run IDs, statuses and times only.
# Usage: m6-index-run.sh MIRROR_SHA ARM [GPU...]
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
shift 2
GPU_LIST=("$@")
[ ${#GPU_LIST[@]} -gt 0 ] || GPU_LIST=(4 5 6 7)
for g in "${GPU_LIST[@]}"; do [[ "$g" =~ ^[4-7]$ ]] || { echo "node D GPU4-7 only, not $g" >&2; exit 2; }; done
n=${#GPU_LIST[@]}
M=/data/dev2/src/$SHA-src_training_decision2
L=$M/src/training/decision2/v2/eval/ix1/launch.sh
R=/data/dev2/private/eval/index021/ix1
GPUS=$(for k in 0 1 2 3 4 5 6 7; do printf '%s ' "${GPU_LIST[$((k % n))]}"; done)
CACHE=$R/parity/DEV2.0-27B/cache-frozen
[ -f "$L" ] || { echo "missing mirror $SHA" >&2; exit 2; }
[ -f "$R/parity/$ARM/parity.json" ] || { echo "no parity gate for $ARM" >&2; exit 2; }
loop() {  # i: GPU GPU_LIST[i] runs shards i, i+n, ... < 8
  local i=$1 k e g=${GPU_LIST[$1]}
  sleep $((300 * i))
  for ((k = i; k < 8; k += n)); do
    echo "$(date -u +%FT%TZ) $ARM shard $k on GPU$g: start"
    (cd "$M/src/training/decision2" && bash "$L" run --src "$M" --model "$ARM" --gpus "${GPUS% }" --only "$k" \
      --run "$R/runs/$ARM" --rows-dir "$R/panel-8" --cache "$CACHE") ||
      { echo "$(date -u +%FT%TZ) $ARM shard $k: launch failed"; return 1; }
    while [ ! -f "$R/runs/$ARM/shard-$k/end_epoch" ]; do sleep 60; done
    e=$(cat "$R/runs/$ARM/shard-$k/exit_code")
    echo "$(date -u +%FT%TZ) $ARM shard $k on GPU$g: exit $e"
    [ "$e" = 0 ] || return 1
  done
}
echo "m6 index run $SHA $ARM on GPU ${GPU_LIST[*]}: start $(date -u +%FT%TZ)"
pids=()
for ((i = 0; i < n; i++)); do
  loop "$i" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "m6 index run $ARM complete (status $status): $(date -u +%FT%TZ)"
exit "$status"
