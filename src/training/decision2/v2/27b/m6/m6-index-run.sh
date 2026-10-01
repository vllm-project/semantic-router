#!/usr/bin/env bash
# ~27B M6 private Index full run on node D GPU4-7 (node side; m6-index.sh run starts it detached). The eight shards
# of IX1's panel-8 with IX1's launch.sh (run --only k): shard k and then shard k+4 on GPU 4+k, each with a copy of
# A20r's frozen autotune cache (parity/DEV2.0-27B/cache-frozen, as IX1's M5-L128 diagnostic). The four GPU loops
# start 300 s apart so at most one 27B shard loads at a time; a loop stops at the first shard that does not end with
# exit code 0 (rerun with launch.sh resume, IX1's procedure). Prints run IDs, statuses and times only.
# Usage: m6-index-run.sh MIRROR_SHA ARM
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
M=/data/dev2/src/$SHA-src_training_decision2
L=$M/src/training/decision2/v2/eval/ix1/launch.sh
R=/data/dev2/private/eval/index021/ix1
GPUS="4 5 6 7 4 5 6 7"
CACHE=$R/parity/DEV2.0-27B/cache-frozen
[ -f "$L" ] || { echo "missing mirror $SHA" >&2; exit 2; }
[ -f "$R/parity/$ARM/parity.json" ] || { echo "no parity gate for $ARM" >&2; exit 2; }
loop() {  # i: GPU 4+i runs shard i, then shard i+4
  local i=$1 k e
  sleep $((300 * i))
  for k in "$i" $((i + 4)); do
    echo "$(date -u +%FT%TZ) $ARM shard $k on GPU$((4 + i)): start"
    (cd "$M/src/training/decision2" && bash "$L" run --src "$M" --model "$ARM" --gpus "$GPUS" --only "$k" \
      --run "$R/runs/$ARM" --rows-dir "$R/panel-8" --cache "$CACHE") ||
      { echo "$(date -u +%FT%TZ) $ARM shard $k: launch failed"; return 1; }
    while [ ! -f "$R/runs/$ARM/shard-$k/end_epoch" ]; do sleep 60; done
    e=$(cat "$R/runs/$ARM/shard-$k/exit_code")
    echo "$(date -u +%FT%TZ) $ARM shard $k on GPU$((4 + i)): exit $e"
    [ "$e" = 0 ] || return 1
  done
}
echo "m6 index run $SHA $ARM: start $(date -u +%FT%TZ)"
pids=()
for i in 0 1 2 3; do
  loop "$i" &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "m6 index run $ARM complete (status $status): $(date -u +%FT%TZ)"
exit "$status"
