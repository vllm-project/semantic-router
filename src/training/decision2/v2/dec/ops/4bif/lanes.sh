#!/usr/bin/env bash
# Node side: one full IX1 run of NAME over an n-way panel on fewer GPUs than shards. PLAN lists each GPU with the
# shards it runs one after another ("4:0,4 5:1,5 ..."); shard k always runs on the GPU at position k of the n-way list
# launch.sh is given (launch.sh run --only k), so receipts read as a normal sharded run. A lane stops at its first
# failure; the other lanes finish their current shard. Usage: lanes.sh MIRROR_DIR NAME PANEL PLAN
set -uo pipefail
M=$1 NAME=$2 PANEL=$3 PLAN=$4
R=/data/dev2/private/eval/index021/ix1
S=$M/src/training/decision2
L=$S/v2/eval/ix1/launch.sh
n=$(find "$R/$PANEL" -maxdepth 1 -name 'shard-0-of-*.jsonl.gz' | sed 's/.*-of-\([0-9]*\)\.jsonl\.gz/\1/')
[[ "$n" =~ ^[0-9]+$ ]] || { echo "no shards in $R/$PANEL" >&2; exit 2; }
declare -a GPU_OF
for lane in $PLAN; do
  g=${lane%%:*}
  IFS=, read -r -a ks <<< "${lane#*:}"
  for k in "${ks[@]}"; do GPU_OF[k]=$g; done
done
for ((k = 0; k < n; k++)); do [ -n "${GPU_OF[$k]:-}" ] || { echo "shard $k has no GPU in the plan" >&2; exit 2; }; done
GPUS="${GPU_OF[*]}"
log() { echo "$(date -u +%FT%TZ) $*"; }
lane() {
  local k e
  for k in "$@"; do
    log "start shard $k on gpu${GPU_OF[$k]}"
    (cd "$S" && bash "$L" run --src "$M" --model "$NAME" --gpus "$GPUS" --only "$k" --run "$R/runs/$NAME" \
      --rows-dir "$R/$PANEL" --cache "$R/parity/$NAME/cache-frozen") || { log "launch of shard $k failed"; return 1; }
    while [[ ! -f "$R/runs/$NAME/shard-$k/end_epoch" ]]; do sleep 30; done
    e=$(cat "$R/runs/$NAME/shard-$k/exit_code"); log "shard $k exit $e"
    [[ "$e" == 0 ]] || return 1
  done
}
log "$NAME: $n shards on GPUs $GPUS (plan $PLAN)"
first=1
for lane in $PLAN; do
  IFS=, read -r -a ks <<< "${lane#*:}"
  lane "${ks[@]}" &
  # the first shard of a fresh run copies the cache and loads alone; launch.sh waits for room afterwards
  [ "$first" = 1 ] && sleep 60 && first=0
done
wait
log "$NAME: all lanes done"
