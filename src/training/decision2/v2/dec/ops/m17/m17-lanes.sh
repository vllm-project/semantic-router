#!/usr/bin/env bash
# Decoder M17 (amendment 1), node side: the remaining panel shards of one IX1 diagnostic package, one lane per GPU.
# A lane waits until the GPU's earlier shard of this package has ended, then runs its shards one after another, each
# as its own `launch.sh run --only k` with shard k on the lane's GPU and the parity gate's frozen cache. A shard that
# ends with a non-zero exit code stops its lane (never rerun); the other lanes go on.
#
# usage: m17-lanes.sh MIRROR_DIR NAME PANEL "GPU:k,k [GPU:k ...]" ["GPU:k ..." (shards already running per GPU)]
set -u
M=$1 NAME=$2 PANEL=$3 PLAN=$4 RUNNING=${5:-}
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
RUN=$R/runs/$NAME
log() { echo "$(date -u +%FT%TZ) $*"; }
ended() { [ -f "$RUN/shard-$1/end_epoch" ]; }
lane() {  # <gpu> <shards comma list> [<shard already running on the gpu>]
  local g=$1 ks=$2 before=${3:-} k order
  if [ -n "$before" ]; then
    until ended "$before"; do sleep 30; done
    log "$NAME shard $before on gpu$g ended (exit $(cat "$RUN/shard-$before/exit_code"))"
  fi
  for k in ${ks//,/ }; do
    order="$g $g $g $g $g $g $g $g"
    if ! (cd "$S" && bash v2/eval/ix1/launch.sh run --src "$M" --model "$NAME" --gpus "$order" --only "$k" --run "$RUN" \
      --rows-dir "$R/$PANEL" --cache "$R/parity/$NAME/cache-frozen"); then
      log "$NAME shard $k launch on gpu$g FAILED; lane stopped"
      return 1
    fi
    log "$NAME shard $k started on gpu$g"
    until ended "$k"; do sleep 30; done
    log "$NAME shard $k on gpu$g ended (exit $(cat "$RUN/shard-$k/exit_code"))"
    [ "$(cat "$RUN/shard-$k/exit_code")" = 0 ] || { log "$NAME lane gpu$g stopped"; return 1; }
  done
}
declare -A BEFORE=()
for x in $RUNNING; do BEFORE[${x%%:*}]=${x#*:}; done
for x in $PLAN; do
  lane "${x%%:*}" "${x#*:}" "${BEFORE[${x%%:*}]:-}" &
done
wait
log "$NAME lanes finished"
