#!/usr/bin/env bash
# IX1 per-GPU chain (node side): when shard k of the previous model has ended with exit code 0 on the
# k-th GPU, start shard k of the next model there with launch.sh run --only k.
#
# Usage: chain.sh --src DIR --gpus "N ..." --rows-dir DIR --after MODEL --models "MODEL ..."
#
# Runs in the foreground (start it with nohup). Each GPU's chain stops at the first shard that does
# not end with exit code 0; the other GPUs continue. Per-GPU logs: <ix1>/logs/chain-gpu<N>.log.
set -euo pipefail
R=/data/dev2/private/eval/index021/ix1
src="" gpus="" rows_dir="" after="" models=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) src="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --rows-dir) rows_dir="$2"; shift 2 ;;
    --after) after="$2"; shift 2 ;;
    --models) models="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$src" && -n "$gpus" && -d "$rows_dir" && -n "$after" && -n "$models" ]] \
  || { sed -n '2,9p' "$0" >&2; exit 2; }
L="$src/src/training/decision2/v2/eval/ix1/launch.sh"
read -r -a gpu_list <<< "$gpus"

chain() {  # shard index
  local k="$1" previous="$after" model finished
  for model in $models; do
    finished="$R/runs/$previous/shard-$k"
    while [[ ! -f "$finished/end_epoch" ]]; do sleep 30; done
    if [[ "$(cat "$finished/exit_code" 2>/dev/null)" != 0 ]]; then
      echo "$(date -u +%FT%TZ) $previous shard $k ended with exit $(cat "$finished/exit_code" 2>/dev/null); chain stops"
      return 1
    fi
    echo "$(date -u +%FT%TZ) start $model shard $k"
    bash "$L" run --src "$src" --model "$model" --gpus "$gpus" --only "$k" --run "$R/runs/$model" \
      --rows-dir "$rows_dir" --cache "$R/parity/$model/cache-frozen"
    previous="$model"
  done
}

for k in "${!gpu_list[@]}"; do
  chain "$k" > "$R/logs/chain-gpu${gpu_list[$k]}.log" 2>&1 &
done
wait
