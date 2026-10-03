#!/usr/bin/env bash
# Complement runs for board submissions (node side): one GPU, one model after another. Each job runs
# the stored IX1 run's own launcher (its mirror), package, image, kit and frozen cache over the
# complement rows (v2.eval.ix1.complement) into $BASE/submit/runs/<model>/shard-0.
#
# Usage: complement_chain.sh --gpu N --rows-dir DIR --job "MODEL MIRROR-SHA CACHE-DIR" [--job ...]
#
# Runs in the foreground (start it with nohup). Skips a model whose complement shard has ended, stops
# at the first job that does not end with exit code 0, and stops before the next job when
# $BASE/submit/STOP-gpu<N> exists. Notes each job in the GPU's lease owner file and marks the lease
# released when the chain ends.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
gpu="" rows_dir="" jobs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --rows-dir) rows_dir="$2"; shift 2 ;;
    --job) jobs+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-7]$ && -f "$rows_dir/panel.json" && -f "$rows_dir/shard-0-of-1.jsonl.gz" && ${#jobs[@]} -gt 0 ]] \
  || { sed -n '2,11p' "$0" >&2; exit 2; }
lease="/data/dev2/leases/gpu$gpu.lock/owner"
for job in "${jobs[@]}"; do
  read -r model mirror cache <<< "$job"
  src="/data/dev2/src/$mirror-src_training_decision2"
  run="$BASE/submit/runs/$model"
  if [[ -f "$run/shard-0/end_epoch" ]]; then
    echo "$(date -u +%FT%TZ) $model already ended (exit $(cat "$run/shard-0/exit_code"))"
    continue
  fi
  if [[ -e "$BASE/submit/STOP-gpu$gpu" ]]; then
    echo "$(date -u +%FT%TZ) STOP-gpu$gpu present; chain ends before $model"
    exit 0
  fi
  echo "$(date -u +%FT%TZ) start $model on gpu$gpu"
  bash "$src/src/training/decision2/v2/eval/ix1/launch.sh" run --src "$src" --model "$model" --gpus "$gpu" \
    --run "$run" --rows-dir "$rows_dir" --cache "$cache"
  printf 'note=index-submit complement of %s (board submission worker)\n' "$model" >> "$lease"
  while [[ ! -f "$run/shard-0/end_epoch" ]]; do sleep 30; done
  code="$(cat "$run/shard-0/exit_code")"
  echo "$(date -u +%FT%TZ) $model exit $code: $(cat "$run/shard-0/status.json")"
  [[ "$code" == 0 ]] || exit 1
done
printf 'track=eval-ix1\nstatus=released\npurpose=index-submit complement chain ended\nend_utc=%s\n' \
  "$(date -u +%FT%TZ)" > "$lease"
echo "$(date -u +%FT%TZ) chain done"
