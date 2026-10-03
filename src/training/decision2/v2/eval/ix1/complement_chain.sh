#!/usr/bin/env bash
# Complement runs for board submissions (node side): one GPU, one job after another. Each job runs the
# stored IX1 run's own launcher (its mirror), package, image, kit and frozen cache over one shard of the
# complement rows (v2.eval.ix1.complement) into $BASE/submit/runs/<model>/shard-<k>.
#
# Usage: complement_chain.sh --gpu N --rows-dir DIR [--after-log FILE --after-grep TEXT]
#                            --job "MODEL MIRROR-SHA CACHE-DIR [SHARD]" [--job ...]
#
# Runs in the foreground (start it with nohup). SHARD (default 0) indexes the rows dir's
# shard-<k>-of-<n> files. First waits until --after-log contains --after-grep, if given. Skips a job
# whose shard has ended, stops at the first job that does not end with exit code 0, and stops before
# the next job when $BASE/submit/STOP-gpu<N> exists. Notes each job in the GPU's lease owner file and
# marks the lease released when the chain ends.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
gpu="" rows_dir="" after_log="" after_grep="" jobs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --rows-dir) rows_dir="$2"; shift 2 ;;
    --after-log) after_log="$2"; shift 2 ;;
    --after-grep) after_grep="$2"; shift 2 ;;
    --job) jobs+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-7]$ && -f "$rows_dir/panel.json" && ${#jobs[@]} -gt 0 ]] || { sed -n '2,13p' "$0" >&2; exit 2; }
n=$(ls "$rows_dir"/shard-*-of-*.jsonl.gz | wc -l)
[[ -f "$rows_dir/shard-0-of-$n.jsonl.gz" ]] || { echo "$rows_dir has no $n-way shards" >&2; exit 2; }
gpus="$(printf "$gpu %.0s" $(seq "$n"))"
lease="/data/dev2/leases/gpu$gpu.lock/owner"
if [[ -n "$after_log" ]]; then
  echo "$(date -u +%FT%TZ) waiting for '$after_grep' in $after_log"
  until grep -qF "$after_grep" "$after_log" 2>/dev/null; do sleep 20; done
fi
for job in "${jobs[@]}"; do
  read -r model mirror cache shard <<< "$job"
  shard="${shard:-0}"
  (( shard < n )) || { echo "shard $shard is not in $rows_dir" >&2; exit 2; }
  src="/data/dev2/src/$mirror-src_training_decision2"
  run="$BASE/submit/runs/$model"
  if [[ -f "$run/shard-$shard/end_epoch" ]]; then
    echo "$(date -u +%FT%TZ) $model shard $shard already ended (exit $(cat "$run/shard-$shard/exit_code"))"
    continue
  fi
  if [[ -e "$BASE/submit/STOP-gpu$gpu" ]]; then
    echo "$(date -u +%FT%TZ) STOP-gpu$gpu present; chain ends before $model shard $shard"
    exit 0
  fi
  echo "$(date -u +%FT%TZ) start $model shard $shard of $n on gpu$gpu"
  bash "$src/src/training/decision2/v2/eval/ix1/launch.sh" run --src "$src" --model "$model" --gpus "${gpus% }" \
    --only "$shard" --run "$run" --rows-dir "$rows_dir" --cache "$cache"
  printf 'note=index-submit complement of %s shard %s (board submission worker)\n' "$model" "$shard" >> "$lease"
  while [[ ! -f "$run/shard-$shard/end_epoch" ]]; do sleep 30; done
  code="$(cat "$run/shard-$shard/exit_code")"
  echo "$(date -u +%FT%TZ) $model shard $shard exit $code: $(cat "$run/shard-$shard/status.json")"
  [[ "$code" == 0 ]] || exit 1
done
printf 'track=eval-ix1\nstatus=released\npurpose=index-submit complement chain ended\nend_utc=%s\n' \
  "$(date -u +%FT%TZ)" > "$lease"
echo "$(date -u +%FT%TZ) chain done"
