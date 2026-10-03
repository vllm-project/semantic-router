#!/usr/bin/env bash
# Release spot checks for board submissions (node side): one GPU, one released revision after another,
# each over the same sample rows (v2.eval.ix1.submission sample) with launch.sh extra into
# $BASE/submit/spot/<model>/extra-release.
#
# Usage: spot_chain.sh --gpu N --src DIR --rows FILE [--after-log FILE --after-grep TEXT]
#                      --job "MODEL CACHE-DIR" [--job ...]
#
# Runs in the foreground (start it with nohup). First waits until --after-log contains --after-grep
# (e.g. the complement chain's "chain done", written after it released the lease). Skips a model whose
# spot run has ended, stops at the first failure or when $BASE/submit/STOP-gpu<N> exists, and marks the
# GPU's lease released at the end.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
gpu="" src="" rows="" after_log="" after_grep="" jobs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --src) src="$2"; shift 2 ;;
    --rows) rows="$2"; shift 2 ;;
    --after-log) after_log="$2"; shift 2 ;;
    --after-grep) after_grep="$2"; shift 2 ;;
    --job) jobs+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-7]$ && -f "$src/.dev2-mirror.json" && -f "$rows" && ${#jobs[@]} -gt 0 ]] || { sed -n '2,12p' "$0" >&2; exit 2; }
L="$src/src/training/decision2/v2/eval/ix1/launch.sh"
lease="/data/dev2/leases/gpu$gpu.lock/owner"
if [[ -n "$after_log" ]]; then
  echo "$(date -u +%FT%TZ) waiting for the --after-grep text in $after_log"
  until grep -qF "$after_grep" "$after_log" 2>/dev/null; do sleep 20; done
fi
for job in "${jobs[@]}"; do
  read -r model cache <<< "$job"
  run="$BASE/submit/spot/$model"
  if [[ -f "$run/extra-release/end_epoch" ]]; then
    echo "$(date -u +%FT%TZ) $model already ended (exit $(cat "$run/extra-release/exit_code"))"
    continue
  fi
  [[ -e "$BASE/submit/STOP-gpu$gpu" ]] && { echo "$(date -u +%FT%TZ) STOP-gpu$gpu present; chain ends before $model"; exit 0; }
  echo "$(date -u +%FT%TZ) start $model on gpu$gpu"
  code=0
  bash "$L" extra --src "$src" --model "$model" --gpu "$gpu" --run "$run" --rows "$rows" --cache "$cache" --tag release || code=$?
  printf 'note=index-submit release spot check of %s (board submission worker)\n' "$model" >> "$lease"
  echo "$(date -u +%FT%TZ) $model exit $code: $(cat "$run/extra-release/status.json" 2>/dev/null)"
  [[ "$code" == 0 ]] || exit 1
done
printf 'track=eval-ix1\nstatus=released\npurpose=index-submit spot chain ended\nend_utc=%s\n' "$(date -u +%FT%TZ)" > "$lease"
echo "$(date -u +%FT%TZ) chain done"
