#!/usr/bin/env bash
# Start a sharded complement run (launch.sh run) once the listed GPUs are released by their owner.
#
# Usage: complement_after_release.sh --gpus "N ..." --src DIR --model NAME --rows-dir DIR --cache DIR
#                                    [--deadline-hours H]
#
# Runs in the foreground (start it with nohup). Waits until every listed GPU's lease owner file says
# status=released (key=value or JSON) and the GPU is idle (use <= 5 %, VRAM <= 2 GiB), then writes eval-ix1 leases for them
# (launch.sh only takes eval-ix1 leases) and runs launch.sh run into $BASE/submit/runs/<model>. Gives up at
# the deadline (default 6 h) or when $BASE/submit/STOP-wait exists.
set -euo pipefail
BASE=/data/dev2/private/eval/index021
gpus="" src="" model="" rows_dir="" cache="" deadline_hours=6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --src) src="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --rows-dir) rows_dir="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --deadline-hours) deadline_hours="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$gpus" && -f "$src/.dev2-mirror.json" && -n "$model" && -f "$rows_dir/panel.json" && -d "$cache" ]] \
  || { sed -n '2,10p' "$0" >&2; exit 2; }
read -r -a gpu_list <<< "$gpus"
deadline=$(( $(date +%s) + deadline_hours * 3600 ))

ready() {  # gpu
  local owner="/data/dev2/leases/gpu$1.lock/owner"
  grep -Eq 'status"?[[:space:]]*[=:][[:space:]]*"?released' "$owner" 2>/dev/null || return 1
  rocm-smi --showuse --showmeminfo vram --json | python3 -c '
import json, sys
card = json.load(sys.stdin)["card" + sys.argv[1]]
sys.exit(0 if float(card["GPU use (%)"]) <= 5 and int(card["VRAM Total Used Memory (B)"]) <= 2 * 2**30 else 1)
' "$1"
}

echo "$(date -u +%FT%TZ) waiting for gpus $gpus"
while :; do
  [[ -e "$BASE/submit/STOP-wait" ]] && { echo "$(date -u +%FT%TZ) STOP-wait present; giving up"; exit 0; }
  (( $(date +%s) < deadline )) || { echo "$(date -u +%FT%TZ) deadline reached; giving up"; exit 1; }
  all=1
  for g in "${gpu_list[@]}"; do ready "$g" || { all=0; break; }; done
  (( all )) && break
  sleep 60
done
for g in "${gpu_list[@]}"; do
  printf 'track=eval-ix1\npurpose=index-submit complement of %s (board submission worker)\nstart_utc=%s\n' \
    "$model" "$(date -u +%FT%TZ)" > "/data/dev2/leases/gpu$g.lock/owner"
done
echo "$(date -u +%FT%TZ) gpus released and idle; starting $model"
bash "$src/src/training/decision2/v2/eval/ix1/launch.sh" run --src "$src" --model "$model" --gpus "$gpus" \
  --run "$BASE/submit/runs/$model" --rows-dir "$rows_dir" --cache "$cache"
for g in "${gpu_list[@]}"; do
  printf 'note=index-submit complement of %s (board submission worker)\n' "$model" >> "/data/dev2/leases/gpu$g.lock/owner"
done
