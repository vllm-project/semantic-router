#!/usr/bin/env bash
# HT-DEV v2 collection lane for one node-A GPU (shared lease eval-htdev2).
# usage: lane.sh GPU OPS_DIR BUDGET_GPU_H
# Pops the next job key from OPS_DIR/queue.txt (one "key est_gpu_h" per line) and runs
# OPS_DIR/jobs/<key>.sh GPU. A job starts only when the GPU shows < 1.5 GB VRAM in use and
# the GPU-hours already spent under /data/dev2/runs/eval/htdev2/{collect,smoke} plus the
# job's estimate stay within BUDGET_GPU_H. Each job is limited to 30 minutes.
set -uo pipefail
GPU=$1 OPS=$2 BUDGET=$3
Q=$OPS/queue.txt LOCK=$OPS/queue.lock RES=$OPS/results.tsv
mkdir -p "$OPS/logs"

used_gb() { rocm-smi -d "$GPU" --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2 {printf "%.1f", $3/1e9}'; }
spent_h() {
  python3 - <<'EOF'
import glob, json
print(f"{sum(json.load(open(p))['gpu_hours'] for p in glob.glob('/data/dev2/runs/eval/htdev2/*/*/GPU-TIME.json')):.4f}")
EOF
}
log() { echo "$(date -u +%FT%TZ) gpu$GPU $*" >> "$OPS/lanes.log"; }

pick() {
  local used spent key est
  used=$(used_gb); [ -n "$used" ] || return 1
  awk -v u="$used" 'BEGIN {exit !(u < 1.5)}' || return 1
  spent=$(spent_h)
  exec 9>"$LOCK"; flock 9
  local blocked=1 entries entry
  mapfile -t entries < "$Q"
  for entry in "${entries[@]}"; do
    read -r key est <<< "$entry"
    [ -n "$key" ] || continue
    if awk -v s="$spent" -v e="$est" -v b="$BUDGET" 'BEGIN {exit !(s + e > b)}'; then
      continue
    fi
    blocked=0
    printf '%s\n' "${entries[@]}" | grep -v "^$key " > "$Q.tmp"
    mv "$Q.tmp" "$Q"
    flock -u 9
    echo "$key $used"
    return 0
  done
  flock -u 9
  [ "$blocked" = 1 ] && echo BUDGET
  return 1
}

log "lane start (budget $BUDGET GPU-h)"
while [ -s "$Q" ]; do
  if ! picked=$(pick); then
    if [ "$picked" = BUDGET ]; then log "stop: budget $BUDGET GPU-h reached for every queued job"; break; fi
    sleep 30; continue
  fi
  read -r key used <<< "$picked"
  log "start $key vram_used_before=${used}GB"
  start=$(date -u +%FT%TZ)
  timeout -s TERM 1800 bash "$OPS/jobs/$key.sh" "$GPU" > "$OPS/logs/$key.log" 2>&1
  code=$?
  docker ps --filter "name=dev2-eval-gpu$GPU-" -q | xargs -r docker stop > /dev/null 2>&1
  printf '%s\t%s\t%s\t%s\t%s\n' "$key" "$GPU" "$start" "$(date -u +%FT%TZ)" "$code" >> "$RES"
  log "end $key exit=$code"
  sleep 10
done
log "lane done"
