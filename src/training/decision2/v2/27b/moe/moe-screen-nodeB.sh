#!/usr/bin/env bash
# 27B MoE Stage A screen, node B side (host; detached). As each cell's checkpoint 892 becomes available (node B cells
# locally, node A cells relayed into /data/dev2/xfer/27b-moe/relay/<cell>/ with a verified SHA256SUMS), it runs the
# T = 1 readout on node B GPU7 against the dense matched reference's HT-DEV v2 predictions. A cell whose driver has
# ended without checkpoint 892 (or whose relay dir says ABSENT) is absent. Then screen_rules.py writes SCREEN.json,
# which is copied to /data/dev2/xfer/27b-moe/screen/ for node A, and node B's dropped cells are stopped.
# Usage: moe-screen-nodeB.sh MIRROR
set -euo pipefail
MIR=$1
S=/data/dev2/src/$MIR/src/training/decision2
R=/data/dev2/runs/27b-moe
X=/data/dev2/xfer/27b-moe
REF=$R/readouts/A20r-s1-c892
CELLS=("MOE-Git-s1:gemma-4-26B-A4B-it:a" "MOE-Qit-s1:Qwen3.5-35B-A3B:a" "MOE-Qpt-s1:Qwen3.5-35B-A3B-Base:b")
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date -u +%FT%TZ) $*"; }
driver_ended() { [ -d "$R/$1" ] && flock -n "$R/$1/.driver.lock" true; }
state() {  # CELL NODE -> "ready PATH" | "absent" | "wait"
  local cell=$1 node=$2 dir
  if [ "$node" = b ]; then
    dir=$R/$cell/full/run/checkpoint-0000892
    if [ -f "$dir/decision_config.json" ]; then echo "ready $dir"
    elif driver_ended "$cell"; then echo absent
    else echo wait; fi
  else
    dir=$X/relay/$cell
    if [ -f "$dir/ABSENT" ]; then echo absent
    elif [ -f "$dir/RELAYED" ] && (cd "$dir" && sha256sum -c --quiet SHA256SUMS); then echo "ready $dir/checkpoint-0000892"
    else echo wait; fi
  fi
}
[ -f "$REF/READOUT.json" ] || { log "dense matched reference readout missing: $REF"; exit 2; }
mkdir -p "$R/screen" "$X/screen"
declare -A seen=()
while [ "${#seen[@]}" -lt "${#CELLS[@]}" ]; do
  progressed=0
  for spec in "${CELLS[@]}"; do
    IFS=: read -r cell base node <<< "$spec"
    [ -z "${seen[$cell]:-}" ] || continue
    read -r status path <<< "$(state "$cell" "$node")"
    case "$status" in
      absent) log "$cell absent at the screen point"; seen[$cell]="absent"; progressed=1 ;;
      ready)
        if [ ! -f "$R/readouts/$cell-c892/READOUT.json" ]; then
          log "readout $cell from $path"
          bash "$S/v2/27b/moe/moe-readout.sh" b 7 "$cell-c892" "$path" "/data/dev2/models/moe/$base" "$MIR" \
            "$REF/output/ht-dev2.predictions.jsonl"
        fi
        seen[$cell]="read"; progressed=1 ;;
    esac
  done
  [ "$progressed" = 1 ] || sleep 300
done
args=()
for spec in "${CELLS[@]}"; do
  IFS=: read -r cell base node <<< "$spec"
  [ "${seen[$cell]}" = read ] && args+=(--cell "$cell=$base=$R/readouts/$cell-c892/READOUT.json")
done
if [ "${#args[@]}" -gt 0 ]; then
  python3 -m v2.27b.moe.screen_rules --reference "$REF/READOUT.json" "${args[@]}" --output "$R/screen/SCREEN.json"
else
  printf '{"schema": "decision2-27b-moe-screen/1", "continue": [], "seed2": null, "stop": [], "note": "no cell reached the screen point"}\n' > "$R/screen/SCREEN.json"
fi
cp "$R/screen/SCREEN.json" "$X/screen/SCREEN.json"
python3 - "$R/screen/SCREEN.json" <<'EOF' | while read -r cell; do
import json, sys
screen = json.load(open(sys.argv[1]))
for name in screen.get("stop", []):
    if name == "MOE-Qpt-s1":
        print(name)
EOF
  log "screen stops $cell"
  echo "screen rule (SCREEN.json)" > "$R/$cell/STOP"
  docker stop -t 60 "d2-27b-moe-$cell-full" || true
done
log "screen node B done: $(python3 -c "import json; d=json.load(open('$R/screen/SCREEN.json')); print(d['continue'], d['seed2'])")"
