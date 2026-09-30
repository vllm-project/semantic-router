#!/usr/bin/env bash
# 27B MoE Stage A screen, node A side (host; detached). Relays each node A cell's checkpoint 892 (without
# trainer_state.pt, with a SHA256SUMS list) to node B over the temporary rsync-only link, or marks the cell ABSENT if
# its driver ended without it. Then waits for node B's SCREEN.json and applies it on node A: dropped node A cells get a
# STOP file and their full container is stopped; the seed-2 cell's second seed (20260928) starts on node A GPU4.
# Usage: moe-screen-nodeA.sh MIRROR
set -euo pipefail
MIR=$1
S=/data/dev2/src/$MIR/src/training/decision2
R=/data/dev2/runs/27b-moe
KEY=/data/dev2/tmp/27b-moe-xfer
STAGE=/data/dev2/tmp/27b-moe-relay
CELLS=("MOE-Git-s1:gemma-4-26B-A4B-it" "MOE-Qit-s1:Qwen3.5-35B-A3B")
PEER=$(cat "$KEY/peer")
RS=(rsync -a --mkpath -e "ssh -i $KEY/id_ed25519 -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes")
log() { echo "$(date -u +%FT%TZ) $*"; }
driver_ended() { [ -d "$R/$1" ] && flock -n "$R/$1/.driver.lock" true; }
declare -A done_cells=()
while [ "${#done_cells[@]}" -lt "${#CELLS[@]}" ]; do
  for spec in "${CELLS[@]}"; do
    IFS=: read -r cell _ <<< "$spec"
    [ -z "${done_cells[$cell]:-}" ] || continue
    ckpt=$R/$cell/full/run/checkpoint-0000892
    out=$STAGE/$cell
    if [ -f "$ckpt/decision_config.json" ]; then
      rm -rf "$out" && mkdir -p "$out/checkpoint-0000892"
      rsync -a --exclude trainer_state.pt "$ckpt/" "$out/checkpoint-0000892/"
      (cd "$out" && find checkpoint-0000892 -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
      "${RS[@]}" "$out/" "root@$PEER:relay/$cell/"
      date -u +%FT%TZ > "$out/RELAYED"
      "${RS[@]}" "$out/RELAYED" "root@$PEER:relay/$cell/RELAYED"
      log "relayed $cell checkpoint-0000892 ($(wc -l < "$out/SHA256SUMS") files)"
      done_cells[$cell]=relayed
    elif driver_ended "$cell"; then
      mkdir -p "$out" && date -u +%FT%TZ > "$out/ABSENT"
      "${RS[@]}" "$out/ABSENT" "root@$PEER:relay/$cell/ABSENT"
      log "$cell ended without checkpoint-0000892: ABSENT"
      done_cells[$cell]=absent
    fi
  done
  [ "${#done_cells[@]}" -lt "${#CELLS[@]}" ] && sleep 120
done
mkdir -p "$R/screen"
until "${RS[@]}" "root@$PEER:screen/SCREEN.json" "$R/screen/SCREEN.json" 2> /dev/null; do sleep 300; done
log "SCREEN.json received"
readarray -t actions < <(python3 - "$R/screen/SCREEN.json" <<'EOF'
import json, sys
screen = json.load(open(sys.argv[1]))
for name in screen.get("stop", []):
    if name in ("MOE-Git-s1", "MOE-Qit-s1"):
        print(f"stop {name}")
seed2 = screen.get("seed2")
if seed2:
    print(f"seed2 {seed2} {screen['cells'][seed2]['base']}")
EOF
)
for action in "${actions[@]}"; do
  read -r verb cell base <<< "$action"
  if [ "$verb" = stop ]; then
    log "screen stops $cell"
    echo "screen rule (SCREEN.json)" > "$R/$cell/STOP"
    docker stop -t 60 "d2-27b-moe-$cell-full" || true
  elif [ "$verb" = seed2 ]; then
    arm=${cell%-s1}-s2
    log "screen starts $arm on node A GPU4 ($base)"
    mkdir -p "$R/$arm"
    cd "$S"
    echo "=== $(date -u +%FT%TZ) $arm node a GPU4 base $base seed 20260928 experts grouped_mm (screen seed 2) mirror $MIR" >> "$R/$arm/driver.log"
    EXPERTS=grouped_mm setsid nohup bash v2/27b/moe/moe-arm.sh a "$arm" 4 "$base" 20260928 "$MIR" admit,onestep,reload,full \
      >> "$R/$arm/driver.log" 2>&1 < /dev/null &
  fi
done
log "screen node A done"
