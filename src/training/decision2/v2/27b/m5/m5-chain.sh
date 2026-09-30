#!/usr/bin/env bash
# ~27B M5 post-training chain on node B (host side): waits for an arm's two seeds, then builds and reads out its
# candidates through m5-tail.sh. Formal runs and the B1 decision are separate, recorded steps.
# Usage: m5-chain.sh MIRROR_SHA ARM GPU [GPU2]
#   ARM FF20:  wait for FF20-s1 (node B) and the FF20-s2 relay (node A) -> pull FF20-s2 -> soup M5-FF20 -> readout
#              M5-FF20 on GPU -> devgates M5-FF20
#   ARM FF20H: the same for FF20H, plus soup M5-SX (FF20-s1, FF20-s2, FF20H-s1, FF20H-s2) and its readout on GPU2 ->
#              devgates M5-FF20 M5-FF20H M5-SX
# A seed missing because of a stop rule (STOP-<ARM> or a failed arm-seed) ends the chain with a record; nothing reruns.
set -euo pipefail
echo "m5 chain $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM} GPU=${3:?GPU} GPU2=${4:-}
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
TAIL=$S/v2/27b/m5/m5-tail.sh
R=/data/dev2/runs/27b/m5
KEY=/data/dev2/tmp/27b-m5-xfer
case "$ARM" in FF20 | FF20H) ;; *) echo "ARM must be FF20 or FF20H" >&2; exit 2 ;; esac
[ "$ARM" = FF20 ] || [ -n "$GPU2" ] || { echo "FF20H needs a second GPU for M5-SX's readout" >&2; exit 2; }
best() {  # RUN_ROOT -> its BEST checkpoint path (node B runs)
  local run
  run=$1/full/$(cat "$1/full/RUN_DIR")
  echo "$run/$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$run/BEST.json")"
}
relay_ready() {  # ARM-SEED: node A's relay directory has its SHA-256 list
  rsync --list-only -e "ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes" \
    "root@$(cat "$KEY/peer"):relay/$1/SHA256SUMS" > /dev/null 2>&1
}
wait_for() {  # ARM: its node B seed complete and its node A seed relayed
  while :; do
    [ -e "$R/STOP-$1" ] && { echo "$(date -u +%FT%TZ) $1 stopped: $(cat "$R/STOP-$1")"; exit 3; }
    local s1=no s2=no
    [ -f "$R/$1-s1/full/RUN_DIR" ] && s1=yes
    relay_ready "$1-s2" && s2=yes
    [ "$s1:$s2" = yes:yes ] && return
    sleep 300
  done
}
wait_for "$ARM"
echo "$(date -u +%FT%TZ) $ARM seeds ready"
bash "$TAIL" pull "$SHA" "$ARM-s2"
S1=$(best "$R/$ARM-s1") S2=$R/relay/$ARM-s2/checkpoint
echo "$(date -u +%FT%TZ) soup M5-$ARM from $S1 and $S2"
[ -f "$R/M5-$ARM/verify.json" ] || bash "$TAIL" soup "$SHA" "M5-$ARM" "$S1" "$S2"
names=("M5-$ARM")
if [ "$ARM" = FF20H ]; then
  F1=$(best "$R/FF20-s1") F2=$R/relay/FF20-s2/checkpoint
  [ -f "$R/M5-SX/verify.json" ] || bash "$TAIL" soup "$SHA" M5-SX "$F1" "$F2" "$S1" "$S2"
  bash "$TAIL" readout "$SHA" M5-SX "$R/M5-SX/checkpoint" "$GPU2" > "$R/logs/readout-M5-SX.log" 2>&1 &
  sx=$!
  names=(M5-FF20 M5-FF20H M5-SX)
fi
bash "$TAIL" readout "$SHA" "M5-$ARM" "$R/M5-$ARM/checkpoint" "$GPU"
[ -z "${sx:-}" ] || wait "$sx"
available=()
for n in "${names[@]}"; do [ -f "$R/readouts/$n/READOUT-M4B.json" ] && available+=("$n"); done
bash "$TAIL" devgates "$SHA" "${available[@]}"
echo "m5 chain $ARM complete: $(date -u +%FT%TZ)"
