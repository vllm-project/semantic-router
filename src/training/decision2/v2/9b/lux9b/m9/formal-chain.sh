#!/usr/bin/env bash
# 9B M9 formal chain on node A (prereg "Formal and successor"). Waits for the development rules
# (select/9b-finalists.json). No finalist: stops. Otherwise: (1) the formal-path parity run of DEV2.0-9B's weights
# (formal.sh C0F on GPU6) and its answer comparison with the stored bar run release/dev2-8b-t1-derived
# (formal-m9/C0F.parity.json: 0 answer differences -> the stored run is the bar); (2) after the manual marker
# status/formal.GO (written once the newest COORDINATION notes are re-read and the finalists' lock record is pushed),
# formal.sh for each finalist (first on GPU6, second on GPU7, in parallel). A failed step stops its branch.
#
# Stage 2 (amendment 2): M9_STAGE=2 reads select/9b-finalists-s2.json and waits for status/formal-s2.GO.
# Stage 3 (amendment 3): M9_STAGE=3 reads select/9b-finalists-s3.json and waits for status/formal-s3.GO.
# Stage 4 (amendment 4): M9_STAGE=4 reads select/9b-finalists-s4.json and waits for status/formal-s4.GO.
#
# usage: [M9_STAGE=2|3|4] M9_NODE=a formal-chain.sh launch|run <mirror-dir>
set -u
MODE=$1 SRC=$2
STAGE=${M9_STAGE:-1}
case $STAGE in
  2) RULES=9b-finalists-s2 GO=formal-s2.GO TAG=-s2 ;;
  3) RULES=9b-finalists-s3 GO=formal-s3.GO TAG=-s3 ;;
  4) RULES=9b-finalists-s4 GO=formal-s4.GO TAG=-s4 ;;
  *) RULES=9b-finalists GO=formal.GO TAG="" ;;
esac
M=/data/dev2/runs/9b/m9
ST=$M/status
F=/data/dev2/runs/9b/formal-m9
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
CODE=/data/dev2/src/$SRC/src/training/decision2
LUX=/data/decision20-20260926/models/Decision-1.0-Lux-9B
BASE=/data/dev2/models/Qwen--Qwen3.5-9B-Base/68c46c4b3498877f3ef123c856ecfde50c39f404
T1=/data/dev2/runs/release/dev2-8b-t1-derived
mkdir -p "$M/chains" "$M/logs" "$ST" "$F"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/formal$TAG.lock" 2> /dev/null || { echo "formal chain$TAG already launched"; exit 0; }
  M9_STAGE=$STAGE M9_NODE=a setsid nohup bash "$0" run "$SRC" > "$M/logs/formal-chain$TAG.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/formal$TAG.pid"
  echo "$(date -u +%FT%TZ) M9 formal chain$TAG launched from $SRC (pid $(cat "$M/chains/formal$TAG.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) formal-chain$TAG $*" | tee -a "$M/OPERATIONS.log"; }
n=0
until [ -f "$M/select/$RULES.json" ]; do
  n=$((n + 1))
  [ $((n % 30)) = 0 ] && log "waiting for the development rules"
  sleep 60
done
mapfile -t FIN < <(python3 -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["finalists"]))' "$M/select/$RULES.json")
[ ${#FIN[@]} -gt 0 ] && [ -n "${FIN[0]}" ] || { log "no finalist: no formal run"; exit 0; }
log "finalists: ${FIN[*]}"
if [ ! -f "$F/C0F.parity.json" ]; then
  bash "$OPS/formal.sh" 6 C0F /data/dev2/runs/9b/m4/K-a13-build/soup "$LUX" "DEV2.0-9B (M9 formal-path parity)" \
    > "$M/logs/formal-C0F.log" 2>&1 || { log "parity run C0F failed (see logs/formal-C0F.log)"; exit 1; }
  python3 "$CODE/v2/dec/ops/m10/m10_compare.py" \
    --pair "typed-final=$F/C0F-16k/output/typed-final.predictions.jsonl,$T1/output/typed-final.predictions.jsonl" \
    --pair "css15=$F/C0F-16k/output/css15.predictions.jsonl,$T1/output/css15.predictions.jsonl" \
    --pair "public231=$F/C0F-16k/output/public231.predictions.jsonl,$T1/output/public231.predictions.jsonl" \
    --output "$F/C0F.parity.json" > "$M/logs/formal-C0F-parity.log" 2>&1 || { log "parity comparison failed"; exit 1; }
  log "C0F parity vs the stored bar: $(tail -c 400 "$M/logs/formal-C0F-parity.log" | tr '\n' ' ')"
fi
n=0
until [ -f "$ST/$GO" ]; do
  n=$((n + 1))
  [ $((n % 30)) = 0 ] && log "waiting for status/$GO (COORDINATION re-read + lock record)"
  sleep 60
done
gpus=(6 7)
pids=()
for i in "${!FIN[@]}"; do
  a=${FIN[$i]}
  art=$(cat "$M/soup/$a/DONE" 2> /dev/null) || { log "$a has no artifact on node A"; continue; }
  case $a in L9 | L9IB | L9IBX) src=$BASE ;; *) src=$LUX ;; esac
  bash "$OPS/formal.sh" "${gpus[$i]}" "$a" "$art" "$src" "9B M9 $a" > "$M/logs/formal-$a.log" 2>&1 &
  pids+=($!)
  log "formal $a started on GPU${gpus[$i]} (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; done
log "formal runs finished: $(for a in "${FIN[@]}"; do printf '%s=%s ' "$a" "$(test -f "$F/$a.gates/successor.json" && echo summary || echo missing)"; done)"
