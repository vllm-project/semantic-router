#!/usr/bin/env bash
# 9B M9 stage-2 training chains on node C (amendment 2), one per GPU, each under the GPU's flock (so a chain on a GPU
# whose stage-1 chain still runs waits for it):
#   GPU5 "L9IB:1"   GPU2 "L9IB:2"   GPU4 "L9IBX:1"   GPU1 "L9IBX:2" (after node C's L9 merges: soup/L9/DONE or FAILED)
# Arms (the L9 recipe: LoRA r128 from Qwen3.5-9B-Base + the shared-init candidate head, own-Lux KL 1.0 on x60 rows):
#   L9IB  = x60 + IB1-r3 + IB2 (IB rows gold only, --teacher-partial);
#   L9IBX = the same without the in-distribution families isarc, w2c, hover, gsm2.
# TRAIN / teacher are re-hashed against data/READY2.json before each seed; seeds 20260926 / 20260927. Stop rules as in
# stage 1: a failed preflight stops the arm; a seed stops at 5.0 GPU-h; no seed starts once the node's M9 GPU-hours
# reach 100. Markers: status/m9-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M9_NODE=c chains2.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M9_NODE:?set M9_NODE=c}
[ "$NODE" = c ] || { echo "M9 training chains run on node C" >&2; exit 2; }
M=/data/dev2/runs/9b/m9
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
mkdir -p "$C" "$ST" "$M/logs"
case $GPU in
  5) ITEMS="L9IB:1" ;;
  2) ITEMS="L9IB:2" ;;
  4) ITEMS="L9IBX:1" ;;
  1) ITEMS="L9IBX:2" ;;
  *) echo "no M9 stage-2 chain for node C GPU$GPU" >&2; exit 2 ;;
esac
TAG=c$GPU-s2

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M9 chain $TAG already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M9 stage-2 chain $TAG launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

BASE=/models/Qwen--Qwen3.5-9B-Base/68c46c4b3498877f3ef123c856ecfde50c39f404
BASE_REV=68c46c4b3498877f3ef123c856ecfde50c39f404
SEEDS=(20260926 20260927)
COMMON=(--teacher-kl-weight 1.0 --teacher-partial --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05
  --epochs 1 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --eval-batch 2
  --checkpoint-schedule even8 --selection matrix-v1 --init base --revision "$BASE_REV" --train-mode lora
  --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4 --head-lr 1e-4 --head-init-seed 20261001
  --max-length 8192)
SEED_CAP=5.0 TOTAL_GATE=100
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=9b-m9\nstatus=%s\npurpose=9B M9 %s (COORDINATION 2026-10-01 15:55)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
lease_free() {
  [ -f "$LEASE_DIR/owner" ] || return 0
  grep -qs '^track=9b-m9' "$LEASE_DIR/owner" && return 0
  grep -qsE '^status=(idle|released)' "$LEASE_DIR/owner"
}
ready() {  # <data dir name>: TRAIN and teacher equal data/READY2.json
  [ -f "$M/data/READY2.json" ] || return 1
  python3 - "$M/data/READY2.json" "$1" "$M/data/$1/train.jsonl" "$M/data/$1/teacher.jsonl" << 'EOF'
import hashlib, json, sys
lock, name, train, teacher = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
sys.exit(0 if sha(train) == lock[name]["train_sha256"] and sha(teacher) == lock[name]["teacher_sha256"] else 1)
EOF
}
terminal() { [ -f "$ST/m9-$1-s$2.DONE" ] || [ -f "$ST/m9-$1-s$2.FAILED" ] || [ -f "$ST/m9-$1-s$2.STOPPED" ]; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd data n=0
  seed=${SEEDS[$((i - 1))]} r=m9-$g-s$i
  case $g in L9IB) data=l9ib ;; L9IBX) data=l9ibx ;; esac
  terminal "$g" "$i" && return 0
  if [ "$GPU" = 1 ]; then
    until [ -f "$M/soup/L9/DONE" ] || [ -f "$M/soup/L9/FAILED" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for node C's L9 merges on GPU1"
      n=$((n + 1))
      sleep 60
    done
  fi
  if ! lease_free; then
    echo "not started: gpu$GPU lease is held by another track" > "$ST/$r.STOPPED"
    log "$r not started: lease of GPU$GPU not free"
    return 0
  fi
  if ! ready "$data"; then
    echo "TRAIN or teacher differs from data/READY2.json (or no lock); not started" > "$ST/$r.STOPPED"
    log "$r not started: data lock check failed"
    return 0
  fi
  if grep -qs "preflight failed" "$ST"/m9-"$g"-s?.FAILED 2> /dev/null; then
    echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
    log "$r not started: arm $g stopped by a failed preflight"
    return 0
  fi
  used=$(python3 "$OPS/gpuh.py" total --running)
  if ! gt "$TOTAL_GATE" "$used"; then
    echo "not started: node M9 GPU-h $used reached $TOTAL_GATE" > "$ST/$r.STOPPED"
    log "$r not started: node GPU-h $used >= $TOTAL_GATE"
    return 0
  fi
  lease busy "$r (training; seed cap $SEED_CAP GPU-h)" 260
  log "start $r on node C GPU$GPU (seed $seed; node M9 GPU-h used $used)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print(($(date -u +%s) - $t0) / 3600)")" "$SEED_CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the seed cap $SEED_CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m9-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  M9_NODE=$NODE M9_PREWARM=0 bash "$OPS/arm.sh" "$r" "$GPU" "$SRC" "$BASE" -- \
    --train "/runs/m9/data/$data/train.jsonl" --teacher "/runs/m9/data/$data/teacher.jsonl" "${COMMON[@]}" --seed "$seed"
  kill "$wd" 2> /dev/null
  if grep -qE "^[^ ]+ $r full run complete" "$M/arms/OPERATIONS.log"; then
    echo "complete $(date -u +%FT%TZ)" > "$ST/$r.DONE"
    log "$r DONE"
  elif grep -qE "^[^ ]+ $r (zero-step FAILED|one-step FAILED|gate job FAILED|preflight FAIL)" "$M/arms/OPERATIONS.log"; then
    echo "preflight failed" > "$ST/$r.FAILED"
    log "$r FAILED (preflight)"
  elif [ -f "$ST/$r.capstop" ]; then
    echo "stopped at the seed cap" > "$ST/$r.STOPPED"
    log "$r STOPPED (cap)"
  else
    echo "full run failed" > "$ST/$r.FAILED"
    log "$r FAILED (full run)"
  fi
}

for it in $ITEMS; do
  item "${it%%:*}" "${it##*:}"
done
lease idle "stage-2 chain finished; GPU idle (merges may follow)" 30
log "chain finished"
