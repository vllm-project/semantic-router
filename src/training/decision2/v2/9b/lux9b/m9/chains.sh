#!/usr/bin/env bash
# 9B M9 stage-1 training chains on node C (prereg lux9b-m9-prereg-2026-10-01.md), one per GPU. A chain holds its
# GPU's flock for its whole run, keeps the GPU's lease owner file current and runs its item:
#   GPU1 "L9:1" (alone first: its one-step run writes status/prewarm.DONE)   GPU2 "L9:2"
#   GPU3 "L9L:1"   GPU4 "L9L:2"   GPU5 "B0:1" (zero-step only: the untrained base, label-token readout)
# Arms: L9 = LoRA r128 from Qwen3.5-9B-Base + fresh candidate head (shared init seed); L9L = the same LoRA on Lux 1.0
# (Lux's head continued). Both: x60 TRAIN + own-Lux targets on every row (K recipe), re-hashed against
# data/READY.json before each seed; seeds 20260926 / 20260927.
# Stop rules: a failed preflight stops the arm (no rerun, no replacement seed); a seed stops at 4.5 GPU-h; no seed
# starts once the node's M9 GPU-hours reach 100. Markers: /data/dev2/runs/9b/m9/status/m9-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M9_NODE=c chains.sh launch <mirror-dir> <gpu>
#        M9_NODE=c chains.sh run <mirror-dir> <gpu>   (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M9_NODE:?set M9_NODE=c}
[ "$NODE" = c ] || { echo "M9 training chains run on node C" >&2; exit 2; }
M=/data/dev2/runs/9b/m9
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9
mkdir -p "$C" "$ST" "$M/logs"
case $GPU in
  1) ITEMS="L9:1" ;;
  2) ITEMS="L9:2" ;;
  3) ITEMS="L9L:1" ;;
  4) ITEMS="L9L:2" ;;
  5) ITEMS="B0:1" ;;
  *) echo "no M9 chain for node C GPU$GPU" >&2; exit 2 ;;
esac
TAG=c$GPU

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M9 chain $TAG already launched"; exit 0; }
  M9_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M9 chain $TAG launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

BASE=/models/Qwen--Qwen3.5-9B-Base/68c46c4b3498877f3ef123c856ecfde50c39f404
BASE_REV=68c46c4b3498877f3ef123c856ecfde50c39f404
SEEDS=(20260926 20260927)
HEAD_INIT_SEED=20261001
TRAIN=$M/data/x60/train.jsonl
TEACHER=$M/data/x60/teacher.jsonl
COMMON=(--train /runs/m9/data/x60/train.jsonl --teacher /runs/m9/data/x60/teacher.jsonl --teacher-kl-weight 1.0
  --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05 --epochs 1 --batching tokens --max-batch-tokens 32768
  --max-batch-rows 64 --update-rows 64 --eval-batch 2 --checkpoint-schedule even8 --selection matrix-v1)
LORA=(--train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4)
SEED_CAP=4.5 TOTAL_GATE=100
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=9b-m9\nstatus=%s\npurpose=9B M9 %s (COORDINATION 2026-10-01 14:45)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
lease_free() {  # the GPU's owner file is absent, ours, idle or released
  [ -f "$LEASE_DIR/owner" ] || return 0
  grep -qs '^track=9b-m9' "$LEASE_DIR/owner" && return 0
  grep -qsE '^status=(idle|released)' "$LEASE_DIR/owner"
}
ready() {  # TRAIN and teacher equal the data lock
  [ -f "$M/data/READY.json" ] || return 1
  python3 - "$M/data/READY.json" "$TRAIN" "$TEACHER" << 'EOF'
import hashlib, json, sys
lock = json.load(open(sys.argv[1]))
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
sys.exit(0 if sha(sys.argv[2]) == lock["train_sha256"] and sha(sys.argv[3]) == lock["teacher_sha256"] else 1)
EOF
}
args_for() {  # <ARM>: start path, then the arm's trainer arguments
  case $1 in
    L9) echo "$BASE --init base --revision $BASE_REV ${LORA[*]} --head-lr 1e-4 --head-init-seed $HEAD_INIT_SEED --max-length 8192" ;;
    L9L) echo "/lux --init decision1 ${LORA[*]} --head-lr 1e-4 --max-length 8192" ;;
    B0) echo "$BASE --zero-only --init base --revision $BASE_REV ${LORA[*]} --readout label_token --max-length 9216" ;;
  esac
}
terminal() { [ -f "$ST/m9-$1-s$2.DONE" ] || [ -f "$ST/m9-$1-s$2.FAILED" ] || [ -f "$ST/m9-$1-s$2.STOPPED" ]; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd start zero="" n=0
  seed=${SEEDS[$((i - 1))]} r=m9-$g-s$i
  terminal "$g" "$i" && return 0
  if ! lease_free; then
    echo "not started: gpu$GPU lease is held by another track" > "$ST/$r.STOPPED"
    log "$r not started: lease of GPU$GPU not free ($(head -c 200 "$LEASE_DIR/owner" | tr '\n' ' '))"
    return 0
  fi
  if ! ready; then
    echo "TRAIN or teacher differs from data/READY.json (or no lock); not started" > "$ST/$r.STOPPED"
    log "$r not started: data lock check failed"
    return 0
  fi
  if [ "$r" != m9-L9-s1 ]; then
    until [ -f "$ST/prewarm.DONE" ] || [ -f "$ST/m9-L9-s1.FAILED" ] || [ -f "$ST/m9-L9-s1.STOPPED" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the pre-warm run (L9-s1 one-step)"
      n=$((n + 1))
      sleep 60
    done
  fi
  if grep -qs "preflight failed" "$ST"/m9-"$g"-s?.FAILED 2> /dev/null; then
    echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
    log "$r not started: arm $g stopped by a failed preflight"
    return 0
  fi
  used=$(gpuh total)
  if ! gt "$TOTAL_GATE" "$used"; then
    echo "not started: node M9 GPU-h $used reached $TOTAL_GATE" > "$ST/$r.STOPPED"
    log "$r not started: node GPU-h $used >= $TOTAL_GATE"
    return 0
  fi
  lease busy "$r (training; seed cap $SEED_CAP GPU-h)" 240
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
  read -r start rest <<< "$(args_for "$g")"
  if [[ $rest == --zero-only* ]]; then zero=--zero-only; rest=${rest#--zero-only }; fi
  # shellcheck disable=SC2086
  M9_NODE=$NODE M9_PREWARM=$([ "$r" = m9-L9-s1 ] && echo 1 || echo 0) \
    bash "$OPS/arm.sh" "$r" "$GPU" "$SRC" "$start" $zero -- "${COMMON[@]}" $rest --seed "$seed"
  kill "$wd" 2> /dev/null
  if grep -qE "^[^ ]+ $r (full run complete|zero-only seed complete)" "$M/arms/OPERATIONS.log"; then
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
lease idle "chain finished; GPU idle (merges may follow)" 30
log "chain finished"
