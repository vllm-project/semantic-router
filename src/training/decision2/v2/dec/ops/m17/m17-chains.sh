#!/usr/bin/env bash
# Decoder M17 training chains (prereg dec-m17-prereg-2026-10-02.md, "Arms" / "GPUs, order, budget"), one per node-F
# GPU. A chain holds its GPU's flock for its whole run, keeps the GPU's lease owner file current and runs its item:
#   GPU6 "4b-LHS10SD:1" (pre-warms F's 4B train cache)  GPU7 "4b-LHS10SD:2"  GPU2 "4b-LHS17SD:1"  GPU3 "4b-LHS17SD:2"
# Recipe (M13 4b-LHA10SD's, unchanged): LoRA r128 / alpha 256 / dropout .05 from Qwen3.5-4B-Base@1001bb4d with a fresh
# head (seed 20261001), LoRA / head LR 1e-4; KL 1.0 to the released LH's self-distillation targets on the kept released
# rows (the arm's teacher-s.jsonl), --teacher-partial (IB rows gold only).
# Every seed re-hashes its TRAIN and teacher against data/READY-m17.json; seeds 20260926 / 20260927.
# Pre-warm: the pre-warm seed's preflight runs alone on the 4B train cache; the other seeds wait for
# status/warm-4b-f. Stop rules: a failed preflight stops the arm (no rerun, no replacement seed); a seed stops at the
# arm cap (5.0 GPU-h); no seed starts above the M17 gate (45 GPU-h).
# Markers: /data/dev2/runs/dec/m17/status/m17-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# Stage 2 (M17_STAGE=2; prereg dec-m17-stage2-prereg-2026-10-02.md): the same recipe and stop rules on data/4b-s2,
# locked by data/READY-m17s2.json; an arm with weights.jsonl adds --example-weights (UP); the M17 gate on node F is 50
# GPU-h. The node's 4B train cache is already warm (stage 1), so no stage-2 item pre-warms. M17_STAGE=3 is stage 2's
# wave 2 (arm (a), data/4b-s3, READY-m17s3.json); its chain waits on the GPU's flock until wave 1's chain ends.
# M17_STAGE=4 is wave 3 on the released 4b-LHA10SDML base (amendment 2; data/4b-s4, READY-m17s4.json).
# usage: M17_NODE=f [M17_STAGE=2] m17-chains.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M17_NODE:?set M17_NODE=f}
STAGE=${M17_STAGE:-1}
M=/data/dev2/runs/dec/m17
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
mkdir -p "$C" "$ST" "$M/logs"
PREWARMS=""
DATA=4b READY=READY-m17.json GATE=45 TAG=$NODE$GPU
case $STAGE:$NODE:$GPU in
  1:f:6) ITEMS="4b-LHS10SD:1" PREWARMS="4b-LHS10SD:1" ;;
  1:f:7) ITEMS="4b-LHS10SD:2" ;;
  1:f:2) ITEMS="4b-LHS17SD:1" ;;
  1:f:3) ITEMS="4b-LHS17SD:2" ;;
  2:f:2) ITEMS="4b-LHS17UP:1" ;;
  2:f:3) ITEMS="4b-LHS17UP:2" ;;
  2:f:6) ITEMS="4b-LHS23SD:1" ;;
  2:f:7) ITEMS="4b-LHS23SD:2" ;;
  3:f:2) ITEMS="4b-LHS17IB4:1" ;;
  3:f:3) ITEMS="4b-LHS17IB4:2" ;;
  3:f:6) ITEMS="4b-LHS17IB4X:1" ;;
  3:f:7) ITEMS="4b-LHS17IB4X:2" ;;
  4:f:2) ITEMS="4b-SDMLIB4:1" ;;
  4:f:3) ITEMS="4b-SDMLIB4:2" ;;
  4:f:6) ITEMS="4b-LHS17ML:1" ;;
  4:f:7) ITEMS="4b-LHS17ML:2" ;;
  *) echo "no M17 stage-$STAGE chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
case $STAGE in
  2) DATA=4b-s2 READY=READY-m17s2.json GATE=50 TAG=s2-$NODE$GPU ;;
  3) DATA=4b-s3 READY=READY-m17s3.json GATE=50 TAG=s3-$NODE$GPU ;;
  4) DATA=4b-s4 READY=READY-m17s4.json GATE=50 TAG=s4-$NODE$GPU ;;
esac

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M17 chain $TAG already launched"; exit 0; }
  M17_NODE=$NODE M17_STAGE=$STAGE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M17 chain $TAG ($ITEMS) launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

REV_4b=1001bb4d826a52d1f399e183466143f4da7b741b
BASE_4b=/models/Qwen--Qwen3.5-4B-Base/$REV_4b
SEEDS=(20260926 20260927)
BATCH="--batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
ARM_CAP=5.0 WARM_WAIT_MIN=360
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m17_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=dec-m17\nstatus=%s\npurpose=decoder M17 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
ready() {  # <ARM>: TRAIN and teacher equal the committed data lock
  python3 - "$M/data/$READY" "$1" "$M/data/$DATA/$1/train.jsonl" "$M/data/$DATA/$1/teacher-s.jsonl" \
    "$M/data/$DATA/$1/weights.jsonl" << 'EOF'
import hashlib, json, sys
lock_path, arm, train, teacher, weights = sys.argv[1:]
import os
try:
    lock = json.load(open(lock_path))
except OSError:
    sys.exit(1)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
ok = arm in lock["arms"] and arm in lock["teachers"]
ok = ok and sha(train) == lock["arms"][arm] and sha(teacher) == lock["teachers"][arm]
want = lock.get("weights", {}).get(arm)
ok = ok and (sha(weights) == want if want else not os.path.exists(weights))
sys.exit(0 if ok else 1)
EOF
}
args_for() {  # <ARM>: start path, then the trainer arguments
  local d=/runs/m17/data/$DATA/$1 w=""
  [ -f "$M/data/$DATA/$1/weights.jsonl" ] && w="--example-weights $d/weights.jsonl "
  echo "$BASE_4b --train $d/train.jsonl ${w}--teacher $d/teacher-s.jsonl --teacher-kl-weight 1.0 --teacher-partial $BATCH --init base --revision $REV_4b --train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4 --head-lr 1e-4 --head-init-seed 20261001"
}
terminal() { [ -f "$ST/m17-$1-s$2.DONE" ] || [ -f "$ST/m17-$1-s$2.FAILED" ] || [ -f "$ST/m17-$1-s$2.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd start rest n pw=0 warm=$ST/warm-4b-$NODE
  [[ " $PREWARMS " == *" $1:$2 "* ]] && pw=1
  seed=${SEEDS[$((i - 1))]} r=m17-$g-s$i
  terminal "$g" "$i" && return 0
  ready "$g" || { stop "$r" "TRAIN, teacher or weights differ from data/$READY (or no lock)"; return 0; }
  if grep -qs "preflight failed" "$ST"/m17-"$g"-s?.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $g failed"
    return 0
  fi
  gt "$(gpuh total)" "$GATE" && { stop "$r" "M17 GPU-h on this node above the gate $GATE"; return 0; }
  if [ "$pw" != 1 ]; then
    n=0
    lease busy "$r waits for the 4b pre-warm (training next)" 60
    until [ -f "$warm" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the node's 4b pre-warm marker"
      n=$((n + 1))
      [ $n -gt $WARM_WAIT_MIN ] && { stop "$r" "the 4b pre-warm marker never appeared"; return 0; }
      sleep 60
    done
    if grep -qs "preflight failed" "$ST"/m17-"$g"-s?.FAILED 2> /dev/null; then
      stop "$r" "a preflight of arm $g failed"
      return 0
    fi
  fi
  used=$(gpuh arm "$g")
  lease busy "$r (training; arm cap $ARM_CAP GPU-h, arm used $used)" 180
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; arm used $used of $ARM_CAP)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print($used + ($(date -u +%s) - $t0) / 3600)")" "$ARM_CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $ARM_CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m17-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  read -r start rest <<< "$(args_for "$g")"
  local env=(M17_NODE="$NODE" M17_CACHE="4b-train")
  [ "$pw" = 1 ] && env+=(M17_WARM_MARKER="$warm")
  # shellcheck disable=SC2086
  env "${env[@]}" bash "$OPS/m17-arm.sh" "$r" "$GPU" "$SRC" "$start" -- $rest --seed "$seed"
  kill "$wd" 2> /dev/null
  [ "$pw" = 1 ] && [ ! -f "$warm" ] && echo "$r ended before its preflight finished $(date -u +%FT%TZ)" > "$warm"
  if grep -qE "^[^ ]+ $r full run complete" "$M/arms/OPERATIONS.log"; then
    echo "complete $(date -u +%FT%TZ)" > "$ST/$r.DONE"
    log "$r DONE"
  elif grep -qE "^[^ ]+ $r (zero-step FAILED|one-step FAILED|gate job FAILED|preflight FAIL)" "$M/arms/OPERATIONS.log"; then
    echo "preflight failed" > "$ST/$r.FAILED"
    log "$r FAILED (preflight)"
  elif [ -f "$ST/$r.capstop" ]; then
    echo "stopped at the arm cap" > "$ST/$r.STOPPED"
    log "$r STOPPED (cap)"
  else
    echo "full run failed" > "$ST/$r.FAILED"
    log "$r FAILED (full run)"
  fi
}

for it in $ITEMS; do
  item "${it%%:*}" "${it##*:}"
done
lease idle "chain finished; GPU idle (readouts may follow)" 30
log "chain finished"
