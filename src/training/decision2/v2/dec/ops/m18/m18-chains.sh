#!/usr/bin/env bash
# Decoder M18 training chains (prereg dec-m18-prereg-2026-10-02.md, "Part B"), one per node-F GPU. A chain waits for
# the node's M18 Index pool to finish (the GPU is shared with it), then holds its GPU's flock for its whole run, keeps
# the GPU's lease owner file current (track=dec-m18) and runs its items in order:
#   GPU4 "2b-RS17UP:1 2b-RAUPM:1" (pre-warms F's 2B train cache)  GPU5 "2b-RS17UP:2 2b-RAUPM:2"
# Recipe (M14 2b-RAUP's): full fine-tune from Sol 1.0@ce0c018a (--init decision1), backbone LR 5e-6 / head LR 5e-5,
# the arm's RAUP weights (--example-weights), --teacher-partial (IB rows gold only); teacher: 2b-RS17UP the swap's SD
# targets at KL 1.0, 2b-RAUPM the own-Sol teacher at KL 0.5.
# Every seed re-hashes its TRAIN, weights and teacher against data/READY-m18.json; seeds 20260926 / 20260927.
# Stop rules: a failed preflight stops the arm; a seed stops at the arm cap (2.5 GPU-h); no seed starts above the
# training gate (24 GPU-h on this node). Markers: /data/dev2/runs/dec/m18/status/m18-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M18_NODE=f m18-chains.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M18_NODE:?set M18_NODE=f}
M=/data/dev2/runs/dec/m18
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m18
mkdir -p "$C" "$ST" "$M/logs"
PREWARMS=""
case $NODE:$GPU in
  f:4) ITEMS="2b-RS17UP:1 2b-RAUPM:1" PREWARMS="2b-RS17UP:1" ;;
  f:5) ITEMS="2b-RS17UP:2 2b-RAUPM:2" ;;
  *) echo "no M18 chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
TAG=$NODE$GPU

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M18 chain $TAG already launched"; exit 0; }
  M18_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M18 chain $TAG ($ITEMS) launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

ONE_2b=/models/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
SEEDS=(20260926 20260927)
BATCH="--batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
GATE=24 ARM_CAP=2.5 WARM_WAIT_MIN=360 POOL_WAIT_MIN=240
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m18_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=dec-m18\nstatus=%s\npurpose=decoder M18 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
ready() {  # <ARM>: TRAIN, weights and teacher equal the committed data lock
  python3 - "$M/data/READY-m18.json" "$1" "$M/data/2b/$1" << 'EOF'
import hashlib, json, sys
lock_path, arm, d = sys.argv[1:]
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
want = lock["arms"].get(arm)
ok = bool(want) and all(sha(f"{d}/{k}.jsonl") == want[k] for k in ("train", "weights", "teacher"))
sys.exit(0 if ok else 1)
EOF
}
args_for() {  # <ARM>: start path, then the trainer arguments
  local d=/runs/m18/data/2b/$1 kl
  case $1 in 2b-RS17UP) kl=1.0 ;; 2b-RAUPM) kl=0.5 ;; esac
  echo "$ONE_2b --train $d/train.jsonl --example-weights $d/weights.jsonl --teacher $d/teacher.jsonl --teacher-kl-weight $kl --teacher-partial $BATCH --init decision1 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5"
}
terminal() { [ -f "$ST/m18-$1-s$2.DONE" ] || [ -f "$ST/m18-$1-s$2.FAILED" ] || [ -f "$ST/m18-$1-s$2.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

n=0
while pgrep -f m18_ixpool.py > /dev/null; do
  [ $((n % 15)) = 0 ] && log "waits for the node's M18 Index pool to finish"
  n=$((n + 1))
  [ $n -gt $POOL_WAIT_MIN ] && { log "the Index pool is still running after $POOL_WAIT_MIN min; chain ends"; exit 1; }
  sleep 60
done

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd start rest n pw=0 warm=$ST/warm-2b-$NODE
  [[ " $PREWARMS " == *" $1:$2 "* ]] && pw=1
  seed=${SEEDS[$((i - 1))]} r=m18-$g-s$i
  terminal "$g" "$i" && return 0
  ready "$g" || { stop "$r" "TRAIN, weights or teacher differ from data/READY-m18.json (or no lock)"; return 0; }
  if grep -qs "preflight failed" "$ST"/m18-"$g"-s?.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $g failed"
    return 0
  fi
  gt "$(gpuh total)" "$GATE" && { stop "$r" "M18 GPU-h on this node above the gate $GATE"; return 0; }
  if [ "$pw" != 1 ]; then
    n=0
    lease busy "$r waits for the 2b pre-warm (training next)" 60
    until [ -f "$warm" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the node's 2b pre-warm marker"
      n=$((n + 1))
      [ $n -gt $WARM_WAIT_MIN ] && { stop "$r" "the 2b pre-warm marker never appeared"; return 0; }
      sleep 60
    done
    if grep -qs "preflight failed" "$ST"/m18-"$g"-s?.FAILED 2> /dev/null; then
      stop "$r" "a preflight of arm $g failed"
      return 0
    fi
  fi
  used=$(gpuh arm "$g")
  lease busy "$r (training; arm cap $ARM_CAP GPU-h, arm used $used)" 120
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; arm used $used of $ARM_CAP)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print($used + ($(date -u +%s) - $t0) / 3600)")" "$ARM_CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $ARM_CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m18-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  read -r start rest <<< "$(args_for "$g")"
  local env=(M18_NODE="$NODE" M18_CACHE="2b-train")
  [ "$pw" = 1 ] && env+=(M18_WARM_MARKER="$warm")
  # shellcheck disable=SC2086
  env "${env[@]}" bash "$OPS/m18-arm.sh" "$r" "$GPU" "$SRC" "$start" -- $rest --seed "$seed"
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
lease idle "chain finished; GPU idle" 30
log "chain finished"
