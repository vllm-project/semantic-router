#!/usr/bin/env bash
# Decoder M11 stage-2 training chains (prereg dec-m11-stage2-prereg-2026-10-01.md), co-tenant on the stage-1 readout
# GPUs:  node F GPU7 "4b-LHB:1 4b-LHB:2"   node E GPU3 "4b-LHBx:1 4b-LHBx:2".
# The LH recipe (LoRA r128 from Qwen3.5-4B-Base + fresh head, own-Lux KL 1.0 partial) on the arm's stage-2 TRAIN,
# re-hashed against data/READY-4b-s2.json before each seed; seeds 20260926 / 20260927. Seed 1 pre-warms the node's
# 4b-train cache (its preflight runs before seed 2 starts on the same GPU). A chain holds chains/gpuN-s2.flock and
# writes its lease entry to gpuN.lock/owner.dec-m11-s2 (the GPU's owner file stays M11's stage-1 entry); a seed starts
# only with >= 60 GB free VRAM (co-tenant rule). Stop rules: a failed preflight stops the arm; a seed stops at the arm
# cap (6 GPU-h); no seed starts above the node's 45 GPU-h training gate.
# Markers: /data/dev2/runs/dec/m11/status/m11-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M11_NODE=e|f m11-s2chains.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M11_NODE:?set M11_NODE=e or f}
M=/data/dev2/runs/dec/m11
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m11
mkdir -p "$C" "$ST" "$M/logs"
case $NODE:$GPU in
  f:7) ARM=4b-LHB ;;
  e:3) ARM=4b-LHBx ;;
  *) echo "no M11 stage-2 chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
TAG=s2-$NODE$GPU

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M11 chain $TAG already launched"; exit 0; }
  M11_NODE=$NODE setsid nohup flock "$C/gpu$GPU-s2.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M11 stage-2 chain $TAG ($ARM) launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

REV=1001bb4d826a52d1f399e183466143f4da7b741b
BASE=/models/Qwen--Qwen3.5-4B-Base/$REV
TRAIN=$M/data/s2/$ARM/train.jsonl
TEACHER=/data/dev2/runs/dec/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl
SEEDS=(20260926 20260927)
CAP=6.0 NODE_GATE=45 MIN_FREE_GB=60
ENTRY=/data/dev2/leases/gpu$GPU.lock/owner.dec-m11-s2
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m11_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  printf 'track=dec-m11\nstatus=%s\npurpose=decoder M11 stage 2 %s (co-tenant with stage-1 readouts)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$ENTRY"
}
vram_free_gb() {
  rocm-smi -d "$GPU" --showmeminfo vram | awk -F': ' '/Total Memory/ {t=$NF} /Total Used Memory/ {u=$NF} END {printf "%d\n", (t-u)/1e9}'
}
ready() {
  python3 - "$M/data/READY-4b-s2.json" "$ARM" "$TRAIN" "$TEACHER" << 'EOF'
import hashlib, json, sys
try:
    lock = json.load(open(sys.argv[1]))
except OSError:
    sys.exit(1)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
sys.exit(0 if sha(sys.argv[3]) == lock["arms"][sys.argv[2]] and sha(sys.argv[4]) == lock["teacher_sha256"] else 1)
EOF
}
terminal() { [ -f "$ST/m11-$ARM-s$1.DONE" ] || [ -f "$ST/m11-$ARM-s$1.FAILED" ] || [ -f "$ST/m11-$ARM-s$1.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

item() {  # <seed index>
  local i=$1 seed r used t0 wd free warm=$ST/warm-4b-$NODE
  seed=${SEEDS[$((i - 1))]} r=m11-$ARM-s$i
  terminal "$i" && return 0
  ready || { stop "$r" "TRAIN or teacher differs from data/READY-4b-s2.json (or no lock)"; return 0; }
  if grep -qs "preflight failed" "$ST"/m11-"$ARM"-s?.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $ARM failed"
    return 0
  fi
  gt "$(gpuh total)" "$NODE_GATE" && { stop "$r" "node M11 GPU-h above the training gate $NODE_GATE"; return 0; }
  if [ "$i" != 1 ] && [ ! -f "$warm" ]; then
    stop "$r" "the node's 4b pre-warm marker is missing"
    return 0
  fi
  free=$(vram_free_gb)
  [ "$free" -ge $MIN_FREE_GB ] || { stop "$r" "GPU$GPU has $free GB free VRAM (< $MIN_FREE_GB)"; return 0; }
  used=$(gpuh arm "$ARM")
  lease busy "$r (training; arm cap $CAP GPU-h, arm used $used)" 150
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; $free GB free; arm used $used of $CAP)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print($used + ($(date -u +%s) - $t0) / 3600)")" "$CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m11-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  local env=(M11_NODE="$NODE" M11_CACHE=4b-train)
  [ "$i" = 1 ] && env+=(M11_WARM_MARKER="$warm")
  env "${env[@]}" bash "$OPS/m11-arm.sh" "$r" "$GPU" "$SRC" "$BASE" -- \
    --train "/runs/m11/data/s2/$ARM/train.jsonl" \
    --teacher /runs/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl --teacher-kl-weight 1.0 --teacher-partial \
    --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 \
    --init base --revision $REV --train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 \
    --lora-lr 1e-4 --head-lr 1e-4 --head-init-seed 20261001 --seed "$seed"
  kill "$wd" 2> /dev/null
  [ "$i" = 1 ] && [ ! -f "$warm" ] && echo "$r ended before its preflight finished $(date -u +%FT%TZ)" > "$warm"
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

for i in 1 2; do item "$i"; done
lease idle "chain finished" 30
log "chain finished"
