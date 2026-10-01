#!/usr/bin/env bash
# Decoder M13 training chains (prereg dec-m13-prereg-2026-10-01.md, "Training" / "GPUs and order"), one per M13 GPU.
# A chain holds its GPU's flock for its whole run, keeps the GPU's lease owner file current and runs its items:
#   node F: GPU2 "4b-LHA5:1 2b-RASD:1" (pre-warm 4b, then 2b)  GPU3 "4b-LHA5:2 2b-RASD:2"
#           GPU6 "4b-LHA10SD:1"  GPU7 "4b-LHA10SD:2"
#   node E: GPU0 "08b-RAAG:1" (pre-warms 08b)  GPU1 "08b-RAAG:2"  GPU2 "08b-RASD:1"  GPU3 "08b-RASD:2"
# Recipes (each tier's released recipe, as M12): 4b = LH (LoRA r128 / alpha 256 from Qwen3.5-4B-Base + fresh head,
# LoRA / head LR 1e-4, KL 1.0 --teacher-partial): 4b-LHA5 keeps LH's own-Lux teacher, 4b-LHA10SD uses the LH
# self-distillation targets instead; 2b = S2T (Sol 1.0 full fine-tune, backbone 5e-6 / head 5e-5, --teacher-partial):
# 2b-RASD uses the DEV2.0-2B self-distillation targets at KL 1.0 instead of M12's own-Sol 0.5; 08b = E8F (Eos 1.0 full
# fine-tune, backbone 1e-5 / head 1e-4): 08b-RAAG has no teacher (as 08b-RA), 08b-RASD the DEV2.0-0.8B targets at
# KL 1.0 --teacher-partial. Self-distillation targets cover exactly the tier's released TRAIN rows; IB rows stay gold.
# Every seed re-hashes its TRAIN (and teacher) against data/READY-m13.json; seeds 20260926 / 20260927.
# Pre-warm: on each (node, tier) the pre-warm seed's preflight runs alone on that tier's train cache; the node's other
# seeds of that tier wait for status/warm-<tier>-<node>. Stop rules: a failed preflight stops the arm (no rerun, no
# replacement seed); a seed stops at its arm cap; no seed starts above the node's training gate (30 GPU-h).
# Markers: /data/dev2/runs/dec/m13/status/m13-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M13_NODE=e|f m13-chains.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M13_NODE:?set M13_NODE=e or f}
M=/data/dev2/runs/dec/m13
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m13
mkdir -p "$C" "$ST" "$M/logs"
PREWARMS=""
case $NODE:$GPU in
  f:2) ITEMS="4b-LHA5:1 2b-RASD:1" PREWARMS="4b-LHA5:1 2b-RASD:1" ;;
  f:3) ITEMS="4b-LHA5:2 2b-RASD:2" ;;
  f:6) ITEMS="4b-LHA10SD:1" ;;
  f:7) ITEMS="4b-LHA10SD:2" ;;
  e:0) ITEMS="08b-RAAG:1" PREWARMS="08b-RAAG:1" ;;
  e:1) ITEMS="08b-RAAG:2" ;;
  e:2) ITEMS="08b-RASD:1" ;;
  e:3) ITEMS="08b-RASD:2" ;;
  *) echo "no M13 chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
TAG=$NODE$GPU

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M13 chain $TAG already launched"; exit 0; }
  M13_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M13 chain $TAG ($ITEMS) launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

REV_4b=1001bb4d826a52d1f399e183466143f4da7b741b
BASE_4b=/models/Qwen--Qwen3.5-4B-Base/$REV_4b
ONE_2b=/models/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
ONE_08b=/models/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab
LUX_4b=/runs/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl
teacher_of() {  # <ARM>: the arm's teacher file (container path), empty for none
  case $1 in
    4b-LHA5) echo "$LUX_4b" ;;
    4b-LHA10SD) echo /runs/m13/data/4b/teacher-sd.jsonl ;;
    2b-RASD) echo /runs/m13/data/2b/teacher-sd.jsonl ;;
    08b-RASD) echo /runs/m13/data/08b/teacher-sd.jsonl ;;
    08b-RAAG) echo "" ;;
  esac
}
SEEDS=(20260926 20260927)
BATCH="--batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64"
NODE_GATE=30 WARM_WAIT_MIN=360
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m13_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=dec-m13\nstatus=%s\npurpose=decoder M13 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
host() { [ -n "$1" ] && echo "/data/dev2/runs/dec/${1#/runs/}"; }
ready() {  # <ARM> <tier>: TRAIN (and the arm's teacher) equal the committed data lock
  python3 - "$M/data/READY-m13.json" "$1" "$M/data/$2/$1/train.jsonl" "$(host "$(teacher_of "$1")")" << 'EOF'
import hashlib, json, os, sys
lock_path, arm, train, teacher = sys.argv[1:]
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
ok = arm in lock["arms"] and sha(train) == lock["arms"][arm]
want = lock["teachers"].get(arm)
if want or teacher:
    ok = ok and bool(want) and os.path.isfile(teacher) and sha(teacher) == want
sys.exit(0 if ok else 1)
EOF
}
args_for() {  # <ARM> <tier>: start path, then the trainer arguments
  local a=$1 t=$2 train=/runs/m13/data/$2/$1/train.jsonl teach
  teach=$(teacher_of "$a")
  [ -n "$teach" ] && teach="--teacher $teach --teacher-kl-weight 1.0 --teacher-partial"
  case $t in
    4b) echo "$BASE_4b --train $train $teach $BATCH --init base --revision $REV_4b --train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4 --head-lr 1e-4 --head-init-seed 20261001" ;;
    2b) echo "$ONE_2b --train $train $teach $BATCH --init decision1 --train-mode full --backbone-lr 5e-6 --head-lr 5e-5" ;;
    08b) echo "$ONE_08b --train $train $teach $BATCH --init decision1 --train-mode full --backbone-lr 1e-5 --head-lr 1e-4" ;;
  esac
}
arm_cap() { case $1 in 4b-*) echo 5.0 ;; 2b-*) echo 2.5 ;; 08b-*) echo 5.5 ;; esac; }
terminal() { [ -f "$ST/m13-$1-s$2.DONE" ] || [ -f "$ST/m13-$1-s$2.FAILED" ] || [ -f "$ST/m13-$1-s$2.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 t=${1%%-*} seed r used t0 wd start rest cap warm n pw=0
  [[ " $PREWARMS " == *" $1:$2 "* ]] && pw=1
  seed=${SEEDS[$((i - 1))]} r=m13-$g-s$i cap=$(arm_cap "$g") warm=$ST/warm-$t-$NODE
  terminal "$g" "$i" && return 0
  ready "$g" "$t" || { stop "$r" "TRAIN or teacher differs from data/READY-m13.json (or no lock)"; return 0; }
  if grep -qs "preflight failed" "$ST"/m13-"$g"-s?.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $g failed"
    return 0
  fi
  gt "$(gpuh total)" "$NODE_GATE" && { stop "$r" "node M13 GPU-h above the training gate $NODE_GATE"; return 0; }
  if [ "$pw" != 1 ]; then
    n=0
    lease busy "$r waits for the $t pre-warm (training next)" 60
    until [ -f "$warm" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the node's $t pre-warm marker"
      n=$((n + 1))
      [ $n -gt $WARM_WAIT_MIN ] && { stop "$r" "the $t pre-warm marker never appeared"; return 0; }
      sleep 60
    done
    if grep -qs "preflight failed" "$ST"/m13-"$g"-s?.FAILED 2> /dev/null; then
      stop "$r" "a preflight of arm $g failed"
      return 0
    fi
  fi
  used=$(gpuh arm "$g")
  lease busy "$r (training; arm cap $cap GPU-h, arm used $used)" 180
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; arm used $used of $cap)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print($used + ($(date -u +%s) - $t0) / 3600)")" "$cap"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $cap GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m13-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  read -r start rest <<< "$(args_for "$g" "$t")"
  local env=(M13_NODE="$NODE" M13_CACHE="$t-train")
  [ "$pw" = 1 ] && env+=(M13_WARM_MARKER="$warm")
  # shellcheck disable=SC2086
  env "${env[@]}" bash "$OPS/m13-arm.sh" "$r" "$GPU" "$SRC" "$start" -- $rest --seed "$seed"
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
