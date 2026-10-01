#!/usr/bin/env bash
# Decoder M10 training chains (prereg dec-m10-prereg-2026-10-01.md), one per M10 GPU. A chain holds its GPU's flock
# for its whole run, keeps the GPU's lease owner file current and runs its items in order:
#   node E: GPU0 "LT:1 NT:1"  GPU1 "LT:2 NT:2"  GPU2 "LT:3 NT:3"        (GPU3: references and readouts)
#   node F: GPU2 "LH:1"  GPU3 "LH:2"  GPU4 "LH:3"  GPU5 "FB:1"  GPU6 "FB:2"  GPU7 "FB:3"
# Arms: LH = LoRA r128 from Qwen3.5-4B-Base + candidate head; LT = the same LoRA + label-token readout; FB = full
# fine-tune from the base at half the recipe's backbone LR + candidate head; NT (optional second wave) = the N4XF
# recipe from Nox 1.0 with the label-token readout. Every arm: the released 4B mixture (TRAIN and own-Lux teacher
# re-hashed against data/READY.json before each seed), N4XF's batching, seeds 20260926 / 27 / 28.
# Stop rules: a failed preflight stops the arm (no rerun, no replacement seed); a seed stops at its arm cap; no NT
# seed starts unless LT's three seeds passed their preflights and the node's M10 GPU-hours are below the NT gate.
# Markers: /data/dev2/runs/dec/m10/status/m10-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M10_NODE=e|f m10-chains.sh launch <mirror-dir> <gpu>
#        M10_NODE=e|f m10-chains.sh run <mirror-dir> <gpu>     (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M10_NODE:?set M10_NODE=e or f}
M=/data/dev2/runs/dec/m10
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m10
mkdir -p "$C" "$ST" "$M/logs"
case $NODE:$GPU in
  e:0) ITEMS="LT:1 NT:1" ;;
  e:1) ITEMS="LT:2 NT:2" ;;
  e:2) ITEMS="LT:3 NT:3" ;;
  f:2) ITEMS="LH:1" ;;
  f:3) ITEMS="LH:2" ;;
  f:4) ITEMS="LH:3" ;;
  f:5) ITEMS="FB:1" ;;
  f:6) ITEMS="FB:2" ;;
  f:7) ITEMS="FB:3" ;;
  *) echo "no M10 chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$NODE$GPU.lock" 2> /dev/null || { echo "M10 chain $NODE$GPU already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$NODE$GPU.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$NODE$GPU.pid"
  echo "$(date -u +%FT%TZ) M10 chain $NODE$GPU launched from $SRC (pid $(cat "$C/chain-$NODE$GPU.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

BASE=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
BASE_REV=1001bb4d826a52d1f399e183466143f4da7b741b
NOX=/models/Decision-1.0-Nox-4B/cde2a68dbaa557ea65dc458104d410a0802ee259
SEEDS=(20260926 20260927 20260928)
HEAD_INIT_SEED=20261001
TRAIN=$M/data/m10-4b-base/train.jsonl
TEACHER=$M/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl
COMMON=(--train /runs/m10/data/m10-4b-base/train.jsonl
  --teacher /runs/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl --teacher-kl-weight 1.0 --teacher-partial
  --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64)
LORA=(--train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4)
CAP=6.0 NT_GATE=40
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$NODE$GPU $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m10_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=dec-m10\nstatus=%s\npurpose=decoder M10 %s (COORDINATION 2026-10-01 11:20)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
ready() {  # TRAIN and teacher equal the committed data lock
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
    LH) echo "$BASE --init base --revision $BASE_REV ${LORA[*]} --head-lr 1e-4 --head-init-seed $HEAD_INIT_SEED" ;;
    LT) echo "$BASE --init base --revision $BASE_REV ${LORA[*]} --readout label_token" ;;
    FB) echo "$BASE --init base --revision $BASE_REV --train-mode full --backbone-lr 2.5e-6 --head-lr 1e-4 --head-init-seed $HEAD_INIT_SEED" ;;
    NT) echo "$NOX --init decision1 --train-mode full --backbone-lr 5e-6 --readout label_token" ;;
  esac
}
terminal() { [ -f "$ST/m10-$1-s$2.DONE" ] || [ -f "$ST/m10-$1-s$2.FAILED" ] || [ -f "$ST/m10-$1-s$2.STOPPED" ]; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd s start
  seed=${SEEDS[$((i - 1))]} r=m10-$g-s$i
  terminal "$g" "$i" && return 0
  if ! ready; then
    echo "TRAIN or teacher differs from data/READY.json (or no lock); not started" > "$ST/$r.STOPPED"
    log "$r not started: data lock check failed"
    return 0
  fi
  if grep -qs "preflight failed" "$ST"/m10-"$g"-s?.FAILED 2> /dev/null; then
    echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
    log "$r not started: arm $g stopped by a failed preflight"
    return 0
  fi
  if [ "$g" = NT ]; then
    for s in 1 2 3; do
      if ! grep -qs "^[^ ]* m10-LT-s$s preflight PASS" "$M/arms/OPERATIONS.log"; then
        echo "not started: LT-s$s did not pass its preflight (NT is the optional second wave)" > "$ST/$r.STOPPED"
        log "$r not started: LT-s$s preflight not PASS"
        return 0
      fi
    done
    if gt "$(gpuh total)" "$NT_GATE"; then
      echo "not started: node M10 GPU-h above the NT gate $NT_GATE" > "$ST/$r.STOPPED"
      log "$r not started: node GPU-h above $NT_GATE"
      return 0
    fi
  fi
  used=$(gpuh arm "$g")
  lease busy "$r (training; arm cap $CAP GPU-h, arm used $used)" 120
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; arm used $used of $CAP)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print($used + ($(date -u +%s) - $t0) / 3600)")" "$CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the arm cap $CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m10-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  read -r start rest <<< "$(args_for "$g")"
  # shellcheck disable=SC2086
  M10_NODE=$NODE bash "$OPS/m10-arm.sh" "$r" "$GPU" "$SRC" "$start" -- "${COMMON[@]}" $rest --seed "$seed"
  kill "$wd" 2> /dev/null
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
