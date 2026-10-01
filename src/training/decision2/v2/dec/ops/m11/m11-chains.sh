#!/usr/bin/env bash
# shellcheck disable=SC2034  # the per-tier values (NAME_<tier>) are read by name through tv()
# Decoder M11 training chains (prereg dec-m11-prereg-2026-10-01.md, stage 1), one per M11 GPU. A chain holds its GPU's
# flock for its whole run, keeps the GPU's lease owner file current and runs its items in order:
#   node F: GPU2 "2b-LH:1"  GPU3 "2b-LH:2"  GPU4 "2b-LH:3"  GPU5 "08b-LH:1"  GPU6 "08b-LH:2"  GPU7 "08b-LH:3"
#   node E: GPU0 "2b-NT:1 08b-NT:1"  GPU1 "2b-NT:2 08b-NT:2"  GPU2 "2b-NT:3 08b-NT:3"   (GPU3: references, readouts)
# Arms: <t>-LH = LoRA r128 from Qwen3.5-<t>-Base + candidate head; <t>-NT = the tier's released recipe from its 1.0
# model (Sol / Eos) with the label-token readout. Every arm: the tier's released r2-clean TRAIN (and, at 2B, own-Sol
# targets) re-hashed against data/READY-<tier>.json before each seed; seeds 20260926 / 27 / 28.
# Pre-warm (prereg "Autotune caches and pre-warm"): seed 1 of a tier's arm on a node fills that node's <tier>-train
# cache alone; the node's other seeds of that tier wait for status/warm-<tier> (written by m11-arm.sh once seed 1's
# preflight stages finished) before they start.
# Stop rules: a failed preflight stops the arm (no rerun, no replacement seed); a seed stops at its arm cap; no NT seed
# starts above the node's NT gate; no seed starts above the node's training gate (45 GPU-h per node, so both nodes
# together stay below the prereg's 95 GPU-h).
# Markers: /data/dev2/runs/dec/m11/status/m11-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M11_NODE=e|f m11-chains.sh launch <mirror-dir> <gpu>
#        M11_NODE=e|f m11-chains.sh run <mirror-dir> <gpu>   (the chain body, started by launch under flock)
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M11_NODE:?set M11_NODE=e or f}
M=/data/dev2/runs/dec/m11
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m11
mkdir -p "$C" "$ST" "$M/logs"
case $NODE:$GPU in
  f:2) ITEMS="2b-LH:1" ;;
  f:3) ITEMS="2b-LH:2" ;;
  f:4) ITEMS="2b-LH:3" ;;
  f:5) ITEMS="08b-LH:1" ;;
  f:6) ITEMS="08b-LH:2" ;;
  f:7) ITEMS="08b-LH:3" ;;
  e:0) ITEMS="2b-NT:1 08b-NT:1" ;;
  e:1) ITEMS="2b-NT:2 08b-NT:2" ;;
  e:2) ITEMS="2b-NT:3 08b-NT:3" ;;
  *) echo "no M11 chain for node $NODE GPU$GPU" >&2; exit 2 ;;
esac
TAG=$NODE$GPU

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M11 chain $TAG already launched"; exit 0; }
  M11_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M11 chain $TAG launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

REV_2b=b1485b2fa6dfa1287294f269f5fb618e03d52d7c
REV_08b=dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68
BASE_2b=/models/Qwen--Qwen3.5-2B-Base/$REV_2b
BASE_08b=/models/Qwen--Qwen3.5-0.8B-Base/$REV_08b
ONE_2b=/models/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
ONE_08b=/models/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab
NT_LR_2b=5e-6 NT_LR_08b=1e-5
TRAIN_2b=$M/inputs/m4-v2m-ret-r2/train.jsonl
TRAIN_08b=$M/inputs/m6-e8f-r2clean/train.jsonl
TEACHER_2b=$M/data/2b/teacher.jsonl TEACHER_08b=
SEEDS=(20260926 20260927 20260928)
HEAD_INIT_SEED=20261001
BATCH=(--batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64)
LORA=(--train-mode lora --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05 --lora-lr 1e-4)
NT_GATE=30 NODE_GATE=45 WARM_WAIT_MIN=360
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$NODE$GPU $*" | tee -a "$M/OPERATIONS.log"; }
gpuh() { python3 "$OPS/m11_gpuh.py" "$@"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=dec-m11\nstatus=%s\npurpose=decoder M11 %s (COORDINATION 2026-10-01 14:45)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
tv() { local n=$1_$2; echo "${!n}"; }  # tv <NAME> <tier>
ready() {  # <tier>: TRAIN (and teacher) equal the committed data lock; prints the NT max length
  python3 - "$M/data/READY-$1.json" "$(tv TRAIN "$1")" "$(tv TEACHER "$1")" << 'EOF'
import hashlib, json, os, sys
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
ok = sha(sys.argv[2]) == lock["train_sha256"]
if lock.get("teacher_sha256"):
    ok = ok and os.path.isfile(sys.argv[3]) and sha(sys.argv[3]) == lock["teacher_sha256"]
print(lock["nt_max_length"])
sys.exit(0 if ok else 1)
EOF
}
args_for() {  # <tier> <kind> <nt max length>: start path, then the arm's trainer arguments
  local t=$1 k=$2 obj
  case $t in
    2b) obj="--train /runs/m11/inputs/m4-v2m-ret-r2/train.jsonl --teacher /runs/m11/data/2b/teacher.jsonl --teacher-kl-weight 0.5" ;;
    08b) obj="--train /runs/m11/inputs/m6-e8f-r2clean/train.jsonl" ;;
  esac
  case $k in
    LH) echo "$(tv BASE "$t") $obj ${BATCH[*]} --init base --revision $(tv REV "$t") ${LORA[*]} --head-lr 1e-4 --head-init-seed $HEAD_INIT_SEED" ;;
    NT) echo "$(tv ONE "$t") $obj ${BATCH[*]} --init decision1 --train-mode full --backbone-lr $(tv NT_LR "$t") --readout label_token --max-length $3" ;;
  esac
}
arm_cap() { case $1 in 2b-*) echo 7.0 ;; 08b-*) echo 9.0 ;; esac; }
terminal() { [ -f "$ST/m11-$1-s$2.DONE" ] || [ -f "$ST/m11-$1-s$2.FAILED" ] || [ -f "$ST/m11-$1-s$2.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 t=${1%%-*} k=${1##*-} seed r used t0 wd start rest ntlen cap warm n
  seed=${SEEDS[$((i - 1))]} r=m11-$g-s$i cap=$(arm_cap "$g") warm=$ST/warm-$t
  terminal "$g" "$i" && return 0
  if ! ntlen=$(ready "$t"); then
    stop "$r" "TRAIN or teacher differs from data/READY-$t.json (or no lock)"
    return 0
  fi
  if grep -qs "preflight failed" "$ST"/m11-"$g"-s?.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $g failed"
    return 0
  fi
  if [ "$k" = NT ] && gt "$(gpuh total)" "$NT_GATE"; then
    stop "$r" "node M11 GPU-h above the NT gate $NT_GATE"
    return 0
  fi
  if gt "$(gpuh total)" "$NODE_GATE"; then
    stop "$r" "node M11 GPU-h above the training gate $NODE_GATE"
    return 0
  fi
  if [ "$i" != 1 ]; then
    n=0
    lease busy "$r waits for the $t pre-warm (training next)" 60
    until [ -f "$warm" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the node's $t pre-warm marker"
      n=$((n + 1))
      [ $n -gt $WARM_WAIT_MIN ] && { stop "$r" "the $t pre-warm marker never appeared"; return 0; }
      sleep 60
    done
    if grep -qs "preflight failed" "$ST"/m11-"$g"-s?.FAILED 2> /dev/null; then
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
        docker ps --format '{{.Names}}' | grep -E "^m11-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  read -r start rest <<< "$(args_for "$t" "$k" "$ntlen")"
  local env=(M11_NODE="$NODE" M11_CACHE="$t-train")
  [ "$i" = 1 ] && env+=(M11_WARM_MARKER="$warm")
  # shellcheck disable=SC2086
  env "${env[@]}" bash "$OPS/m11-arm.sh" "$r" "$GPU" "$SRC" "$start" -- $rest --seed "$seed"
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

for it in $ITEMS; do
  item "${it%%:*}" "${it##*:}"
done
lease idle "chain finished; GPU idle (readouts may follow)" 30
log "chain finished"
