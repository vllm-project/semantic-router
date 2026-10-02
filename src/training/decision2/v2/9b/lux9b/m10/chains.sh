#!/usr/bin/env bash
# 9B M10 training chains on node B (prereg lux9b-m10-prereg-2026-10-02.md), one per GPU, each under the GPU's flock:
#   GPU3 "KUP:1 KIBM:3"  (KUP-s1 pre-warms: its preflights run alone; status/prewarm.DONE after its one-step run)
#   GPU2 "KUP:2"   GPU4 "KUP:3"   GPU6 "KIBM:1"   GPU7 "KIBM:2"   (after the pre-warm marker)
#   M10_PHASE=2 (amendments 1 / 3): GPU2 / 4 "KSW:1" / "KSW:2" (launched by ksw.sh teach once KSW is locked;
#   English-only x60 cut, IB1-r3 minus `sentfin` + IB2, K-a13IB self-distillation targets on the x60 rows) and
#   GPU6 / 7 "KIB4:1" / "KIB4:2" (K-a13IB construction, IB1 `sentfin` out, IB4 phase 1 in, x60 re-cut to match)
# Arms = the K-a13IB recipe (full fine-tuning of Lux 1.0, own-Lux KL 1.0 on x60 rows, IB rows gold only, CE + 0.5
# Brier, backbone LR 1e-5, head LR 1e-4, seeds 20260926 / 1 / 2) at K-a13's 60,183,732 native tokens:
#   KUP  = K-a13IB's TRAIN byte for byte, x60 rows loss weight 1.5, IB rows 1 (--example-weights);
#   KIBM = K-a13IB's construction with IB1 `sentfin` dropped and IB3-r2 `mqa` added (x60 cut re-done to match).
# TRAIN / teacher / weights are re-hashed against data/READY-m10.json before each seed; a seed whose arm is not in it
# yet waits (≤ 4 h). Stop rules: a failed preflight stops the arm; a seed stops at 4.5 GPU-h; no seed starts once the
# node's M10 GPU-hours reach 50; a GPU whose lease another track holds (not idle / released) is not used.
# Markers: status/m10-<ARM>-s<i>.{DONE,FAILED,STOPPED}.
# usage: M10_NODE=b chains.sh launch|run <mirror-dir> <gpu>
set -u
MODE=$1 SRC=$2 GPU=$3
NODE=${M10_NODE:?set M10_NODE=b}
[ "$NODE" = b ] || { echo "M10 training chains run on node B" >&2; exit 2; }
M=/data/dev2/runs/9b/m10
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10
mkdir -p "$C" "$ST" "$M/logs"
PHASE=${M10_PHASE:-1}
case $PHASE:$GPU in
  1:3) ITEMS="KUP:1 KIBM:3" ;;
  1:2) ITEMS="KUP:2" ;;
  1:4) ITEMS="KUP:3" ;;
  1:6) ITEMS="KIBM:1" ;;
  1:7) ITEMS="KIBM:2" ;;
  2:2) ITEMS="KSW:1" ;;
  2:4) ITEMS="KSW:2" ;;
  2:6) ITEMS="KIB4:1" ;;
  2:7) ITEMS="KIB4:2" ;;
  *) echo "no M10 phase-$PHASE chain for node B GPU$GPU" >&2; exit 2 ;;
esac
TAG=b$GPU
[ "$PHASE" = 1 ] || TAG=b$GPU-p$PHASE

if [ "$MODE" = launch ]; then
  mkdir "$C/launch-$TAG.lock" 2> /dev/null || { echo "M10 chain $TAG already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) M10 chain $TAG launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

SEEDS=(20260926 1 2)
COMMON=(--teacher-kl-weight 1.0 --teacher-partial --brier-weight 0.5 --weight-decay 0.01 --warmup-ratio 0.05
  --epochs 1 --batching tokens --max-batch-tokens 32768 --max-batch-rows 64 --update-rows 64 --eval-batch 2
  --checkpoint-schedule even8 --selection matrix-v1 --init decision1 --train-mode full --backbone-lr 1e-5
  --head-lr 1e-4 --max-length 8192)
SEED_CAP=4.5 TOTAL_GATE=50 MARK=prewarm.DONE LOCK=$M/data/READY-m10.json
LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$TAG $*" | tee -a "$M/OPERATIONS.log"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  printf 'track=9b-m10\nstatus=%s\npurpose=9B M10 %s (worker 7e1c9ce8; COORDINATION 2026-10-02 12:30 / 12:40)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
lease_free() {
  [ -f "$LEASE_DIR/owner" ] || return 0
  grep -qs '^track=9b-m10' "$LEASE_DIR/owner" && return 0
  grep -qsE '^status=(idle|released)' "$LEASE_DIR/owner"
}
in_lock() { [ -f "$LOCK" ] && python3 -c 'import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1]))["arms"] else 1)' "$LOCK" "$1"; }
ready() {  # <ARM>: every file of the arm equals data/READY-m10.json
  python3 - "$LOCK" "$1" "$M" << 'EOF'
import hashlib, json, sys
lock, arm, m = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
entry = lock["arms"][arm]
sys.exit(0 if all(sha(f"{m}/{rel}") == want for rel, want in entry["files"].items()) else 1)
EOF
}
arm_args() {  # <ARM>: train_dec data flags (container paths)
  python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["arms"][sys.argv[2]]["args"]))' "$LOCK" "$1"
}
terminal() { [ -f "$ST/m10-$1-s$2.DONE" ] || [ -f "$ST/m10-$1-s$2.FAILED" ] || [ -f "$ST/m10-$1-s$2.STOPPED" ]; }

item() {  # <ARM> <seed index>
  local g=$1 i=$2 seed r used t0 wd n=0
  seed=${SEEDS[$((i - 1))]} r=m10-$g-s$i
  terminal "$g" "$i" && return 0
  if [ "$r" != m10-KUP-s1 ]; then
    until [ -f "$ST/$MARK" ] || [ -f "$ST/m10-KUP-s1.FAILED" ] || [ -f "$ST/m10-KUP-s1.STOPPED" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the pre-warm run (KUP-s1 one-step)"
      n=$((n + 1))
      sleep 60
    done
  fi
  n=0
  until in_lock "$g"; do
    if [ "$n" -ge 240 ]; then
      echo "not started: arm $g not in data/READY-m10.json after 4 h" > "$ST/$r.STOPPED"
      log "$r not started: no data lock entry"
      return 0
    fi
    [ $((n % 15)) = 0 ] && log "$r waits for its data lock entry"
    n=$((n + 1))
    sleep 60
  done
  if ! lease_free; then
    echo "not started: gpu$GPU lease is held by another track" > "$ST/$r.STOPPED"
    log "$r not started: lease of GPU$GPU not free"
    return 0
  fi
  if ! ready "$g"; then
    echo "TRAIN / teacher / weights differ from data/READY-m10.json; not started" > "$ST/$r.STOPPED"
    log "$r not started: data lock check failed"
    return 0
  fi
  if grep -qs "preflight failed" "$ST"/m10-"$g"-s?.FAILED 2> /dev/null; then
    echo "not started: a preflight of arm $g failed" > "$ST/$r.STOPPED"
    log "$r not started: arm $g stopped by a failed preflight"
    return 0
  fi
  used=$(python3 "$OPS/gpuh.py" total --running)
  if ! gt "$TOTAL_GATE" "$used"; then
    echo "not started: node M10 GPU-h $used reached $TOTAL_GATE" > "$ST/$r.STOPPED"
    log "$r not started: node GPU-h $used >= $TOTAL_GATE"
    return 0
  fi
  lease busy "$r (training; seed cap $SEED_CAP GPU-h)" 200
  log "start $r on node B GPU$GPU (seed $seed; node M10 GPU-h used $used)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print(($(date -u +%s) - $t0) / 3600)")" "$SEED_CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the seed cap $SEED_CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^m10-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  # shellcheck disable=SC2046
  M10_NODE=$NODE M10_PREWARM=$([ "$r" = m10-KUP-s1 ] && echo 1 || echo 0) M10_PREWARM_MARK=$MARK \
    bash "$OPS/arm.sh" "$r" "$GPU" "$SRC" /lux -- $(arm_args "$g") "${COMMON[@]}" --seed "$seed"
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
lease idle "chain $TAG finished; GPU idle" 30
log "chain finished"
