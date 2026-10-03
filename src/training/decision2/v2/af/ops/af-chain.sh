#!/usr/bin/env bash
# Arm factory training chain on one GPU (prereg v2/af/records/af-prereg-2026-10-02.md). A chain holds its GPU's flock
# for its whole run, keeps the GPU's lease owner file current (track=arm-factory) and runs its items in order. An item
# is ARM:sN:SEED (run name <size>-<ARM>-sN; recipe and data from af_arms.py). Per item:
#   - skip it if it has a terminal marker; wait <= 4 h for its data lock entry; stop it if its data differs from the
#     node's lock, if a preflight of the same arm already failed, or if this node's training GPU-h reached the gate;
#   - wait for the node's pre-warm marker unless AF_PREWARM=1 names this chain's first item as the pre-warm seed
#     (its preflights run alone on the node's Triton cache);
#   - take the GPU only if its lease is free (absent, released / idle, or arm-factory); wait <= 60 min, else stop;
#   - run af-arm.sh under a watchdog that stops the seed at the seed cap (4B 2.5, 9B 4.5 GPU-h).
# A failed preflight stops its arm (no rerun, no replacement seed). Markers:
# /data/dev2/runs/af/<size>/status/<run>.{DONE,FAILED,STOPPED}.
#
# usage: AF_NODE=a|b|c|f [AF_SIZE=9b] [AF_CAP=<GPU-h>] [AF_PREWARM=1] af-chain.sh launch|run <mirror-dir> <gpu> <ARM:sN:SEED>...
set -u
MODE=$1 SRC=$2 GPU=$3
shift 3
ITEMS=("$@")
NODE=${AF_NODE:?set AF_NODE}
case $NODE:${AF_SIZE:-} in  # node gates: prereg, raised by amendments 4, 5, 7, 8 and 9 (COORDINATION 2026-10-03 00:00-04:58)
  a:*) SIZE=9b CACHE=9b-train CAP=4.5 GATE=32 ;;
  c:*) SIZE=4b CACHE=4b-train CAP=2.5 GATE=44 ;;
  f:*) SIZE=4b CACHE=4b-train CAP=2.5 GATE=9 ;;
  b:9b) SIZE=9b CACHE=9b-train CAP=4.5 GATE=28 ;;
  b:*) SIZE=4b CACHE=4b-train CAP=2.5 GATE=4 ;;
  *) echo "unknown node $NODE" >&2; exit 2 ;;
esac
CAP=${AF_CAP:-$CAP}  # amendment 5: a preregistered per-arm seed cap (two-epoch arms)
M=/data/dev2/runs/af/$SIZE
C=$M/chains ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/af/ops
mkdir -p "$C" "$ST" "$M/logs"
TAG=$NODE$GPU-$(date -u +%H%M%S)

if [ "$MODE" = launch ]; then
  [ ${#ITEMS[@]} -gt 0 ] || { echo "no items" >&2; exit 2; }
  AF_NODE=$NODE AF_SIZE=${AF_SIZE:-} AF_CAP=${AF_CAP:-} AF_PREWARM=${AF_PREWARM:-0} setsid nohup flock "$C/gpu$GPU.flock" bash "$0" run "$SRC" "$GPU" \
    "${ITEMS[@]}" > "$M/logs/chain-$TAG.log" 2>&1 < /dev/null &
  echo $! > "$C/chain-$TAG.pid"
  echo "$(date -u +%FT%TZ) chain $TAG (${ITEMS[*]}) launched from $SRC (pid $(cat "$C/chain-$TAG.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }

LEASE_DIR=/data/dev2/leases/gpu$GPU.lock
log() { echo "$(date -u +%FT%TZ) chain-$NODE$GPU $*" | tee -a "$M/OPERATIONS.log"; }
gt() { python3 -c "import sys;sys.exit(0 if float(sys.argv[1])>float(sys.argv[2]) else 1)" "$1" "$2"; }
lease_free() {
  [ -f "$LEASE_DIR/owner" ] || return 0
  grep -qs '^track=arm-factory' "$LEASE_DIR/owner" && return 0
  grep -qsE '^status=(idle|released)' "$LEASE_DIR/owner"
}
lease() {  # <status> <purpose> <minutes>
  mkdir -p "$LEASE_DIR"
  if [ -f "$LEASE_DIR/owner" ] && ! grep -qs '^track=arm-factory' "$LEASE_DIR/owner"; then
    mv "$LEASE_DIR/owner" "$LEASE_DIR/owner.prev-af-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  printf 'track=arm-factory\nstatus=%s\npurpose=arm factory %s (COORDINATION 2026-10-02 22:00)\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$1" "$2" "$(date -u +%FT%TZ)" "$(date -u -d "+$3 min" +%FT%TZ)" > "$LEASE_DIR/owner"
}
terminal() { [ -f "$ST/$1.DONE" ] || [ -f "$ST/$1.FAILED" ] || [ -f "$ST/$1.STOPPED" ]; }
stop() { echo "$2" > "$ST/$1.STOPPED"; log "$1 not started: $2"; }

first=1
item() {  # <ARM> <sN> <SEED>
  local arm=$1 sn=$2 seed=$3 r used t0 wd n=0 size start rest pw=0 warm=$ST/warm-$NODE
  r=$SIZE-$arm-$sn
  [ "$first" = 1 ] && [ "${AF_PREWARM:-0}" = 1 ] && pw=1
  first=0
  terminal "$r" && return 0
  until python3 "$OPS/af_arms.py" locked "$arm"; do
    [ $((n % 15)) = 0 ] && log "$r waits for its data lock entry"
    n=$((n + 1))
    [ $n -gt 240 ] && { stop "$r" "no data lock entry after 4 h"; return 0; }
    sleep 60
  done
  n=0
  python3 "$OPS/af_arms.py" ready "$arm" || { stop "$r" "data differs from data/READY-af.json"; return 0; }
  if grep -qs "preflight failed" "$ST/$SIZE-$arm"-s*.FAILED 2> /dev/null; then
    stop "$r" "a preflight of arm $arm failed"
    return 0
  fi
  used=$(python3 "$OPS/af_gpuh.py" prefix "$SIZE-" --root "$M/arms" --running)
  gt "$used" "$GATE" && { stop "$r" "node arm-factory training GPU-h $used above the node gate $GATE"; return 0; }
  if [ "$pw" != 1 ]; then
    lease_free && lease busy "$r waits for the node's pre-warm marker (training next)" 60
    until [ -f "$warm" ]; do
      [ $((n % 15)) = 0 ] && log "$r waits for the node's pre-warm marker"
      n=$((n + 1))
      [ $n -gt 240 ] && { stop "$r" "the pre-warm marker never appeared"; return 0; }
      sleep 60
    done
  fi
  n=0
  until lease_free; do
    [ $((n % 15)) = 0 ] && log "$r waits for the GPU$GPU lease (held by another track)"
    n=$((n + 1))
    [ $n -gt 60 ] && { stop "$r" "the GPU$GPU lease stayed with another track"; return 0; }
    sleep 60
  done
  read -r size start rest <<< "$(python3 "$OPS/af_arms.py" recipe "$arm")"
  [ "$size" = "$SIZE" ] || { stop "$r" "arm $arm is $size, node $NODE trains $SIZE"; return 0; }
  lease busy "$r (training; seed cap $CAP GPU-h)" "$(python3 -c "print(int($CAP*60))")"
  log "start $r on node ${NODE^^} GPU$GPU (seed $seed; node training GPU-h used $used of gate $GATE)"
  t0=$(date -u +%s)
  (
    while sleep 60; do
      if gt "$(python3 -c "print(($(date -u +%s) - $t0) / 3600)")" "$CAP"; then
        touch "$ST/$r.capstop"
        log "$r reached the seed cap $CAP GPU-h; stopping its containers"
        docker ps --format '{{.Names}}' | grep -E "^af-$r-" | xargs -r docker stop > /dev/null
      fi
    done
  ) &
  wd=$!
  local env=(AF_NODE="$NODE" AF_SIZE="$SIZE" AF_CACHE="$CACHE")
  [ "$pw" = 1 ] && env+=(AF_WARM_MARKER="$warm")
  # shellcheck disable=SC2086
  env "${env[@]}" bash "$OPS/af-arm.sh" "$r" "$GPU" "$SRC" "$SIZE" "$start" -- $rest --seed "$seed"
  kill "$wd" 2> /dev/null
  [ "$pw" = 1 ] && [ ! -f "$warm" ] && echo "$r ended before its preflight finished $(date -u +%FT%TZ)" > "$warm"
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

for it in "${ITEMS[@]}"; do
  IFS=: read -r a s e <<< "$it"
  item "$a" "$s" "$e"
done
lease released "chain $NODE$GPU finished; GPU free" 0
log "chain finished"
