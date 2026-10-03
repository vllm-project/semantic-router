#!/usr/bin/env bash
# 9B M10 cross-arm points (amendment 5; CPU only, node A or B): the uniform FP32 soup of k frozen arm soups and m copies
# of the pinned Lux 1.0 zero-step member, so each arm carries alpha / k and Lux 1.0 carries 1 - alpha
# (a33: m = 2k, a25: m = 3k, a40: m = 3k / 2 for even k). Arms: an M10 arm whose soup is on this node
# (soup/<ARM>/build/<ARM>, built by post.sh or copied with ix.sh soupcopy) or KIB, K-a13IB's arm soup
# (/runs/m9/soup/KIB/build/KIB-soup on node B). Output soup/<NAME>/build/<NAME> with DONE and the build log, as post.sh
# writes, so ix.sh ship takes it. A failed build writes soup/<NAME>/FAILED and is never rerun.
#
# usage: M10_NODE=a|b xarm.sh launch|run <mirror-dir> <NAME> <a33|a25|a40> <ARM>...
set -u
MODE=$1 SRC=$2 NAME=$3 POINT=$4
shift 4
ARMS=("$@")
NODE=${M10_NODE:?set M10_NODE=a or b}
M=/data/dev2/runs/9b/m10
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10
LUXRUN=m10-KUP-s1
[ "$NODE" = a ] && LUXRUN=m10-KX-s1
LUXHOST=$M/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXCK=/runs/m10/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXSUMS=${M10_LUX_SUMS:-$M/inputs/lux-zero-m9-KIB-s1.sha256}
out=$M/soup/$NAME
[[ "$NAME" =~ ^X[1-6]-a(25|33|40)$ && "$NAME" == *"-$POINT" ]] || { echo "NAME is X<1-6>-<point>" >&2; exit 2; }
k=${#ARMS[@]}
case $POINT in
  a33) nlux=$((2 * k)) ;;
  a25) nlux=$((3 * k)) ;;
  a40) (( k % 2 == 0 )) || { echo "a40 needs an even number of arms" >&2; exit 2; }; nlux=$((3 * k / 2)) ;;
  *) echo "point a33|a25|a40" >&2; exit 2 ;;
esac
(( k >= 2 )) || { echo "at least two arms" >&2; exit 2; }
if [ "$MODE" = launch ]; then
  mkdir -p "$M/chains" "$M/logs" "$M/soup"
  mkdir "$M/chains/xarm-$NAME.lock" 2> /dev/null || { echo "cross-arm $NAME already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$NAME" "$POINT" "${ARMS[@]}" > "$M/logs/xarm-$NAME.log" 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) M10 cross-arm $NAME ($POINT of ${ARMS[*]}) launched from $SRC (pid $!)" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) xarm-$NAME $*" | tee -a "$M/OPERATIONS.log"; }
members=()
for arm in "${ARMS[@]}"; do
  if [ "$arm" = KIB ]; then
    p=/runs/m9/soup/KIB/build/KIB-soup h=/data/dev2/runs/9b/m9/soup/KIB/build/KIB-soup
  else
    p=/runs/m10/soup/$arm/build/$arm h=$M/soup/$arm/build/$arm
  fi
  [ -f "$h/decision_config.json" ] || { log "no arm soup $arm at $h"; exit 1; }
  members+=("$p")
done
[ -f "$LUXSUMS" ] || { log "missing pinned Lux member SHA-256 list $LUXSUMS"; exit 1; }
(cd "$LUXHOST" && sha256sum -c --quiet "$LUXSUMS") || { log "Lux zero-step member differs from K-a13IB's"; exit 1; }
for ((i = 0; i < nlux; i++)); do members+=("$LUXCK"); done
[ -f "$out/DONE" ] && { log "already built"; exit 0; }
[ -f "$out/FAILED" ] && { log "failed before; not rerun"; exit 1; }
mkdir -p "$out"
args=()
for p in "${members[@]}"; do args+=(--member "$p"); done
if M10_NODE=$NODE bash "$OPS/launch.sh" "soup-$NAME" "$SRC" "$out/build" --cpu -- -m v2.dec.soup "${args[@]}" \
  --output "/out/$NAME"; then
  printf '%s\n' "${members[@]}" > "$out/members.txt"
  echo "$out/build/$NAME" > "$out/DONE"
  log "built $NAME (${ARMS[*]} at $POINT): $(tail -c 300 "$out/build.stdout.log" | tr '\n' ' ')"
else
  echo "soup build failed (see $out/build.stderr.log)" > "$out/FAILED"
  log "FAILED: $NAME"
  exit 1
fi
