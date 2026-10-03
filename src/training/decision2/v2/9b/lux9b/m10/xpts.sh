#!/usr/bin/env bash
# 9B M10 amendment 7 cross-arm averages, node side (CPU only, node A or B): the uniform FP32 soup (v2.dec.soup) of
# built points on this node, each member carrying weight 1 / n. Every member is an M10 point or arm soup
# (soup/<P>/build/<P>: post.sh, xarm.sh, ix.sh soupcopy, or a linked arm-factory point):
#   X7-a40 = [KIB4-a40, KX-a40, KSW-a40]   Y1 = [F*, KIB4-a40]
#   X8-a40 = [KIB4-a40, KSW-a40]           Y2 = [F*, KIB4-a40, X5-a33]   (F* = AF-<name>, the best factory point)
# Amendment 13 (half learning rates): HLR4 = [KIB4H, AF-KIB4-lrhh] (four half-LR seeds), HLR4-a60 = [HLR4 x 3, LUX x 2],
#   HLR4-a80 = [HLR4 x 4, LUX], KIB4-lrhh-a80 = [AF-KIB4-lrhh x 4, LUX] (revision 1); wave 2: HLR4-a100 / HLR4-a50,
#   LRX6-aNN, KIB4-lrh-aNN (node A, [AF-KIB4-lrh x k, LUX x m]). The member LUX is the pinned Lux 1.0 zero-step
#   checkpoint (post.sh's: node B m10-KUP-s1, node A m10-KX-s1), checked against lux-zero-m9-KIB-s1.sha256 first.
# The output is soup/<NAME>/build/<NAME> with DONE, members.txt and the build log, as post.sh writes it, so ix.sh ship
# takes it. A failed build writes soup/<NAME>/FAILED and is never rerun.
#
#   link AFNAME  (node A or B) the arm factory's built point runs/af/9b/soup/AFNAME/build/AFNAME, hard-linked (read-only use)
#                as soup/AF-AFNAME/build/AF-AFNAME with equal SHA-256 lists; its MODEL_SHA256 is copied next to it.
#                m10/formal.sh and this script then take AF-AFNAME like any M10 point. M10_LINK_AS=<name> links it
#                under that M10 name instead (amendment 8: a factory point that M10 measures as M10-<name>-bf16).
#
# usage: M10_NODE=a|b xpts.sh launch|run <mirror-dir> <NAME> <MEMBER>... | M10_NODE=a xpts.sh link <mirror-dir> <AFNAME>
set -u
MODE=$1 SRC=$2 NAME=$3
shift 3
NODE=${M10_NODE:?set M10_NODE=a or b}
M=/data/dev2/runs/9b/m10
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10
log() { echo "$(date -u +%FT%TZ) xpts-$NAME $*" | tee -a "$M/OPERATIONS.log"; }
if [ "$MODE" = link ]; then
  [[ "$NODE" == [ab] ]] || { echo "arm-factory 9B points are on node A (batch 1) or node B (batches 2-3)" >&2; exit 2; }
  as=${M10_LINK_AS:-AF-$NAME}
  af=/data/dev2/runs/af/9b/soup/$NAME out=$M/soup/$as
  [ -f "$af/DONE" ] && [ -f "$af/MODEL_SHA256" ] && [ -f "$af/build/$NAME/decision_config.json" ] \
    || { echo "no built arm-factory point $NAME" >&2; exit 3; }
  [ ! -e "$out" ] || { echo "$out exists" >&2; exit 3; }
  mkdir -p "$out/build"
  cp -al "$af/build/$NAME" "$out/build/$as"
  a=$(cd "$af/build/$NAME" && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2)
  b=$(cd "$out/build/$as" && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2)
  [ -n "$a" ] && [ "$a" = "$b" ] || { rm -rf "$out"; echo "linked copy differs" >&2; exit 3; }
  cp "$af/MODEL_SHA256" "$out/MODEL_SHA256"
  cp "$af/members.txt" "$out/members.af.txt" 2> /dev/null || true
  printf '{"model_sha256": "%s"}\n' "$(tr -d '[:space:]' < "$af/MODEL_SHA256")" > "$out/build.stdout.log"
  echo "$out/build/$as" > "$out/DONE"
  log "linked arm-factory point $NAME as $as ($(wc -l <<< "$a") files, SHA-256 lists equal, model $(cut -c1-12 "$out/MODEL_SHA256"))"
  exit 0
fi
MEMBERS=("$@")
out=$M/soup/$NAME
case $NAME in
  X7-a40 | X8-a40 | Y1 | Y2) ;;
  HLR4 | HLR4-a50 | HLR4-a60 | HLR4-a80 | HLR4-a100 | LRX6-a50 | LRX6-a60 | LRX6-a80 | LRX6-a100) ;;
  KIB4-lrhh-a80 | KIB4-lrh-a50 | KIB4-lrh-a60 | KIB4-lrh-a80 | KIB4-lrh-a100) ;;
  *) echo "NAME is X7-a40, X8-a40, Y1, Y2 (amendment 7) or an amendment-13 point" >&2; exit 2 ;;
esac
(( ${#MEMBERS[@]} >= 2 )) || { echo "at least two members" >&2; exit 2; }
if [ "$MODE" = launch ]; then
  mkdir -p "$M/chains" "$M/logs" "$M/soup"
  mkdir "$M/chains/xpts-$NAME.lock" 2> /dev/null || { echo "cross-arm $NAME already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$NAME" "${MEMBERS[@]}" > "$M/logs/xpts-$NAME.log" 2>&1 < /dev/null &
  echo "$(date -u +%FT%TZ) M10 cross-arm $NAME (uniform of ${MEMBERS[*]}) launched from $SRC (pid $!)" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ -f "$out/DONE" ] && { log "already built"; exit 0; }
[ -f "$out/FAILED" ] && { log "failed before; not rerun"; exit 1; }
args=()
luxrun=m10-KUP-s1
[ "$NODE" = a ] && luxrun=m10-KX-s1
luxhost=$M/arms/pre/$luxrun-zero/checkpoint-0000000
luxok=0
for p in "${MEMBERS[@]}"; do
  if [ "$p" = LUX ]; then
    if [ "$luxok" = 0 ]; then
      (cd "$luxhost" && sha256sum -c --quiet "$M/inputs/lux-zero-m9-KIB-s1.sha256") \
        || { log "Lux zero-step member differs from K-a13IB's"; exit 1; }
      luxok=1
    fi
    args+=(--member "/runs/m10/arms/pre/$luxrun-zero/checkpoint-0000000")
    continue
  fi
  d=$(cat "$M/soup/$p/DONE" 2> /dev/null)
  [ "$d" = "$M/soup/$p/build/$p" ] && [ -f "$d/decision_config.json" ] || { log "no built point $p on node ${NODE^^}"; exit 1; }
  args+=(--member "/runs/m10/soup/$p/build/$p")
done
mkdir -p "$out"
if M10_NODE=$NODE bash "$OPS/launch.sh" "xpts-$NAME" "$SRC" "$out/build" --cpu -- -m v2.dec.soup "${args[@]}" \
  --output "/out/$NAME"; then
  printf '%s\n' "${MEMBERS[@]}" > "$out/members.txt"
  echo "$out/build/$NAME" > "$out/DONE"
  log "built $NAME (uniform of ${MEMBERS[*]}): $(tail -c 300 "$out/build.stdout.log" | tr '\n' ' ')"
else
  echo "soup build failed (see $out/build.stderr.log)" > "$out/FAILED"
  log "FAILED: $NAME"
  exit 1
fi
