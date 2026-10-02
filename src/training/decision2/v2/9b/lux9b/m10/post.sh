#!/usr/bin/env bash
# 9B M10 post-training chain on node B for one arm (CPU only; prereg "Candidates"): wait until every seed of ARM has a
# terminal marker, then
#   soup/<ARM>/build/<ARM>-soup   the uniform FP32 soup of the finished seeds' BEST checkpoints (one seed: that seed)
#   soup/<ARM>-a33|a25|a40/build/ the K-a13 construction W(alpha) = uniform FP32 soup of [S x p, Lux x q]:
#                                 a33 = [S, Lux, Lux], a25 = [S, Lux, Lux, Lux], a40 = [S, S, Lux, Lux, Lux]
# The Lux 1.0 member is m10-KUP-s1's zero-step checkpoint (written by the M10 trainer from Lux 1.0 bd45a30a); its
# backbone shards and head must be byte-identical to node C's m9-KIB-s1 zero-step checkpoint, K-a13IB's Lux member
# (SHA-256 list pinned below). A failed step writes soup/<name>/FAILED and is never rerun.
#
# usage: M10_NODE=b post.sh launch|run <mirror-dir> <ARM>
set -u
MODE=$1 SRC=$2 ARM=$3
NODE=${M10_NODE:?set M10_NODE=b}
M=/data/dev2/runs/9b/m10
ST=$M/status
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10
LUXRUN=m10-KUP-s1
[ "$NODE" = a ] && LUXRUN=m10-KX-s1
LUXHOST=$M/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXCK=/runs/m10/arms/pre/$LUXRUN-zero/checkpoint-0000000
LUXSUMS=${M10_LUX_SUMS:-$M/inputs/lux-zero-m9-KIB-s1.sha256}
mkdir -p "$M/chains" "$M/logs" "$M/soup"
if [ "$MODE" = launch ]; then
  mkdir "$M/chains/post-$ARM.lock" 2> /dev/null || { echo "post chain $ARM already launched"; exit 0; }
  M10_NODE=$NODE setsid nohup bash "$0" run "$SRC" "$ARM" > "$M/logs/post-$ARM.log" 2>&1 < /dev/null &
  echo $! > "$M/chains/post-$ARM.pid"
  echo "$(date -u +%FT%TZ) M10 post chain $ARM launched from $SRC (pid $(cat "$M/chains/post-$ARM.pid"))" \
    | tee -a "$M/OPERATIONS.log"
  exit 0
fi
log() { echo "$(date -u +%FT%TZ) post-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
terminal() { [ -f "$ST/m10-$ARM-s$1.DONE" ] || [ -f "$ST/m10-$ARM-s$1.FAILED" ] || [ -f "$ST/m10-$ARM-s$1.STOPPED" ]; }
case $ARM in KSW | KIB4) SEEDS="1 2" ;; *) SEEDS="1 2 3" ;; esac  # two-seed arms: amendment 3
n=0
log "waiting for the seeds of $ARM"
all_terminal() { local s; for s in $SEEDS; do terminal "$s" || return 1; done; }
until all_terminal; do
  n=$((n + 1))
  [ $((n % 30)) = 0 ] && log "still waiting for the seeds of $ARM"
  sleep 60
done
build() {  # <name> <member container path>...
  local name=$1 out=$M/soup/$1
  shift
  [ -f "$out/DONE" ] && return 0
  [ -f "$out/FAILED" ] && return 1
  mkdir -p "$out"
  args=()
  for p in "$@"; do args+=(--member "$p"); done
  if M10_NODE=$NODE bash "$OPS/launch.sh" "soup-$name" "$SRC" "$out/build" --cpu -- -m v2.dec.soup "${args[@]}" \
    --output "/out/$name"; then
    printf '%s\n' "$@" > "$out/members.txt"
    echo "$out/build/$name" > "$out/DONE"
    log "built $name: $(tail -c 300 "$out/build.stdout.log" | tr '\n' ' ')"
  else
    echo "soup build failed (see $out/build.stderr.log)" > "$out/FAILED"
    log "FAILED: $name"
    return 1
  fi
}
members=()
for s in $SEEDS; do
  [ -f "$ST/m10-$ARM-s$s.DONE" ] || continue
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$M/arms/full/m10-$ARM-s$s/BEST.json")
  [ -f "$M/arms/full/m10-$ARM-s$s/$best/decision_config.json" ] || { log "s$s BEST $best is not a checkpoint"; exit 1; }
  members+=("/runs/m10/arms/full/m10-$ARM-s$s/$best")
  echo "s$s $best" >> "$M/soup/$ARM.seeds.txt"
done
case ${#members[@]} in
  0) log "no finished seed; no artifact"; exit 1 ;;
  1) S=${members[0]}
     log "one finished seed: the arm artifact is $S (disclosed)" ;;
  *) build "$ARM" "${members[@]}" || exit 1
     S=/runs/m10/soup/$ARM/build/$ARM ;;
esac
[ -f "$LUXSUMS" ] || { log "missing pinned Lux member SHA-256 list $LUXSUMS"; exit 1; }
(cd "$LUXHOST" && sha256sum -c --quiet "$LUXSUMS") || { log "Lux zero-step member differs from K-a13IB's"; exit 1; }
build "$ARM-a33" "$S" "$LUXCK" "$LUXCK"
build "$ARM-a25" "$S" "$LUXCK" "$LUXCK" "$LUXCK"
build "$ARM-a40" "$S" "$S" "$LUXCK" "$LUXCK" "$LUXCK"
log "post chain finished"
