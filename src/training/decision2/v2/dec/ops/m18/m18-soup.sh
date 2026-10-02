#!/usr/bin/env bash
# Decoder M18 arm artifact (prereg "Part B"): the uniform FP32 soup of both seeds' BEST checkpoints (full fine-tune
# seeds, souped directly as M14's 2b-RAUP). An arm with fewer than two finished seeds has no soup. Outputs:
# /data/dev2/runs/dec/m18/soup/<ARM>/build/<ARM>-soup, markers soup/<ARM>/{DONE,FAILED}; DONE holds the soup path.
#
# usage: M18_NODE=a|f m18-soup.sh <mirror-dir> <ARM>
set -u
SRC=$1 ARM=$2
NODE=${M18_NODE:?set M18_NODE=a or f}
M=/data/dev2/runs/dec/m18
ST=$M/status
OUT=$M/soup/$ARM
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m18
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  2b-RS17UP | 2b-RAUPM | 2b-RAM | 08b-RAM) ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
rm -f "$OUT/members.txt"
for s in 1 2; do
  [ -f "$ST/m18-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m18-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  members+=(--member "/runs/m18/arms/full/m18-$ARM-s$s/$best")
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge 4 ] || fail "fewer than two finished seeds"
M18_NODE=$NODE bash "$OPS/m18-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
