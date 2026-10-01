#!/usr/bin/env bash
# 9B M9 arm artifact (prereg "Merge and soup") on node C: each finished LoRA seed's BEST is merged into its pinned start
# in FP32 on an M9 GPU (v2/dec/ops/m10/m10_merge.py, with its 128-row SELECT agreement check), then the uniform FP32
# soup of the merged seeds (v2.dec.soup, CPU). One finished seed: that merged seed is the artifact (disclosed).
# Outputs: /data/dev2/runs/9b/m9/soup/<ARM>/{merged/s<i>,build/<ARM>-soup}, markers soup/<ARM>/{DONE,FAILED}
# (DONE holds the artifact's host path).
#
# usage: M9_NODE=c soup.sh <mirror-dir> <ARM> <gpu>
set -u
SRC=$1 ARM=$2 GPU=$3
NODE=${M9_NODE:?set M9_NODE=c}
M=/data/dev2/runs/9b/m9
ST=$M/status
OUT=$M/soup/$ARM
L=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m9/launch.sh
BASE=/models/Qwen--Qwen3.5-9B-Base/68c46c4b3498877f3ef123c856ecfde50c39f404
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  L9 | L9IB | L9IBX) source=$BASE ;;
  L9L) source=/lux ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
for s in 1 2; do
  [ -f "$ST/m9-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m9-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  merged=$OUT/merged/s$s
  if [ ! -f "$merged/merge_check.json" ]; then
    M9_NODE=$NODE bash "$L" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" --gpu "$GPU" -- \
      v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m9/arms/full/m9-$ARM-s$s/$best" --source-path "$source" \
      --select /data/select.jsonl --output "/out/s$s" || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
    mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
    log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
  fi
  members+=(--member "/runs/m9/soup/$ARM/merged/s$s")
  echo "s$s $best" >> "$OUT/members.txt"
done
case ${#members[@]} in
  0) fail "no finished seed" ;;
  2) echo "$OUT/merged/${members[1]##*/}" > "$OUT/DONE"
     log "one finished seed: the artifact is $(cat "$OUT/DONE") (disclosed)" ;;
  *) M9_NODE=$NODE bash "$L" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
       --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
     echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
     log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')" ;;
esac
