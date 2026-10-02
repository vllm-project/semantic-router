#!/usr/bin/env bash
# Decoder M17 arm artifact (prereg "Arms"): the uniform FP32 soup of both seeds' BEST checkpoints. Each LoRA seed is
# first merged into a full FP32 checkpoint on an M17 GPU (M10's m10_merge.py, with its SELECT agreement check; the 4B
# read cache). An arm with fewer than two finished seeds has no soup. Outputs:
# /data/dev2/runs/dec/m17/soup/<ARM>/{merged/s<i>,build/<ARM>-soup}, markers soup/<ARM>/{DONE,FAILED}; DONE holds the
# soup path, MODEL_SHA256 its model digest (v2.dec.soup's dec_fingerprint). Stage 2's interpolation points
# 4b-<X>-m50 are the uniform FP32 average of the 4b-LHS17SD soup and the 4b-<X> soup (CPU; <gpu> unused).
#
# usage: M17_NODE=f m17-soup.sh <mirror-dir> <ARM> <gpu>
set -u
SRC=$1 ARM=$2 GPU=$3
NODE=${M17_NODE:?set M17_NODE=f}
M=/data/dev2/runs/dec/m17
ST=$M/status
OUT=$M/soup/$ARM
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  4b-LHS10SD | 4b-LHS17SD | 4b-LHS17UP | 4b-LHS23SD | 4b-LHS17IB4 | 4b-LHS17IB4X)
    source=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b ;;
  4b-*-m50)
    x=${ARM%-m50}
    for a in 4b-LHS17SD "$x"; do
      [ -f "$M/soup/$a/DONE" ] || fail "no soup of $a"
      echo "$a $(cat "$M/soup/$a/MODEL_SHA256")" >> "$OUT/members.txt"
    done
    M17_NODE=$NODE bash "$OPS/m17-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup \
      --member /runs/m17/soup/4b-LHS17SD/build/4b-LHS17SD-soup --member "/runs/m17/soup/$x/build/$x-soup" \
      --output "/out/$ARM-soup" || fail "interpolation build failed (see $OUT/build.stderr.log)"
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$OUT/build.stdout.log" > "$OUT/MODEL_SHA256" || fail "no model_sha256 in the interpolation output"
    echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
    log "built $OUT/build/$ARM-soup (uniform .5 / .5 of 4b-LHS17SD and $x)"
    exit 0 ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
rm -f "$OUT/members.txt"
for s in 1 2; do
  [ -f "$ST/m17-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m17-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  merged=$OUT/merged/s$s
  if [ ! -f "$merged/merge_check.json" ]; then
    M17_NODE=$NODE M17_CACHE=4b-read bash "$OPS/m17-launch.sh" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" \
      --gpu "$GPU" -- v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m17/arms/full/m17-$ARM-s$s/$best" \
      --source-path "$source" --select /data/select.jsonl --output "/out/s$s" \
      || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
    mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
    log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
  fi
  members+=(--member "/runs/m17/soup/$ARM/merged/s$s")
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge 4 ] || fail "fewer than two finished seeds"
M17_NODE=$NODE bash "$OPS/m17-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
  "$OUT/build.stdout.log" > "$OUT/MODEL_SHA256" || fail "no model_sha256 in the soup output"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
