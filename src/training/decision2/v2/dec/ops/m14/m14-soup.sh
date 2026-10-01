#!/usr/bin/env bash
# Decoder M14 arm artifact (prereg "Arms"): the uniform FP32 soup of the seeds' BEST checkpoints. LoRA seeds (4B
# 4b-LHA10UP) are first merged into full FP32 checkpoints on an M14 GPU (M10's m10_merge.py, with its SELECT agreement
# check; the tier's read cache); full fine-tune seeds (2b-RAUP, 08b-RAUP) are souped directly. An arm with fewer than
# two finished seeds has no soup. Outputs: /data/dev2/runs/dec/m14/soup/<ARM>/{merged/s<i>,build/<ARM>-soup},
# markers soup/<ARM>/{DONE,FAILED}.
#
# usage: M14_NODE=a|b m14-soup.sh <mirror-dir> <ARM> <gpu>
set -u
SRC=$1 ARM=$2 GPU=$3
NODE=${M14_NODE:?set M14_NODE=a or b}
M=/data/dev2/runs/dec/m14
ST=$M/status
OUT=$M/soup/$ARM
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m14
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  4b-LHA10UP) lora=1 source=/hf/models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b ;;
  2b-RAUP | 08b-RAUP) lora=0 ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
rm -f "$OUT/members.txt"
for s in 1 2; do
  [ -f "$ST/m14-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m14-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  if [ $lora = 1 ]; then
    merged=$OUT/merged/s$s
    if [ ! -f "$merged/merge_check.json" ]; then
      M14_NODE=$NODE M14_CACHE=${ARM%%-*}-read bash "$OPS/m14-launch.sh" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" \
        --gpu "$GPU" -- v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m14/arms/full/m14-$ARM-s$s/$best" \
        --source-path "$source" --select /data/select.jsonl --output "/out/s$s" \
        || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
      mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
      log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
    fi
    members+=(--member "/runs/m14/soup/$ARM/merged/s$s")
  else
    members+=(--member "/runs/m14/arms/full/m14-$ARM-s$s/$best")
  fi
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge 4 ] || fail "fewer than two finished seeds"
M14_NODE=$NODE bash "$OPS/m14-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
