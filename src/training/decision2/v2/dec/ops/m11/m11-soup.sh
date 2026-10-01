#!/usr/bin/env bash
# Decoder M11 arm artifact (prereg "Training"): the uniform FP32 soup of the seeds' BEST checkpoints. LoRA seeds
# (<t>-LH, stage 2's 4b-LHB / 4b-LHBx) are first merged into full FP32 checkpoints on an M11 GPU (M10's m10_merge.py, with its SELECT agreement
# check; the tier's read cache); full seeds (<t>-NT) are souped directly. An arm with fewer than two finished seeds has
# no soup (two are disclosed). Outputs: /data/dev2/runs/dec/m11/soup/<ARM>/{merged/s<i>,build/<ARM>-soup}, markers
# soup/<ARM>/{DONE,FAILED}.
#
# usage: M11_NODE=e|f m11-soup.sh <mirror-dir> <ARM> <gpu>
set -u
SRC=$1 ARM=$2 GPU=$3
NODE=${M11_NODE:?set M11_NODE=e or f}
M=/data/dev2/runs/dec/m11
ST=$M/status
OUT=$M/soup/$ARM
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m11
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  2b-LH) lora=1 source=/models/Qwen--Qwen3.5-2B-Base/b1485b2fa6dfa1287294f269f5fb618e03d52d7c ;;
  08b-LH) lora=1 source=/models/Qwen--Qwen3.5-0.8B-Base/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 ;;
  4b-LHB | 4b-LHBx) lora=1 source=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b ;;
  2b-NT | 08b-NT) lora=0 ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
rm -f "$OUT/members.txt"
for s in 1 2 3; do
  [ -f "$ST/m11-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m11-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  if [ $lora = 1 ]; then
    merged=$OUT/merged/s$s
    if [ ! -f "$merged/merge_check.json" ]; then
      M11_NODE=$NODE M11_CACHE=${ARM%%-*}-read bash "$OPS/m11-launch.sh" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" \
        --gpu "$GPU" -- v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m11/arms/full/m11-$ARM-s$s/$best" \
        --source-path "$source" --select /data/select.jsonl --output "/out/s$s" \
        || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
      mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
      log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
    fi
    members+=(--member "/runs/m11/soup/$ARM/merged/s$s")
  else
    members+=(--member "/runs/m11/arms/full/m11-$ARM-s$s/$best")
  fi
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge 4 ] || fail "fewer than two finished seeds"
M11_NODE=$NODE bash "$OPS/m11-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
