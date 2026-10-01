#!/usr/bin/env bash
# Decoder M10 arm artifact (prereg "Arm artifact"): the uniform FP32 soup of the seeds' BEST checkpoints. LoRA seeds
# (LH, LT2) are first merged into full FP32 checkpoints on an M10 GPU (m10_merge.py, with its SELECT agreement check);
# full seeds (FB, NT2) are souped directly. An arm with fewer than two finished seeds has no soup (two are disclosed).
# Outputs: /data/dev2/runs/dec/m10/soup/<ARM>/{merged/s<i>,<ARM>-soup}, markers soup/<ARM>/{DONE,FAILED}.
#
# usage: M10_NODE=e|f m10-soup.sh <mirror-dir> <ARM> <gpu>
set -u
SRC=$1 ARM=$2 GPU=$3
NODE=${M10_NODE:?set M10_NODE=e or f}
M=/data/dev2/runs/dec/m10
ST=$M/status
OUT=$M/soup/$ARM
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m10
BASE=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
NOX=/models/Decision-1.0-Nox-4B/cde2a68dbaa557ea65dc458104d410a0802ee259
mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) soup-$ARM $*" | tee -a "$M/OPERATIONS.log"; }
fail() { echo "$*" > "$OUT/FAILED"; log "FAILED: $*"; exit 1; }
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
case $ARM in
  LH | LT2) lora=1 source=$BASE ;;
  FB) lora=0 source=$BASE ;;
  NT2) lora=0 source=$NOX ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
for s in 1 2 3; do
  [ -f "$ST/m10-$ARM-s$s.DONE" ] || continue
  run=$M/arms/full/m10-$ARM-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  if [ $lora = 1 ]; then
    merged=$OUT/merged/s$s
    if [ ! -f "$merged/merge_check.json" ]; then
      M10_NODE=$NODE bash "$OPS/m10-launch.sh" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" --gpu "$GPU" -- \
        v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m10/arms/full/m10-$ARM-s$s/$best" --source-path "$source" \
        --select /data/select.jsonl --output "/out/s$s" || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
      mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
      log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
    fi
    members+=(--member "/runs/m10/soup/$ARM/merged/s$s")
  else
    members+=(--member "/runs/m10/arms/full/m10-$ARM-s$s/$best")
  fi
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge 4 ] || fail "fewer than two finished seeds"
M10_NODE=$NODE bash "$OPS/m10-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
