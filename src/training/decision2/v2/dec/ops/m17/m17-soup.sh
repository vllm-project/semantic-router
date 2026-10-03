#!/usr/bin/env bash
# Decoder M17 arm artifact (prereg "Arms"): the uniform FP32 soup of both seeds' BEST checkpoints. Each LoRA seed is
# first merged into a full FP32 checkpoint on an M17 GPU (M10's m10_merge.py, with its SELECT agreement check; the 4B
# read cache). An arm with fewer than two finished seeds has no soup. Outputs:
# /data/dev2/runs/dec/m17/soup/<ARM>/{merged/s<i>,build/<ARM>-soup}, markers soup/<ARM>/{DONE,FAILED}; DONE holds the
# soup path, MODEL_SHA256 its model digest (v2.dec.soup's dec_fingerprint). Stage 2's interpolation points
# 4b-<X>-m50 are the uniform FP32 average of the 4b-LHS17SD soup and the 4b-<X> soup (CPU; <gpu> unused); wave 3's
# (amendment 2) average with the released M15 4b-LHA10SDML soup instead, and 4b-SDMLxS17-m50 is the average of the
# 4b-LHA10SDML and 4b-LHS17SD soups. 4b-LHS17IB4-x3 / 4b-SDMLIB4-x3 (amendment 3) are the soups of seeds 1-3 of
# their arm; seeds 1-2 reuse the arm's merged checkpoints when its members.txt names the same BEST checkpoint.
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
SDML_FP32=1b51567523426b896ae50afeefca4f133c2e3c220c349ebfb9b9c630600dd19d
[ -f "$OUT/DONE" ] && { log "already built"; exit 0; }
[ -f "$OUT/FAILED" ] && { log "failed earlier; not rebuilt"; exit 1; }
TR=$ARM SEEDN="1 2" NEED=4
case $ARM in
  4b-LHS10SD | 4b-LHS17SD | 4b-LHS17UP | 4b-LHS23SD | 4b-LHS17IB4 | 4b-LHS17IB4X | 4b-SDMLIB4 | 4b-LHS17ML)
    source=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b ;;
  4b-LHS17IB4-x3 | 4b-SDMLIB4-x3)
    TR=${ARM%-x3} SEEDN="1 2 3" NEED=6
    source=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b ;;
  4b-*-m50 | 4b-SDMLxS17xIB4 | 4b-SDMLxALL | 4b-SDMLxALL9 | 4b-SDMLxALL15)
    case $ARM in
      4b-SDMLxS17xIB4) list="4b-LHA10SDML 4b-LHS17SD 4b-LHS17IB4" ;;
      4b-SDMLxALL) list="4b-LHA10SDML 4b-LHS17SD 4b-LHS17UP 4b-LHS17IB4 4b-LHS17IB4X 4b-SDMLIB4 4b-LHS17ML" ;;
      4b-SDMLxALL9) list="4b-LHA10SDML 4b-LHS10SD 4b-LHS17SD 4b-LHS17UP 4b-LHS23SD 4b-LHS17IB4-x3 4b-LHS17IB4X 4b-SDMLIB4-x3 4b-LHS17ML" ;;
      4b-SDMLxALL15) list="4b-LHA10SDML 4b-LHS10SD 4b-LHS17SD 4b-LHS17UP 4b-LHS23SD 4b-LHS17IB4-x3 4b-LHS17IB4X 4b-SDMLIB4-x3 4b-LHS17ML m10/LH m11/4b-LHB m12/4b-LHA m12/4b-LHA10 m13/4b-LHA10SD m13/4b-LHA5" ;;
      *)
        x=${ARM%-m50} base=4b-LHS17SD
        case $x in 4b-SDMLIB4 | 4b-LHS17ML) base=4b-LHA10SDML ;; 4b-SDMLxS17) base=4b-LHA10SDML x=4b-LHS17SD ;; esac
        list="$base $x" ;;
    esac
    rm -f "$OUT/members.txt"
    for a in $list; do
      if [ "$a" = 4b-LHA10SDML ]; then
        [ "$(cat /data/dev2/runs/dec/m15/soup/4b-LHA10SDML/DONE)" = /data/dev2/runs/dec/m15/soup/4b-LHA10SDML/build/4b-LHA10SDML-soup ] \
          || fail "no M15 soup of 4b-LHA10SDML"
        echo "$a $SDML_FP32" >> "$OUT/members.txt"
        continue
      fi
      if [[ $a == m1[0-3]/* ]]; then
        d=/data/dev2/runs/dec/${a%%/*}/soup/${a#*/}
        [ "$(cat "$d/DONE" 2> /dev/null)" = "$d/build/${a#*/}-soup" ] || fail "no soup of $a"
        echo "$a $d/build/${a#*/}-soup" >> "$OUT/members.txt"
        continue
      fi
      [ -f "$M/soup/$a/DONE" ] || fail "no soup of $a"
      echo "$a $(cat "$M/soup/$a/MODEL_SHA256")" >> "$OUT/members.txt"
    done
    member() {
      case $1 in
        4b-LHA10SDML) echo /runs/m15/soup/4b-LHA10SDML/build/4b-LHA10SDML-soup ;;
        m1[0-3]/*) echo "/runs/${1%%/*}/soup/${1#*/}/build/${1#*/}-soup" ;;
        *) echo "/runs/m17/soup/$1/build/$1-soup" ;;
      esac
    }
    mm=()
    for a in $list; do mm+=(--member "$(member "$a")"); done
    M17_NODE=$NODE bash "$OPS/m17-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${mm[@]}" \
      --output "/out/$ARM-soup" || fail "interpolation build failed (see $OUT/build.stderr.log)"
    python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
      "$OUT/build.stdout.log" > "$OUT/MODEL_SHA256" || fail "no model_sha256 in the interpolation output"
    echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
    log "built $OUT/build/$ARM-soup (uniform average of $list)"
    exit 0 ;;
  *) fail "unknown arm $ARM" ;;
esac
members=()
rm -f "$OUT/members.txt"
for s in $SEEDN; do
  [ -f "$ST/m17-$TR-s$s.DONE" ] || continue
  run=$M/arms/full/m17-$TR-s$s
  best=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$run/BEST.json")
  if [ "$TR" != "$ARM" ] && [ -f "$M/soup/$TR/merged/s$s/merge_check.json" ] \
    && grep -qx "s$s $best" "$M/soup/$TR/members.txt"; then
    members+=(--member "/runs/m17/soup/$TR/merged/s$s")
    echo "s$s $best" >> "$OUT/members.txt"
    continue
  fi
  merged=$OUT/merged/s$s
  if [ ! -f "$merged/merge_check.json" ]; then
    M17_NODE=$NODE M17_CACHE=4b-read bash "$OPS/m17-launch.sh" "merge-$ARM-s$s" "$SRC" "$OUT/merge-s$s" \
      --gpu "$GPU" -- v2/dec/ops/m10/m10_merge.py --checkpoint "/runs/m17/arms/full/m17-$TR-s$s/$best" \
      --source-path "$source" --select /data/select.jsonl --output "/out/s$s" \
      || fail "merge of s$s failed (see $OUT/merge-s$s.stderr.log)"
    mkdir -p "$OUT/merged" && mv "$OUT/merge-s$s/s$s" "$merged"
    log "s$s merged ($best): $(tail -c 300 "$OUT/merge-s$s.stdout.log" | tr '\n' ' ')"
  fi
  members+=(--member "/runs/m17/soup/$ARM/merged/s$s")
  echo "s$s $best" >> "$OUT/members.txt"
done
[ ${#members[@]} -ge "$NEED" ] || fail "fewer than $((NEED / 2)) finished seeds"
M17_NODE=$NODE bash "$OPS/m17-launch.sh" "soup-$ARM" "$SRC" "$OUT/build" --cpu -- -m v2.dec.soup "${members[@]}" \
  --output "/out/$ARM-soup" || fail "soup build failed (see $OUT/build.stderr.log)"
python3 -c 'import json,sys; print(json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])["model_sha256"])' \
  "$OUT/build.stdout.log" > "$OUT/MODEL_SHA256" || fail "no model_sha256 in the soup output"
echo "$OUT/build/$ARM-soup" > "$OUT/DONE"
log "built $OUT/build/$ARM-soup: $(tail -c 300 "$OUT/build.stdout.log" | tr '\n' ' ')"
