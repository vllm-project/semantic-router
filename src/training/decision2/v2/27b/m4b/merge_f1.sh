#!/usr/bin/env bash
# M4b F1M: F1 (the M3-A rank-16 LoRA soup) merged into the pinned base in FP32 with F1's head (node B host; CPU
# containers of the pinned image, no GPU, no network).
#   1. training.model.materialize (PEFT safe merge, FP32) -> /data/dev2/runs/27b/m4b/F1M/checkpoint and
#      checkpoint.materialization.json
#   2. v2.27b.m4b.interp_full merge-check (float64: every merged projection within 1e-6 relative of
#      base + scale*B@A, every other text tensor equal to the base in FP32, head bitwise) -> F1M/merge-check.json
# Usage: merge_f1.sh MIRROR_SHA [STAGES]   (STAGES: materialize,check; default both)
set -euo pipefail
echo "m4b merge_f1 $* start $(date -u +%FT%TZ)"

MIRROR=$1 STAGES=${2:-materialize,check}
S=/data/dev2/src/$MIRROR/src/training/decision2
[ -d "$S" ] || S=/data/dev2/src/$MIRROR-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m4b/interp_full.py" ] || { echo "no m4b code in mirror $MIRROR" >&2; exit 2; }
S=$(cd "$S" && pwd -P)
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
BASE=${BASE:-/data/decision20-20260926/models/Qwen3.8-27B}
F1=${F1:-/data/dev2/runs/27b/M3-A-soup/soup/checkpoint}
OUT=/data/dev2/runs/27b/m4b/F1M
CPUS=${MERGE_CPUS:-16}
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
[ -f "$F1/soup_manifest.json" ] || { echo "missing F1 soup $F1" >&2; exit 2; }
mkdir -p "$OUT"
export TMPDIR=/data/dev2/tmp
cpu() {  # NAME ARGS...: python3 ARGS in a CPU-only container with the code, base and F1 read-only
  local name=$1
  shift
  docker run --rm --name "d2-27b-m4b-$name" --network none --cpus "$CPUS" -e "OMP_NUM_THREADS=$CPUS" \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e "PYTHONPATH=$S" -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 --mount "type=bind,src=$S,dst=$S,readonly" \
    --mount "type=bind,src=$BASE,dst=$BASE,readonly" --mount "type=bind,src=$F1,dst=$F1,readonly" \
    --mount "type=bind,src=$OUT,dst=$OUT" -w "$S" --entrypoint python3 "$IMAGE" "$@"
}

if has materialize; then
  [ ! -e "$OUT/checkpoint" ] || { echo "$OUT/checkpoint exists" >&2; exit 66; }
  cpu f1m-materialize -m training.model.materialize --checkpoint "$F1" --source-path "$BASE" \
    --output "$OUT/checkpoint" --device cpu 2>&1 | tee "$OUT/materialize.log"
fi
if has check; then
  cpu f1m-check -m v2.27b.m4b.interp_full merge-check --merged "$OUT/checkpoint" --lora "$F1" \
    --source-path "$BASE" --output "$OUT/merge-check.json" 2>&1 | tee "$OUT/merge-check.log"
fi
echo "m4b merge_f1 stages $STAGES complete"
