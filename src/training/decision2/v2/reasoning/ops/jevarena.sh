#!/usr/bin/env bash
# Reasoning track: post-key same-panel JevArena collection of one FP32 candidate checkpoint, exactly as the released
# 4B's formal run m17-4b-LRHxALL (v2.dec.infer_dec, image dbe5f32b, 16,384-token limit, T = 1, panels typed-final,
# css15, public231): gold-free on this node; scoring (seal / report / compare) happens where the gold panels live.
#
# usage: jevarena.sh <mirror-dir> <gpu> <NAME> <checkpoint-host-path> [--max-items N]
set -euo pipefail
src=$1 gpu=$2 name=$3 ckpt=$4
shift 4
S=/data/dev2/src/$src/src/training/decision2
OUT=/data/dev2/runs/reasoning/formal/$name
SOURCE=/data/dev2/models/Decision-1.0-Nox-4B/cde2a68dbaa557ea65dc458104d410a0802ee259
[[ -f $ckpt/decision_config.json ]] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[[ -e $OUT ]] && { echo "$OUT exists" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")" /data/dev2/runs/reasoning/triton-cache/formal-4b
"$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track reasoning-eval --src "$src" --run-dir "$OUT" \
  --model-dir "$ckpt" --mount "$SOURCE" --isolate --image decision20-train-fast:latest \
  --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=/data/dev2/runs/reasoning/triton-cache/formal-4b \
  --mount-rw /data/dev2/runs/reasoning/triton-cache/formal-4b --purpose "reasoning JevArena $name" -- \
  --adapter-spec "$S/v2/dec/ops/m5/m5-adapter-infer-dec-t1.json" --model-path "$ckpt" --revision "rsn-$name" \
  --extra source="$SOURCE" --extra model_id="decision2-rsn-$name" --extra max_length=16384 "$@"
