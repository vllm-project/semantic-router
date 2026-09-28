#!/usr/bin/env bash
# usage: formal.sh SHA GPU cand NAME CHECKPOINT CAL_DIR LABEL
#        formal.sh SHA GPU lux1
# Formal post-key same-panel collection on node A with the eval track's frozen runner at the
# 16,384-token limit and the frozen Milestone 3 autotune cache copy (formal-m3/triton-cache):
# gold-free smoke, typed FINAL + CSS15 + public 231, then mlx-diag; then seal, report, paired
# compare against the Lux1 16K comparator (eval m1/d1-lux1-autotune-cache) and, when present,
# the same-renderer Lux1 16K control. "lux1" collects that control (shared from_decision1
# renderer, package temperature) once.
set -uo pipefail
sha=$1; gpu=$2; mode=$3; shift 3
SRC=$sha-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
F=/data/dev2/runs/9b/formal-m3
LUX=/data/decision20-20260926/models/Decision-1.0-Lux-9B
LUX_REV=bd45a30aee8c84032791c245c70f86dee5389cc8
COMPARATOR=/data/dev2/runs/eval/m1/d1-lux1-autotune-cache
TC=$F/triton-cache
CACHE=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
[ -d "$TC" ] || { echo "frozen cache $TC missing" >&2; exit 2; }
waitidle() {
  for _ in $(seq 1 60); do
    v=$(rocm-smi -d "$gpu" --showmemuse | awk -F': ' '/VRAM%/ {v=$NF} END {print v}')
    [ "$v" = 0 ] && return 0
    sleep 5
  done
  return 1
}
sp() { (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" python3 -m v2.eval.same_panel "$@"); }
mlx() {
  (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" python3 -m v2.eval.multilingual_panel score \
    --panel /data/dev2/private/panels/mlx-diag-v1 --predictions "$1/output/mlx-diag.predictions.jsonl" \
    --output "$1/mlx-diag.score.json")
}
if [ "$mode" = lux1 ]; then
  lux() {
    out=$1; shift
    "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 9b-clm --src "$SRC" --run-dir "$F/$out" --model-dir "$LUX" \
      "${CACHE[@]}" --purpose "9B M3 Lux1 16K same-renderer control" -- \
      --adapter-spec "$S/v2/9b/lux9b/adapter-spec-lux1-infer1p0.json" --model-path "$LUX" --revision "$LUX_REV" \
      --extra model_id=llm-semantic-router/Decision-1.0-Lux-9B --extra max_length=16384 "$@"
  }
  lux lux1-16k-shared-smoke --max-items 8 || exit 1; waitidle
  lux lux1-16k-shared || exit 1; waitidle
  lux lux1-16k-shared-mlx --panels mlx-diag || exit 1
  sp seal --run-dir "$F/lux1-16k-shared" || exit 1
  sp report --run-dir "$F/lux1-16k-shared" --label "post-key same-panel" --tier 9B --family decision1 \
    --model-id llm-semantic-router/Decision-1.0-Lux-9B --revision "$LUX_REV" --loaded-parameters 7940895744 \
    --parameter-source "Lux 1.0 deployed" || exit 1
  sp compare --run-dir "$F/lux1-16k-shared" --comparator-run-dir "$COMPARATOR" --left-name Lux1-16K-shared \
    --right-name Lux1-16K || exit 1
  mlx "$F/lux1-16k-shared-mlx"
  echo "done lux1-16k-shared"
  exit 0
fi
[ "$mode" = cand ] || { echo "mode must be cand or lux1" >&2; exit 2; }
name=$1; ckpt=$2; cal=$3; label=$4
rev="m3-$name-$(basename "$ckpt")"
lora=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trainable_by_group"]["lora"])' "$(dirname "$ckpt")/provenance.json")
cand() {
  out=$1; shift
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 9b-clm --src "$SRC" --run-dir "$F/$out" --model-dir "$ckpt" \
    --mount "$LUX" --mount "$cal" "${CACHE[@]}" --purpose "9B M3 formal $name" -- \
    --adapter-spec "$S/v2/dec/adapter-spec-infer-dec.json" --model-path "$ckpt" --revision "$rev" \
    --extra source="$LUX" --extra model_id="decision2-9b-m3-$name" --extra max_length=16384 \
    --extra calibration="$cal/calibration.json" "$@"
}
cand "$name-smoke" --max-items 8 || exit 1; waitidle
cand "$name-16k" || exit 1; waitidle
cand "$name-16k-mlx" --panels mlx-diag || exit 1
sp seal --run-dir "$F/$name-16k" || exit 1
sp report --run-dir "$F/$name-16k" --label "$label" --tier 9B --family decision2 --model-id "decision2-9b-m3-$name" \
  --revision "$rev" --loaded-parameters $((7940895744 + lora)) \
  --parameter-source "Lux 1.0 deployed 7,940,895,744 + LoRA $lora" || exit 1
sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$COMPARATOR" --left-name "$name" --right-name Lux1-16K || exit 1
if [ -f "$F/lux1-16k-shared/SEAL.json" ]; then
  sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$F/lux1-16k-shared" --left-name "$name" \
    --right-name Lux1-16K-shared
fi
mlx "$F/$name-16k-mlx"
echo "done $name"
