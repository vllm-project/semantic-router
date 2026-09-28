#!/usr/bin/env bash
# usage: formal.sh SHA GPU NAME CHECKPOINT CAL_DIR LABEL
# Formal post-key same-panel collection of one Milestone 4 candidate on node A with the eval
# track's frozen runner (default image, as the comparators) at the 16,384-token limit: gold-free
# smoke, typed FINAL + CSS15 + public 231, then mlx-diag; then seal, report and paired compare
# against the Lux1 16K comparator (eval m1/d1-lux1-autotune-cache) and the same-renderer Lux1
# 16K control (formal-m3/lux1-16k-shared). Runs land in /data/dev2/runs/9b/formal-m4/NAME-16k
# (+ -smoke, -mlx). The autotune cache is formal-m4/triton-cache: one copy of the frozen
# formal-m3/triton-cache (the cache the same-renderer control used), made on first use and never
# refreshed; its tree hash is recorded before and after every collection in NAME-cache.jsonl.
# CHECKPOINT and CAL_DIR are host paths (a full checkpoint; the directory with calibration.json).
set -uo pipefail
sha=$1; gpu=$2; name=$3; ckpt=$4; cal=$5; label=$6
SRC=$sha-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
F=/data/dev2/runs/9b/formal-m4
F3=/data/dev2/runs/9b/formal-m3
LUX=/data/decision20-20260926/models/Decision-1.0-Lux-9B
COMPARATOR=/data/dev2/runs/eval/m1/d1-lux1-autotune-cache
SHARED=$F3/lux1-16k-shared
TC=$F/triton-cache
[ -f "$ckpt/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[ -f "$cal/calibration.json" ] || { echo "no calibration.json in $cal" >&2; exit 2; }
[ -f "$SHARED/SEAL.json" ] || { echo "same-renderer control $SHARED is not sealed" >&2; exit 2; }
[ ! -e "$F/$name-16k" ] || { echo "$F/$name-16k exists" >&2; exit 66; }
tree_sha() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
mkdir -p "$F"
(
  flock 8
  if [ ! -d "$TC" ]; then
    [ -d "$F3/triton-cache" ] || { echo "frozen cache $F3/triton-cache missing" >&2; exit 2; }
    rm -rf "$TC.pending"
    cp -a "$F3/triton-cache" "$TC.pending"
    printf '{"source": "%s", "copied_utc": "%s", "files": %s, "source_tree_sha256": "%s", "tree_sha256": "%s"}\n' \
      "$F3/triton-cache" "$(date -u +%FT%TZ)" "$(find "$TC.pending" -type f | wc -l)" \
      "$(tree_sha "$F3/triton-cache")" "$(tree_sha "$TC.pending")" > "$F/triton-cache.copy.json"
    mv "$TC.pending" "$TC"
  fi
) 8>"$F/.triton-copy.lock" || exit 2
CACHE=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
cache_note() {
  printf '{"step": "%s", "utc": "%s", "files": %s, "tree_sha256": "%s"}\n' "$1" "$(date -u +%FT%TZ)" \
    "$(find "$TC" -type f | wc -l)" "$(tree_sha "$TC")" >> "$F/$name-cache.jsonl"
}
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
rev="m4-$name-$(basename "$ckpt")"
params_source="full Lux 1.0 architecture (backbone + head), 7,940,895,744"
cand() {
  out=$1; shift
  cache_note "before $out"
  "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track 9b-clm --src "$SRC" --run-dir "$F/$out" --model-dir "$ckpt" \
    --mount "$LUX" --mount "$cal" "${CACHE[@]}" --purpose "9B M4 formal $name" -- \
    --adapter-spec "$S/v2/dec/adapter-spec-infer-dec.json" --model-path "$ckpt" --revision "$rev" \
    --extra source="$LUX" --extra model_id="decision2-9b-m4-$name" --extra max_length=16384 \
    --extra calibration="$cal/calibration.json" "$@"
  status=$?
  cache_note "after $out"
  return $status
}
cand "$name-smoke" --max-items 8 || exit 1; waitidle
cand "$name-16k" || exit 1; waitidle
cand "$name-16k-mlx" --panels mlx-diag || exit 1
sp seal --run-dir "$F/$name-16k" || exit 1
sp report --run-dir "$F/$name-16k" --label "$label" --tier 9B --family decision2 --model-id "decision2-9b-m4-$name" \
  --revision "$rev" --loaded-parameters 7940895744 --parameter-source "$params_source" || exit 1
sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$COMPARATOR" --left-name "$name" --right-name Lux1-16K || exit 1
sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$SHARED" --left-name "$name" \
  --right-name Lux1-16K-shared || exit 1
mlx "$F/$name-16k-mlx"
echo "done $name"
