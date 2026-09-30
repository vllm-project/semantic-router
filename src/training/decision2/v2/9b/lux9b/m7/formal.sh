#!/usr/bin/env bash
# usage: formal.sh RUNNER_SHA GPU NAME CHECKPOINT CAL_DIR LABEL
# Formal post-key same-panel collection of one Milestone 7 candidate on node A with the eval
# runner of the mirror RUNNER_SHA (run_same_panel / seal / report / compare / gates; the
# incumbent's was 3277dec9d) at the 16,384-token limit: gold-free smoke (--max-items 8), typed
# FINAL + CSS15 + public 231, then mlx-diag; seal, report and paired compare against the Lux1
# 16K comparator (eval m1/d1-lux1-autotune-cache) and the same-renderer Lux1 16K control
# (formal-m3/lux1-16k-shared). Runs land in /data/dev2/runs/9b/formal-m7/NAME-16k (+ -smoke,
# -mlx). The autotune cache is formal-m7/triton-cache: one copy of the frozen
# formal-m3/triton-cache (tree af623300..., refused otherwise), made on first use and never
# refreshed; its tree hash goes to NAME-cache.jsonl before and after every collection. Then CPU
# steps into formal-m7/NAME.gates/: gates paired vs the released DEV2.0-9B T = 1 run
# (PAIRED-vs-DEV2.0-9B-T1.json) and vs Nimble v2 (PAIRED-vs-Nimble2.json), gates types
# (types.json), lux9b.mlx_paired vs formal-m4/K-a13-16k-mlx (MLX-PAIRED-vs-DEV2.0-9B-T1.json),
# lux9b.score_levels (score-levels.json; both from this script's own mirror), successor-rule item 7
# gates public231 vs the released T = 1 run (PUBLIC231-vs-DEV2.0-9B-T1.json; own mirror) and successor.json
# (numbers only, no verdict). CHECKPOINT and CAL_DIR are host paths (a full checkpoint; the
# directory with calibration.json). On GPU7 each collection waits while the eval track's C1
# lease entry is active. The GPU lease owner must say track=9b-m7 (chain-step.sh writes it).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
rsha=$1; gpu=$2; name=$3; ckpt=$4; cal=$5; label=$6
SRC=$rsha-src_training_decision2
S=$(code_dir "$rsha")
OWN=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
F=$RUNS/formal-m7
F3=$RUNS/formal-m3
F4=$RUNS/formal-m4
LUX=$DATA/decision20-20260926/models/Decision-1.0-Lux-9B
COMPARATOR=$DATA/dev2/runs/eval/m1/d1-lux1-autotune-cache
SHARED=$F3/lux1-16k-shared
T1=$DATA/dev2/runs/release/dev2-8b-t1-derived
NIMBLE=$DATA/dev2/runs/eval/m2/q6-nimble2
INC_MLX=$F4/K-a13-16k-mlx
MLX_PANEL=$DATA/dev2/private/panels/mlx-diag-v1
FROZEN=${D2_FROZEN_CACHE_SHA:-af623300d71a8fdb9a6a5588d25e17356da3610d57710b08b5830a0c5d6efb6f}
TC=$F/triton-cache
G=$F/$name.gates
[ -d "$S" ] || { echo "runner mirror $S missing" >&2; exit 2; }
[ -f "$ckpt/decision_config.json" ] || { echo "no checkpoint at $ckpt" >&2; exit 2; }
[ -f "$cal/calibration.json" ] || { echo "no calibration.json in $cal" >&2; exit 2; }
for d in "$SHARED" "$T1"; do [ -f "$d/SEAL.json" ] || { echo "comparator $d is not sealed" >&2; exit 2; }; done
[ ! -e "$F/$name-16k" ] || { echo "$F/$name-16k exists" >&2; exit 66; }
mkdir -p "$F"
(
  flock 8
  if [ ! -d "$TC" ]; then
    [ -d "$F3/triton-cache" ] || { echo "frozen cache $F3/triton-cache missing" >&2; exit 2; }
    src_tree=$(tree_sha "$F3/triton-cache")
    [ "$src_tree" = "$FROZEN" ] || { echo "frozen cache tree $src_tree != $FROZEN" >&2; exit 2; }
    rm -rf "$TC.pending"
    cp -a "$F3/triton-cache" "$TC.pending"
    printf '{"source": "%s", "copied_utc": "%s", "files": %s, "source_tree_sha256": "%s", "tree_sha256": "%s"}\n' \
      "$F3/triton-cache" "$(date -u +%FT%TZ)" "$(find "$TC.pending" -type f | wc -l)" \
      "$src_tree" "$(tree_sha "$TC.pending")" > "$F/triton-cache.copy.json"
    mv "$TC.pending" "$TC"
  fi
) 8>"$F/.triton-copy.lock" || exit 2
CACHE=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
cache_note() {
  printf '{"step": "%s", "utc": "%s", "files": %s, "tree_sha256": "%s"}\n' "$1" "$(date -u +%FT%TZ)" \
    "$(find "$TC" -type f | wc -l)" "$(tree_sha "$TC")" >> "$F/$name-cache.jsonl"
}
waitidle() {
  [ "${DRY_RUN:-0}" != 1 ] || return 0
  for _ in $(seq 1 60); do
    v=$(rocm-smi -d "$gpu" --showmemuse | awk -F': ' '/VRAM%/ {v=$NF} END {print v}')
    [ "$v" = 0 ] && return 0
    sleep 5
  done
  return 1
}
sp() { (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" dry python3 -m v2.eval.same_panel "$@"); }
gates() { (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" dry python3 -m v2.eval.gates "$@"); }
lux9b() { (cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN:$OWN/v2/9b" dry python3 -m "$@"); }
gates6() { (cd "$OWN" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$OWN" dry python3 -m v2.eval.gates "$@"); }
mlx() {
  (cd "$S" && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$S" dry python3 -m v2.eval.multilingual_panel score \
    --panel "$MLX_PANEL" --predictions "$1/output/mlx-diag.predictions.jsonl" \
    --output "$1/mlx-diag.score.json")
}
rev="m7-$name-$(basename "$ckpt")"
model_id="decision2-9b-m7-$name"
params_source="full Lux 1.0 architecture (backbone + head), 7,940,895,744"
python3 - "$F/$name.inputs.json" "$name" "$ckpt" "$cal" "$rsha" "$OWN" "$label" "$rev" <<'EOF'
import hashlib, json, sys
out, name, ckpt, cal, rsha, own, label, rev = sys.argv[1:]
sha = hashlib.sha256(open(f"{cal}/calibration.json", "rb").read()).hexdigest()
json.dump({"name": name, "checkpoint": ckpt, "calibration_dir": cal, "calibration_sha256": sha,
           "runner_sha": rsha, "own_mirror": own, "label": label, "revision": rev}, open(out, "w"), indent=1)
EOF
cand() {
  out=$1; shift
  yield_gpu7 "$gpu"
  cache_note "before $out"
  dry "$S/v2/eval/run_same_panel.sh" --gpu "$gpu" --track "$TRACK" --src "$SRC" --run-dir "$F/$out" \
    --model-dir "$ckpt" --mount "$LUX" --mount "$cal" "${CACHE[@]}" --purpose "9B M7 formal $name" -- \
    --adapter-spec "$S/v2/dec/adapter-spec-infer-dec.json" --model-path "$ckpt" --revision "$rev" \
    --extra source="$LUX" --extra model_id="$model_id" --extra max_length=16384 \
    --extra calibration="$cal/calibration.json" "$@"
  status=$?
  cache_note "after $out"
  return $status
}
cand "$name-smoke" --max-items 8 || exit 1; waitidle
cand "$name-16k" || exit 1; waitidle
cand "$name-16k-mlx" --panels mlx-diag || exit 1
sp seal --run-dir "$F/$name-16k" || exit 1
sp report --run-dir "$F/$name-16k" --label "$label" --tier 9B --family decision2 --model-id "$model_id" \
  --revision "$rev" --loaded-parameters 7940895744 --parameter-source "$params_source" || exit 1
sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$COMPARATOR" --left-name "$name" --right-name Lux1-16K || exit 1
sp compare --run-dir "$F/$name-16k" --comparator-run-dir "$SHARED" --left-name "$name" \
  --right-name Lux1-16K-shared || exit 1
fail=0
mlx "$F/$name-16k-mlx" || fail=1
mkdir -p "$G"
gates paired --left "$F/$name-16k" --right "$T1" --left-name "$name" --right-name DEV2.0-9B-T1 \
  --output "$G/PAIRED-vs-DEV2.0-9B-T1.json" || fail=1
gates paired --left "$F/$name-16k" --right "$NIMBLE" --left-name "$name" --right-name Nimble2 \
  --output "$G/PAIRED-vs-Nimble2.json" || fail=1
gates types --run "$F/$name-16k" --label "$name" --output "$G/types.json" || fail=1
gates6 public231 --left "$F/$name-16k" --right "$T1" --left-name "$name" --right-name DEV2.0-9B-T1 \
  --output "$G/PUBLIC231-vs-DEV2.0-9B-T1.json" || fail=1
lux9b lux9b.mlx_paired --left "$F/$name-16k-mlx" --right "$INC_MLX" --panel "$MLX_PANEL" --left-name "$name" \
  --right-name DEV2.0-9B-T1 --output "$G/MLX-PAIRED-vs-DEV2.0-9B-T1.json" || fail=1
lux9b lux9b.score_levels --run-dir "$F/$name-16k" --output "$G/score-levels.json" || fail=1
successor_summary "$G/successor.json" "$name" "$F/$name-16k" "$G" "$F/$name-16k/PAIRED-vs-Lux1-16K.json" \
  "$G/MLX-PAIRED-vs-DEV2.0-9B-T1.json" "$G/score-levels.json" || fail=1
[ "$fail" = 0 ] || { echo "formal $name: a CPU step after sealing failed (see above)" >&2; exit 1; }
echo "done $name"
