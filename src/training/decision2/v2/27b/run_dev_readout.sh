#!/usr/bin/env bash
# Kernel-path development readout of one ~27B checkpoint (Milestone 3; node B host side; never a
# release score). Every GPU stage runs with the image FLA / causal-conv1d kernels verified and its
# own fresh copy of the frozen Triton autotune cache (<stage dir>/triton-cache, hash checked before,
# triton-cache.post.json after).
# Usage: run_dev_readout.sh RUN GPU SRC CHECKPOINT LIMIT FROZEN_CACHE CACHE_SHA LABEL
#   RUN        output directory under /data/dev2/runs/27b (for example M3-A-s1/readout-kernel-32768)
#   CHECKPOINT an arm's BEST checkpoint or a soup (qwen-adapter on the pinned base BASE)
# Stages (READOUT_STAGES, default cal,collect,aho,score):
#   cal      CAL698 per-type temperatures at LIMIT (v2.27b.kernel_readout fit, launch.py):
#            cal/calibration.json, cal/cal.probs.jsonl (calibrated), cal/cal.raw.probs.jsonl, plus
#            cal/select.probs.jsonl on the kernel path when TRAIN_RUN is unset (a soup)
#   collect  typed DEV + CSS pilot through the eval runner with the kernel adapter at LIMIT and
#            cal/calibration.json (output/typed-dev|css-pilot.predictions.jsonl)
#   aho      AHO slices (AHO=NAME=HOST_ROWS,...) at TRAIN_RUN's BEST with v2.27b.aho_eval at LIMIT
#   score    v2.eval.dev_readout (typed DEV, CSS pilot, SELECT700) -> READOUT.json, and CAL698
#            metrics of cal/cal.probs.jsonl -> cal698.summary.json
# TRAIN_RUN: the completed trainer run whose BEST is CHECKPOINT; its SELECT700 probabilities are the
# trainer's own (select-step-<BEST>-predictions.jsonl) and AHO reads its BEST. BASE, CAL_FILE and
# CAL_SHA256 override the pinned base, CAL698 and its hash. DRY_RUN=1: see kernel_common.sh (the
# trainer SELECT extraction still runs; scoring is printed and argchecked).
set -euo pipefail

RUN=$1 GPU=$2 SRC=$3 CKPT=$4 LIMIT=$5 FROZEN=$6 CACHE_SHA=$7 LABEL=$8
S=/data/dev2/src/$SRC/src/training/decision2
OUT=/data/dev2/runs/27b/$RUN
STAGES=${READOUT_STAGES:-cal,collect,aho,score}
TRAIN_RUN=${TRAIN_RUN:-}
AHO=${AHO:-}
SELECT_ROWS=/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl
NAME=d2-27b-$(printf '%s' "$RUN" | tr -c 'A-Za-z0-9_.-' -)
case "$GPU" in 5 | 6 | 7) ;; *) echo "GPU$GPU: launch.py maps only node B GPU5-7" >&2; exit 2 ;; esac
[[ "$CACHE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "CACHE_SHA must be a full SHA-256" >&2; exit 2; }
[ -d "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S
source "$S/v2/27b/kernel_common.sh"
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
need "$BASE" "$CAL_FILE" "$SELECT_ROWS" "$FROZEN"
if has aho && [ -n "$AHO" ] && [ -z "$TRAIN_RUN" ]; then
  echo "the aho stage reads TRAIN_RUN's BEST; unset AHO for a soup" >&2
  exit 2
fi
mkdir -p "$OUT/receipts"

if [ -n "$TRAIN_RUN" ]; then
  python3 - "$TRAIN_RUN" "$CKPT" <<'EOF'
import json, os, sys
run, ckpt = sys.argv[1:]
best = json.load(open(os.path.join(run, "BEST.json")))["checkpoint"]
if os.path.realpath(os.path.join(run, best)) != os.path.realpath(ckpt):
    raise SystemExit(f"CHECKPOINT is not {run}/{best}")
EOF
fi

if has cal; then
  mkdir -p "$OUT/cal"
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/cal/triton-cache"
  select=()
  [ -n "$TRAIN_RUN" ] || select=(--select /data/select.jsonl)
  status=0
  launcher "$NAME-cal" 1.0 "M3 kernel CAL698 fit at $LIMIT" "$OUT/receipts/cal.json" \
    "$OUT/cal/triton-cache" -- --mount "$CKPT:$CKPT" --mount "$CAL_FILE:/data/cal.jsonl" \
    --mount "$SELECT_ROWS:/data/select.jsonl" --mount "$OUT/cal:$OUT/cal:rw" -- \
    python3 -m v2.27b.kernel_readout fit --checkpoint "$CKPT" --source-path "$BASE" \
    --cal /data/cal.jsonl --cal-sha256 "$CAL_SHA256" --max-length "$LIMIT" \
    "${select[@]}" --out-dir "$OUT/cal" || status=$?
  cache_finish "$OUT/cal/triton-cache"
  [ "$status" = 0 ] || exit "$status"
fi
if has collect; then
  REVISION="checkpoint-sha256:$(model_sha "$OUT/cal/calibration.json")"
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/triton-cache"
  status=0
  runner "$OUT" "$CKPT" "$OUT/triton-cache" "27b $RUN kernel-path development readout" -- \
    --revision "$REVISION" --extra "source=$BASE" --extra "calibration=$OUT/cal/calibration.json" \
    --extra "max_length=$LIMIT" --panels typed-dev,css-pilot || status=$?
  cache_finish "$OUT/triton-cache"
  [ "$status" = 0 ] || exit "$status"
fi
if has aho && [ -n "$AHO" ]; then
  mkdir -p "$OUT/aho"
  mounts=() slices=()
  IFS=, read -r -a specs <<< "$AHO"
  for spec in "${specs[@]}"; do
    [ -n "$spec" ] || continue
    mounts+=(--mount "${spec#*=}:/data/aho-${spec%%=*}.jsonl")
    slices+=(--slice "${spec%%=*}=/data/aho-${spec%%=*}.jsonl")
  done
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/aho/triton-cache"
  status=0
  launcher "$NAME-aho" 1.0 "M3 kernel AHO readout at $LIMIT" "$OUT/receipts/aho.json" \
    "$OUT/aho/triton-cache" -- --mount "$TRAIN_RUN:$TRAIN_RUN" "${mounts[@]}" \
    --mount "$OUT/aho:$OUT/aho:rw" -- \
    python3 -m v2.27b.kernel_readout exec --runtime "$OUT/aho/runtime.json" -- v2.27b.aho_eval \
    --run-dir "$TRAIN_RUN" --source-path "$BASE" "${slices[@]}" --max-length "$LIMIT" \
    --out-dir "$OUT/aho" || status=$?
  cache_finish "$OUT/aho/triton-cache"
  [ "$status" = 0 ] || exit "$status"
fi
if has score; then
  if [ -n "$TRAIN_RUN" ]; then
    SELECT_PROBS=$OUT/select.probs.jsonl
    [ -f "$SELECT_PROBS" ] || python3 -m v2.27b.kernel_readout select-from-trainer \
      --run-dir "$TRAIN_RUN" --rows "$SELECT_ROWS" --output "$SELECT_PROBS"
  else
    SELECT_PROBS=$OUT/cal/select.probs.jsonl
  fi
  score=(python3 -m v2.eval.dev_readout --run-dir "$OUT" --select "$SELECT_PROBS" --label "$LABEL"
    --output "$OUT/READOUT.json")
  summary=(python3 -m v2.27b.kernel_readout cal-summary --rows "$CAL_FILE" --cal-sha256 "$CAL_SHA256"
    --probabilities "$OUT/cal/cal.probs.jsonl" --output "$OUT/cal698.summary.json")
  dry "${score[@]}"
  dry "${summary[@]}"
  if [ "$DRY_RUN" = 1 ]; then
    argcheck "${score[@]:2}"
    argcheck "${summary[@]:2}"
  fi
fi
echo "kernel dev readout $RUN stages $STAGES complete"
