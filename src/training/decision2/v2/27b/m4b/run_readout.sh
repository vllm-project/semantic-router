#!/usr/bin/env bash
# M4b kernel-path development readout of one full checkpoint (node B host side, one GPU of GPU0-2; never
# a release score). v2/27b/run_dev_readout.sh's stages with launch3.py and track 27b-m4b (m4b/common3.sh):
# every GPU stage on its own fresh verified copy of the frozen readout cache 583241fb (FROZEN, CACHE_SHA).
# Usage: run_readout.sh NAME CKPT GPU MIRROR_SHA
#   NAME        output directory /data/dev2/runs/27b/m4b/readouts/NAME (A1-s1, A2-soup, F1M, T-13, ...)
#   CKPT        a full checkpoint (arm-seed BEST, full soup, F1M or theta(alpha))
#   MIRROR_SHA  code commit; the mirror /data/dev2/src/<sha>[-src_training_decision2]
# Stages (STAGES, default verify,cal,collect,aho,score,summary):
#   verify   mirror record, inputs, the frozen cache's tree hash and the GPU lease (no GPU, no writes)
#   cal      CAL698 per-type temperatures at 32,768 (kernel_readout fit via launch3.py) -> cal/; SELECT700 on
#            the kernel path (cal/select.probs.jsonl) when TRAIN_RUN is unset (soup, F1M, theta)
#   collect  typed DEV + CSS pilot through the eval runner, 27B kernel adapter, cal/calibration.json -> output/
#   aho      AHO slices (AHO=NAME=HOST_ROWS,...) at TRAIN_RUN's BEST (v2.27b.aho_eval via launch3.py) -> aho/
#   score    v2.eval.dev_readout (typed DEV, CSS pilot, SELECT700) -> READOUT.json; CAL698 -> cal698.summary.json
#   summary  m4b_rules summary -> READOUT-M4B.json (P_dev, T_dev, H_pilot, H3, per type, per family)
# TRAIN_RUN: the trainer run whose BEST.json names CKPT (arm-seeds): SELECT700 from its own predictions.
# AHO defaults to the M3 slices A6g, A6h, A7 when TRAIN_RUN is set. DRY_RUN=1: see kernel_common.sh.
set -euo pipefail
echo "m4b readout $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

NAME=${1:?NAME} CKPT=${2:?CKPT} GPU=${3:?GPU} MIRROR_SHA=${4:?MIRROR_SHA}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
SRC=$MIRROR_SHA
[ -d "/data/dev2/src/$SRC" ] || SRC=$MIRROR_SHA-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
LIMIT=32768
FROZEN=${FROZEN:-/data/dev2/runs/27b/m3-warm-32768/triton-cache}
CACHE_SHA=${CACHE_SHA:-583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf}
STAGES=${STAGES:-verify,cal,collect,aho,score,summary}
TRAIN_RUN=${TRAIN_RUN:-}
DATA=/data/dev2/private/27b/m3-data/mixtures-m3-1
AHO=${AHO-${TRAIN_RUN:+A6g=$DATA/aho-A6g.jsonl,A6h=$DATA/aho-A6h.jsonl,A7=$DATA/aho-A7.jsonl}}
SELECT_ROWS=/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl
LABEL=${LABEL:-M4b $NAME}
[[ "$CACHE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "CACHE_SHA must be a full SHA-256" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
source "$S/v2/27b/kernel_common.sh"
source "$S/v2/27b/m4b/common3.sh"
OUT=$M4B/readouts/$NAME
JOB=d2-m4b-$NAME
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
if has aho && [ -n "$AHO" ] && [ -z "$TRAIN_RUN" ]; then
  echo "the aho stage reads TRAIN_RUN's BEST; unset AHO without a trainer run" >&2
  exit 2
fi

check_best() {
  python3 - "$TRAIN_RUN" "$CKPT" <<'EOF'
import json, os, sys
run, ckpt = sys.argv[1:]
best = json.load(open(os.path.join(run, "BEST.json")))["checkpoint"]
if os.path.realpath(os.path.join(run, best)) != os.path.realpath(ckpt):
    raise SystemExit(f"CHECKPOINT is not {run}/{best}")
print(f"TRAIN_RUN BEST {best}")
EOF
}
check_full() {
  python3 - "$CKPT" <<'EOF'
import json, pathlib, sys
ckpt = pathlib.Path(sys.argv[1])
config = json.loads((ckpt / "decision_config.json").read_text())
if config.get("checkpoint_format", "full") != "full" or not (ckpt / "backbone").is_dir():
    raise SystemExit(f"{ckpt} is not a full DecisionModel checkpoint")
EOF
}

if has verify; then
  verify_mirror "$MIRROR_SHA"
  need "$CKPT" "$BASE" "$CAL_FILE" "$SELECT_ROWS" "$FROZEN" "$PANEL_ROOT"
  check_full
  [ -z "$TRAIN_RUN" ] || check_best
  verify_cache "$FROZEN" "$CACHE_SHA"
  [ "$DRY_RUN" = 1 ] || verify_lease
fi
[ -z "$TRAIN_RUN" ] || check_best >/dev/null
mkdir -p "$OUT/receipts"

if has cal; then
  mkdir -p "$OUT/cal"
  cache_copy "$FROZEN" "$CACHE_SHA" "$OUT/cal/triton-cache"
  select=()
  [ -n "$TRAIN_RUN" ] || select=(--select /data/select.jsonl)
  status=0
  launcher "$JOB-cal" 1.0 "M4b $NAME kernel CAL698 fit at $LIMIT" "$OUT/receipts/cal.json" \
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
  runner "$OUT" "$CKPT" "$OUT/triton-cache" "27b-m4b $NAME kernel-path development readout" -- \
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
  launcher "$JOB-aho" 1.0 "M4b $NAME kernel AHO readout at $LIMIT" "$OUT/receipts/aho.json" \
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
if has summary; then
  dry python3 -m v2.27b.m4b.m4b_rules summary --readout-dir "$OUT" --checkpoint "$CKPT" \
    --output "$OUT/READOUT-M4B.json"
fi
echo "m4b readout $NAME stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
