#!/usr/bin/env bash
# Kernel-path formal post-key same-panel run of one ~27B LoRA checkpoint (Milestone 3; node B host
# side): the eval track's frozen runner with the kernel adapter (image FLA / causal-conv1d verified
# in the collector), one fresh copy of a frozen Triton autotune cache per collection, gold-free
# seal, report, paired comparisons. run_formal_typed.sh stays the Milestone 2 reference path.
# Usage: run_formal_kernel.sh RUN GPU SRC CHECKPOINT CALIBRATION MAX_LENGTH LABEL STAGES FROZEN_CACHE CACHE_SHA [COMPARATOR=DIR]...
#   SRC           mirror directory name under /data/dev2/src (<sha>[-src_training_decision2])
#   CHECKPOINT    candidate checkpoint directory; CALIBRATION its adopted CAL report at MAX_LENGTH
#   STAGES        comma list of smoke (8 items per formal panel into RUN-smoke), collect, score
#   FROZEN_CACHE  frozen autotune cache (never written); CACHE_SHA its tree hash (v2/27b/triton_cache.py)
#   COMPARATOR    NAME=RUN_DIR of a sealed same-panel run (paired bootstrap, 5,000 draws)
# Each collection copies FROZEN_CACHE to <dir>/triton-cache (cp -a), refuses to start unless the
# copy's tree hash is CACHE_SHA, and writes <dir>/triton-cache.post.json afterwards (post-run hash,
# added/changed/removed files). The formal run's post-run cache is the one a release run reuses.
# LOADED_PARAMETERS / PARAMETER_SOURCE default to a rank-8 LoRA arm checkpoint; set them for a soup.
# Every comparator must be sealed before the score stage starts; a sealed run is not resealed, so
# the score stage can be repeated. DRY_RUN=1: see kernel_common.sh (CALIBRATION may not exist yet).
set -euo pipefail

RUN=$1 GPU=$2 SRC=$3 CKPT=$4 CAL=$5 MAXLEN=$6 LABEL=$7 STAGES=$8 FROZEN=$9 CACHE_SHA=${10}
shift 10
S=/data/dev2/src/$SRC/src/training/decision2
OUT=/data/dev2/runs/27b/$RUN
LOADED_PARAMETERS=${LOADED_PARAMETERS:-25688227840}
PARAMETER_SOURCE=${PARAMETER_SOURCE:-pinned base text backbone 25,624,600,064 + LoRA 58,363,904 + head 5,263,872 (safetensors headers)}
case "$GPU" in 5 | 6 | 7) ;; *) echo "GPU$GPU is outside the ~27B allocation" >&2; exit 2 ;; esac
[[ "$CACHE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "CACHE_SHA must be a full SHA-256" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S
source "$S/v2/27b/kernel_common.sh"
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
need "$CKPT" "$BASE" "$FROZEN"
[ "$DRY_RUN" = 1 ] || need "$CAL"
if has score; then
  for spec in "$@"; do
    [ -f "${spec#*=}/SEAL.json" ] && continue
    [ "$DRY_RUN" = 1 ] || { echo "comparator ${spec%%=*} is not sealed: ${spec#*=}" >&2; exit 2; }
    echo "dry-run: comparator ${spec%%=*} is not sealed yet: ${spec#*=}" >&2
  done
fi

REVISION="checkpoint-sha256:$(model_sha "$CAL")"

collect() {  # run-dir [--max-items N]
  local dir=$1 status=0
  shift
  mkdir -p "$dir"
  cache_copy "$FROZEN" "$CACHE_SHA" "$dir/triton-cache"
  runner "$dir" "$CKPT" "$dir/triton-cache" "27b $RUN kernel-path post-key same-panel collection" \
    --mount "$(dirname "$CAL")" -- --revision "$REVISION" --extra "source=$BASE" \
    --extra "calibration=$CAL" --extra "max_length=$MAXLEN" "$@" || status=$?
  cache_finish "$dir/triton-cache"
  return "$status"
}

if has smoke; then
  collect "$OUT-smoke" --max-items 8
fi
if has collect; then
  collect "$OUT"
fi
if has score; then
  score() {  # same_panel ARGS...: run, or print and argcheck with DRY_RUN=1
    dry python3 -m v2.eval.same_panel "$@"
    [ "$DRY_RUN" != 1 ] || argcheck v2.eval.same_panel "$@"
  }
  [ -f "$OUT/SEAL.json" ] || score seal --run-dir "$OUT"
  score report --run-dir "$OUT" --label "$LABEL" --tier 27B \
    --family decision2 --model-id llm-semantic-router/DEV2.0-27B --revision "$REVISION" \
    --loaded-parameters "$LOADED_PARAMETERS" --parameter-source "$PARAMETER_SOURCE"
  for spec in "$@"; do
    score compare --run-dir "$OUT" --comparator-run-dir "${spec#*=}" \
      --left-name "$LABEL" --right-name "${spec%%=*}"
  done
fi
echo "kernel formal driver $RUN stages $STAGES complete"
