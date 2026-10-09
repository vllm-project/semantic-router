#!/usr/bin/env bash
# OpenAI-compatible vLLM server for ws-synth (generator / verifier / judge / teacher).
# Configured by environment so one script serves every model:
#   MODEL_DIR, SERVED_NAME, TP (default 8), MAX_LEN (default 32768), REASONING_PARSER (optional),
#   EXTRA_ARGS (optional), PORT (default 8000), VLLM_ROCM_USE_AITER (default 1).
set -euo pipefail
: "${MODEL_DIR:?}" "${SERVED_NAME:?}"
TP=${TP:-8}
MAX_LEN=${MAX_LEN:-32768}
PORT=${PORT:-8000}
LOG_DIR=/data/d25/vega/synth/logs
CACHE=/data/d25/vega/synth/cache
mkdir -p "$LOG_DIR" "$CACHE/triton" "$CACHE/vllm" "$CACHE/aiter"
export VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1}
export TRITON_CACHE_DIR=$CACHE/triton
export VLLM_CACHE_ROOT=$CACHE/vllm
export AITER_JIT_DIR=${AITER_JIT_DIR:-$CACHE/aiter}
export HF_HUB_OFFLINE=1
args=(
  "$MODEL_DIR" --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN"
  --enable-prefix-caching --max-num-seqs "${MAX_SEQS:-512}" --max-logprobs 300
  --gpu-memory-utilization "${GPU_UTIL:-0.90}"
)
[[ -n "${REASONING_PARSER:-}" ]] && args+=(--reasoning-parser "$REASONING_PARSER")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)
log="$LOG_DIR/server-${SERVED_NAME}-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "vllm serve ${args[*]}" | tee "$log"
rocm-smi --showmeminfo vram 2>/dev/null | tail -12 | tee -a "$log" || true
vllm serve "${args[@]}" 2>&1 | tee -a "$log"
