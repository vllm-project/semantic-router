#!/usr/bin/env bash
# Raw cache speed on one GPU: prompts of 4k to 32k tokens sent twice against three servers in turn, without a
# prefix cache, with vLLM's GPU prefix cache, and with LMCache's CPU tier and the GPU prefix cache off.
# Rows are appended to WORK/prefix_ttft.jsonl.
set -eu
. "$(cd "$(dirname "$0")" && pwd)/common.sh"
COMMON=(--max-model-len 40960 --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8000)
mkdir -p "$WORK/prefix_ttft"

arm() {  # $1 label, the rest: the server command
  local label=$1
  shift
  start_server "$WORK/prefix_ttft/server_$label.log" "$@"
  ready 8000 || { echo "server failed: $label"; stop_servers; return 1; }
  python3 "$HERE/prefix_ttft.py" --model "$MODEL" --label "$label" --out "$WORK/prefix_ttft.jsonl"
  stop_servers
}

arm none vllm serve "$MODEL" "${COMMON[@]}" --no-enable-prefix-caching
arm gpu vllm serve "$MODEL" "${COMMON[@]}" --enable-prefix-caching
arm lmcache env "${LMCACHE_ENV[@]}" vllm serve "$MODEL" "${COMMON[@]}" --no-enable-prefix-caching \
  --kv-transfer-config "$LMCACHE_CONNECTOR"
