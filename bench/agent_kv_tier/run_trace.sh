#!/usr/bin/env bash
# One GPU, two arms: vLLM's GPU prefix cache alone, and the same server with LMCache's CPU tier. The tester
# replays the SemiAnalysis sessions and adds users while time to first token stays under 60 s, so the arms
# compare the load each carries at the same latency limit. Outputs go to WORK/trace/<arm>.
set -eu
. "$(cd "$(dirname "$0")" && pwd)/common.sh"
COMMON=(--max-model-len 131072 --gpu-memory-utilization 0.85 --host 127.0.0.1 --port 8000 --hf-overrides "$YARN"
  --enable-prefix-caching)

arm() {  # $1 label, the rest: the server command
  local out=$WORK/trace/$1
  shift
  mkdir -p "$out"
  start_server "$out/server.log" "$@"
  ready 8000 || { echo "server failed: $out"; stop_servers; return 1; }
  run_tester 8000 "$out"
  stop_servers
  echo "arm done $out"
}

for a in ${ARMS:-gpu lmcache}; do
  case $a in
    gpu) arm gpu vllm serve "$MODEL" "${COMMON[@]}" ;;
    lmcache) arm lmcache env "${LMCACHE_ENV[@]}" vllm serve "$MODEL" "${COMMON[@]}" --kv-transfer-config "$LMCACHE_CONNECTOR" ;;
    *) echo "unknown arm $a" >&2; exit 2 ;;
  esac
done
