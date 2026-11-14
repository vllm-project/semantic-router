#!/usr/bin/env bash
# Two replicas on one GPU (0.42 of its memory each, GPU prefix cache only) behind placement_proxy.py, one arm per
# placement policy. Replicas restart for every arm so no arm inherits another's cache. They share one GPU's
# compute, so this measures placement and a split KV pool, not added compute. MAX_USERS=4 keeps the sessions
# within the two pools. Outputs go to WORK/placement/<policy>[_u<MAX_USERS>].
set -eu
. "$(cd "$(dirname "$0")" && pwd)/common.sh"
COMMON=(--max-model-len 131072 --gpu-memory-utilization 0.42 --host 127.0.0.1 --hf-overrides "$YARN"
  --enable-prefix-caching)
SUFFIX=""
[ "$MAX_USERS" = 16 ] || SUFFIX=_u$MAX_USERS

for policy in ${ARMS:-round_robin least_request session sticky}; do
  out=$WORK/placement/$policy$SUFFIX
  mkdir -p "$out"
  for r in 1 2; do
    start_server "$out/server$r.log" vllm serve "$MODEL" "${COMMON[@]}" --port "800$r"
    ready "800$r" || { echo "replica $r failed: $out"; stop_servers; continue 2; }
  done
  start_server "$out/proxy.log" python3 "$HERE/placement_proxy.py" "$policy" 8000 http://127.0.0.1:8001,http://127.0.0.1:8002
  ready 8000 || { echo "proxy failed: $out"; stop_servers; continue; }
  run_tester 8000 "$out"
  stop_servers
  echo "arm done $out"
done
