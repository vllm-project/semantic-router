# shellcheck shell=bash
# Shared settings for the run scripts. Source it, do not run it.
# The servers are started and stopped by process id, so other vLLM servers on the machine are left alone.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK=${WORK:-$PWD/agent-kv-tier-work}
MODEL=${MODEL:-$WORK/models/Qwen3-14B}
TRACES=${TRACES:-$WORK/traces}
TESTER_DIR=${TESTER_DIR:-$WORK/kv-cache-tester}
DURATION=${DURATION:-1200}
MAX_USERS=${MAX_USERS:-16}
# Qwen3-14B extended to 131,072 tokens with YaRN. vLLM 0.29 with Transformers 5 reads rope_parameters and
# ignores the older rope_scaling override.
YARN='{"max_position_embeddings":131072,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}'
LMCACHE_ENV=(PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 "LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_CPU_GB:-64}")
LMCACHE_CONNECTOR='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
PIDS=()

start_server() {  # $1 log file, the rest: the command
  local log=$1
  shift
  "$@" >"$log" 2>&1 &
  PIDS+=($!)
}

stop_servers() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  PIDS=()
  sleep 10
}
trap stop_servers EXIT

ready() {  # $1 port
  for _ in $(seq 1 180); do
    curl -sf -o /dev/null "http://127.0.0.1:$1/v1/models" && return 0
    sleep 10
  done
  return 1
}

run_tester() {  # $1 port, $2 output directory
  python3 "$TESTER_DIR/trace_replay_tester.py" --api-endpoint "http://127.0.0.1:$1" --trace-directory "$TRACES" \
    --tokenizer "$MODEL" --start-users 4 --max-users "$MAX_USERS" --max-ttft 60 --test-duration "$DURATION" \
    --max-context 100000 --warm-prefix-pct 0.5 --timing-strategy think-only --recycle --seed 0 --trace-seed 0 \
    --prompt-seed 0 --skip-graphs --no-color --output-dir "$2" >"$2/client.log" 2>&1
}
