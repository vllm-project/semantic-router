#!/usr/bin/env bash
# Keep the vLLM server of serve_vllm.sh up until $STOP_FILE exists, restarting it whenever it exits.
# vLLM's API server exits cleanly (code 0) after an engine failure, so a plain Job would just complete
# and leave the generator waiting. Env: everything serve_vllm.sh reads, plus STOP_FILE.
set -uo pipefail
: "${STOP_FILE:?}"
here=$(cd "$(dirname "$0")" && pwd)
stop_server() {
  pkill -TERM -P "$1" 2>/dev/null
  kill -TERM "$1" 2>/dev/null
  sleep 45
  pkill -KILL -f "vllm serve" 2>/dev/null
  wait "$1" 2>/dev/null
}
n=0
while [[ ! -f $STOP_FILE ]]; do
  n=$((n + 1))
  echo "$(date -u +%FT%TZ) server start #$n"
  bash "$here/serve_vllm.sh" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -f $STOP_FILE ]]; then
      echo "$(date -u +%FT%TZ) $STOP_FILE found: stopping the server"
      stop_server "$pid"
      exit 0
    fi
    sleep 30
  done
  wait "$pid"
  echo "$(date -u +%FT%TZ) server exited with code $?; restarting in 30 s"
  pkill -KILL -f "vllm serve" 2>/dev/null
  sleep 30
done
echo "$(date -u +%FT%TZ) $STOP_FILE present: nothing to serve"
