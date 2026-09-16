#!/usr/bin/env bash
# Start the linux/riscv64 router under qemu-user and prove /health plus one classify.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"
export VLLM_SR_CONFIG_BASE_DIR="${REPO_ROOT}"

ROUTER_BIN=${RISCV_ROUTER_BIN:?}
CONFIG=${RISCV_ROUTER_CONFIG:?}
QEMU=${RISCV_QEMU:?}
SYSROOT=${RISCV_SYSROOT:?}
LIBDIR=${RISCV_CANDLE_LIBDIR:?}
API_PORT=${RISCV_ROUTER_API_PORT:-18080}
EXTPROC_PORT=${RISCV_ROUTER_EXTPROC_PORT:-15051}
METRICS_PORT=${RISCV_ROUTER_METRICS_PORT:-19190}
HEALTH_TIMEOUT=${RISCV_ROUTER_HEALTH_TIMEOUT:-90}
READY_TIMEOUT=${RISCV_ROUTER_READY_TIMEOUT:-1200}
CLASSIFY_TIMEOUT=${RISCV_ROUTER_CLASSIFY_TIMEOUT:-600}
API_URL="http://127.0.0.1:${API_PORT}"

log=$(mktemp)
pid=""
cleanup() {
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  echo "----- router log -----"
  cat "${log}"
  rm -f "${log}"
}
trap cleanup EXIT

export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Starting RISC-V router under qemu-user on api-port ${API_PORT}"
"${QEMU}" -L "${SYSROOT}" "${ROUTER_BIN}" \
  -config="${CONFIG}" \
  -port="${EXTPROC_PORT}" \
  -api-port="${API_PORT}" \
  -metrics-port="${METRICS_PORT}" \
  -enable-api=true \
  >"${log}" 2>&1 &
pid=$!

wait_http() {
  local path="$1"
  local seconds="$2"
  local elapsed=0
  while (( elapsed < seconds )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "router process exited before ${path}" >&2
      return 1
    fi
    if curl -sf --max-time 5 "${API_URL}${path}" >/dev/null; then
      echo "${path} ready after ${elapsed}s"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "timed out waiting for ${path} after ${seconds}s" >&2
  return 1
}

wait_http /health "${HEALTH_TIMEOUT}"
wait_http /ready "${READY_TIMEOUT}"

echo "POST /api/v1/diagnostics/classify/intent"
resp=$(curl -sf --max-time "${CLASSIFY_TIMEOUT}" \
  -H "Content-Type: application/json" \
  -d '{"text":"What is photosynthesis?"}' \
  "${API_URL}/api/v1/diagnostics/classify/intent")
echo "${resp}"

python3 - "${resp}" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
category = (data.get("classification") or {}).get("category") or ""
decision = data.get("routing_decision") or ""
if not category or decision == "placeholder_response":
    raise SystemExit(
        f"classify did not use the domain classifier: category={category!r} routing_decision={decision!r}"
    )
print(f"classified category={category}")
PY
