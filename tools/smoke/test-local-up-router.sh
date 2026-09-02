#!/usr/bin/env bash

set -euo pipefail

SR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${SR_ROOT}"

export LD_LIBRARY_PATH="${SR_ROOT}/candle-binding/target/release:${SR_ROOT}/ml-binding/target/release:${SR_ROOT}/nlp-binding/target/release${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

ROUTER_CONFIG=${ROUTER_CONFIG:-"e2e/config/config.e2e.yaml"}
MOCK_PYTHON=${MOCK_PYTHON:-"python3"}
MOCK_PORT=8000
ENVOY_PORT=8801
WAIT_STARTUP_SECS=${WAIT_STARTUP_SECS:-900}
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/local-up-smoke.XXXXXX")

MOCK_PID=""
HARNESS_PID=""

log() { echo "[local-up-smoke] $*"; }

fail() {
  log "FAIL: $*"
  log "--- harness log (tail) ---"; tail -30 "${WORK_DIR}/harness.log" 2>/dev/null || true
  log "--- router log (tail) ---"; tail -30 "${WORK_DIR}/router.log" 2>/dev/null || true
  log "--- envoy log (tail) ---"; tail -30 "${WORK_DIR}/envoy.log" 2>/dev/null || true
  exit 1
}

cleanup() {
  if [[ -n "${HARNESS_PID}" ]] && kill -0 "${HARNESS_PID}" 2>/dev/null; then
    kill -INT "${HARNESS_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${HARNESS_PID}" 2>/dev/null || break
      sleep 1
    done
    kill "${HARNESS_PID}" 2>/dev/null || true
  fi
  if [[ -n "${MOCK_PID}" ]]; then
    kill "${MOCK_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

log "Setting up mock vLLM backend venv..."
"${MOCK_PYTHON}" -m venv "${WORK_DIR}/mock-venv"
"${WORK_DIR}/mock-venv/bin/pip" install --quiet -r tools/mock-vllm/requirements.txt

log "Starting mock vLLM backend on :${MOCK_PORT}..."
(cd tools/mock-vllm && exec "${WORK_DIR}/mock-venv/bin/uvicorn" app:app \
  --host 127.0.0.1 --port "${MOCK_PORT}") > "${WORK_DIR}/mock.log" 2>&1 &
MOCK_PID=$!
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:${MOCK_PORT}/health" >/dev/null 2>&1 && break
  [[ "$i" == 30 ]] && fail "mock backend did not become healthy; see ${WORK_DIR}/mock.log"
  sleep 1
done
log "Mock backend is healthy."

BIN_ARGS=()
if [[ -x bin/router ]]; then
  BIN_ARGS=(-o bin)
else
  log "bin/router not found; the harness will build it."
fi

log "Starting local-up-router.sh (config: ${ROUTER_CONFIG})..."
LOG_DIR="${WORK_DIR}" ROUTER_CONFIG="${ROUTER_CONFIG}" \
  tools/dev/local-up-router.sh "${BIN_ARGS[@]+"${BIN_ARGS[@]}"}" \
  > "${WORK_DIR}/harness.log" 2>&1 &
HARNESS_PID=$!

deadline=$((SECONDS + WAIT_STARTUP_SECS))
until grep -q "The local semantic router is running" "${WORK_DIR}/harness.log" 2>/dev/null; do
  kill -0 "${HARNESS_PID}" 2>/dev/null || fail "harness exited before becoming ready"
  [[ ${SECONDS} -ge ${deadline} ]] && fail "harness not ready after ${WAIT_STARTUP_SECS}s"
  sleep 5
done
log "Harness is up."

log "Sending /v1/chat/completions through Envoy (:${ENVOY_PORT})..."
status=""
routed=""
for _ in $(seq 1 24); do
  status=$(curl -s -o "${WORK_DIR}/response.json" -w '%{http_code}' --max-time 60 \
    -X POST "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Model-A","messages":[{"role":"user","content":"smoke test"}]}') || status=""
  sleep 2
  if [[ "${status}" == "200" ]] &&
     grep -q '"selected_model":"Model-A"' "${WORK_DIR}/envoy.log" 2>/dev/null &&
     grep -q '"upstream_cluster":"vllm_backend_cluster"' "${WORK_DIR}/envoy.log" 2>/dev/null; then
    routed=yes
    break
  fi
  sleep 3
done
[[ "${status}" == "200" ]] || fail "expected HTTP 200 through Envoy, got '${status:-none}' (a 403 here means the request was rejected before reaching the router)"
[[ "${routed}" == "yes" ]] || fail "Envoy access log never showed selected_model=Model-A via vllm_backend_cluster; requests returned 200 but the router (ext_proc) did not process them"
log "Got HTTP 200; response: $(head -c 200 "${WORK_DIR}/response.json")"

log "PASS: request reached the mock backend through Envoy and was processed by the router."
