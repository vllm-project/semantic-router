#!/usr/bin/env bash

set -euo pipefail

SR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${SR_ROOT}"

SOAK_LOG_DIR=${SOAK_LOG_DIR:-/tmp/soak-logs}
SOAK_VENV_DIR=${SOAK_VENV_DIR:-${SOAK_LOG_DIR}/venv}
SOAK_ROUTER_BASE_CONFIG=e2e/config/config.e2e.yaml
SOAK_CONFIG=${SOAK_CONFIG:-}
SOAK_OUT_DIR=${SOAK_OUT_DIR:-soak-results/$(date -u +%Y-%m-%dT%H-%M-%SZ)}
SOAK_STREAMING=${SOAK_STREAMING:-0}

SOAK_MOCK_PORT=${SOAK_MOCK_PORT:-8010}
SOAK_BACKEND_PORT=${SOAK_BACKEND_PORT:-8000}
SOAK_DELAY_MS=${SOAK_DELAY_MS:-1500}
SOAK_DELAY_JITTER_MS=${SOAK_DELAY_JITTER_MS:-500}
SOAK_STREAM_TTFT_MS=${SOAK_STREAM_TTFT_MS:-500}
SOAK_STREAM_TTFT_JITTER_MS=${SOAK_STREAM_TTFT_JITTER_MS:-0}
SOAK_STREAM_INTERVAL_MS=${SOAK_STREAM_INTERVAL_MS:-50}
SOAK_STREAM_FRAMES=${SOAK_STREAM_FRAMES:-64}

SOAK_METRICS_PORT=${SOAK_METRICS_PORT:-9190}
SOAK_PPROF_PORT=${SOAK_PPROF_PORT:-6060}
SOAK_ENVOY_PORT=${SOAK_ENVOY_PORT:-8801}
SOAK_ENVOY_ADMIN_PORT=${SOAK_ENVOY_ADMIN_PORT:-19000}
SOAK_ENVOY_DEFAULT_CONFIG=deploy/local/envoy.yaml
SOAK_ENVOY_CONFIG=${SOAK_ENVOY_CONFIG:-${SOAK_ENVOY_DEFAULT_CONFIG}}
SOAK_ENVOY_VERSION=${SOAK_ENVOY_VERSION:-1.35.4}
SOAK_FUNC_E_VERSION=${SOAK_FUNC_E_VERSION:-v1.3.0}

PYTHON_BIN=${PYTHON_BIN:-}

MOCK_PID=
PROXY_PID=
ROUTER_PID=
ENVOY_PID=

log() {
  echo "[soak $(date '+%H:%M:%S')] $*"
}

die() {
  echo "[soak] error: $*" >&2
  exit 1
}

# shellcheck disable=SC2317,SC2329
stop_pid() {
  local name=$1 pid=$2
  [[ -z "${pid}" ]] && return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  log "stopping ${name} (pid ${pid})"
  local child
  while read -r child; do
    if [[ -n "${child}" ]]; then
      kill "${child}" 2>/dev/null || true
    fi
  done < <(pgrep -P "${pid}" 2>/dev/null || true)
  kill "${pid}" 2>/dev/null || true
  local i
  for i in $(seq 1 20); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 0.5
  done
  log "${name} did not exit, sending SIGKILL"
  kill -9 "${pid}" 2>/dev/null || true
}

# shellcheck disable=SC2317,SC2329
cleanup() {
  local status=$?
  stop_pid envoy "${ENVOY_PID}"
  stop_pid router "${ROUTER_PID}"
  stop_pid fault-proxy "${PROXY_PID}"
  stop_pid mock-vllm "${MOCK_PID}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_url() {
  local name=$1 url=$2 tries=${3:-60}
  local i
  for i in $(seq 1 "${tries}"); do
    if curl --max-time 2 -gksf -o /dev/null "${url}"; then
      log "${name} ready after ${i}s (${url})"
      return 0
    fi
    sleep 1
  done
  die "timed out waiting for ${name} at ${url}; see ${SOAK_LOG_DIR}"
}

PYTHON_MAX_MINOR=13
python_is_supported() {
  "$1" -c "import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, ${PYTHON_MAX_MINOR}) else 1)" 2>/dev/null
}

select_python() {
  local path
  if [[ -n "${PYTHON_BIN}" ]]; then
    path=$(command -v "${PYTHON_BIN}" 2>/dev/null) ||
      die "PYTHON_BIN=${PYTHON_BIN} not found"
    python_is_supported "${path}" ||
      die "PYTHON_BIN=${PYTHON_BIN} is $(${path} -V 2>&1); tools/mock-vllm needs CPython 3.9-3.${PYTHON_MAX_MINOR}"
    echo "${path}"
    return
  fi
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    path=$(command -v "${candidate}" 2>/dev/null) || continue
    if python_is_supported "${path}"; then
      echo "${path}"
      return
    fi
  done
  die "no CPython 3.9-3.${PYTHON_MAX_MINOR} on PATH; tools/mock-vllm pins pydantic 2.9.2, which has no wheels for newer interpreters. Set PYTHON_BIN=/path/to/python3.13"
}

require_port_free() {
  local name=$1 port=$2
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    exec 3>&- 3<&-
    die "port ${port} (${name}) is already in use; stop the other process first"
  fi
}

case "${SOAK_STREAMING}" in
  0)
    SOAK_RESPONSE_MODE=buffered
    SOAK_PROXY_DELAY_MS=${SOAK_DELAY_MS}
    SOAK_PROXY_DELAY_JITTER_MS=${SOAK_DELAY_JITTER_MS}
    SOAK_PROXY_STREAM_INTERVAL_MS=0
    SOAK_PROXY_STREAM_FRAMES=0
    ;;
  1)
    SOAK_RESPONSE_MODE=streaming
    SOAK_PROXY_DELAY_MS=${SOAK_STREAM_TTFT_MS}
    SOAK_PROXY_DELAY_JITTER_MS=${SOAK_STREAM_TTFT_JITTER_MS}
    SOAK_PROXY_STREAM_INTERVAL_MS=${SOAK_STREAM_INTERVAL_MS}
    SOAK_PROXY_STREAM_FRAMES=${SOAK_STREAM_FRAMES}
    ;;
  *)
    die "SOAK_STREAMING must be 0 or 1 (got ${SOAK_STREAMING})"
    ;;
esac

mkdir -p "${SOAK_LOG_DIR}"

[[ -x bin/router ]] || die "bin/router missing; run 'make soak-local' (or 'make build-router') first"
[[ -x bin/soak ]] || die "bin/soak missing; run 'make soak-local' (or 'cd e2e && go build -o ../bin/soak ./cmd/soak') first"
if [[ -n "${SOAK_CONFIG}" ]]; then
  [[ -f "${SOAK_CONFIG}" ]] || die "router config ${SOAK_CONFIG} not found"
else
  [[ -f "${SOAK_ROUTER_BASE_CONFIG}" ]] || die "base router config ${SOAK_ROUTER_BASE_CONFIG} not found"
fi
command -v curl >/dev/null || die "curl is required"
PYTHON_BIN=$(select_python)
log "using python ${PYTHON_BIN} ($(${PYTHON_BIN} -V 2>&1))"

for spec in "mock-vllm:${SOAK_MOCK_PORT}" "fault-proxy:${SOAK_BACKEND_PORT}" \
  "router-metrics:${SOAK_METRICS_PORT}" "router-pprof:${SOAK_PPROF_PORT}" \
  "envoy:${SOAK_ENVOY_PORT}" "envoy-admin:${SOAK_ENVOY_ADMIN_PORT}"; do
  require_port_free "${spec%%:*}" "${spec##*:}"
done

if [[ -x "${SOAK_VENV_DIR}/bin/python" ]] && ! python_is_supported "${SOAK_VENV_DIR}/bin/python"; then
  log "recreating venv: ${SOAK_VENV_DIR} was built with an unsupported interpreter"
  rm -rf "${SOAK_VENV_DIR}"
fi
if [[ ! -x "${SOAK_VENV_DIR}/bin/python" ]]; then
  log "creating venv at ${SOAK_VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${SOAK_VENV_DIR}"
fi
log "installing tools/mock-vllm requirements"
"${SOAK_VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
"${SOAK_VENV_DIR}/bin/python" -m pip install --quiet --only-binary=:all: \
  pyyaml -r tools/mock-vllm/requirements.txt

log "starting mock-vllm on :${SOAK_MOCK_PORT}"
(
  cd tools/mock-vllm
  exec "${SOAK_VENV_DIR}/bin/python" -m uvicorn app:app \
    --host 127.0.0.1 --port "${SOAK_MOCK_PORT}" --log-level warning
) >"${SOAK_LOG_DIR}/mock-vllm.log" 2>&1 &
MOCK_PID=$!
wait_for_url mock-vllm "http://127.0.0.1:${SOAK_MOCK_PORT}/openapi.json"

log "starting fault proxy on :${SOAK_BACKEND_PORT} (response ${SOAK_RESPONSE_MODE}, first-byte delay ${SOAK_PROXY_DELAY_MS}ms + 0..${SOAK_PROXY_DELAY_JITTER_MS}ms jitter, stream interval ${SOAK_PROXY_STREAM_INTERVAL_MS}ms, content frames ${SOAK_PROXY_STREAM_FRAMES})"
"${SOAK_VENV_DIR}/bin/python" bench/openai_fault_proxy.py \
  --listen-host 127.0.0.1 \
  --listen-port "${SOAK_BACKEND_PORT}" \
  --upstream-base-url "http://127.0.0.1:${SOAK_MOCK_PORT}" \
  --delay-ms "${SOAK_PROXY_DELAY_MS}" \
  --delay-jitter-ms "${SOAK_PROXY_DELAY_JITTER_MS}" \
  --stream-interval-ms "${SOAK_PROXY_STREAM_INTERVAL_MS}" \
  --stream-frames "${SOAK_PROXY_STREAM_FRAMES}" \
  >"${SOAK_LOG_DIR}/fault-proxy.log" 2>&1 &
PROXY_PID=$!
wait_for_url fault-proxy "http://127.0.0.1:${SOAK_BACKEND_PORT}/health"

export LD_LIBRARY_PATH="${SR_ROOT}/candle-binding/target/release:${SR_ROOT}/ml-binding/target/release:${SR_ROOT}/nlp-binding/target/release${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DYLD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

if [[ -z "${SOAK_CONFIG}" ]]; then
  derived_router="${SOAK_LOG_DIR}/config.soak.yaml"
  log "deriving ${derived_router} from ${SOAK_ROUTER_BASE_CONFIG} (backends -> :${SOAK_BACKEND_PORT}, profiling on :${SOAK_PPROF_PORT}, response_cache off)"
  "${SOAK_VENV_DIR}/bin/python" - "${SOAK_ROUTER_BASE_CONFIG}" "${derived_router}" "${SOAK_BACKEND_PORT}" "${SOAK_PPROF_PORT}" <<'PY'
import sys

import yaml

src, dst, backend_port, pprof_port = sys.argv[1:5]
with open(src) as handle:
    doc = yaml.safe_load(handle)

rewritten = 0
for model in doc["providers"]["models"]:
    for ref in model.get("backend_refs", []):
        ref["endpoint"] = f"127.0.0.1:{backend_port}"
        rewritten += 1
if rewritten == 0:
    sys.exit(f"no backend_refs in {src}; nothing was rewritten")

doc["global"].setdefault("services", {})["observability"] = {
    "profiling": {"enabled": True, "port": int(pprof_port), "bind": "127.0.0.1"}
}
doc["global"]["stores"]["response_cache"]["enabled"] = False

with open(dst, "w") as handle:
    yaml.safe_dump(doc, handle, sort_keys=False)
PY
  SOAK_CONFIG="${derived_router}"
fi

log "downloading models declared by ${SOAK_CONFIG} (idempotent)"
./bin/router -config "${SOAK_CONFIG}" --download-only >"${SOAK_LOG_DIR}/download-models.log" 2>&1 ||
  die "model download failed; see ${SOAK_LOG_DIR}/download-models.log"

log "starting router with ${SOAK_CONFIG}"
./bin/router -config "${SOAK_CONFIG}" >"${SOAK_LOG_DIR}/router.log" 2>&1 &
ROUTER_PID=$!
wait_for_url router-metrics "http://127.0.0.1:${SOAK_METRICS_PORT}/metrics" 300
wait_for_url router-pprof "http://127.0.0.1:${SOAK_PPROF_PORT}/debug/pprof/" 30

if [[ "${SOAK_ENVOY_CONFIG}" == "${SOAK_ENVOY_DEFAULT_CONFIG}" ]]; then
  derived="${SOAK_LOG_DIR}/envoy.soak.yaml"
  log "deriving ${derived} from ${SOAK_ENVOY_CONFIG} (drop ext_authz if present, bind loopback)"
  "${SOAK_VENV_DIR}/bin/python" - "${SOAK_ENVOY_CONFIG}" "${derived}" <<'PY'
import sys

import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as handle:
    doc = yaml.safe_load(handle)

for listener in doc["static_resources"]["listeners"]:
    sock = listener.get("address", {}).get("socket_address")
    if sock and sock.get("address") not in (None, "127.0.0.1", "::1"):
        sock["address"] = "127.0.0.1"
    for chain in listener.get("filter_chains", []):
        for network_filter in chain.get("filters", []):
            typed = network_filter.get("typed_config", {})
            http_filters = typed.get("http_filters")
            if not http_filters:
                continue
            kept = [
                f for f in http_filters
                if f.get("name") != "envoy.filters.http.ext_authz"
            ]
            typed["http_filters"] = kept

with open(dst, "w") as handle:
    yaml.safe_dump(doc, handle, sort_keys=False)
PY
  SOAK_ENVOY_CONFIG="${derived}"
fi

FUNC_E_BIN=$(command -v func-e || true)
if [[ -z "${FUNC_E_BIN}" ]]; then
  if [[ ! -x bin/func-e ]]; then
    log "installing func-e ${SOAK_FUNC_E_VERSION} into bin/"
    curl -fsSL https://func-e.io/install.sh | bash -s -- -b bin "${SOAK_FUNC_E_VERSION}"
  fi
  FUNC_E_BIN="${SR_ROOT}/bin/func-e"
fi
log "using func-e at ${FUNC_E_BIN}"
"${FUNC_E_BIN}" use "${SOAK_ENVOY_VERSION}" >"${SOAK_LOG_DIR}/func-e.log" 2>&1

log "starting envoy on :${SOAK_ENVOY_PORT} with ${SOAK_ENVOY_CONFIG}"
"${FUNC_E_BIN}" run --config-path "${SOAK_ENVOY_CONFIG}" \
  >"${SOAK_LOG_DIR}/envoy.log" 2>&1 &
ENVOY_PID=$!
wait_for_url envoy "http://127.0.0.1:${SOAK_ENVOY_ADMIN_PORT}/ready" 60

mkdir -p "${SOAK_OUT_DIR}"
cp "${SOAK_CONFIG}" "${SOAK_OUT_DIR}/router-config.yaml"
cp "${SOAK_ENVOY_CONFIG}" "${SOAK_OUT_DIR}/envoy-config.yaml"
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  GIT_SHA="${GIT_SHA}-dirty"
fi
{
  echo "git_sha=${GIT_SHA}"
  echo "response_mode=${SOAK_RESPONSE_MODE}"
  echo "delay_ms=${SOAK_PROXY_DELAY_MS}"
  echo "delay_jitter_ms=${SOAK_PROXY_DELAY_JITTER_MS}"
  echo "stream_interval_ms=${SOAK_PROXY_STREAM_INTERVAL_MS}"
  echo "stream_content_frames=${SOAK_PROXY_STREAM_FRAMES}"
  echo "envoy_version=${SOAK_ENVOY_VERSION}"
  echo "router_config=${SOAK_CONFIG}"
  echo "envoy_config=${SOAK_ENVOY_CONFIG}"
  echo "platform=$(uname -sm)"
} >"${SOAK_OUT_DIR}/run-env.txt"

log "running soak harness (router pid ${ROUTER_PID}), results -> ${SOAK_OUT_DIR}"
SOAK_HARNESS_ARGS=()
if [[ "${SOAK_STREAMING}" == "1" ]]; then
  SOAK_HARNESS_ARGS+=("-stream")
fi
set +e
./bin/soak \
  -gateway-url "http://127.0.0.1:${SOAK_ENVOY_PORT}" \
  -metrics-url "http://127.0.0.1:${SOAK_METRICS_PORT}/metrics" \
  -pprof-url "http://127.0.0.1:${SOAK_PPROF_PORT}" \
  -router-pid "${ROUTER_PID}" \
  -out "${SOAK_OUT_DIR}" \
  "${SOAK_HARNESS_ARGS[@]}" \
  "$@"
SOAK_STATUS=$?
set -e

cat <<EOF

soak run finished with exit code ${SOAK_STATUS}
  artifacts: ${SR_ROOT}/${SOAK_OUT_DIR#"${SR_ROOT}/"}
  logs:      ${SOAK_LOG_DIR} (router.log, envoy.log, fault-proxy.log, mock-vllm.log)
EOF

exit "${SOAK_STATUS}"
