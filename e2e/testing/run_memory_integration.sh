#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/vllm-project/semantic-router}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
VLLM_SR_IMAGE="${VLLM_SR_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr:latest}"
# Match the CLI's normalized stack identity and validated host ports. A manual
# invocation gets its own name; CI supplies a deterministic job-scoped name.
export VLLM_SR_STACK_NAME="${VLLM_SR_STACK_NAME:-vsr-memory-${BASHPID}}"
export VLLM_SR_PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-0}"
export CONTAINER_RUNTIME
layout_variables="$(python3 - <<'PY_LAYOUT'
import shlex
from cli.runtime_stack import resolve_runtime_stack
layout = resolve_runtime_stack()
values = {
    "VLLM_SR_STACK_NAME": layout.stack_name,
    "VLLM_SR_NETWORK": layout.network_name,
    "VLLM_SR_DATA_NETWORK": layout.data_network_name,
    "ROUTER_CONTAINER": layout.router_container_name,
    "ENVOY_CONTAINER": layout.envoy_container_name,
    "DASHBOARD_CONTAINER": layout.dashboard_container_name,
    "MILVUS_CONTAINER_NAME": layout.milvus_container_name,
    "MILVUS_HOST_PORT": layout.milvus_port,
    "MILVUS_HEALTH_PORT": layout.host_port(9091, name="milvus_health_port"),
    "PROVIDER_MOCKER_HOST_PORT": layout.host_port(8000, name="provider_mocker_port"),
    "ROUTER_API_HEALTH_URL": f"http://localhost:{layout.api_port}/ready",
    "ROUTER_ENDPOINT": f"http://localhost:{layout.host_port(8888, name='memory_listener_port')}",
    "STACK_CONTAINERS": " ".join((*layout.runtime_container_names,
        *layout.storage_container_names, layout.jaeger_container_name,
        layout.prometheus_container_name, layout.grafana_container_name)),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY_LAYOUT
)"
eval "${layout_variables}"
export VLLM_SR_STACK_NAME VLLM_SR_PORT_OFFSET
PROVIDER_MOCKER_CONTAINER="${VLLM_SR_STACK_NAME}-provider-mocker"
TEST_DIR="${MEMORY_TEST_DIR:-$(mktemp -d -t vsr-memory-test-XXXXXX)}"
mkdir -p "${TEST_DIR}"
TEST_DIR="$(cd "${TEST_DIR}" && pwd)"
if [[ -n "$(ls -A "${TEST_DIR}")" ]]; then
    echo "Memory test directory must be empty: ${TEST_DIR}" >&2
    exit 1
fi
PID_FILE="${TEST_DIR}/serve.pid"
SERVE_LOG="${TEST_DIR}/serve.log"
CONFIG_FILE="${TEST_DIR}/config.yaml"
KEEP_TEST_DIR="${KEEP_MEMORY_TEST_DIR:-0}"
ARTIFACT_DIR="${MEMORY_TEST_ARTIFACT_DIR:-${REPO_ROOT}/logs}/memory-${VLLM_SR_STACK_NAME}"
MODEL_DIR="${MEMORY_TEST_MODEL_DIR:-${TEST_DIR}/models}"
if [[ "${MODEL_DIR}" != /* ]]; then
    MODEL_DIR="${REPO_ROOT}/${MODEL_DIR}"
fi
MODEL_MOUNT_DIR="${TEST_DIR}/models"
USE_DETERMINISTIC_MEMORY_EMBEDDINGS="${USE_DETERMINISTIC_MEMORY_EMBEDDINGS:-0}"
export MILVUS_CONTAINER_NAME MILVUS_HOST_PORT MILVUS_HEALTH_PORT
export MILVUS_DATA_DIR="${TEST_DIR}/milvus-data"
export MILVUS_BIND_ADDRESS=127.0.0.1

VLLM_SR_PID=""
STACK_STARTED=0
MILVUS_STARTED=0
PROVIDER_STARTED=0
NETWORK_CREATED=0

# A named collision belongs to another invocation. Never adopt it, stop it, or
# collect its logs; CLI stop removes every resource with this stack identity.
for container in ${STACK_CONTAINERS} "${PROVIDER_MOCKER_CONTAINER}"; do
    if "${CONTAINER_RUNTIME}" inspect "${container}" >/dev/null 2>&1; then
        echo "Refusing to reuse existing memory-test container: ${container}" >&2
        exit 1
    fi
done
for network in "${VLLM_SR_NETWORK}" "${VLLM_SR_DATA_NETWORK}"; do
    if "${CONTAINER_RUNTIME}" network inspect "${network}" >/dev/null 2>&1; then
        echo "Refusing to reuse existing memory-test network: ${network}" >&2
        exit 1
    fi
done

reclaim_test_dir_permissions() {
    local host_uid host_gid

    host_uid="$(id -u)"
    host_gid="$(id -g)"

    "${CONTAINER_RUNTIME}" run --rm --user root \
        -v "${TEST_DIR}:/artifacts" \
        --entrypoint /bin/sh \
        "${VLLM_SR_IMAGE}" \
        -c "chown -R ${host_uid}:${host_gid} /artifacts || chmod -R a+rwX /artifacts" \
        >/dev/null 2>&1
}

remove_test_dir() {
    if [[ ! -d "${TEST_DIR}" ]]; then
        return 0
    fi

    if rm -rf "${TEST_DIR}" 2>/dev/null; then
        return 0
    fi

    reclaim_test_dir_permissions || return 1
    rm -rf "${TEST_DIR}"
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    # Capture only this invocation's resources before either serve or stop can
    # remove them. Preserve artifacts independently of disposable runtime state.
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    if [[ "${STACK_STARTED}" == "1" ]]; then
        for container in ${STACK_CONTAINERS}; do
            "${CONTAINER_RUNTIME}" logs "${container}" >"${ARTIFACT_DIR}/${container}.predump.log" 2>&1 || true
        done
    fi
    if [[ "${PROVIDER_STARTED}" == "1" ]]; then
        "${CONTAINER_RUNTIME}" logs "${PROVIDER_MOCKER_CONTAINER}" >"${ARTIFACT_DIR}/${PROVIDER_MOCKER_CONTAINER}.predump.log" 2>&1 || true
    fi
    if [[ "${MILVUS_STARTED}" == "1" ]]; then
        "${CONTAINER_RUNTIME}" logs "${MILVUS_CONTAINER_NAME}" >"${ARTIFACT_DIR}/${MILVUS_CONTAINER_NAME}.predump.log" 2>&1 || true
    fi
    [[ ! -f "${SERVE_LOG}" ]] || cp "${SERVE_LOG}" "${ARTIFACT_DIR}/serve.log" || true
    [[ ! -f "${TEST_DIR}/router-startup.log" ]] || cp "${TEST_DIR}/router-startup.log" "${ARTIFACT_DIR}/router-startup.log" || true

    if [[ -n "${VLLM_SR_PID}" ]] && kill -0 "${VLLM_SR_PID}" 2>/dev/null; then
        kill "${VLLM_SR_PID}" 2>/dev/null || true
        wait "${VLLM_SR_PID}" 2>/dev/null || true
    fi
    if [[ "${PROVIDER_STARTED}" == "1" ]]; then
        "${CONTAINER_RUNTIME}" stop "${PROVIDER_MOCKER_CONTAINER}" >/dev/null 2>&1 || true
        "${CONTAINER_RUNTIME}" rm "${PROVIDER_MOCKER_CONTAINER}" >/dev/null 2>&1 || true
    fi
    if [[ "${MILVUS_STARTED}" == "1" ]]; then
        make -C "${REPO_ROOT}" stop-milvus >/dev/null 2>&1 || true
    fi
    if [[ "${STACK_STARTED}" == "1" ]]; then
        (cd "${TEST_DIR}" && vllm-sr stop) >/dev/null 2>&1 || true
    fi
    if [[ "${NETWORK_CREATED}" == "1" ]]; then
        "${CONTAINER_RUNTIME}" network rm "${VLLM_SR_NETWORK}" >/dev/null 2>&1 || true
    fi

    if [[ "${KEEP_TEST_DIR}" == "1" ]]; then
        echo "Preserving memory integration artifacts at ${TEST_DIR}"
    elif ! remove_test_dir; then
        echo "Warning: failed to clean up memory integration artifacts at ${TEST_DIR}" >&2
    fi
    return "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Using memory integration temp dir: ${TEST_DIR}"

python3 -m pip install -U requests pymilvus

prepare_model_dir() {
    mkdir -p "${MODEL_DIR}"
    if [[ "${MODEL_DIR}" == "${MODEL_MOUNT_DIR}" ]]; then
        return 0
    fi

    rm -rf "${MODEL_MOUNT_DIR}"
    ln -s "${MODEL_DIR}" "${MODEL_MOUNT_DIR}"
}

prepare_model_dir
echo "Using memory integration model dir: ${MODEL_DIR}"
if [[ "${USE_DETERMINISTIC_MEMORY_EMBEDDINGS}" == "1" ]]; then
    export VLLM_SR_DETERMINISTIC_EMBEDDINGS=1
    echo "Using deterministic memory embeddings"
else
    echo "Router startup will download the configured Vela model at its registered revision"
fi
MILVUS_STARTED=1
make -C "${REPO_ROOT}" start-milvus

# Double-check Milvus readiness with pymilvus probe (gRPC-level, not just HTTP)
echo "Verifying Milvus gRPC readiness via pymilvus..."
for attempt in $(seq 1 30); do
    if python3 -c "
from pymilvus import connections
try:
    connections.connect('default', host='localhost', port='${MILVUS_HOST_PORT}', timeout=5)
    connections.disconnect('default')
    print('Milvus gRPC connection verified')
except Exception as e:
    raise SystemExit(1)
" 2>/dev/null; then
        break
    fi
    if [ "${attempt}" -eq 30 ]; then
        echo "ERROR: Milvus gRPC not ready after 30 attempts"
        "${CONTAINER_RUNTIME}" logs "${MILVUS_CONTAINER_NAME}" 2>&1 | tail -30 || true
        exit 1
    fi
    sleep 2
done

cp "${REPO_ROOT}/e2e/config/config.memory-user.yaml" "${CONFIG_FILE}"
python3 - "${CONFIG_FILE}" "${PROVIDER_MOCKER_CONTAINER}" "${MILVUS_CONTAINER_NAME}" <<'PY_CONFIG'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text().replace("host.docker.internal:8000", f"{sys.argv[2]}:8000")
text = text.replace("host.docker.internal:19530", f"{sys.argv[3]}:19530")
path.write_text(text)
PY_CONFIG

"${CONTAINER_RUNTIME}" network create "${VLLM_SR_NETWORK}" >/dev/null
NETWORK_CREATED=1
"${CONTAINER_RUNTIME}" network connect "${VLLM_SR_NETWORK}" "${MILVUS_CONTAINER_NAME}"
echo "Milvus connected to ${VLLM_SR_NETWORK} as ${MILVUS_CONTAINER_NAME}"

"${CONTAINER_RUNTIME}" run -d --name "${PROVIDER_MOCKER_CONTAINER}" \
    --network "${VLLM_SR_NETWORK}" \
    -p "127.0.0.1:${PROVIDER_MOCKER_HOST_PORT}:8000" \
    -e PROVIDER_MOCKER_SCENARIO=memory \
    -e PROVIDER_MOCKER_MODEL=qwen3 \
    "${PROVIDER_MOCKER_IMAGE:-semantic-router-ci/provider-mocker:e2e-test}" >/dev/null
PROVIDER_STARTED=1

for _ in $(seq 1 30); do
    if curl -s "http://localhost:${PROVIDER_MOCKER_HOST_PORT}/health" >/dev/null 2>&1; then
        echo "provider-mocker ready"
        break
    fi

    if ! "${CONTAINER_RUNTIME}" ps --filter "name=^${PROVIDER_MOCKER_CONTAINER}$" --format '{{.Names}}' | grep -Fxq "${PROVIDER_MOCKER_CONTAINER}"; then
        echo "provider-mocker container exited unexpectedly"
        "${CONTAINER_RUNTIME}" logs "${PROVIDER_MOCKER_CONTAINER}" || true
        exit 1
    fi

    sleep 1
done

if ! curl -s "http://localhost:${PROVIDER_MOCKER_HOST_PORT}/health" >/dev/null 2>&1; then
    echo "provider-mocker did not become healthy"
    "${CONTAINER_RUNTIME}" logs "${PROVIDER_MOCKER_CONTAINER}" || true
    exit 1
fi

STACK_STARTED=1
(
    cd "${TEST_DIR}"
    vllm-sr serve --config config.yaml --image "${VLLM_SR_IMAGE}" --image-pull-policy never >"${SERVE_LOG}" 2>&1 &
    echo "$!" >"${PID_FILE}"
)

if [[ ! -s "${PID_FILE}" ]]; then
    echo "Failed to capture vllm-sr serve PID"
    cat "${SERVE_LOG}" || true
    exit 1
fi

VLLM_SR_PID="$(cat "${PID_FILE}")"

for _ in $(seq 1 300); do
    http_code="$(curl -s -o /dev/null -w "%{http_code}" "${ROUTER_API_HEALTH_URL}" 2>/dev/null || echo "000")"
    if [[ "${http_code}" == "200" ]]; then
        echo "vllm-sr router API ready"
        break
    fi

    if ! kill -0 "${VLLM_SR_PID}" 2>/dev/null; then
        echo "vllm-sr serve exited unexpectedly"
        cat "${SERVE_LOG}" || true
        exit 1
    fi

    sleep 2
done

http_code="$(curl -s -o /dev/null -w "%{http_code}" "${ROUTER_API_HEALTH_URL}" 2>/dev/null || echo "000")"
if [[ "${http_code}" != "200" ]]; then
    echo "vllm-sr router API did not become healthy"
    cat "${SERVE_LOG}" || true
    exit 1
fi

# The running model determines the physical vector namespace. Read the store
# that actually initialized instead of recomputing its identity in the test or
# querying the old logical collection. This fresh stack must have one store;
# missing or conflicting initialization events are a failed prerequisite.
router_startup_log="${TEST_DIR}/router-startup.log"
"${CONTAINER_RUNTIME}" logs "${ROUTER_CONTAINER}" >"${router_startup_log}" 2>&1
memory_collection="$(python3 - "${router_startup_log}" <<'PY'
import json
import re
import sys
from pathlib import Path

collections = set()
for line in Path(sys.argv[1]).read_text().splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    if event.get("component") != "memory" or event.get("event") != "milvus_store_initialized":
        continue
    name = event.get("collection_name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise SystemExit("Invalid initialized memory collection")
    collections.add(name)
if len(collections) != 1:
    raise SystemExit("Expected exactly one initialized Milvus memory collection")
print(collections.pop())
PY
)"

cd "${REPO_ROOT}/e2e/testing"
PYTHONUNBUFFERED=1 \
ROUTER_ENDPOINT="${ROUTER_ENDPOINT}" \
ROUTER_HEALTH_ENDPOINT="${ROUTER_API_HEALTH_URL}" \
MILVUS_ADDRESS="localhost:${MILVUS_HOST_PORT}" \
MILVUS_COLLECTION="${memory_collection}" \
python3 09-memory-features-test.py
