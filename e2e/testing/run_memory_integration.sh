#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/vllm-project/semantic-router}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
VLLM_SR_IMAGE="${VLLM_SR_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr:latest}"
VLLM_SR_STACK_NAME="${VLLM_SR_STACK_NAME:-vllm-sr-memory-$$}"
VLLM_SR_PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-0}"
VLLM_SR_RUN_ID="${VLLM_SR_RUN_ID:-memory-$$}"
export VLLM_SR_STACK_NAME VLLM_SR_PORT_OFFSET VLLM_SR_RUN_ID
if [[ "${VLLM_SR_STACK_NAME}" == "vllm-sr" ]]; then
    VLLM_SR_NETWORK="${VLLM_SR_NETWORK:-vllm-sr-network}"
    VLLM_SR_ROUTER_CONTAINER_NAME="vllm-sr-router-container"
    VLLM_SR_ENVOY_CONTAINER_NAME="vllm-sr-envoy-container"
    VLLM_SR_DASHBOARD_CONTAINER_NAME="vllm-sr-dashboard-container"
else
    VLLM_SR_NETWORK="${VLLM_SR_NETWORK:-${VLLM_SR_STACK_NAME}-vllm-sr-network}"
    VLLM_SR_ROUTER_CONTAINER_NAME="${VLLM_SR_STACK_NAME}-vllm-sr-router-container"
    VLLM_SR_ENVOY_CONTAINER_NAME="${VLLM_SR_STACK_NAME}-vllm-sr-envoy-container"
    VLLM_SR_DASHBOARD_CONTAINER_NAME="${VLLM_SR_STACK_NAME}-vllm-sr-dashboard-container"
fi
STACK_SUFFIX="${VLLM_SR_STACK_NAME}-memory"
LLM_KATAN_CONTAINER_NAME="${LLM_KATAN_CONTAINER_NAME:-${STACK_SUFFIX}-llm-katan}"
MILVUS_CONTAINER_NAME="${MILVUS_CONTAINER_NAME:-${STACK_SUFFIX}-milvus}"
LLM_KATAN_HOST_PORT="${LLM_KATAN_HOST_PORT:-$((8000 + VLLM_SR_PORT_OFFSET))}"
MILVUS_PORT=$((19530 + VLLM_SR_PORT_OFFSET))
MILVUS_HEALTH_PORT=$((9091 + VLLM_SR_PORT_OFFSET))
ROUTER_ENDPOINT_PORT=$((8888 + VLLM_SR_PORT_OFFSET))

TEST_DIR="${MEMORY_TEST_DIR:-$(mktemp -d -t vsr-memory-test-XXXXXX)}"
PID_FILE="${TEST_DIR}/serve.pid"
SERVE_LOG="${TEST_DIR}/serve.log"
CONFIG_FILE="${TEST_DIR}/config.yaml"
KEEP_TEST_DIR="${KEEP_MEMORY_TEST_DIR:-0}"
ROUTER_API_HEALTH_URL="${ROUTER_API_HEALTH_URL:-http://localhost:$((8080 + VLLM_SR_PORT_OFFSET))/ready}"
MODEL_DIR="${MEMORY_TEST_MODEL_DIR:-${TEST_DIR}/models}"
if [[ "${MODEL_DIR}" != /* ]]; then
    MODEL_DIR="${REPO_ROOT}/${MODEL_DIR}"
fi
MODEL_MOUNT_DIR="${TEST_DIR}/models"
USE_DETERMINISTIC_MEMORY_EMBEDDINGS="${USE_DETERMINISTIC_MEMORY_EMBEDDINGS:-0}"

VLLM_SR_PID=""

container_owned_by_run() {
    local container_name="$1"
    local labels
    labels="$(${CONTAINER_RUNTIME} inspect --format '{{index .Config.Labels "com.vllm.semantic-router.managed"}} {{index .Config.Labels "com.vllm.semantic-router.stack"}} {{index .Config.Labels "com.vllm.semantic-router.run"}}' "${container_name}" 2>/dev/null || true)"
    [[ "${labels}" == "true ${VLLM_SR_STACK_NAME} ${VLLM_SR_RUN_ID}" ]]
}

network_owned_by_run() {
    local network_name="$1"
    local labels
    labels="$(${CONTAINER_RUNTIME} network inspect --format '{{index .Labels "com.vllm.semantic-router.managed"}} {{index .Labels "com.vllm.semantic-router.stack"}} {{index .Labels "com.vllm.semantic-router.run"}}' "${network_name}" 2>/dev/null || true)"
    [[ "${labels}" == "true ${VLLM_SR_STACK_NAME} ${VLLM_SR_RUN_ID}" ]]
}

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

    if [[ -z "${VLLM_SR_PID}" && -f "${PID_FILE}" ]]; then
        VLLM_SR_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
    fi

    if [[ -n "${VLLM_SR_PID}" ]] && kill -0 "${VLLM_SR_PID}" 2>/dev/null; then
        kill "${VLLM_SR_PID}" 2>/dev/null || true
        wait "${VLLM_SR_PID}" 2>/dev/null || true
    fi

    # Dump container logs BEFORE stopping/removing them so CI can collect them.
    local log_dump_dir="${REPO_ROOT}/logs"
    mkdir -p "${log_dump_dir}" 2>/dev/null || true
    for c in "${VLLM_SR_ROUTER_CONTAINER_NAME}" "${VLLM_SR_ENVOY_CONTAINER_NAME}" "${VLLM_SR_DASHBOARD_CONTAINER_NAME}" "${LLM_KATAN_CONTAINER_NAME}" "${MILVUS_CONTAINER_NAME}"; do
        "${CONTAINER_RUNTIME}" logs "$c" > "${log_dump_dir}/${c}.predump.log" 2>&1 || true
    done

    if container_owned_by_run "${LLM_KATAN_CONTAINER_NAME}"; then
        "${CONTAINER_RUNTIME}" stop "${LLM_KATAN_CONTAINER_NAME}" >/dev/null 2>&1 || true
        "${CONTAINER_RUNTIME}" rm "${LLM_KATAN_CONTAINER_NAME}" >/dev/null 2>&1 || true
    elif "${CONTAINER_RUNTIME}" inspect "${LLM_KATAN_CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "Refusing to remove ${LLM_KATAN_CONTAINER_NAME}: ownership labels do not match" >&2
    fi
    make -C "${REPO_ROOT}" stop-milvus \
        MILVUS_CONTAINER_NAME="${MILVUS_CONTAINER_NAME}" \
        MILVUS_DATA_DIR="${TEST_DIR}/milvus-data" \
        MILVUS_STACK_NAME="${VLLM_SR_STACK_NAME}" \
        MILVUS_RUN_ID="${VLLM_SR_RUN_ID}" >/dev/null || true

    # A failure before serve starts must not let vllm-sr stop target an older
    # stack that happens to use the caller-supplied name.
    if [[ -n "${VLLM_SR_PID}" ]]; then
        vllm-sr stop >/dev/null 2>&1 || true
    fi

    # vllm-sr stop normally removes this network. Remove a pre-serve leftover
    # only after this integration run proves exact stack/run ownership.
    if network_owned_by_run "${VLLM_SR_NETWORK}"; then
        "${CONTAINER_RUNTIME}" network rm "${VLLM_SR_NETWORK}" >/dev/null 2>&1 || true
    elif "${CONTAINER_RUNTIME}" network inspect "${VLLM_SR_NETWORK}" >/dev/null 2>&1; then
        echo "Refusing to remove ${VLLM_SR_NETWORK}: ownership labels do not match" >&2
    fi

    if [[ "${KEEP_TEST_DIR}" == "1" ]]; then
        echo "Preserving memory integration artifacts at ${TEST_DIR}"
    else
        if ! remove_test_dir; then
            echo "Warning: failed to clean up memory integration artifacts at ${TEST_DIR}" >&2
            echo "Set KEEP_MEMORY_TEST_DIR=1 to inspect the leftover files manually." >&2
        fi
    fi

    return "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "Using memory integration temp dir: ${TEST_DIR}"

if [[ "${USE_DETERMINISTIC_MEMORY_EMBEDDINGS}" == "1" ]]; then
    python3 -m pip install -U requests pymilvus
else
    python3 -m pip install -U "huggingface_hub[cli]" hf_transfer requests pymilvus
fi

prepare_model_dir() {
    mkdir -p "${MODEL_DIR}"
    if [[ "${MODEL_DIR}" == "${MODEL_MOUNT_DIR}" ]]; then
        return 0
    fi

    rm -rf "${MODEL_MOUNT_DIR}"
    ln -s "${MODEL_DIR}" "${MODEL_MOUNT_DIR}"
}

download_hf_snapshot() {
    local repo_id="$1"
    local local_dir="$2"
    local required="${3:-required}"
    local max_attempts="${HF_DOWNLOAD_ATTEMPTS:-6}"
    local attempt delay exit_code marker

    if ! [[ "${max_attempts}" =~ ^[0-9]+$ ]] || (( max_attempts < 1 )); then
        max_attempts=6
    fi

    marker="${local_dir}/.vsr-download-complete"
    if [[ -f "${marker}" ]]; then
        echo "Using cached Hugging Face model ${repo_id} from ${local_dir}"
        return 0
    fi

    mkdir -p "${local_dir}"
    exit_code=1
    for attempt in $(seq 1 "${max_attempts}"); do
        echo "Downloading Hugging Face model ${repo_id} to ${local_dir} (attempt ${attempt}/${max_attempts})"
        if HF_HUB_ENABLE_HF_TRANSFER=1 python3 - "${repo_id}" "${local_dir}" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
PY
        then
            touch "${marker}"
            return 0
        else
            exit_code=$?
        fi

        if (( attempt == max_attempts )); then
            break
        fi

        delay=$((attempt * attempt * 10))
        if (( delay > 120 )); then
            delay=120
        fi
        echo "Hugging Face download failed for ${repo_id}; retrying in ${delay}s" >&2
        sleep "${delay}"
    done

    if [[ "${required}" == "optional" ]]; then
        echo "Warning: ${repo_id} download failed; router will skip it" >&2
        return 0
    fi

    echo "ERROR: failed to download required Hugging Face model ${repo_id}" >&2
    return "${exit_code}"
}

prepare_model_dir
echo "Using memory integration model dir: ${MODEL_DIR}"
# Detect requested embedding model from the e2e config so we can make a best-effort
# attempt to ensure a compatible model is available during CI runs. This avoids
# silent mismatches between the config and the model the test script downloads.
CONFIG_EMBEDDING_MODEL="$(grep -m1 '^ *embedding_model:' "${REPO_ROOT}/e2e/config/config.memory-user.yaml" 2>/dev/null | awk -F: '{print $2}' | tr -d ' \"')"
if [[ -z "${CONFIG_EMBEDDING_MODEL}" ]]; then
    CONFIG_EMBEDDING_MODEL="mmbert"
fi
if [[ "${CONFIG_EMBEDDING_MODEL}" != "mmbert" ]]; then
    echo "Note: config requests embedding_model='${CONFIG_EMBEDDING_MODEL}'. For CI stability we will still ensure the mmbert embeddings model is present unless deterministic mode is explicitly requested."
fi
if [[ "${USE_DETERMINISTIC_MEMORY_EMBEDDINGS}" == "1" ]]; then
    export VLLM_SR_DETERMINISTIC_EMBEDDINGS=1
    echo "Using deterministic memory embeddings for CI; skipping Hugging Face model download"
else
    echo "Attempting to download Hugging Face model for embeddings (will fall back to deterministic on failure)"
    # Ensure the mmbert model used by the CI harness is available. Tests and
    # configs may accidentally request a different model; providing mmbert keeps
    # the CI stable and compatible with the rest of the harness (collection dims, etc.).
    if download_hf_snapshot "llm-semantic-router/mmbert-embed-32k-2d-matryoshka" "${MODEL_DIR}/mmbert-embed-32k-2d-matryoshka"; then
        echo "Hugging Face model downloaded successfully"
    else
        if [[ "${USE_DETERMINISTIC_MEMORY_EMBEDDINGS}" == "1" ]]; then
            echo "Warning: Hugging Face model download failed; using deterministic embeddings due to USE_DETERMINISTIC_MEMORY_EMBEDDINGS=1"
            export VLLM_SR_DETERMINISTIC_EMBEDDINGS=1
        else
            echo "ERROR: Hugging Face model download failed and deterministic fallback is disabled for CI. Exiting." >&2
            exit 1
        fi
    fi
fi
make -C "${REPO_ROOT}" start-milvus \
    MILVUS_CONTAINER_NAME="${MILVUS_CONTAINER_NAME}" \
    MILVUS_BIND_HOST="127.0.0.1" \
    MILVUS_PORT="${MILVUS_PORT}" \
    MILVUS_HEALTH_PORT="${MILVUS_HEALTH_PORT}" \
    MILVUS_DATA_DIR="${TEST_DIR}/milvus-data" \
    MILVUS_STACK_NAME="${VLLM_SR_STACK_NAME}" \
    MILVUS_RUN_ID="${VLLM_SR_RUN_ID}"

# Double-check Milvus readiness with pymilvus probe (gRPC-level, not just HTTP)
echo "Verifying Milvus gRPC readiness via pymilvus..."
for attempt in $(seq 1 30); do
    if python3 -c "
from pymilvus import connections
try:
    connections.connect('default', host='localhost', port=${MILVUS_PORT}, timeout=5)
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
python3 -c 'from pathlib import Path; path = Path("'"${CONFIG_FILE}"'"); t = path.read_text(); t = t.replace("host.docker.internal:8000", "'"${LLM_KATAN_CONTAINER_NAME}"':8000"); t = t.replace("host.docker.internal:19530", "'"${MILVUS_CONTAINER_NAME}"':19530"); path.write_text(t)'

if ! "${CONTAINER_RUNTIME}" network inspect "${VLLM_SR_NETWORK}" >/dev/null 2>&1; then
    "${CONTAINER_RUNTIME}" network create \
        --label com.vllm.semantic-router.managed=true \
        --label com.vllm.semantic-router.stack="${VLLM_SR_STACK_NAME}" \
        --label com.vllm.semantic-router.run="${VLLM_SR_RUN_ID}" \
        "${VLLM_SR_NETWORK}" >/dev/null
elif ! network_owned_by_run "${VLLM_SR_NETWORK}"; then
    echo "Refusing to use ${VLLM_SR_NETWORK}: ownership labels do not match" >&2
    exit 1
fi

# Connect the externally-started Milvus to the vllm-sr network so the router
# container can reach it by the name vllm-sr serve expects.
"${CONTAINER_RUNTIME}" network connect --alias "${MILVUS_CONTAINER_NAME}" "${VLLM_SR_NETWORK}" "${MILVUS_CONTAINER_NAME}" 2>/dev/null || true
echo "Milvus connected to ${VLLM_SR_NETWORK} as ${MILVUS_CONTAINER_NAME}"

"${CONTAINER_RUNTIME}" run -d --name "${LLM_KATAN_CONTAINER_NAME}" \
    --label com.vllm.semantic-router.managed=true \
    --label com.vllm.semantic-router.stack="${VLLM_SR_STACK_NAME}" \
    --label com.vllm.semantic-router.run="${VLLM_SR_RUN_ID}" \
    --network "${VLLM_SR_NETWORK}" \
    --network-alias "${LLM_KATAN_CONTAINER_NAME}" \
    -p "127.0.0.1:${LLM_KATAN_HOST_PORT}:8000" \
    "${DOCKER_REGISTRY}/llm-katan:${DOCKER_TAG}" \
    llm-katan --model dummy --host 0.0.0.0 --port 8000 --served-model-name qwen3 --backend echo >/dev/null

for _ in $(seq 1 30); do
    if curl -s "http://localhost:${LLM_KATAN_HOST_PORT}/health" >/dev/null 2>&1; then
        echo "llm-katan ready"
        break
    fi

    if ! "${CONTAINER_RUNTIME}" ps --filter "name=${LLM_KATAN_CONTAINER_NAME}" --format '{{.Names}}' | grep -q "^${LLM_KATAN_CONTAINER_NAME}$"; then
        echo "llm-katan container exited unexpectedly"
        "${CONTAINER_RUNTIME}" logs "${LLM_KATAN_CONTAINER_NAME}" || true
        exit 1
    fi

    sleep 1
done

if ! curl -s "http://localhost:${LLM_KATAN_HOST_PORT}/health" >/dev/null 2>&1; then
    echo "llm-katan did not become healthy"
    "${CONTAINER_RUNTIME}" logs "${LLM_KATAN_CONTAINER_NAME}" || true
    exit 1
fi

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

cd "${REPO_ROOT}/e2e/testing"
PYTHONUNBUFFERED=1 \
ROUTER_ENDPOINT="http://localhost:${ROUTER_ENDPOINT_PORT}" \
ROUTER_HEALTH_ENDPOINT="${ROUTER_API_HEALTH_URL}" \
MILVUS_ADDRESS="localhost:${MILVUS_PORT}" \
MILVUS_COLLECTION=memory_test_ci \
python3 09-memory-features-test.py
