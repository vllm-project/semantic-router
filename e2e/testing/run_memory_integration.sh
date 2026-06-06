#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/vllm-project/semantic-router}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
VLLM_SR_IMAGE="${VLLM_SR_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr:latest}"
VLLM_SR_STACK_NAME="${VLLM_SR_STACK_NAME:-vllm-sr}"
if [[ "${VLLM_SR_STACK_NAME}" == "vllm-sr" ]]; then
    VLLM_SR_NETWORK="${VLLM_SR_NETWORK:-vllm-sr-network}"
else
    VLLM_SR_NETWORK="${VLLM_SR_NETWORK:-${VLLM_SR_STACK_NAME}-vllm-sr-network}"
fi

TEST_DIR="${MEMORY_TEST_DIR:-$(mktemp -d -t vsr-memory-test-XXXXXX)}"
PID_FILE="${TEST_DIR}/serve.pid"
SERVE_LOG="${TEST_DIR}/serve.log"
CONFIG_FILE="${TEST_DIR}/config.yaml"
KEEP_TEST_DIR="${KEEP_MEMORY_TEST_DIR:-0}"
ROUTER_API_HEALTH_URL="${ROUTER_API_HEALTH_URL:-http://localhost:8080/ready}"
MODEL_DIR="${MEMORY_TEST_MODEL_DIR:-${TEST_DIR}/models}"
if [[ "${MODEL_DIR}" != /* ]]; then
    MODEL_DIR="${REPO_ROOT}/${MODEL_DIR}"
fi
MODEL_MOUNT_DIR="${TEST_DIR}/models"
USE_DETERMINISTIC_MEMORY_EMBEDDINGS="${USE_DETERMINISTIC_MEMORY_EMBEDDINGS:-0}"

VLLM_SR_PID=""

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
    for c in vllm-sr-router-container vllm-sr-envoy-container vllm-sr-dashboard-container vllm-sr-container llm-katan milvus-semantic-cache; do
        "${CONTAINER_RUNTIME}" logs "$c" > "${log_dump_dir}/${c}.predump.log" 2>&1 || true
    done

    vllm-sr stop >/dev/null 2>&1 || true
    "${CONTAINER_RUNTIME}" stop llm-katan >/dev/null 2>&1 || true
    "${CONTAINER_RUNTIME}" rm llm-katan >/dev/null 2>&1 || true
    make -C "${REPO_ROOT}" stop-milvus >/dev/null 2>&1 || true

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
    local max_attempts="${HF_DOWNLOAD_ATTEMPTS:-12}"
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
if [[ "${USE_DETERMINISTIC_MEMORY_EMBEDDINGS}" == "1" ]]; then
    export VLLM_SR_DETERMINISTIC_EMBEDDINGS=1
    echo "Using deterministic memory embeddings for CI; skipping Hugging Face model download"
else
    echo "Attempting to download Hugging Face model for embeddings (required for memory integration tests)"
    if download_hf_snapshot "llm-semantic-router/mmbert-embed-32k-2d-matryoshka" "${MODEL_DIR}/mmbert-embed-32k-2d-matryoshka"; then
        echo "Hugging Face model downloaded successfully"
    else
        echo "ERROR: Hugging Face model download failed; aborting memory integration tests" >&2
        exit 1
    fi
fi
make -C "${REPO_ROOT}" start-milvus

# Double-check Milvus readiness with pymilvus probe (gRPC-level, not just HTTP)
echo "Verifying Milvus gRPC readiness via pymilvus..."
for attempt in $(seq 1 30); do
    if python3 -c "
from pymilvus import connections
try:
    connections.connect('default', host='localhost', port='19530', timeout=5)
    connections.disconnect('default')
    print('Milvus gRPC connection verified')
except Exception as e:
    raise SystemExit(1)
" 2>/dev/null; then
        break
    fi
    if [ "${attempt}" -eq 30 ]; then
        echo "ERROR: Milvus gRPC not ready after 30 attempts"
        "${CONTAINER_RUNTIME}" logs milvus-semantic-cache 2>&1 | tail -30 || true
        exit 1
    fi
    sleep 2
done

cp "${REPO_ROOT}/e2e/config/config.memory-user.yaml" "${CONFIG_FILE}"
python3 -c 'from pathlib import Path; path = Path("'"${CONFIG_FILE}"'"); t = path.read_text(); t = t.replace("host.docker.internal:8000", "llm-katan:8000"); t = t.replace("host.docker.internal:19530", "vllm-sr-milvus:19530"); path.write_text(t)'

if ! "${CONTAINER_RUNTIME}" network inspect "${VLLM_SR_NETWORK}" >/dev/null 2>&1; then
    "${CONTAINER_RUNTIME}" network create "${VLLM_SR_NETWORK}" >/dev/null
fi

# Connect the externally-started Milvus to the vllm-sr network so the router
# container can reach it by the name vllm-sr serve expects.
"${CONTAINER_RUNTIME}" network connect --alias vllm-sr-milvus "${VLLM_SR_NETWORK}" milvus-semantic-cache 2>/dev/null || true
echo "Milvus connected to ${VLLM_SR_NETWORK} as vllm-sr-milvus"

"${CONTAINER_RUNTIME}" run -d --name llm-katan \
    --network "${VLLM_SR_NETWORK}" \
    --network-alias llm-katan \
    -p 8000:8000 \
    "${DOCKER_REGISTRY}/llm-katan:${DOCKER_TAG}" \
    llm-katan --model dummy --host 0.0.0.0 --port 8000 --served-model-name qwen3 --backend echo >/dev/null

for _ in $(seq 1 30); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "llm-katan ready"
        break
    fi

    if ! "${CONTAINER_RUNTIME}" ps --filter "name=llm-katan" --format '{{.Names}}' | grep -q '^llm-katan$'; then
        echo "llm-katan container exited unexpectedly"
        "${CONTAINER_RUNTIME}" logs llm-katan || true
        exit 1
    fi

    sleep 1
done

if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "llm-katan did not become healthy"
    "${CONTAINER_RUNTIME}" logs llm-katan || true
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
ROUTER_ENDPOINT=http://localhost:8888 \
ROUTER_HEALTH_ENDPOINT="${ROUTER_API_HEALTH_URL}" \
MILVUS_ADDRESS=localhost:19530 \
MILVUS_COLLECTION=memory_test_ci \
python3 09-memory-features-test.py
