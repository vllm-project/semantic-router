#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/vllm-project/semantic-router}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
VLLM_SR_IMAGE="${VLLM_SR_IMAGE:-ghcr.io/vllm-project/semantic-router/vllm-sr:latest}"

TEST_DIR="${MEMORY_TEST_DIR:-$(mktemp -d -t vsr-memory-test-XXXXXX)}"
PID_FILE="${TEST_DIR}/serve.pid"
SERVE_LOG="${TEST_DIR}/serve.log"
CONFIG_FILE="${TEST_DIR}/config.yaml"
KEEP_TEST_DIR="${KEEP_MEMORY_TEST_DIR:-0}"
ROUTER_API_HEALTH_URL="${ROUTER_API_HEALTH_URL:-http://localhost:8080/health}"

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

python3 -m pip install -U "huggingface_hub[cli]" hf_transfer requests pymilvus

mkdir -p "${TEST_DIR}/models"
HF_HUB_ENABLE_HF_TRANSFER=1 \
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L12-v2', local_dir='${TEST_DIR}/models/mom-embedding-light', local_dir_use_symlinks=False)"

make -C "${REPO_ROOT}" start-milvus
cp "${REPO_ROOT}/e2e/config/config.memory-user.yaml" "${CONFIG_FILE}"

"${CONTAINER_RUNTIME}" run -d --name llm-katan --network host \
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

for _ in $(seq 1 180); do
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
MILVUS_ADDRESS=localhost:19530 \
MILVUS_COLLECTION=memory_test_ci \
python3 09-memory-features-test.py
