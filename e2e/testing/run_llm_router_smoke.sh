#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

STACK_NAME="${STACK_NAME:-llm-router-smoke}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/e2e/config/config.llm-router-mock.yaml}"
ROUTER_URL="${ROUTER_URL:-http://localhost:8899/v1/chat/completions}"
MOCK_ROUTER_HOST="${MOCK_ROUTER_HOST:-0.0.0.0}"
MOCK_ROUTER_PORT="${MOCK_ROUTER_PORT:-9001}"
MOCK_VLLM_HOST="${MOCK_VLLM_HOST:-0.0.0.0}"
MOCK_VLLM_PORT="${MOCK_VLLM_PORT:-8002}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

ARTIFACT_DIR="$(mktemp -d -t llm-router-smoke-XXXXXX)"
MOCK_ROUTER_LOG="${ARTIFACT_DIR}/mock-llm-router.log"
MOCK_VLLM_LOG="${ARTIFACT_DIR}/mock-vllm.log"
SERVE_LOG="${ARTIFACT_DIR}/serve.log"
MOCK_ROUTER_PID=""
MOCK_VLLM_PID=""

cleanup() {
    local exit_code=$?

    if [[ -n "${MOCK_ROUTER_PID}" ]] && kill -0 "${MOCK_ROUTER_PID}" 2>/dev/null; then
        kill "${MOCK_ROUTER_PID}" 2>/dev/null || true
        wait "${MOCK_ROUTER_PID}" 2>/dev/null || true
    fi

    if [[ -n "${MOCK_VLLM_PID}" ]] && kill -0 "${MOCK_VLLM_PID}" 2>/dev/null; then
        kill "${MOCK_VLLM_PID}" 2>/dev/null || true
        wait "${MOCK_VLLM_PID}" 2>/dev/null || true
    fi

    vllm-sr stop >/dev/null 2>&1 || true

    if [[ "${KEEP_ARTIFACTS}" == "1" ]]; then
        echo "Artifacts kept at: ${ARTIFACT_DIR}"
    else
        rm -rf "${ARTIFACT_DIR}" 2>/dev/null || true
    fi

    return "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "Using config: ${CONFIG_FILE}"
echo "Artifacts: ${ARTIFACT_DIR}"

python3 "${REPO_ROOT}/e2e/testing/mock-vllm-simple.py" \
    --host "${MOCK_VLLM_HOST}" \
    --port "${MOCK_VLLM_PORT}" \
    >"${MOCK_VLLM_LOG}" 2>&1 &
MOCK_VLLM_PID=$!

python3 "${REPO_ROOT}/e2e/testing/mock-llm-router-server.py" \
    --host "${MOCK_ROUTER_HOST}" \
    --port "${MOCK_ROUTER_PORT}" \
    >"${MOCK_ROUTER_LOG}" 2>&1 &
MOCK_ROUTER_PID=$!

sleep 2

echo "Starting local stack..."
make -C "${REPO_ROOT}" agent-serve-local \
    ENV=cpu \
    AGENT_STACK_NAME="${STACK_NAME}" \
    AGENT_SERVE_CONFIG="${CONFIG_FILE}" \
    >"${SERVE_LOG}" 2>&1

echo "Sending request to router..."
response="$(
    curl -sS -i "${ROUTER_URL}" \
        -H 'Content-Type: application/json' \
        -d '{
            "model": "MoM",
            "messages": [
                {
                    "role": "user",
                    "content": "Please route this request to the best model for a long context reasoning task."
                }
            ]
        }'
)"
printf '%s\n' "${response}"

if ! grep -q '^HTTP/1.1 200 OK' <<<"${response}"; then
    echo "ERROR: request did not succeed" >&2
    echo "--- serve log ---" >&2
    tail -n 120 "${SERVE_LOG}" >&2 || true
    echo "--- mock llm router log ---" >&2
    tail -n 120 "${MOCK_ROUTER_LOG}" >&2 || true
    exit 1
fi

if ! grep -q 'query=Decision name: llm-router-decision' "${MOCK_ROUTER_LOG}"; then
    echo "ERROR: mock LLM router did not receive the rendered query" >&2
    echo "--- mock llm router log ---" >&2
    tail -n 120 "${MOCK_ROUTER_LOG}" >&2 || true
    exit 1
fi

echo
echo "Mock LLM router log excerpt:"
tail -n 40 "${MOCK_ROUTER_LOG}" || true

