#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECIPES="${RECIPES:-}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT_DIR}/.agent-harness/recipe-conformance}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
WORKSPACE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vllm-sr-recipe-conformance.XXXXXX")"

if [[ -z "${RECIPES}" ]]; then
  echo "RECIPES is required (comma-separated recipe names)" >&2
  exit 2
fi

cleanup() {
  VLLM_SR_STATE_ROOT_DIR="${WORKSPACE_ROOT}" vllm-sr stop >/dev/null 2>&1 || true
}

cleanup_workspace() {
  cleanup
  rm -rf -- "${WORKSPACE_ROOT}"
}

trap cleanup_workspace EXIT

wait_for_router() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if curl --fail --silent "${ROUTER_URL}/ready" >/dev/null; then
      return 0
    fi
    sleep 3
  done
  echo "router did not become ready within ${READY_TIMEOUT_SECONDS}s" >&2
  return 1
}

collect_logs() {
  local recipe="$1"
  local destination="${REPORT_ROOT}/${recipe}"
  mkdir -p "${destination}"
  for container in \
    vllm-sr-router-container \
    vllm-sr-envoy-container \
    vllm-sr-dashboard-container \
    vllm-sr-container; do
    docker logs "${container}" >"${destination}/${container}.log" 2>&1 || true
  done
  docker ps -a >"${destination}/docker-status.txt" 2>&1 || true
}

IFS=',' read -r -a recipe_names <<<"${RECIPES}"
for recipe in "${recipe_names[@]}"; do
  recipe="${recipe//[[:space:]]/}"
  [[ -n "${recipe}" ]] || continue
  config="${ROOT_DIR}/config/recipes/${recipe}/config.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "unknown recipe: ${recipe}" >&2
    exit 2
  fi

  echo "=== recipe conformance: ${recipe} ==="
  cleanup
  cp -- "${config}" "${WORKSPACE_ROOT}/config.yaml"
  if ! POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-router-secret}" \
    VLLM_SR_STATE_ROOT_DIR="${WORKSPACE_ROOT}" \
    vllm-sr serve \
      --image-pull-policy ifnotpresent \
      --minimal; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! wait_for_router; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! python3 "${ROOT_DIR}/tools/agent/scripts/recipe_conformance.py" \
    --output-dir "${REPORT_ROOT}" \
    eval \
    --recipe "${recipe}" \
    --router-url "${ROUTER_URL}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  collect_logs "${recipe}"
done
