#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${VLLM_SR_TEST_ISOLATED:-0}" != "1" ]]; then
  exec python3 "${ROOT_DIR}/tools/dev/with_test_resources.py" --isolate-stack -- bash "${BASH_SOURCE[0]}" "$@"
fi
RECIPES="${RECIPES:-}"
ROUTER_IMAGE="${ROUTER_IMAGE:-}"
VLLM_SR_PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-0}"
if ! [[ "${VLLM_SR_PORT_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "VLLM_SR_PORT_OFFSET must be a non-negative integer" >&2
  exit 2
fi
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:$((8080 + VLLM_SR_PORT_OFFSET))}"
REPORT_ROOT="${REPORT_ROOT:-${VLLM_SR_TEST_OUTPUT_DIR}/recipe-conformance}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
STACK_STARTED=0
CONTAINER_PREFIX="${VLLM_SR_STACK_NAME}-vllm-sr"
if [[ "${VLLM_SR_STACK_NAME}" == "vllm-sr" ]]; then
  CONTAINER_PREFIX="vllm-sr"
fi

if [[ -z "${RECIPES}" ]]; then
  echo "RECIPES is required (comma-separated recipe names)" >&2
  exit 2
fi
if [[ -z "${ROUTER_IMAGE}" ]]; then
  echo "ROUTER_IMAGE is required (the immutable image built for this source tree)" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d -t vsr-recipe-test-XXXXXX)"

cleanup() {
  if [[ "${STACK_STARTED}" == "1" ]]; then
    vllm-sr stop >/dev/null 2>&1 || true
    STACK_STARTED=0
  fi
}
trap 'cleanup; rm -rf "${WORK_DIR}"' EXIT

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
    "${CONTAINER_PREFIX}-router-container" \
    "${CONTAINER_PREFIX}-envoy-container" \
    "${CONTAINER_PREFIX}-dashboard-container" \
    "${CONTAINER_PREFIX}-container"; do
    docker logs "${container}" >"${destination}/${container}.log" 2>&1 || true
  done
  docker ps -a --filter "name=${CONTAINER_PREFIX}-" >"${destination}/docker-status.txt" 2>&1 || true
}

IFS=',' read -r -a recipe_names <<<"${RECIPES}"
for recipe in "${recipe_names[@]}"; do
  recipe="${recipe//[[:space:]]/}"
  [[ -n "${recipe}" ]] || continue
  if ! [[ "${recipe}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "invalid recipe name: ${recipe}" >&2
    exit 2
  fi
  config="${ROOT_DIR}/config/recipes/${recipe}/config.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "unknown recipe: ${recipe}" >&2
    exit 2
  fi

  mkdir -p "${WORK_DIR}/${recipe}"
  cp -R "${ROOT_DIR}/config/recipes/${recipe}/." "${WORK_DIR}/${recipe}/"
  rm -rf "${WORK_DIR}/${recipe}/.vllm-sr"
  config="${WORK_DIR}/${recipe}/config.yaml"

  echo "=== recipe conformance: ${recipe} ==="
  cleanup
  STACK_STARTED=1
  if ! POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-router-secret}" \
    vllm-sr serve \
      --image-pull-policy ifnotpresent \
      --router-image "${ROUTER_IMAGE}" \
      --minimal \
      --config "${config}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! wait_for_router; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! python3 "${ROOT_DIR}/tools/dev/router-calibration/recipe_conformance.py" \
    --output-dir "${REPORT_ROOT}" \
    eval \
    --recipe "${recipe}" \
    --router-url "${ROUTER_URL}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  collect_logs "${recipe}"
done
