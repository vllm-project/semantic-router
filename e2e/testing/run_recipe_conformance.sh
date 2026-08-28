#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECIPES="${RECIPES:-}"
RECIPES_ROOT="${RECIPES_ROOT:-${ROOT_DIR}/config/recipes}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT_DIR}/.agent-harness/recipe-conformance}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"
GENERATED_RECIPE_DIRS=()
READINESS_TOKEN=""

if [[ -z "${RECIPES}" ]]; then
  echo "RECIPES is required (comma-separated recipe names)" >&2
  exit 2
fi
if [[ ! -d "${RECIPES_ROOT}" ]]; then
  echo "recipe root does not exist: ${RECIPES_ROOT}" >&2
  exit 2
fi
RECIPES_ROOT="$(cd "${RECIPES_ROOT}" && pwd)"

cleanup() {
  VLLM_SR_STATE_ROOT_DIR="${ROOT_DIR}" vllm-sr stop >/dev/null 2>&1 || true
  for directory in "${GENERATED_RECIPE_DIRS[@]}"; do
    rm -rf "${directory}"
  done
}
trap cleanup EXIT

wait_for_router() {
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  local curl_args=(--fail --silent)
  if [[ -n "${READINESS_TOKEN}" ]]; then
    curl_args+=(--header "Authorization: Bearer ${READINESS_TOKEN}")
  fi
  while ((SECONDS < deadline)); do
    if curl "${curl_args[@]}" "${ROUTER_URL}/ready" >/dev/null; then
      return 0
    fi
    sleep 3
  done
  echo "router did not become ready within ${READY_TIMEOUT_SECONDS}s" >&2
  return 1
}

configure_management_auth() {
  local recipe="$1"
  local env_name=""
  local capability=""
  local token_index=0
  local token_value=""
  local auth_bindings=""

  READINESS_TOKEN=""
  unset VSR_MGMT_TOKEN
  auth_bindings="$(
    python3 "${ROOT_DIR}/tools/agent/scripts/recipe_conformance.py" \
      --recipes-root "${RECIPES_ROOT}" \
      runtime-auth \
      --recipe "${recipe}"
  )"
  while IFS='|' read -r env_name capability; do
    [[ -n "${env_name}" ]] || continue
    token_index=$((token_index + 1))
    token_value="${!env_name:-}"
    if [[ -z "${token_value}" ]]; then
      printf -v token_value '%064x' "${token_index}"
      printf -v "${env_name}" '%s' "${token_value}"
      export "${env_name?}"
    fi
    if [[ "${capability}" == "ready" && -z "${READINESS_TOKEN}" ]]; then
      READINESS_TOKEN="${token_value}"
      VSR_MGMT_TOKEN="${token_value}"
      export VSR_MGMT_TOKEN
    fi
  done <<<"${auth_bindings}"
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
  config="${RECIPES_ROOT}/${recipe}/config.yaml"
  generated_output="${RECIPES_ROOT}/${recipe}/.vllm-sr"
  GENERATED_RECIPE_DIRS+=("${generated_output}")
  rm -rf "${generated_output}"
  if [[ ! -f "${config}" ]]; then
    echo "unknown recipe: ${recipe}" >&2
    exit 2
  fi

  echo "=== recipe conformance: ${recipe} ==="
  cleanup
  configure_management_auth "${recipe}"
  if ! POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-router-secret}" \
    VLLM_SR_STATE_ROOT_DIR="${ROOT_DIR}" \
    vllm-sr serve \
      --image-pull-policy ifnotpresent \
      --minimal \
      --config "${config}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! wait_for_router; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! python3 "${ROOT_DIR}/tools/agent/scripts/recipe_conformance.py" \
    --recipes-root "${RECIPES_ROOT}" \
    --output-dir "${REPORT_ROOT}" \
    eval \
    --recipe "${recipe}" \
    --router-url "${ROUTER_URL}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  collect_logs "${recipe}"
done
