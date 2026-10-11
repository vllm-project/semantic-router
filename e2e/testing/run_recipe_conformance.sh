#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECIPES="${RECIPES:-}"
RECIPES_ROOT="${RECIPES_ROOT:-${ROOT_DIR}/config/recipes}"
ROUTER_IMAGE="${ROUTER_IMAGE:-}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
VLLM_SR_PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-0}"
if ! [[ "${VLLM_SR_PORT_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "VLLM_SR_PORT_OFFSET must be a non-negative integer" >&2
  exit 2
fi
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:$((8080 + VLLM_SR_PORT_OFFSET))}"
REPORT_ROOT="${REPORT_ROOT:-${ROOT_DIR}/.agent-harness/recipe-conformance}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-900}"
CONFORMANCE="${ROOT_DIR}/tools/calibration/recipe/recipe_conformance.py"
READINESS_TOKEN=""

if [[ -z "${RECIPES}" || -z "${ROUTER_IMAGE}" ]]; then
  echo "RECIPES and ROUTER_IMAGE (built for this source tree) are required" >&2
  exit 2
fi
# Reject an unknown or hardware-only selection before touching any stack.
python3 "${CONFORMANCE}" --recipes-root "${RECIPES_ROOT}" check-cpu --recipes "${RECIPES}"
mkdir -p "${REPORT_ROOT}" "${ROOT_DIR}/.agent-harness/recipe-conformance-runtime"
RUN_ROOT="$(mktemp -d "${ROOT_DIR}/.agent-harness/recipe-conformance-runtime/run-XXXXXX")"
mkdir -p "${ROOT_DIR}/models"
ln -s "${ROOT_DIR}/models" "${RUN_ROOT}/models"
export VLLM_SR_STACK_NAME="${VLLM_SR_STACK_NAME:-recipe-$(basename "${RUN_ROOT}")}"
export VLLM_SR_PORT_OFFSET
export VLLM_SR_STATE_ROOT_DIR="${RUN_ROOT}"
export CONTAINER_RUNTIME
CONTAINERS="$(PYTHONPATH="${ROOT_DIR}/src/vllm-sr${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
from cli.runtime_stack import resolve_runtime_stack
stack = resolve_runtime_stack()
print(stack.router_container_name, stack.envoy_container_name, stack.dashboard_container_name)
PY
)"

cleanup() {
  vllm-sr stop >/dev/null 2>&1 || true
  # Keep the invocation's configuration and state for failure diagnosis. Both
  # live output and generated config stay under ignored .agent-harness paths.
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
  local config="$1" env_name capability token_value auth_bindings
  READINESS_TOKEN=""
  unset VSR_MGMT_TOKEN
  auth_bindings="$(python3 "${CONFORMANCE}" runtime-auth --config "${config}")"
  while IFS='|' read -r env_name capability; do
    [[ -n "${env_name}" ]] || continue
    token_value="${!env_name:-}"
    if [[ -z "${token_value}" ]]; then
      token_value="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
      printf -v "${env_name}" '%s' "${token_value}"
      export "${env_name?}"
    fi
    if [[ "${capability}" == "ready" && -z "${READINESS_TOKEN}" ]]; then
      READINESS_TOKEN="${token_value}"
      export VSR_MGMT_TOKEN="${token_value}"
    fi
  done <<<"${auth_bindings}"
}

collect_logs() {
  local destination="${REPORT_ROOT}/$1" container
  mkdir -p "${destination}"
  for container in ${CONTAINERS}; do
    "${CONTAINER_RUNTIME}" logs "${container}" >"${destination}/${container}.log" 2>&1 || true
    "${CONTAINER_RUNTIME}" inspect "${container}" --format '{{.Id}} {{.Image}} {{.State.Status}}' \
      >>"${destination}/containers.txt" 2>&1 || true
  done
}

IFS=',' read -r -a recipe_names <<<"${RECIPES}"
for recipe in "${recipe_names[@]}"; do
  recipe="${recipe//[[:space:]]/}"
  [[ -n "${recipe}" ]] || continue
  config="${RUN_ROOT}/${recipe}/config.yaml"
  python3 "${CONFORMANCE}" --recipes-root "${RECIPES_ROOT}" \
    prepare-runtime --recipe "${recipe}" --config "${config}"
  echo "=== recipe conformance: ${recipe} ==="
  cleanup
  configure_management_auth "${config}"
  if ! POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-router-secret}" \
    vllm-sr serve --image-pull-policy ifnotpresent --router-image "${ROUTER_IMAGE}" \
      --minimal --config "${config}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! wait_for_router; then
    collect_logs "${recipe}"
    exit 1
  fi
  if ! python3 "${CONFORMANCE}" --recipes-root "${RECIPES_ROOT}" \
    --output-dir "${REPORT_ROOT}" eval --recipe "${recipe}" \
    --runtime-config "${config}" --router-url "${ROUTER_URL}"; then
    collect_logs "${recipe}"
    exit 1
  fi
  collect_logs "${recipe}"
done
