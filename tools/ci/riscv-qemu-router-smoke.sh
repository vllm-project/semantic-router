#!/usr/bin/env bash
# Run the pure-Go linux/riscv64 router under qemu-user with a model runtime on
# the host attached, and prove /health, /ready and one Domain classification
# that the runtime serves through the router.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"

ROUTER_BIN=${RISCV_ROUTER_BIN:?}
QEMU=${RISCV_QEMU:?}
read -r -a RUNTIME <<<"${VLLM_SR_RUNTIME_COMMAND:-vllm-sr-runtime}"
REPORT_DIR=${MODEL_TEST_REPORT_DIR:?}
RUNTIME_PORT=${RISCV_RUNTIME_PORT:-18100}
API_PORT=${RISCV_ROUTER_API_PORT:-18080}
EXTPROC_PORT=${RISCV_ROUTER_EXTPROC_PORT:-15051}
METRICS_PORT=${RISCV_ROUTER_METRICS_PORT:-19190}
READY_TIMEOUT=${RISCV_ROUTER_READY_TIMEOUT:-600}
RUNTIME_URL="http://127.0.0.1:${RUNTIME_PORT}"
API_URL="http://127.0.0.1:${API_PORT}"
DEPLOYMENT=riscv-domain
PROMPT="What is photosynthesis?"
mkdir -p "${REPORT_DIR}"

work=$(mktemp -d)
pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  for log in runtime router; do
    echo "----- ${log} log -----"
    cat "${REPORT_DIR}/${log}.log" 2>/dev/null || true
  done
  rm -rf "${work}"
}
trap cleanup EXIT

wait_http() {
  local name="$1" url="$2" seconds="$3" pid="$4" elapsed=0
  while ((elapsed < seconds)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before ${url} answered" >&2
      return 1
    fi
    if curl -sf --max-time 5 -o "${REPORT_DIR}/${name}.body" \
      -w '%{http_code}' "${url}" >"${REPORT_DIR}/${name}.status"; then
      echo "${name} ready after ${elapsed}s"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "timed out waiting for ${url} after ${seconds}s" >&2
  return 1
}

# A tiny random-weight Domain classifier, served on the host CPU.
"${RUNTIME[@]}" fixture "${work}/domain" --family task_heads --variant sequence
"${RUNTIME[@]}" serve "${work}/domain" --served-model-name "${DEPLOYMENT}" --device cpu \
  --host 127.0.0.1 --port "${RUNTIME_PORT}" >"${REPORT_DIR}/runtime.log" 2>&1 &
pids+=("$!")
wait_http runtime-health "${RUNTIME_URL}/health" 300 "${pids[0]}"
curl -sf --max-time 60 -o "${REPORT_DIR}/runtime-classify.body" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${DEPLOYMENT}\",\"input\":\"${PROMPT}\"}" \
  "${RUNTIME_URL}/v1/classify"

cat >"${REPORT_DIR}/router-config.yaml" <<EOF
version: v0.3
listeners:
  - name: riscv-qemu-http
    address: 0.0.0.0
    port: 18888
    timeout: 60s
providers:
  defaults:
    model: riscv-model
  models:
    - name: riscv-model
      provider_model_id: riscv-model
      backend_refs:
        - name: primary
          provider: vllm
          weight: 100
          endpoint: 127.0.0.1:8000/v1
          protocol: http
routing:
  modelCards:
    - name: riscv-model
      modality: text
  model_bindings:
    domain_classifier:
      deployment: ${DEPLOYMENT}
      contract: label_distribution.v1
  signals:
    domains:
      - name: biology
        description: Biology prompts.
        mmlu_categories: [biology]
      - name: other
        description: Everything else.
        mmlu_categories: [other]
  decisions:
    - name: default-route
      description: Fallback route.
      priority: 10
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: riscv-model
          use_reasoning: false
global:
  router:
    model_selection:
      enabled: false
  services:
    response_api:
      enabled: false
    router_replay:
      enabled: false
    startup_status:
      store_backend: file
    observability:
      tracing:
        enabled: false
  stores:
    response_cache:
      enabled: false
  model_catalog:
    deployments:
      ${DEPLOYMENT}:
        provider: model_runtime
        endpoint: ${RUNTIME_URL}
        device: cpu
EOF

echo "Starting the linux/riscv64 router under qemu-user"
"${QEMU}" "${ROUTER_BIN}" \
  -config="${REPORT_DIR}/router-config.yaml" \
  -port="${EXTPROC_PORT}" \
  -api-port="${API_PORT}" \
  -metrics-port="${METRICS_PORT}" \
  -enable-api=true \
  >"${REPORT_DIR}/router.log" 2>&1 &
pids+=("$!")
wait_http health "${API_URL}/health" 120 "${pids[1]}"
wait_http ready "${API_URL}/ready" "${READY_TIMEOUT}" "${pids[1]}"
classify_status=$(curl -sf --max-time 120 -o "${REPORT_DIR}/classify.body" -w '%{http_code}' \
  -H "Content-Type: application/json" -d "{\"text\":\"${PROMPT}\"}" \
  "${API_URL}/api/v1/diagnostics/classify/intent")

python3 - "${REPORT_DIR}" "${classify_status}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

directory = Path(sys.argv[1])
responses = [
    {
        "path": name,
        "http_status": int((directory / f"{name}.status").read_text()),
        "body": (directory / f"{name}.body").read_text(),
    }
    for name in ("health", "ready")
]
responses.append(
    {
        "path": "classify",
        "http_status": int(sys.argv[2]),
        "body": json.loads((directory / "classify.body").read_text()),
    }
)
report = {
    "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "runtime": json.loads((directory / "runtime-classify.body").read_text()),
    "responses": responses,
}
(directory / "router.json").write_text(json.dumps(report, indent=2) + "\n")
sys.path.insert(0, "tools/ci")
from riscv_evidence import router_cases

router_cases(report, report["source_sha"])
PY
