#!/usr/bin/env bash
# Start the linux/riscv64 router under qemu-user and prove /health plus one classify.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${REPO_ROOT}"
export VLLM_SR_CONFIG_BASE_DIR="${REPO_ROOT}"

ROUTER_BIN=${RISCV_ROUTER_BIN:?}
CONFIG=${RISCV_ROUTER_CONFIG:?}
QEMU=${RISCV_QEMU:?}
SYSROOT=${RISCV_SYSROOT:?}
LIBDIR=${RISCV_CANDLE_LIBDIR:?}
API_PORT=${RISCV_ROUTER_API_PORT:-18080}
EXTPROC_PORT=${RISCV_ROUTER_EXTPROC_PORT:-15051}
METRICS_PORT=${RISCV_ROUTER_METRICS_PORT:-19190}
HEALTH_TIMEOUT=${RISCV_ROUTER_HEALTH_TIMEOUT:-90}
READY_TIMEOUT=${RISCV_ROUTER_READY_TIMEOUT:-1200}
CLASSIFY_TIMEOUT=${RISCV_ROUTER_CLASSIFY_TIMEOUT:-600}
API_URL="http://127.0.0.1:${API_PORT}"
REPORT_DIR=${MODEL_TEST_REPORT_DIR:?}
MODEL_MANIFEST=${MODEL_TEST_MANIFEST:-${REPORT_DIR}/models.json}
mkdir -p "${REPORT_DIR}"
python3 - "${CONFIG}" "${MODEL_MANIFEST}" "${REPORT_DIR}/router-config.yaml" <<'PYCONFIG'
import json
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
manifest = json.loads(Path(sys.argv[2]).read_text())
assert manifest["provider"] == "candle" and len(manifest["models"]) == 1
model = manifest["models"][0]
assert model["name"] == "Domain"
domain = config["global"]["model_catalog"]["modules"]["classifier"]["domain"]
domain["model_id"] = model["path"]
domain["category_mapping_path"] = str(Path(model["path"]) / "category_mapping.json")
Path(sys.argv[3]).write_text(yaml.safe_dump(config, sort_keys=False))
PYCONFIG
CONFIG="${REPORT_DIR}/router-config.yaml"

log=$(mktemp)
pid=""
cleanup() {
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  echo "----- router log -----"
  cat "${log}"
  rm -f "${log}"
}
trap cleanup EXIT

export LD_LIBRARY_PATH="${LIBDIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Starting RISC-V router under qemu-user on api-port ${API_PORT}"
"${QEMU}" -L "${SYSROOT}" "${ROUTER_BIN}" \
  -config="${CONFIG}" \
  -port="${EXTPROC_PORT}" \
  -api-port="${API_PORT}" \
  -metrics-port="${METRICS_PORT}" \
  -enable-api=true \
  >"${log}" 2>&1 &
pid=$!

wait_http() {
  local path="$1"
  local seconds="$2"
  local elapsed=0
  while (( elapsed < seconds )); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "router process exited before ${path}" >&2
      return 1
    fi
    if curl -sf --max-time 5 -o "${REPORT_DIR}${path}.body" \
      -w '%{http_code}' "${API_URL}${path}" >"${REPORT_DIR}${path}.status"; then
      echo "${path} ready after ${elapsed}s"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "timed out waiting for ${path} after ${seconds}s" >&2
  return 1
}

wait_http /health "${HEALTH_TIMEOUT}"
wait_http /ready "${READY_TIMEOUT}"

echo "POST /api/v1/diagnostics/classify/intent"
status=$(curl -sf --max-time "${CLASSIFY_TIMEOUT}" \
  -o "${REPORT_DIR}/classify.body" -w '%{http_code}' \
  -H "Content-Type: application/json" \
  -d '{"text":"What is photosynthesis?"}' \
  "${API_URL}/api/v1/diagnostics/classify/intent")
resp=$(cat "${REPORT_DIR}/classify.body")
echo "${resp}"

preview_status=$(curl -sf --max-time "${CLASSIFY_TIMEOUT}" \
  -o "${REPORT_DIR}/preview.body" -w '%{http_code}' \
  -H "Content-Type: application/json" \
  -d '{"text":"What is photosynthesis?","model":"vllm-sr/auto"}' \
  "${API_URL}/api/v1/routing/preview")

python3 - "${resp}" "${REPORT_DIR}" "${status}" "${preview_status}" <<'PY'
import json
import sys
import subprocess
from pathlib import Path

data = json.loads(sys.argv[1])
category = (data.get("classification") or {}).get("category") or ""
decision = data.get("routing_decision") or ""
if not category or decision == "placeholder_response" or data.get("signal_errors"):
    raise SystemExit(
        f"classify did not use the domain classifier: category={category!r} routing_decision={decision!r}"
    )
print(f"classified category={category}")
directory = Path(sys.argv[2])
responses = [
    {"path": path, "http_status": int((directory / (path + ".status")).read_text()),
     "body": (directory / (path + ".body")).read_text()}
    for path in ("health", "ready")
]
responses.append({"path": "classify", "http_status": int(sys.argv[3]), "body": data})
responses.append({"path": "preview", "http_status": int(sys.argv[4]), "body": json.loads((directory / "preview.body").read_text())})
report = {
    "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "responses": responses,
}
(directory / "router.json").write_text(json.dumps(report, indent=2) + "\n")
sys.path.insert(0, "tools/ci")
from riscv_evidence import router_cases
router_cases(report, report["source_sha"])
PY
