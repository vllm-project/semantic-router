#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE_NAME="response-api"
SUITE_DIR="${REPO_ROOT}/response-api-artifacts"
EXIT_CODES_FILE="${SUITE_DIR}/exit-codes.txt"
AGGREGATED_REPORT_FILE="${SUITE_DIR}/suite-test-report.md"
AGGREGATED_LOG_FILE="${SUITE_DIR}/suite-semantic-router-logs.txt"
PROFILES=(
  "response-api"
  "response-api-redis"
  "response-api-redis-cluster"
)

run_profile() {
  local profile="$1"
  local artifact_dir="${SUITE_DIR}/${profile}"
  local test_exit_code=0

  rm -f "${REPO_ROOT}/test-report.json" "${REPO_ROOT}/test-report.md" "${REPO_ROOT}/semantic-router-logs.txt"

  set +e
  (
    cd "${REPO_ROOT}"
    make e2e-test E2E_PROFILE="${profile}"
  )
  test_exit_code=$?
  set -e

  mkdir -p "${artifact_dir}"
  for file in test-report.json test-report.md semantic-router-logs.txt; do
    if [[ -f "${REPO_ROOT}/${file}" ]]; then
      mv "${REPO_ROOT}/${file}" "${artifact_dir}/${file}"
    fi
  done

  if [[ -f "${artifact_dir}/test-report.md" ]]; then
    {
      printf '## %s\n\n' "${profile}"
      cat "${artifact_dir}/test-report.md"
      printf '\n'
    } >> "${AGGREGATED_REPORT_FILE}"
  fi

  if [[ -f "${artifact_dir}/semantic-router-logs.txt" ]]; then
    {
      printf '===== %s =====\n' "${profile}"
      cat "${artifact_dir}/semantic-router-logs.txt"
      printf '\n'
    } >> "${AGGREGATED_LOG_FILE}"
  fi

  printf '%s=%s\n' "${profile}" "${test_exit_code}" >> "${EXIT_CODES_FILE}"
  return "${test_exit_code}"
}

write_suite_report() {
  python3 - "${EXIT_CODES_FILE}" "${REPO_ROOT}/test-report.json" "${SUITE_NAME}" <<'PY'
import json
import sys
from pathlib import Path

exit_codes_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
suite_name = sys.argv[3]
exit_codes = {}

if exit_codes_path.exists():
    for line in exit_codes_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        exit_codes[name] = int(value)

report_path.write_text(
    json.dumps({"suite": suite_name, "profiles": exit_codes}, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

main() {
  local profile=""
  local suite_exit_code=0

  rm -rf "${SUITE_DIR}"
  mkdir -p "${SUITE_DIR}"
  : > "${EXIT_CODES_FILE}"
  : > "${AGGREGATED_REPORT_FILE}"
  : > "${AGGREGATED_LOG_FILE}"

  for profile in "${PROFILES[@]}"; do
    echo "::group::Run ${profile}"
    if ! run_profile "${profile}"; then
      suite_exit_code=1
    fi
    echo "::endgroup::"
  done

  write_suite_report
  cp "${AGGREGATED_REPORT_FILE}" "${REPO_ROOT}/test-report.md"
  cp "${AGGREGATED_LOG_FILE}" "${REPO_ROOT}/semantic-router-logs.txt"
  exit "${suite_exit_code}"
}

main "$@"
