#!/usr/bin/env bash
# Apply reviewed maintainer board actions from a prior sync workflow run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${ROOT}/tools/agent/scripts"
OUTPUT_DIR="${MAINTAINER_BOARD_OUTPUT_DIR:-${ROOT}/.agent-harness/maintainer}"
ACTIONS_FILE="${OUTPUT_DIR}/proposed-actions.json"
SOURCE_RUN_ID="${MAINTAINER_BOARD_SOURCE_RUN_ID:-}"

write_summary_header() {
  if [[ -z "${GITHUB_STEP_SUMMARY:-}" ]]; then
    return 0
  fi
  {
    echo "## Maintainer Board Apply"
    echo ""
    echo "$1"
    echo ""
  } >>"${GITHUB_STEP_SUMMARY}"
}

if [[ "${GITHUB_EVENT_NAME:-}" != "workflow_dispatch" ]]; then
  write_summary_header "Refusing to apply maintainer board actions outside \`workflow_dispatch\` (event=${GITHUB_EVENT_NAME:-local})."
  echo "Refusing to apply maintainer board actions outside workflow_dispatch (event=${GITHUB_EVENT_NAME:-local})" >&2
  exit 1
fi

if [[ -z "${SOURCE_RUN_ID}" ]]; then
  write_summary_header "Refusing to apply without \`source_run_id\`; download and review a prior sync artifact first."
  echo "Refusing to apply without MAINTAINER_BOARD_SOURCE_RUN_ID (reviewed sync run ID required)" >&2
  exit 1
fi

if [[ ! "${SOURCE_RUN_ID}" =~ ^[0-9]+$ ]]; then
  write_summary_header "Invalid \`source_run_id\`: \`${SOURCE_RUN_ID}\` (must be numeric)."
  echo "Invalid source_run_id: ${SOURCE_RUN_ID} (must be numeric)" >&2
  exit 1
fi

if [[ ! -f "${ACTIONS_FILE}" ]]; then
  write_summary_header "Missing proposed actions file: \`${ACTIONS_FILE}\`."
  echo "Missing proposed actions file: ${ACTIONS_FILE}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
export PYTHONPATH=.

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  python3 maintainer_board_ci_apply.py \
    --actions "${ACTIONS_FILE}" \
    --source-run-id "${SOURCE_RUN_ID}" \
    --summary "${GITHUB_STEP_SUMMARY}"
else
  python3 maintainer_board_ci_apply.py \
    --actions "${ACTIONS_FILE}" \
    --source-run-id "${SOURCE_RUN_ID}"
fi
exit_code=$?

exit "${exit_code}"
