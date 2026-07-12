#!/usr/bin/env bash
# Apply proposed maintainer board actions in CI (workflow_dispatch only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${ROOT}/tools/agent/scripts"
OUTPUT_DIR="${MAINTAINER_BOARD_OUTPUT_DIR:-${ROOT}/.agent-harness/maintainer}"
ACTIONS_FILE="${OUTPUT_DIR}/proposed-actions.json"

if [[ "${GITHUB_EVENT_NAME:-}" != "workflow_dispatch" ]]; then
  echo "Refusing to apply maintainer board actions outside workflow_dispatch (event=${GITHUB_EVENT_NAME:-local})" >&2
  exit 1
fi

if [[ ! -f "${ACTIONS_FILE}" ]]; then
  echo "Missing proposed actions file: ${ACTIONS_FILE}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
export PYTHONPATH=.

action_count="$(python3 - <<'PY' "${ACTIONS_FILE}"
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle)))
PY
)"

if [[ "${action_count}" == "0" ]]; then
  echo "No proposed actions to apply."
  exit 0
fi

echo "Applying ${action_count} proposed maintainer board action(s) from ${ACTIONS_FILE}"
python3 maintainer_board.py apply --actions "${ACTIONS_FILE}" --confirm

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "## Maintainer Board Apply"
    echo ""
    echo "Applied **${action_count}** action(s) from \`proposed-actions.json\`."
    echo ""
    echo "See workflow artifacts for the sync snapshot that produced this payload."
  } >>"${GITHUB_STEP_SUMMARY}"
fi
