#!/usr/bin/env bash
# Generate the maintainer board in CI without mutating GitHub.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${ROOT}/tools/agent/scripts"
OUTPUT_DIR="${MAINTAINER_BOARD_OUTPUT_DIR:-${ROOT}/.agent-harness/maintainer}"
MILESTONE="${MAINTAINER_BOARD_MILESTONE:-}"
ISSUE_LIMIT="${MAINTAINER_BOARD_ISSUE_LIMIT:-}"
PR_LIMIT="${MAINTAINER_BOARD_PR_LIMIT:-}"

cd "${SCRIPT_DIR}"
export PYTHONPATH=.

args=(sync --output-dir "${OUTPUT_DIR}")
if [[ -n "${MILESTONE}" ]]; then
  args+=(--milestone "${MILESTONE}")
fi
if [[ -n "${ISSUE_LIMIT}" ]]; then
  args+=(--issue-limit "${ISSUE_LIMIT}")
fi
if [[ -n "${PR_LIMIT}" ]]; then
  args+=(--pr-limit "${PR_LIMIT}")
fi

python3 maintainer_board.py "${args[@]}"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -f "${OUTPUT_DIR}/today.md" ]]; then
  {
    echo "## Maintainer Board"
    echo ""
    cat "${OUTPUT_DIR}/today.md"
  } >>"${GITHUB_STEP_SUMMARY}"
fi

echo "Maintainer board artifacts written to ${OUTPUT_DIR}"
