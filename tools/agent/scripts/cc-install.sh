#!/usr/bin/env bash
# Session-isolated vllm-sr install wrapper for coding agents.
# Installs to a temp directory so the user's system is untouched.
set -euo pipefail

SESSION_TAG="${VLLM_SR_SESSION_TAG:-$$}"
SESSION_DIR="${TMPDIR:-/tmp}"
SESSION_DIR="${SESSION_DIR%/}/vllm-sr-session-${SESSION_TAG}"

trap '[ $? -ne 0 ] && rm -rf "$SESSION_DIR"' EXIT

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

has_flag() {
  local flag="$1"; shift
  for arg in "$@"; do
    [ "$arg" = "$flag" ] && return 0
  done
  return 1
}

extra_args=()

has_flag "--mode"         "$@" || extra_args+=(--mode cli)
has_flag "--runtime"      "$@" || extra_args+=(--runtime skip)
has_flag "--no-launch"    "$@" || extra_args+=(--no-launch)
has_flag "--install-root" "$@" || extra_args+=(--install-root "$SESSION_DIR")
has_flag "--bin-dir"      "$@" || extra_args+=(--bin-dir "$SESSION_DIR/bin")

echo "=> Session directory: $SESSION_DIR"
echo ""

bash "$REPO_ROOT/install.sh" "${extra_args[@]}" "$@"

echo ""
echo "=> To activate in this shell:"
echo ""
echo "   export PATH=\"$SESSION_DIR/bin:\$PATH\""
echo ""
echo "=> To clean up later:"
echo ""
echo "   rm -rf \"$SESSION_DIR\""
