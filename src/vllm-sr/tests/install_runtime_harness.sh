#!/usr/bin/env bash
# Harness: exercise install.sh runtime selection with stubbed Docker/Podman
# so the behavior tests can inspect the real code path and the persisted
# runtime.env. Invoked by test_install_runtime_behavior.py; not meant to be
# run by hand.
#
# Usage: install_runtime_harness.sh <scenario> <repo-root>
#
# Scenarios:
#   auto-both-ready              Docker + Podman present, --runtime auto
#                                -> Docker must win.
#   auto-podman-only             Only Podman present, --runtime auto
#                                -> Podman fallback must kick in.
#   explicit-docker-both-ready   Docker + Podman present, --runtime docker
#                                -> Docker wins, Podman branch must not run.
#   skip                         --runtime skip -> no runtime.env written.

set -u

SCENARIO="${1:-}"
REPO_ROOT="${2:-}"

if [ -z "$SCENARIO" ] || [ -z "$REPO_ROOT" ]; then
  printf 'usage: install_runtime_harness.sh <scenario> <repo-root>\n' >&2
  exit 2
fi

INSTALL_SH="$REPO_ROOT/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  printf 'install.sh not found at %s\n' "$INSTALL_SH" >&2
  exit 2
fi

STUB_BIN="$(mktemp -d)"
INSTALL_ROOT_TMP="$(mktemp -d)"
cleanup() {
  rm -rf "$STUB_BIN" "$INSTALL_ROOT_TMP"
}
trap cleanup EXIT

# Emit a stub binary that either reports a healthy daemon (ready) or a
# broken/missing one (absent). Only `info` is consulted by install.sh.
write_stub() {
  local name="$1"
  local state="$2"
  local path="$STUB_BIN/$name"
  if [ "$state" = "ready" ]; then
    printf '#!/usr/bin/env bash\nexit 0\n' > "$path"
  else
    printf '#!/usr/bin/env bash\nexit 1\n' > "$path"
  fi
  chmod +x "$path"
}

# Map scenario -> (docker state, podman state, VLLM_SR_RUNTIME env value).
# VLLM_SR_RUNTIME is used because install.sh reads it at source time to set
# REQUESTED_RUNTIME, so setting it before sourcing is the cleanest way to
# drive each branch without re-implementing argv parsing.
case "$SCENARIO" in
  auto-both-ready)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="auto"
    ;;
  auto-podman-only)
    write_stub docker absent
    write_stub podman ready
    export VLLM_SR_RUNTIME="auto"
    ;;
  explicit-docker-both-ready)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="docker"
    ;;
  skip)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="skip"
    ;;
  *)
    printf 'unknown scenario: %s\n' "$SCENARIO" >&2
    exit 2
    ;;
esac

export PATH="$STUB_BIN:$PATH"
export VLLM_SR_INSTALL_ROOT="$INSTALL_ROOT_TMP"

# Source install.sh with its main entrypoint stripped so the harness can
# call individual functions without triggering a real install.
sed '/^main /d' "$INSTALL_SH" > "$INSTALL_ROOT_TMP/install.sh.testable"
# shellcheck source=/dev/null
. "$INSTALL_ROOT_TMP/install.sh.testable"
# install.sh sets `set -euo pipefail` at the top; neutralize for the harness
# so a non-zero stub return does not abort before we can report results.
set +e +u 2>/dev/null || true
set +o pipefail 2>/dev/null || true

# Every scenario above short-circuits before any OS-specific install path,
# so pretend we are on Linux without invoking detect_os (which would die on
# unsupported platforms).
OS_NAME="linux"
MODE="serve"
SELECTED_RUNTIME="${REQUESTED_RUNTIME:-}"

ensure_runtime

# Report the selected runtime and the exact runtime.env contents so the
# Python side can assert both behavior and persisted state.
printf 'SELECTED_RUNTIME=%s\n' "$SELECTED_RUNTIME"
if [ -f "$INSTALL_ROOT_TMP/runtime.env" ]; then
  printf 'RUNTIME_ENV_FILE=present\n'
  cat "$INSTALL_ROOT_TMP/runtime.env"
else
  printf 'RUNTIME_ENV_FILE=absent\n'
fi
