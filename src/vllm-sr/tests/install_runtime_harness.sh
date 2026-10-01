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
#   explicit-podman-both-ready   Docker + Podman present, --runtime podman
#                                -> Podman wins, Docker detection must not run.
#   skip                         --runtime skip -> no runtime.env written.
#   print-command-podman         --runtime podman with both stubs ready
#                                -> asserts printed restart/start commands
#                                   include `--runtime podman`.
#   first-launch-podman          --runtime podman with both stubs ready
#                                -> exercises the complete first-launch sequence
#                                   (serve + dashboard check) with a stubbed
#                                   launcher and asserts both invocations carry
#                                   `--runtime podman`.
#   print-dashboard-offset       --runtime docker with port offset 1000
#                                -> verifies the printed and opened Dashboard
#                                   URL and SSH tunnel use port 9700.

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
# broken/missing one (absent). Each invocation is logged to a call-trace
# file so the Python side can assert which runtime was (or was not) probed.
CALL_TRACE="$INSTALL_ROOT_TMP/calls.log"
write_stub() {
  local name="$1"
  local state="$2"
  local path="$STUB_BIN/$name"
  if [ "$state" = "ready" ]; then
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "%s" >> "%s"\nexit 0\n' "$name" "$CALL_TRACE" > "$path"
  else
    printf '#!/usr/bin/env bash\nprintf "%%s\\n" "%s" >> "%s"\nexit 1\n' "$name" "$CALL_TRACE" > "$path"
  fi
  chmod +x "$path"
}

# Stub for the installed `vllm-sr` launcher. It records its full argv on each
# invocation (one line per call) so the first-launch scenario can assert the
# exact arguments serve and dashboard received.
ARGV_TRACE="$INSTALL_ROOT_TMP/argv.log"
write_vllm_sr_stub() {
  local path="$STUB_BIN/vllm-sr"
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' "$ARGV_TRACE" > "$path"
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
  explicit-podman-both-ready)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="podman"
    ;;
  skip)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="skip"
    ;;
  print-command-podman)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="podman"
    ;;
  first-launch-podman)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="podman"
    ;;
  print-dashboard-offset)
    write_stub docker ready
    write_stub podman ready
    export VLLM_SR_RUNTIME="docker"
    export VLLM_SR_PORT_OFFSET="1000"
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
# Snapshot the call trace once ensure_runtime has finished. Only
# ensure_runtime's own probes are visible here; the print path below appends
# further entries that this line cannot observe.
if [ -f "$CALL_TRACE" ]; then
  printf 'CALLS=%s\n' "$(tr '\n' ',' < "$CALL_TRACE" | sed 's/,$//')"
else
  printf 'CALLS=\n'
fi
if [ -f "$INSTALL_ROOT_TMP/runtime.env" ]; then
  printf 'RUNTIME_ENV_FILE=present\n'
  cat "$INSTALL_ROOT_TMP/runtime.env"
else
  printf 'RUNTIME_ENV_FILE=absent\n'
fi

# For scenarios that need to verify printed commands, invoke the full
# installer print path. print_install_plan() is the only production caller of
# detect_existing_runtime(), so those probes land in CALL_TRACE after the
# CALLS= snapshot above and are only visible in the accumulated trace below.
if [ "$SCENARIO" = "print-command-podman" ]; then
  LAUNCH_PLATFORM=""
  printf '[PRINT_INSTALL_PLAN]\n'
  print_install_plan
  printf '[PRINT_RESTART_COMMAND]\n'
  print_restart_command
  printf '[PRINT_NEXT_STEPS]\n'
  AUTO_LAUNCH_RAN=0
  print_next_steps
fi

# Exercise the complete first-launch sequence (serve + dashboard availability
# check) through a stubbed launcher. Platform/dir resolution and the browser
# step are stubbed inert so the scenario stays deterministic.
if [ "$SCENARIO" = "first-launch-podman" ]; then
  # Route the installer's launcher lookups at the stub dir, and keep platform,
  # directory, and browser resolution inert so the sequence is deterministic.
  BIN_DIR="$STUB_BIN"
  write_vllm_sr_stub
  resolve_launch_platform() { printf '\n'; }
  resolve_launch_dir() { printf '%s\n' "$INSTALL_ROOT_TMP"; }
  open_dashboard_url() { return 0; }
  printf '[FIRST_LAUNCH]\n'
  launch_first_session
  printf '[FIRST_LAUNCH_ARGS]\n'
  if [ -f "$ARGV_TRACE" ]; then
    cat "$ARGV_TRACE"
  fi
fi

if [ "$SCENARIO" = "print-dashboard-offset" ]; then
  detect_primary_ip() { printf '192.0.2.10\n'; }
  detect_host_label() { printf 'fixture.example\n'; }
  is_remote_session() { return 0; }
  resolve_launch_platform() { printf '\n'; }
  BIN_DIR="$STUB_BIN"
  AUTO_LAUNCH_RAN=1
  LAUNCH_PLATFORM=""
  USER="fixture-user"

  # Record the URL passed to the browser without opening a real browser.
  OPEN_TRACE="$INSTALL_ROOT_TMP/opened-url.log"
  cat > "$STUB_BIN/xdg-open" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$1" >> "$OPEN_TRACE"
EOF
  chmod +x "$STUB_BIN/xdg-open"

  printf '[DASHBOARD_ACCESS]\n'
  print_dashboard_access
  printf '[NEXT_STEPS]\n'
  print_next_steps
  open_dashboard_url
  printf 'OPENED_URL=%s\n' "$(cat "$OPEN_TRACE")"
fi

# Accumulated trace for every scenario, taken after the print path has run.
# Scenarios that skip the print path report the same value as CALLS=.
if [ -f "$CALL_TRACE" ]; then
  printf 'CALLS_TOTAL=%s\n' "$(tr '\n' ',' < "$CALL_TRACE" | sed 's/,$//')"
else
  printf 'CALLS_TOTAL=\n'
fi
