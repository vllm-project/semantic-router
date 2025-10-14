#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config-quickstart.yaml"
ROUTER_BIN="${ROOT_DIR}/bin/router"
ROUTER_LOG_DIR="${SCRIPT_DIR}/logs"
ROUTER_LOG_FILE="${ROUTER_LOG_DIR}/router.log"
HEALTH_URL="${QUICKSTART_HEALTH_URL:-http://127.0.0.1:8801/health}"
REQUIRED_COMMANDS=(make go cargo rustc python3 curl)
SKIP_DOWNLOAD=false
SKIP_BUILD=false
SKIP_START=false
ROUTER_HEALTH_TIMEOUT=${QUICKSTART_HEALTH_TIMEOUT:-60}

usage() {
  cat <<'USAGE'
Usage: quickstart.sh [OPTIONS]

Options:
  --skip-download   Do not run `make download-models` (expects assets present)
  --skip-build      Do not run `make build`
  --skip-start      Skip starting the router (useful for smoke builds)
  -h, --help        Show this help message

Environment overrides:
  QUICKSTART_HEALTH_URL       Health probe URL (default http://127.0.0.1:8801/health)
  QUICKSTART_HEALTH_TIMEOUT   Seconds to wait for router health (default 60)
USAGE
}

log() {
  local level="$1"
  shift
  printf '[%s] %s\n' "$level" "$*"
}

die() {
  log "ERROR" "$*"
  exit 1
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --skip-download)
        SKIP_DOWNLOAD=true
        shift
        ;;
      --skip-build)
        SKIP_BUILD=true
        shift
        ;;
      --skip-start)
        SKIP_START=true
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

require_commands() {
  local missing=()
  for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    log "ERROR" "Missing required commands: ${missing[*]}"
    log "ERROR" "Install the missing tooling and re-run quickstart."
    exit 1
  fi
}

ensure_config_ready() {
  if [[ ! -f "$CONFIG_FILE" ]]; then
    die "Expected config at $CONFIG_FILE"
  fi

  if grep -q "TODO" "$CONFIG_FILE"; then
    log "WARN" "Config file contains TODO markers; router bootstrap may fail until populated."
  fi
}

run_make_target() {
  local target="$1"
  shift || true
  log "INFO" "Running make $target"
  (cd "$ROOT_DIR" && make "$target" "$@") || die "make $target failed"
}

build_assets() {
  if [[ "$SKIP_DOWNLOAD" != true ]]; then
    run_make_target download-models
  else
    log "INFO" "Skipping model download as requested"
  fi

  if [[ "$SKIP_BUILD" != true ]]; then
    run_make_target build
  else
    log "INFO" "Skipping build as requested"
  fi
}

wait_for_health() {
  local elapsed=0
  while (( elapsed < ROUTER_HEALTH_TIMEOUT )); do
    if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
      log "INFO" "Router is healthy at $HEALTH_URL"
      return 0
    fi
    sleep 1
    ((elapsed++))
  done

  log "ERROR" "Router failed health check within ${ROUTER_HEALTH_TIMEOUT}s"
  if [[ -f "$ROUTER_LOG_FILE" ]]; then
    log "INFO" "Tail of router log:"
    tail -n 40 "$ROUTER_LOG_FILE"
  fi
  return 1
}

cleanup_router() {
  if [[ -n "${ROUTER_PID:-}" ]] && kill -0 "$ROUTER_PID" >/dev/null 2>&1; then
    log "INFO" "Stopping router (PID $ROUTER_PID)"
    kill "$ROUTER_PID" >/dev/null 2>&1 || true
    wait "$ROUTER_PID" 2>/dev/null || true
  fi
}

start_router() {
  ensure_config_ready
  mkdir -p "$ROUTER_LOG_DIR"

  if [[ ! -x "$ROUTER_BIN" ]]; then
    die "Router binary not found at $ROUTER_BIN; run with --skip-start to build only"
  fi

  log "INFO" "Launching router with $CONFIG_FILE"
  local ld_path="${ROOT_DIR}/candle-binding/target/release"
  (
    cd "$ROOT_DIR" || exit 1
    export LD_LIBRARY_PATH="$ld_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH="$ld_path${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    exec "$ROUTER_BIN" -config="$CONFIG_FILE"
  ) >"$ROUTER_LOG_FILE" 2>&1 &

  ROUTER_PID=$!
  if [[ -z "${ROUTER_PID}" ]]; then
    die "Failed to spawn router"
  fi
  log "INFO" "Router PID: $ROUTER_PID (logs: $ROUTER_LOG_FILE)"

  if ! wait_for_health; then
    cleanup_router
    die "Router did not become healthy"
  fi

  log "INFO" "Router started successfully. Press Ctrl+C to stop."
  wait "$ROUTER_PID"
}

main() {
  parse_args "$@"
  require_commands
  build_assets

  if [[ "$SKIP_START" == true ]]; then
    log "INFO" "Skipping router startup; quickstart build complete"
    exit 0
  fi

  trap 'cleanup_router; exit 130' INT
  trap 'cleanup_router; exit 143' TERM
  trap 'cleanup_router' EXIT

  start_router
}

main "$@"
