#!/usr/bin/env bash

SR_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
cd "${SR_ROOT}"

# This script builds and runs a local router instance along with an Envoy
# proxy configured to route traffic to it. It is intended for development
# and testing purposes in a dockerless environment.

SR_LOG_LEVEL=${SR_LOG_LEVEL:-"info"}
# Dump config at debug level
if [[ "${SR_LOG_LEVEL}" == "debug" ]]; then
  set -x
fi

API_HOST_IP=${API_HOST_IP:-"127.0.0.1"} # Used for all components

# Router proxy (Envoy) settings
# TODO: generate the envoy config dynamically based on 
# the ROUTER_API_PORT and ENVOY_PROXY_PORT values
ENVOY_PROXY_PORT=${ENVOY_PROXY_PORT:-"8801"}
ENVOY_CONFIG=${ENVOY_CONFIG:-"config/envoy.yaml"}
ENVOY_VERSION=${ENVOY_VERSION:-"1.35.4"}

# Router settings
ROUTER_API_PORT=${ROUTER_API_PORT:-"8080"}
ROUTER_CONFIG=${ROUTER_CONFIG:-"config/config.yaml"}

# Script settings
FUNC_E_VERSION=${FUNC_E_VERSION:-"v1.3.0"}
LOG_DIR=${LOG_DIR:-"/tmp"}
WAIT_FOR_URL_ROUTER_SERVER=${WAIT_FOR_URL_ROUTER_SERVER:-60}
MAX_TIME_FOR_URL_ROUTER_SERVER=${MAX_TIME_FOR_URL_ROUTER_SERVER:-1}
set +x

# Stop right away if the build fails
set -e

function usage {
  echo "This script starts a local router instance along with an Envoy proxy."
  echo "Example 0: scripts/local-up-router.sh -h  (this 'help' usage description)"
  echo "Example 1: scripts/local-up-router.sh -o bin (run from make build)"
  echo "Example 2: scripts/local-up-router.sh (build a local copy of the source)"
  echo "Example 3: ROUTER_CONFIG=\"config/config.yaml\" \\"
  echo "           scripts/local-up-router.sh (build a local copy of the source with a specific config file)"
  echo ""
  echo "-d         dry-run: prepare for running commands, then show their command lines instead of running them"
}

### Allow user to supply the source directory.
BIN_OUT=${BIN_OUT:-}
DRY_RUN=
while getopts "dho:O" OPTION
do
    case ${OPTION} in
        d)
            echo "skipping running commands"
            DRY_RUN=1
            ;;
        o)
            echo "skipping build"
            BIN_OUT="${OPTARG}"
            echo "using source ${BIN_OUT}"
            ;;
        h)
            usage
            exit
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

if [ -z "${BIN_OUT}" ]; then
    BIN_OUT="${SR_ROOT}/bin"
    make -C "${SR_ROOT}" build
else
    echo "skipped the build because BIN_OUT was set (${BIN_OUT})"
fi

# run executes the command specified by its parameters if DRY_RUN is empty,
# otherwise it prints them.
#
# The first parameter must be the name of the semantic router components.
# It is only used when printing the command in dry-run mode.
# The second parameter is a log file for the command. It may be empty.
function run {
    local what="$1"
    local log="$2"
    shift
    shift
    if [[ -z "${DRY_RUN}" ]]; then
        if [[ -z "${log}" ]]; then
            "${@}"
        else
            "${@}" >"${log}" 2>&1
        fi
    else
        echo "RUN ${what}: ${*}"
    fi
}
# Downloads the models required by the router.
function download_models {
  echo "Downloading models..."
  run "download-models" "" "${BIN_OUT}/router" \
    --config "${ROUTER_CONFIG}" --download-only
}

function wait_for_url() {
  local url=$1
  local prefix=${2:-}
  local wait=${3:-1}
  local times=${4:-30}
  local maxtime=${5:-1}

  command -v curl >/dev/null || {
    error_log "curl must be installed"
    exit 1
  }

  local i
  for i in $(seq 1 "${times}"); do
    local out
    if out=$(curl --max-time "${maxtime}" -gkfs "${@:6}" "${url}" 2>/dev/null); then
      info_log "On try ${i}, ${prefix}: ${out}"
      return 0
    fi
    sleep "${wait}"
  done
  error_log "Timed out waiting for ${prefix} to answer at ${url}; tried ${times} waiting ${wait} between each"
  return 1
}


# Starts the router in the background, logging to ROUTER_LOG.
function start_router {
  echo "Starting router..."
  ROUTER_LOG=${LOG_DIR}/router.log
  run "router" "${ROUTER_LOG}" "${BIN_OUT}/router" \
    --api-port "${ROUTER_API_PORT}" \
    --config "${ROUTER_CONFIG}" &
  ROUTER_PID=$!

  if [[ -z "${DRY_RUN}" ]]; then
      # Wait for router to come up before launching the rest of the components.
      echo "Waiting for router to come up"
      wait_for_url "http://${API_HOST_IP}:${ROUTER_API_PORT}/health" "router: " 1 "${WAIT_FOR_URL_ROUTER_SERVER}" "${MAX_TIME_FOR_URL_ROUTER_SERVER}" \
          || { echo "check router logs: ${ROUTER_LOG}" ; exit 1 ; }
  fi
}

# Installs func-e into BIN_OUT if it is not already installed there.
function install_func_e_if_needed {
  echo "Checking func-e Installation at ${BIN_OUT}"
  if ! command -v "${BIN_OUT}/func-e" &> /dev/null ; then
    echo "func-e Installation not found at ${BIN_OUT}"
    install_func_e
  fi
}

# Installs func-e into BIN_OUT.
function install_func_e {
  echo "Installing func-e..."
  curl -sSL https://func-e.io/install.sh | bash -s -- -b "${BIN_OUT}" "${FUNC_E_VERSION}"
}

# Starts Envoy in the background, logging to ENVOY_LOG.
function start_envoy {
  echo "Starting Envoy..."
  run "func-e" "" "${BIN_OUT}/func-e" use "${ENVOY_VERSION}"
  ENVOY_LOG=${LOG_DIR}/envoy.log
  run "envoy" "${ENVOY_LOG}" "${BIN_OUT}/func-e" "run" \
    --config-path "${ENVOY_CONFIG}" \
    --component-log-level "ext_proc:trace,router:trace,http:trace" &
  ENVOY_PID=$!
}

# Prints success message with configuration details.
function print_success {
  if [[ -n "${DRY_RUN}" ]]; then
    return
  fi

  echo "The local semantic router is running. Press Ctrl-C to shut it down."

# TODO: add support for dashboard via make dashboard-build-backend and npm run dev
  cat <<EOF
Configurations:
  Router: ${ROUTER_CONFIG}
  Envoy: ${ENVOY_CONFIG}

Endpoints:
  * Router Endpoint: http://${API_HOST_IP}:${ROUTER_API_PORT}
  * Envoy Proxy Endpoint: http://${API_HOST_IP}:${ENVOY_PROXY_PORT}

Logs:
  ${ROUTER_LOG:-}
  ${ENVOY_LOG:-}
EOF
}

# Prints a colored message with an optional prefix.
function print_color {
  message=$1
  prefix=${2:+$2: } # add colon only if defined
  color=${3:-1}     # default is red
  echo -n "$(tput bold)$(tput setaf "${color}")"
  echo "${prefix}${message}"
  echo -n "$(tput sgr0)"
}

function info_log {
  print_color "$1" "I$(date "+%m%d %H:%M:%S")]" 2
}

# Prints a warning log message with timestamp.
function warning_log {
  print_color "$1" "W$(date "+%m%d %H:%M:%S")]" 3
}

# Prints an error log message with timestamp.
function error_log {
  print_color "$1" "E$(date "+%m%d %H:%M:%S")]" 1
}

# Check if all processes are still running. Prints a warning once each time
# a process dies unexpectedly.
function healthcheck {
  if [[ -n "${ROUTER_PID-}" ]] && ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
    warning_log "router terminated unexpectedly, see ${ROUTER_LOG}"
    ROUTER_PID=
  fi

  if [[ -n "${ENVOY_PID-}" ]] && ! kill -0 "${ENVOY_PID}" 2>/dev/null; then
    warning_log "envoy terminated unexpectedly, see ${ENVOY_LOG}"
    ENVOY_PID=
  fi
}

# Reads in stdin and adds it line by line to the array provided. This can be
# used instead of "mapfile -t", and is bash 3 compatible.  If the named array
# exists and is an array, it will be overwritten.  Otherwise it will be unset
# and recreated.
function read-array {
  if [[ -z "$1" ]]; then
    echo "usage: ${FUNCNAME[0]} <varname>" >&2
    return 1
  fi
  if [[ -n $(declare -p "$1" 2>/dev/null) ]]; then
    if ! declare -p "$1" 2>/dev/null | grep -q '^declare -a'; then
      echo "${FUNCNAME[0]}: $1 is defined but isn't an array" >&2
      return 2
    fi
  fi
  # shellcheck disable=SC2034 # this variable _is_ used
  local __read_array_i=0
  while IFS= read -r "$1[__read_array_i++]"; do :; done
  if ! eval "[[ \${$1[--__read_array_i]} ]]"; then
    unset "$1[__read_array_i]" # ensures last element isn't empty
  fi
}

# Cleans up background processes on exit.
function cleanup {
  echo "Cleaning up..."

  [[ -n "${ROUTER_PID-}" ]] && read-array ROUTER_PIDS < <(pgrep -P "${ROUTER_PID}" ; ps -o pid= -p "${ROUTER_PID}")
  [[ -n "${ROUTER_PIDS-}" ]] && kill "${ROUTER_PIDS[@]}" 2>/dev/null

  [[ -n "${ENVOY_PID-}" ]] && read-array ENVOY_PIDS < <(pgrep -P "${ENVOY_PID}" ; ps -o pid= -p "${ENVOY_PID}")
  [[ -n "${ENVOY_PIDS-}" ]] && kill "${ENVOY_PIDS[@]}" 2>/dev/null

  exit 0
}

trap cleanup EXIT
trap cleanup INT

# Main script execution
download_models
start_router
install_func_e_if_needed
start_envoy
print_success

# Health check loop
if [[ -z "${DRY_RUN}" ]]; then
  while true; do sleep 1; healthcheck; done
fi
