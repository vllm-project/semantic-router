#!/usr/bin/env bash
#
# Shared retry helper for network-bound CI steps.
#
# Source this file, then retry any command:
#
#   # shellcheck source=tools/ci/retry.sh
#   source tools/ci/retry.sh
#   retry_run 2 60 docker pull "${IMAGE}"
#
# Wrap a multi-line step in a function and pass its name:
#
#   create_cluster() {
#     kind create cluster --name "${CLUSTER}" --config=- <<'EOF'
#     kind: Cluster
#     apiVersion: kind.x-k8s.io/v1alpha4
#     EOF
#   }
#   retry_run 2 60 create_cluster
#
# The command runs as the condition of an `if`, where bash suspends errexit. A
# function with several commands returns the status of its last command only, so
# a failure in an earlier command would be reported as a success and skipped.
# Chain every step that must succeed with `&&`:
#
#   install_kubectl() {
#     curl -fsSLO "${URL}" && chmod +x kubectl && sudo mv kubectl /usr/local/bin
#   }
#
# Set RETRY_CLEANUP to the name of a function that runs before every retry, so a
# partially-created resource does not poison the next attempt:
#
#   delete_cluster() { kind delete cluster --name "${CLUSTER}" >/dev/null 2>&1 || true; }
#   RETRY_CLEANUP=delete_cluster retry_run 2 60 create_cluster
#
# This exists so retries stay in-repo instead of adding a third-party action.
# Contracts are covered by tools/ci/tests/test_retry_sh.py.

retry_run() {
  if (($# < 3)); then
    printf 'retry_run: usage: retry_run <attempts> <delay-seconds> <command> [args...]\n' >&2
    return 2
  fi

  # bash resolves a variable to the nearest binding in the dynamic scope, so a
  # local here is visible to the retried command. Prefix every internal name to
  # keep a workflow function that assigns `attempts`, `delay`, or `attempt` from
  # rewriting the loop.
  local __retry_run_attempts="$1"
  local __retry_run_delay="$2"
  shift 2

  if [[ ! "${__retry_run_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'retry_run: attempts must be a positive integer, got "%s"\n' "${__retry_run_attempts}" >&2
    return 2
  fi

  if [[ ! "${__retry_run_delay}" =~ ^[0-9]+$ ]]; then
    printf 'retry_run: delay must be a non-negative whole number of seconds, got "%s"\n' "${__retry_run_delay}" >&2
    return 2
  fi

  local __retry_run_attempt
  for ((__retry_run_attempt = 1;
    __retry_run_attempt <= __retry_run_attempts;
    __retry_run_attempt++)); do
    if "$@"; then
      if ((__retry_run_attempt > 1)); then
        printf 'retry_run: "%s" succeeded on attempt %d/%d\n' \
          "$1" "${__retry_run_attempt}" "${__retry_run_attempts}"
      fi
      return 0
    fi

    if ((__retry_run_attempt == __retry_run_attempts)); then
      printf '::error::"%s" failed after %d attempt(s)\n' "$1" "${__retry_run_attempts}" >&2
      return 1
    fi

    printf '::warning::"%s" failed on attempt %d/%d; retrying in %ss\n' \
      "$1" "${__retry_run_attempt}" "${__retry_run_attempts}" "${__retry_run_delay}" >&2

    if [[ -n "${RETRY_CLEANUP:-}" ]]; then
      if ! "${RETRY_CLEANUP}"; then
        printf '::warning::RETRY_CLEANUP "%s" failed; continuing to the retry\n' "${RETRY_CLEANUP}" >&2
      fi
    fi

    sleep "${__retry_run_delay}"
  done
}
