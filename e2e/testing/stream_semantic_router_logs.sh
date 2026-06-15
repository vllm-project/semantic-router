#!/usr/bin/env bash

set -uo pipefail

NAMESPACE="${SEMANTIC_ROUTER_LOG_NAMESPACE:-vllm-semantic-router-system}"
DEPLOYMENT="${SEMANTIC_ROUTER_LOG_DEPLOYMENT:-semantic-router}"
CONTAINER="${SEMANTIC_ROUTER_LOG_CONTAINER:-semantic-router}"
POLL_INTERVAL="${SEMANTIC_ROUTER_LOG_POLL_INTERVAL:-300}"
STATUS_INTERVAL="${SEMANTIC_ROUTER_LOG_STATUS_INTERVAL:-30}"

log() {
  printf '[router-live] %s\n' "$*"
}

print_status_snapshot() {
  log "pod snapshot for namespace ${NAMESPACE}:"
  kubectl get pods -n "${NAMESPACE}" -o wide 2>&1 | sed -u 's/^/[router-live] /' || true
  log "recent events for namespace ${NAMESPACE}:"
  kubectl get events -n "${NAMESPACE}" --sort-by=.lastTimestamp 2>&1 | tail -n 20 | sed -u 's/^/[router-live] /' || true
}

main() {
  local last_status_at=0
  local now=0
  local status=0

  log "waiting for deployment/${DEPLOYMENT} in namespace ${NAMESPACE}"

  while true; do
    now="$(date +%s)"

    if kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" >/dev/null 2>&1; then
      log "following logs from deployment/${DEPLOYMENT} container ${CONTAINER}"
      kubectl logs \
        -f "deployment/${DEPLOYMENT}" \
        -n "${NAMESPACE}" \
        -c "${CONTAINER}" \
        --timestamps=true \
        --tail=20 2>&1 | sed -u 's/^/[router-live] /'
      status=${PIPESTATUS[0]}
      log "log stream exited with status ${status}; retrying in ${POLL_INTERVAL}s"
      print_status_snapshot
      last_status_at="${now}"
    else
      if (( now - last_status_at >= STATUS_INTERVAL )); then
        log "deployment/${DEPLOYMENT} not created yet"
        print_status_snapshot
        last_status_at="${now}"
      fi
    fi

    sleep "${POLL_INTERVAL}"
  done
}

main "$@"
