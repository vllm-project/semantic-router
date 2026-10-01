#!/usr/bin/env bash
# Wait until one of this node's candidate GPUs has no lease owner, then run a command
# with every "{gpu}" argument replaced by that GPU (e.g. gpu3). The command takes the
# lease itself. Usage: wait_and_run1.sh "<indices>" <poll-seconds> <command> [args...]
set -euo pipefail
candidates="$1"; poll="$2"; shift 2
while true; do
  for index in $candidates; do
    if [[ ! -e "/data/dev2/leases/gpu${index}.lock/owner" ]]; then
      args=()
      for arg in "$@"; do args+=("${arg//\{gpu\}/gpu$index}"); done
      echo "$(date -u +%FT%TZ) gpu$index is free; running"
      exec "${args[@]}"
    fi
  done
  sleep "$poll"
done
