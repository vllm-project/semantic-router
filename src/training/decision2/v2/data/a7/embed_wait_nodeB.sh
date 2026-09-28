#!/usr/bin/env bash
# Wait until node B GPU7 is released by its owner and idle, then run embed_nodeB.sh once.
#
# Usage: embed_wait_nodeB.sh <deadline-utc> <out-dir> <protected-manifest> <candidates.jsonl>...
# Polls every 60 s; gives up at the deadline (e.g. 2026-09-28T16:00:00Z) without touching
# the lease. The eligibility checks (owner file says released/idle, no load) are repeated
# inside embed_nodeB.sh, which also writes and restores the lease owner file.
set -euo pipefail

deadline="$(date -u -d "${1:?deadline}" +%s)"
shift
here="$(cd "$(dirname "$0")" && pwd)"
while true; do
  if grep -q -E '"status" *: *"[^"]*(released|idle)[^"]*"|^status=[^ ]*(released|idle)' /data/dev2/leases/gpu7.lock/owner &&
    [[ "$(rocm-smi -d 7 --showuse --showmemuse --json | python3 -c '
import json, sys
card = next(iter(json.load(sys.stdin).values()))
print(card.get("GPU use (%)"), card.get("GPU Memory Allocated (VRAM%)"))')" == "0 0" ]]; then
    echo "$(date -u +%FT%TZ) GPU7 released and idle; starting the scan"
    exec bash "$here/embed_nodeB.sh" "$@"
  fi
  if (( $(date -u +%s) >= deadline )); then
    echo "$(date -u +%FT%TZ) deadline reached; GPU7 still leased or busy" >&2
    exit 3
  fi
  sleep 60
done
