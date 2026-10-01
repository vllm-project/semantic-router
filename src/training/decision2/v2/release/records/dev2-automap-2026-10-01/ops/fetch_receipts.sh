#!/usr/bin/env bash
# Copy the receipts of the newest node E work directories into this record (run locally from the worktree root):
# <key>/{verify,mlx,tf518,release}/ receive receipts/*.json (and extra/*.json); per-prompt answers stay on the node.
# Usage: fetch_receipts.sh <node-e address>
set -euo pipefail
node="${1:?node E address}"
R=src/training/decision2/v2/release/records/dev2-automap-2026-10-01
declare -A keys=([0.6B]=0p6b [0.8B]=0p8b [2B]=2b [4B]=4b [9B]=9b [27B]=27b)
for tier in 0.6B 0.8B 2B 4B 9B 27B; do
  for kind in verify mlx tf518 release; do
    pattern="dev2-automap-$tier-$kind-*"
    [[ "$kind" == release ]] && pattern="dev2-automap-$tier-2026*"
    W=$(ssh -n -o BatchMode=yes "root@$node" "ls -d /data/dev2/runs/release/$pattern 2>/dev/null | tail -1" || true)
    [[ -n "$W" ]] || continue
    dest="$R/${keys[$tier]}/$kind"
    mkdir -p "$dest"
    rsync -a --include='*.json' --exclude='*' "root@$node:$W/receipts/" "$dest/receipts/"
    ssh -n -o BatchMode=yes "root@$node" "test -d $W/extra" && \
      rsync -a --include='*.json' --exclude='*' "root@$node:$W/extra/" "$dest/extra/"
    echo "$tier $kind <- $(basename "$W")"
  done
done
