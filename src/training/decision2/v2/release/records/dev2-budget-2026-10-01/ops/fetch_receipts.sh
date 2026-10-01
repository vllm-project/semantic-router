#!/usr/bin/env bash
# Copy the receipts of the newest node E budget work directories into this record (run locally from the worktree
# root): <key>/{stage,extra,release}/ receive receipts/*.json (and extra/*.json); answers stay on the node.
# Usage: fetch_receipts.sh <node-e address>
set -euo pipefail
node="${1:?node E address}"
R=src/training/decision2/v2/release/records/dev2-budget-2026-10-01
declare -A keys=([0.8B]=0p8b [2B]=2b [9B]=9b [27B]=27b)
for tier in 0.8B 2B 9B 27B; do
  for kind in stage extra release; do
    pattern="dev2-budget-$tier-$kind-*"
    [[ "$kind" == release ]] && pattern="dev2-budget-$tier-2026*"
    W=$(ssh -n -o BatchMode=yes "root@$node" "ls -d /data/dev2/runs/release/$pattern 2>/dev/null | tail -1" || true)
    [[ -n "$W" ]] || continue
    dest="$R/${keys[$tier]}/$kind"
    rm -rf "$dest"
    mkdir -p "$dest"
    rsync -a --include='*.json' --exclude='*' "root@$node:$W/receipts/" "$dest/receipts/"
    if ssh -n -o BatchMode=yes "root@$node" "test -d $W/extra"; then
      rsync -a --include='*.json' --exclude='*' "root@$node:$W/extra/" "$dest/extra/"
    fi
    echo "$tier $kind <- $(basename "$W")"
  done
done
