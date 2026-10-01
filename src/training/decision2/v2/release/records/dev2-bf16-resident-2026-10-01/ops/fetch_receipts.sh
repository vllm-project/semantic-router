#!/usr/bin/env bash
# Copy one release work directory's receipts, post-check receipts and card text into the records (workstation).
# Usage: bash fetch_receipts.sh <node alias> <work dir> <records subdir, e.g. 0p6b/release>
# shellcheck disable=SC2029  # node paths are expanded on the workstation on purpose
set -euo pipefail
alias=$1 work=$2 dest=$3
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
host=$(grep "^$alias=" "$nodes" | cut -d= -f2-)
R=$(cd "$(dirname "$0")/.." && pwd)
name=$(ssh "$host" "ls $work/package | grep -v '\.build$' | head -1")
mkdir -p "$R/$dest/package-text"
ssh "$host" "cd $work && tar -c receipts logs/release.log \$(ls -d extra RELEASE-RECEIPT.json 2>/dev/null)" \
  | tar -x -C "$R/$dest"
ssh "$host" "cd $work/package/$name && tar -c README.md MODEL_MANIFEST.json config.json" | tar -x -C "$R/$dest/package-text"
echo "$dest: $(find "$R/$dest" -type f | wc -l) files"
