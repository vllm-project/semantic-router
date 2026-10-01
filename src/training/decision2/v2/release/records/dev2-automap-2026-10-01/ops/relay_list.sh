#!/usr/bin/env bash
# Relay a list of node paths (one per line; files or directories) from this node to another at the same
# paths, then compare the SHA-256 list of every relayed file on both sides. Run on the source node.
# Usage: relay_list.sh <destination address> <list file>
set -euo pipefail
dest="${1:?destination}" listfile="${2:?list file}"
key=/root/.ssh/d2_temp_cd
mapfile -t paths < "$listfile"
for p in "${paths[@]}"; do
  [[ -e "$p" ]] || { echo "missing $p" >&2; exit 1; }
  ssh -i "$key" -o BatchMode=yes "root@$dest" "mkdir -p '$(dirname "$p")'"
  rsync -a -e "ssh -i $key -o BatchMode=yes" "$p" "root@$dest:$(dirname "$p")/"
done
list() { for p in "${paths[@]}"; do find "$p" -type f -print0; done | sort -z | xargs -0 sha256sum; }
here=$(list | sha256sum | cut -c1-64)
there=$(ssh -i "$key" -o BatchMode=yes "root@$dest" "$(declare -p paths); $(declare -f list); list" | sha256sum | cut -c1-64)
echo "paths=${#paths[@]} files=$(list | wc -l) sha256-list here=$here there=$there"
[[ "$here" == "$there" ]] || { echo "relay differs" >&2; exit 1; }
