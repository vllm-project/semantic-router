#!/usr/bin/env bash
# Replicate the small A20r release and C1 inputs from node B to node A at the same paths (workstation relay:
# tar | xz on node B, xz -d | tar on node A, as event3-stage-nodeA.sh does). Weights never take this path; the
# checkpoint moves through the staging repository (stage_a20r.py). Existing destinations are left as they are.
# Every path is then compared file by file (SHA-256 lists from both nodes) and the lists are written to OUT_DIR.
# Usage: relay_inputs.sh OUT_DIR PATH...   (absolute node-B paths under /data/dev2/runs/27b)
# Node addresses come from ~/.config/decision2/nodes.env and are never printed.
set -euo pipefail

out=$1
shift
mkdir -p "$out"
nodes="$HOME/.config/decision2/nodes.env"
NA=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
NB=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
ssh_a() { ssh -o BatchMode=yes -o ServerAliveInterval=30 "$NA" "$@"; }
ssh_b() { ssh -o BatchMode=yes -o ServerAliveInterval=30 "$NB" "$@"; }
lists() { printf 'cd %q && find %q -type f -print0 | sort -z | xargs -0 -r sha256sum' "$(dirname "$1")" "$(basename "$1")"; }

status=0
for path in "$@"; do
  case $path in /data/dev2/runs/27b/*) ;; *) echo "refused: $path is not under /data/dev2/runs/27b" >&2; exit 2 ;; esac
  q=$(printf %q "$path") parent=$(printf %q "$(dirname "$path")") base=$(printf %q "$(basename "$path")")
  if ssh_a "test -e $q"; then
    echo "$path: exists on node A (left as is)"
  else
    echo "$path: node B -> node A"
    ssh_b "set -o pipefail; tar -C $parent -cf - $base | xz -T0 -6" |
      ssh_a "set -euo pipefail; mkdir -p $parent; t=\$(mktemp -d $parent/.relay.XXXXXX); xz -d | tar -C \"\$t\" -xf -; mv \"\$t\"/$base $q; rmdir \"\$t\""
  fi
  name=$(echo "${path#/data/dev2/runs/27b/}" | tr '/' '_')
  ssh_b "$(lists "$path")" >"$out/$name.nodeB.sha256"
  ssh_a "$(lists "$path")" >"$out/$name.nodeA.sha256"
  if cmp -s "$out/$name.nodeB.sha256" "$out/$name.nodeA.sha256"; then
    echo "$path: $(wc -l <"$out/$name.nodeA.sha256") files, SHA-256 lists equal"
  else
    echo "$path: SHA-256 lists DIFFER" >&2
    status=1
  fi
done
exit $status
