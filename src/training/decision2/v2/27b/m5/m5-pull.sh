#!/usr/bin/env bash
# ~27B M5: pull one node A relay directory (m5-lane.sh: a BEST checkpoint hardlinked with its SHA-256 list) to node B
# over the private node link and check the list on node B (node B host side).
# Usage: m5-pull.sh ARM-SEED   -> /data/dev2/runs/27b/m5/relay/ARM-SEED/{checkpoint,SHA256SUMS,BEST.json,COMPLETE.json}
# The link's key, known_hosts and peer address stay in the private /data/dev2/tmp/27b-m5-xfer/ (m5-state.md); rsync
# paths are relative to node A's rrsync root /data/dev2/xfer/27b-m5.
set -euo pipefail
echo "m5 pull $*: start $(date -u +%FT%TZ)"
NAME=${1:?ARM-SEED}
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ARM-SEED must be one directory name" >&2; exit 2; }
KEY=/data/dev2/tmp/27b-m5-xfer
PEER=$(cat "$KEY/peer")
X="ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
DEST=/data/dev2/runs/27b/m5/relay/$NAME
mkdir -p "$DEST"
rsync -a -e "$X" "root@$PEER:relay/$NAME/" "$DEST/"
(cd "$DEST/checkpoint" && find . -type f | sort | xargs -P 16 -n 4 sha256sum | sort -k2) > "$DEST/SHA256SUMS.nodeB"
diff "$DEST/SHA256SUMS" "$DEST/SHA256SUMS.nodeB"
echo "m5 pull $NAME: $(wc -l < "$DEST/SHA256SUMS") files, SHA-256 lists equal ($(sha256sum "$DEST/SHA256SUMS" | cut -c1-12)): $(date -u +%FT%TZ)"
