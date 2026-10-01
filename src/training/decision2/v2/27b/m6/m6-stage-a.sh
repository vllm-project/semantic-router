#!/usr/bin/env bash
# ~27B M6: stage one sealed formal finalist on node A for item 8 and the release (workstation side; runbook
# m6-handoff-2026-10-01.md §2). Over the M6 node link (node B -> node A, rrsync root /data/dev2/xfer/27b-m6) the
# finalist's m6/ARM/{checkpoint, package, ADOPTION.json, adopt, cal698, soup, formal (output/, triton-cache/, the
# paired comparisons)}, m6/readouts/ARM and the whole m6/gates directory (items 1-7, beats-AutoJev, mlx pairing,
# overlap exposure, verdicts, contrasts) land in stage/ and are moved on node A to the same paths under
# /data/dev2/runs/27b/m6/; node A's own m6/mlx-diag/ARM (scored by m6-mlx-watch.sh) is already there. The SHA-256 lists
# of both nodes must match. Run it after every chain has settled and before m6-link.sh remove.
# Usage: m6-stage-a.sh ARM
set -euo pipefail
ARM=${1:?ARM}
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) B=$(grep '^node-b=' "$NODES" | cut -d= -f2-)
onb() { ssh -o BatchMode=yes "$B" "$@"; }
ona() { ssh -o BatchMode=yes "$A" "$@"; }
R=/data/dev2/runs/27b/m6 X=/data/dev2/xfer/27b-m6 KEY=/data/dev2/tmp/27b-m6-xfer
ITEMS="$ARM/checkpoint $ARM/package $ARM/ADOPTION.json $ARM/adopt $ARM/cal698 $ARM/soup $ARM/formal readouts/$ARM gates"
onb "test -f $R/$ARM/formal/SEAL.json && test -f $R/$ARM/package/PACKAGE.json && test -f $R/$ARM/checkpoint/soup_manifest.json" ||
  { echo "$ARM has no sealed formal run, frozen package or soup on node B" >&2; exit 3; }
onb "test -f $KEY/id_ed25519 && test -f $KEY/peer" || { echo "the M6 node link is gone (m6-link.sh setup)" >&2; exit 3; }
ona "test ! -e $R/$ARM && test ! -e $X/stage" || { echo "node A already has $R/$ARM or $X/stage" >&2; exit 3; }
ona "test -f $R/mlx-diag/$ARM/mlx-diag.score.json" || echo "warning: node A has no scored mlx-diag collection for $ARM" >&2
sums() { echo "cd $R && for i in $ITEMS; do [ -e \$i ] && find \$i -type f; done | LC_ALL=C sort | xargs -P 8 -n 16 sha256sum | sort -k2"; }
echo "$(date -u +%FT%TZ) $ARM: node B -> node A over the M6 link"
onb "set -e; cd $R; X=\"ssh -i $KEY/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$KEY/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes\"; \
  P=root@\$(cat $KEY/peer); for i in $ITEMS; do [ -e \$i ] || continue; rsync -a --mkpath -e \"\$X\" ./\$i \$P:stage/\$(dirname \$i)/; done"
ona "set -e; mkdir -p $R; rsync -a --remove-source-files $X/stage/ $R/; find $X/stage -depth -type d -empty -delete"
b=$(onb "$(sums)") a=$(ona "$(sums)")
[ -n "$b" ] && [ "$b" = "$a" ] || { echo "node A copy differs from node B's" >&2; exit 3; }
echo "$ARM staged on node A: $(wc -l <<< "$b") files under $R, SHA-256 lists equal"
