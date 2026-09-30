#!/usr/bin/env bash
# ~27B M4 node A staging (run on the workstation, since the nodes do not reach each other). Streams node B's
# frozen M4 training files (a20, ar) into node A's /data/dev2/private/27b/m4-data/mixtures-m4-1 and the frozen
# training-cache seed T0 (a verified copy of M3-A-s1's training cache) to the same path on node A, then checks
# every file's SHA-256 and T0's tree hash on both sides. Files already on node A are kept only if they hash equal.
# Streams are gzip-compressed: the workstation link runs at about 0.1-0.2 MB/s.
# Usage: m4-stage-nodeA.sh SRCDIR   (SRCDIR: a directory under node B's /data/dev2/private/27b/m4-data holding
#                                    the byte-identical build, e.g. mixtures-m4-1)
set -euo pipefail
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
SRCDIR=${1:?SRCDIR}
[[ "$SRCDIR" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad SRCDIR $SRCDIR" >&2; exit 2; }
FROM=/data/dev2/private/27b/m4-data/$SRCDIR
TO=/data/dev2/private/27b/m4-data/mixtures-m4-1
T0=/data/dev2/runs/27b/m4-train-cache-T0
T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
FILES="a20.train.jsonl ar.train.jsonl"
TREE='find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64'
on_a() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
on_b() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$B" "$@"; }

hb=$(on_b "cd '$FROM' && sha256sum $FILES")
on_a "mkdir -p '$TO' && chmod 700 /data/dev2/private/27b /data/dev2/private/27b/m4-data '$TO'"
if [ "$(on_a "cd '$TO' && sha256sum $FILES 2>/dev/null" || true)" != "$hb" ]; then
  on_a "cd '$TO' && rm -f $FILES"
  on_b "tar -C '$FROM' -czf - $FILES" | on_a "tar -C '$TO' -xzf - && cd '$TO' && chmod 600 $FILES"
fi
[ "$(on_a "cd '$TO' && sha256sum $FILES")" = "$hb" ] || { echo "relayed mixtures differ" >&2; exit 1; }
echo "mixtures on node A:"
echo "$hb"

[ "$(on_b "cd '$T0' && $TREE")" = "$T0_SHA" ] || { echo "node B T0 does not hash to $T0_SHA" >&2; exit 1; }
if [ "$(on_a "cd '$T0' 2>/dev/null && $TREE" || true)" != "$T0_SHA" ]; then
  on_a "test ! -e '$T0'" || { echo "node A $T0 exists with another tree hash" >&2; exit 1; }
  on_b "tar -C '$(dirname "$T0")' -czf - '$(basename "$T0")' '$(basename "$T0").copy.json'" \
    | on_a "tar -C '$(dirname "$T0")' -xzf - && chmod -R a-w '$T0'"
fi
[ "$(on_a "cd '$T0' && $TREE")" = "$T0_SHA" ] || { echo "node A T0 does not hash to $T0_SHA" >&2; exit 1; }
echo "T0 on node A: $T0_SHA"
