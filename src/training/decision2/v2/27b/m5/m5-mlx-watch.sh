#!/usr/bin/env bash
# ~27B M5 on node A (host side): wait until node B has pushed NAME's mlx-diag collection (m5-tail.sh mlx-push, then
# the marker /data/dev2/xfer/27b-m5/mlx/NAME.PUSHED, written after the push), then score and pair it with
# m5-mlx-nodeA.sh (its pairing JSON is left in the relay directory for node B's mlx-pull). mlx/NAME.SKIP (node B's
# chain made no formal run) ends the wait. Usage: m5-mlx-watch.sh MIRROR_SHA NAME. Detached, log on stdout.
set -euo pipefail
echo "m5 mlx watch $*: start $(date -u +%FT%TZ)"
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
NODEA=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/27b/m5/m5-mlx-nodeA.sh
[ -f "$NODEA" ] || { echo "missing mirror $SHA" >&2; exit 2; }
X=/data/dev2/xfer/27b-m5/mlx
while :; do
  if [ -f "$X/$NAME.SKIP" ]; then
    echo "$(date -u +%FT%TZ) no mlx-diag collection will come: $(cat "$X/$NAME.SKIP")"
    exit 0
  fi
  [ -f "$X/$NAME.PUSHED" ] && break
  sleep 300
done
bash "$NODEA" "$SHA" "$NAME"
echo "m5 mlx watch $NAME complete: $(date -u +%FT%TZ)"
