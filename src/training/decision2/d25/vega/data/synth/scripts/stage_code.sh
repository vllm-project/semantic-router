#!/usr/bin/env bash
# Copy the d25 package (exact worktree subtree, no caches) to /data/d25/vega/src/<tag>/ on a node.
# Usage: stage_code.sh <ssh-target> [tag]   (prints the tag; pods use PYTHONPATH=/data/d25/vega/src/<tag>)
set -euo pipefail
target=${1:?ssh target}
here=$(cd "$(dirname "$0")/../../../../.." && pwd)   # src/training/decision2
tarball=$(mktemp /tmp/d25-src-XXXXXX.tgz)
tar -C "$here" --exclude='__pycache__' --exclude='*.pyc' -czf "$tarball" d25
tag=${2:-synth-$(date -u +%Y%m%d%H%M)-$(sha256sum "$tarball" | cut -c1-8)}
scp -q -o BatchMode=yes "$tarball" "$target:/tmp/$tag.tgz"
ssh -o BatchMode=yes "$target" "mkdir -p /data/d25/vega/src/$tag && tar -C /data/d25/vega/src/$tag -xzf /tmp/$tag.tgz && rm /tmp/$tag.tgz"
rm -f "$tarball"
echo "$tag"
