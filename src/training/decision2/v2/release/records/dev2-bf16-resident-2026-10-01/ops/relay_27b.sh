#!/usr/bin/env bash
# DEV2.0-27B inputs on node B for the BF16-resident rollout (workstation relay, as the A20r release did). The three
# small successor-gate evidence files the release spec names that exist only on node A are copied to the same paths
# on node B (never overwritten; SHA-256 equal on both nodes afterwards); the A20r checkpoint's SHA-256 list is
# compared across the nodes; the released revision is downloaded on node B with the real `hf download` into a fresh
# directory (the bench's old runtime) and its MODEL_MANIFEST.json checked against the released manifest.
# Usage (workstation): bash relay_27b.sh <mirror commit>
# shellcheck disable=SC2029  # node paths are expanded on the workstation on purpose
set -euo pipefail
sha=${1:?mirror commit}
nodes=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$nodes" | cut -d= -f2-)
B=$(grep '^node-b=' "$nodes" | cut -d= -f2-)
files=(
  /data/dev2/runs/27b/m4-mlx/M4-A20r-soup/mlx-diag.score.json
  /data/dev2/runs/27b/m4-mlx/mlx-paired-M4-A20r-soup-vs-DEV2.0-27B.json
  /data/dev2/runs/eval/c1-postkey/27B/dev2-27b-a20r-20260930T000642Z/SUMMARY.json
)
for f in "${files[@]}"; do
  a=$(ssh "$A" "sha256sum '$f'" | cut -c1-64)
  if ! ssh "$B" "test -e '$f'"; then
    ssh "$A" "cat '$f'" | ssh "$B" "mkdir -p '$(dirname "$f")' && cat > '$f.relay' && mv -n '$f.relay' '$f'"
  fi
  b=$(ssh "$B" "sha256sum '$f'" | cut -c1-64)
  [[ "$a" == "$b" ]] || { echo "differs on node B: $f" >&2; exit 1; }
  echo "equal $a $f"
done
ck=/data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint
list='find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64'
la=$(ssh "$A" "cd $ck && $list")
lb=$(ssh "$B" "cd $ck && $list")
[[ "$la" == "$lb" ]] || { echo "the A20r checkpoint differs between the nodes" >&2; exit 1; }
echo "checkpoint sha256 list equal on both nodes: $la"
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
dest=/data/dev2/runs/release/inputs/dev2-27b-a20r-download
ssh "$B" "set -e
if [ ! -e $dest/DEV2.0-27B ]; then
  mkdir -p $dest && cd $S && HF_HUB_CACHE=/data/dev2/hf-cache /data/dev2/tools/hf-cli/bin/python -m v2.release.hub download \
    --repo llm-semantic-router/DEV2.0-27B --revision 5323310327e52d4eadd119cd10accac9b106c97d \
    --dest $dest/DEV2.0-27B --output $dest/download.json
fi
test \"\$(sha256sum < $dest/DEV2.0-27B/MODEL_MANIFEST.json | cut -c1-64)\" = 82c71c2e232be71e842784cf8f305b1de7d81ff67a58fe11b4464f88b2e62d37
echo download ok $dest/DEV2.0-27B"
