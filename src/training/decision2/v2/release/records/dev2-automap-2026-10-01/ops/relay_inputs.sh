#!/usr/bin/env bash
# Relay the auto_map parity inputs from node A to node E at the same paths (run on node A).
# Usage: relay_inputs.sh <node-e address> [--base]
#   copies the gold-free prompts of the four scored panels, the sealed predictions and frozen Triton autotune
#   caches the BF16-resident rollout's parity uses, and (--base) the pinned Qwen3.8-27B base snapshot; then
#   compares the SHA-256 list of every relayed file on both sides. Gold never leaves node A.
set -euo pipefail
dest="${1:?node E address}"
base="${2:-}"
key=/root/.ssh/d2_temp_cd
G=/data/dev2/private/panels/goldfree
paths=(
  "$G/typed-final.prompts.jsonl" "$G/css15.prompts.jsonl" "$G/public231.prompts.jsonl" "$G/mlx-diag.prompts.jsonl"
  /data/dev2/runs/06b/m8/formal/m8-s5-b05/output
  /data/dev2/runs/06b/m8/formal/m8-s5-b05-mlx/output
  /data/dev2/runs/release/inputs/dev2-0p8b-t1/derived
  /data/dev2/runs/release/inputs/dev2-2b-t1/derived
  /data/dev2/runs/release/inputs/dev2-4b-t1/derived
  /data/dev2/runs/release/inputs/dev2-8b-t1/derived
  /data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
  /data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
  /data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
  /data/dev2/runs/dec/formal/m4/m4-N4XF-soup-triton
  /data/dev2/runs/9b/formal-m4/triton-cache
  /data/dev2/runs/27b/M4-A20r-soup/formal/output
  /data/dev2/runs/27b/m4-mlx/M4-A20r-soup/output
  /data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache
)
[[ "$base" == --base ]] && paths+=(/data/dev2/hf-cache/models--Qwen--Qwen3.8-27B)
for p in "${paths[@]}"; do
  [[ -e "$p" ]] || { echo "missing $p" >&2; exit 1; }
  ssh -i "$key" -o BatchMode=yes "root@$dest" "mkdir -p '$(dirname "$p")'"
  rsync -a -e "ssh -i $key -o BatchMode=yes" "$p" "root@$dest:$(dirname "$p")/"
done
list() { for p in "${paths[@]}"; do find "$p" -type f -print0; done | sort -z | xargs -0 sha256sum; }
here=$(list | sha256sum | cut -c1-64)
there=$(ssh -i "$key" -o BatchMode=yes "root@$dest" "$(declare -p paths); $(declare -f list); list" | sha256sum | cut -c1-64)
echo "files=$(list | wc -l) sha256-list here=$here there=$there"
[[ "$here" == "$there" ]] || { echo "relay differs" >&2; exit 1; }
