#!/usr/bin/env bash
# PN1 dev-slice validity collection on node A GPU1 (prereg sec. 7, amendment 4 part B). Runs only from an
# exact mirror /data/dev2/src/<sha>. Formal node-A settings of N4XF's mlx-diag run: image f83b1d10, 16,384
# tokens, no truncation, offline, a cp -a copy of that run's Triton cache per job. The container sees only
# the gold-free prompt file of its job, never gold.
#
# Usage:
#   pn1_validity.sh smoke  <run-root>                 N4XF on the first 20 mlx-diag prompts (parity preflight)
#   pn1_validity.sh n4xf   <run-root> <prompts.jsonl>
#   pn1_validity.sh nox    <run-root> <prompts.jsonl>
set -euo pipefail

GPU=1
IMAGE=f83b1d10f14d
SHA="$(basename "$(cd "$(dirname "$0")/../../../../../.." && pwd)")"
SRC=/data/dev2/src/$SHA
CACHE=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-triton
N4XF=/data/dev2/runs/dec/m4/formal-candidates/staging-e8656221/m4/N4XF-soup
NOX_SOURCE=/data/dev2/hf-cache/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/cde2a68dbaa557ea65dc458104d410a0802ee259
NOX_PACKAGE=/data/dev2/hf-cache/models--llm-semantic-router--Decision-1.0-Nox-4B/snapshots/0bb833504965c0eabdb9630b7bbd385cb2fe5cd4
LEASE=/data/dev2/leases/gpu$GPU.lock

lease() {
  printf 'track=data\npurpose=PN1 dev-slice validity (%s), recorded co-tenant under the PN1 lend (coordinator 20:45)\nstatus=%s\nupdated_utc=%s\nsource_commit=%s\n' \
    "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SHA" > "$LEASE/owner.data"
}

run() {
  local job=$1 root=$2 prompts=$3
  shift 3
  local dir=$root/$job
  [ ! -e "$dir" ] || { echo "exists: $dir" >&2; exit 1; }
  mkdir -p "$dir/input" "$dir/output"
  cp "$prompts" "$dir/input/prompts.jsonl"
  cp -a "$CACHE" "$dir/triton"
  lease "$job" busy
  local t0 t1 code=0
  t0=$(date +%s.%N)
  docker run --rm --name "pn1-$job" --network none \
    --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES=$GPU -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_AUTOTUNING=1 \
    -e TRITON_CACHE_DIR="$dir/triton" -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/opt/decision-fla:. \
    -v "$SRC:$SRC:ro" -v /data/dev2/hf-cache:/data/dev2/hf-cache:ro -v "$N4XF:$N4XF:ro" \
    -v "$dir:$dir" -w "$SRC/src/training/decision2" --entrypoint python3 "$IMAGE" \
    "$@" --input "$dir/input/prompts.jsonl" --output "$dir/output/predictions.jsonl" \
    > "$dir/collect.log" 2>&1 || code=$?
  t1=$(date +%s.%N)
  python3 - "$dir/GPU-TIME.json" "$t0" "$t1" "$code" "$SHA" "$IMAGE" <<'EOF'
import json, sys
path, t0, t1, code, sha, image = sys.argv[1:]
wall = float(t1) - float(t0)
json.dump({"gpu": 1, "gpus": 1, "node": "node A", "image": image, "source_commit": sha, "shared": True,
           "wall_seconds": wall, "gpu_hours": wall / 3600, "exit_code": int(code)}, open(path, "x"), indent=2)
EOF
  lease "$job" "done (exit $code)"
  return "$code"
}

n4xf_args=(-m v2.dec.infer_dec --checkpoint "$N4XF/checkpoint" --source-path "$NOX_SOURCE"
  --model-id decision2-dec-m4-N4XF-soup --model-revision m4-N4XF-soup-e86562218928fbb77f8071b45646962f63049fad
  --max-length 16384 --calibration "$N4XF/cal698-16k/calibration.json")
nox_args=(-m v2.dec.infer_1p0 --package "$NOX_PACKAGE" --model-id llm-semantic-router/Decision-1.0-Nox-4B
  --model-revision 0bb833504965c0eabdb9630b7bbd385cb2fe5cd4 --max-length 16384 --package-temperatures)

case "${1:-}" in
  smoke)
    head -n 20 /data/dev2/private/panels/goldfree/mlx-diag.prompts.jsonl > "$2.smoke-prompts.jsonl"
    run smoke "$2" "$2.smoke-prompts.jsonl" "${n4xf_args[@]}" ;;
  n4xf) run n4xf "$2" "$3" "${n4xf_args[@]}" ;;
  nox) run nox "$2" "$3" "${nox_args[@]}" ;;
  *) sed -n '2,11p' "$0" >&2; exit 2 ;;
esac
