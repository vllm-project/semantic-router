#!/usr/bin/env bash
# usage: job.sh SHA GPU NAME PURPOSE EXPECTED_MIN -- python3-args...
# One GPU job in the pinned trainer image (FLA overlay, persisted Triton autotune cache) from
# the exact subtree mirror of SHA. Run dir /data/dev2/runs/9b/m3/NAME (container /out).
# Read-only mounts: /code mirror, /model Lux 1.0, /d10 rights-clean data, /hfds the private
# dataset cache (snapshots/<revision>/...), /m3 earlier Milestone 3 runs and data, /panels
# gold-free development prompts.
set -euo pipefail
sha=$1; gpu=$2; name=$3; purpose=$4; expected=$5; shift 5; [ "$1" = "--" ] && shift
M3=/data/dev2/runs/9b/m3
MIRROR=/data/dev2/src/$sha-src_training_decision2
S=$MIRROR/src/training/decision2
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$MIRROR/.dev2-mirror.json")
exec "$S/v2/9b/lux9b/m3/run_gpu.sh" "$gpu" "$M3/$name" "$purpose" "$expected" -- \
  -e PYTHONPATH=/code:/opt/decision-fla -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/triton \
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image" \
  --mount type=bind,src="$M3/triton-cache",dst=/triton \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src=/data/decision20-20260926/models/Decision-1.0-Lux-9B,dst=/model,readonly \
  --mount type=bind,src=/data/decision20-20260926/data,dst=/d10,readonly \
  --mount type=bind,src=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data,dst=/hfds,readonly \
  --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly \
  --mount type=bind,src="$M3/$name",dst=/out \
  -w /code "$image" python3 "$@"
