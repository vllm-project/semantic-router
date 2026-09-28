#!/usr/bin/env bash
# usage: job.sh SHA GPU NAME PURPOSE EXPECTED_MIN -- python3-args...
# One Milestone 4 GPU job in the pinned trainer image (FLA overlay) from the exact subtree
# mirror of SHA, through the M3 lease wrapper (m3/run_gpu.sh). Run dir
# /data/dev2/runs/9b/m4/NAME (container /out). Read-only mounts: /code mirror, /model Lux 1.0,
# /d10 rights-clean data, /hfc the HF cache, /m4 earlier Milestone 4 runs and data, /m3 the
# Milestone 3 runs (D-line members, the Lux full checkpoint), /panels gold-free development
# prompts. The M4 training autotune cache (/triton) starts as a copy of the M3 one, made once
# and recorded in m4/triton-cache.init.json.
set -euo pipefail
sha=$1; gpu=$2; name=$3; purpose=$4; expected=$5; shift 5; [ "$1" = "--" ] && shift
M3=/data/dev2/runs/9b/m3
M4=/data/dev2/runs/9b/m4
MIRROR=/data/dev2/src/$sha-src_training_decision2
S=$MIRROR/src/training/decision2
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
[ -d "$S" ] || { echo "mirror $S missing" >&2; exit 2; }
tree_sha() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
mkdir -p "$M4"
(
  flock 8
  if [ ! -d "$M4/triton-cache" ]; then
    rm -rf "$M4/triton-cache.pending"
    cp -a "$M3/triton-cache" "$M4/triton-cache.pending"
    printf '{"source": "%s", "copied_utc": "%s", "files": %s, "tree_sha256": "%s"}\n' \
      "$M3/triton-cache" "$(date -u +%FT%TZ)" "$(find "$M4/triton-cache.pending" -type f | wc -l)" \
      "$(tree_sha "$M4/triton-cache.pending")" > "$M4/triton-cache.init.json"
    mv "$M4/triton-cache.pending" "$M4/triton-cache"
  fi
) 8>"$M4/.triton-init.lock"
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$MIRROR/.dev2-mirror.json")
exec "$S/v2/9b/lux9b/m3/run_gpu.sh" "$gpu" "$M4/$name" "$purpose" "$expected" -- \
  -e PYTHONPATH=/code:/opt/decision-fla -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/triton \
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image" \
  --mount type=bind,src="$M4/triton-cache",dst=/triton \
  --mount type=bind,src="$S",dst=/code,readonly \
  --mount type=bind,src=/data/decision20-20260926/models/Decision-1.0-Lux-9B,dst=/model,readonly \
  --mount type=bind,src=/data/decision20-20260926/data,dst=/d10,readonly \
  --mount type=bind,src=/data/dev2/hf-cache,dst=/hfc,readonly \
  --mount type=bind,src="$M4",dst=/m4,readonly \
  --mount type=bind,src="$M3",dst=/m3,readonly \
  --mount type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly \
  --mount type=bind,src="$M4/$name",dst=/out \
  -w /code "$image" python3 "$@"
