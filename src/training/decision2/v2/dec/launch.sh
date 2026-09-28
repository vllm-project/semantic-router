#!/usr/bin/env bash
# Run one decoder-track job in the pinned training image on the track's GPU.
#
# usage: launch.sh <job-name> <source-sha> <output-dir> [--cpu] -- <python3 args...>
#
# The exact source mirror /data/dev2/src/<sha> is mounted read-only as /code,
# the HF cache as /hf, rights-clean partitions as /data and gold-free prompt
# panels as /panels. Only <output-dir> and the shared Triton cache are writable.
# A receipt with start/end UTC, exit status and the full docker argv is written
# beside the output.
set -euo pipefail

name=$1 sha=$2 out=$3
shift 3
cpu=0
if [[ ${1:-} == --cpu ]]; then cpu=1; shift; fi
[[ ${1:-} == -- ]] && shift

# Defaults are node A GPU5 (PCI 0000:ab:00.0). On node B set DEC_IMAGE, DEC_RENDER
# and DEC_GPU_LABEL to an allocated GPU and that node's qualified image.
image=${DEC_IMAGE:-sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54}
render=${DEC_RENDER:-/dev/dri/renderD169}
gpu_label=${DEC_GPU_LABEL:-node A GPU5}
data=${DEC_DATA:-/data/decision20-20260926/data/hf-private-decision20-clean-v2}
src=/data/dev2/src/$sha
[[ -d $src/src/training/decision2 ]] || { echo "missing exact mirror $src" >&2; exit 2; }
# Subtree mirrors (<sha>-src_training_decision2) record commit and tree in .dev2-mirror.json.
if [[ -f $src/.dev2-mirror.json ]]; then
  sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$src/.dev2-mirror.json")
  tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$src/.dev2-mirror.json")
else
  tree=$(cat "$src/TREE" 2>/dev/null || echo unknown)
fi
mkdir -p "$out"
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }

argv=(docker run --name "dec-$name" --rm --network none --shm-size 16g
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$src/src/training/decision2,dst=/code,readonly"
  --mount "type=bind,src=/data/dev2/hf-cache,dst=/hf,readonly"
  --mount "type=bind,src=$data,dst=/data,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec,dst=/runs,readonly"
  --mount "type=bind,src=$out,dst=/out")
if [[ $cpu == 0 ]]; then
  argv+=(--device /dev/kfd --device "$render" -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0)
  # Every GPU job shares one persisted Triton kernel + autotune cache per node
  # and image, so training, reload and readout processes pick the same kernel
  # configurations (FLA does not reproduce across processes without it).
  image_hex=${image#sha256:}
  triton_cache=${DEC_TRITON_CACHE:-/data/dev2/runs/dec/triton-cache/${image_hex:0:12}}
  mkdir -p "$triton_cache"
  argv+=(--mount "type=bind,src=$triton_cache,dst=/triton-cache"
    -e TRITON_CACHE_DIR=/triton-cache -e TRITON_CACHE_AUTOTUNING=1)
fi
argv+=(-w /code "$image" python3 "$@")

start=$(date -u +%FT%TZ)
set +e
"${argv[@]}" >"$out.stdout.log" 2>"$out.stderr.log"
status=$?
set -e
end=$(date -u +%FT%TZ)
python3 - "$receipt" "$name" "$sha" "$tree" "$image" "$start" "$end" "$status" "$cpu" "$gpu_label" "${argv[@]}" <<'EOF'
import json, sys
path, name, sha, tree, image, start, end, status, cpu, gpu, *argv = sys.argv[1:]
json.dump({"job": name, "source_commit": sha, "source_tree": tree, "image_id": image,
           "start_utc": start, "end_utc": end, "exit_status": int(status),
           "gpu": None if cpu == "1" else gpu, "docker_argv": argv},
          open(path, "x"), indent=2)
EOF
exit $status
