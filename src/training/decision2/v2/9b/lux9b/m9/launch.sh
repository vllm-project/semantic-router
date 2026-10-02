#!/usr/bin/env bash
# 9B M9: run one job in the pinned 9B image on one M9 GPU (node C GPU1-7, node A GPU6-7), or on CPU.
#
# usage: M9_NODE=c|a launch.sh <job> <mirror-dir> <out-dir> (--cpu | --gpu N) -- <python3 args...>
# On node A, M9_FORMAL_GPUS (default "6 7") widens the GPU list for a caller that holds another GPU (formal.sh).
#
# GPU isolation (nodes C-F rule, also used on node A): the container gets /dev/kfd plus only its GPU's render node,
# resolved from the GPU's PCI address, with ROCR_VISIBLE_DEVICES=0 and --network none. The GPU's lease owner file
# must name "9b-m9". Mounts (read-only unless noted): the exact mirror's src/training/decision2 as /code,
# /data/dev2/models as /models, /data/dev2/runs/9b as /runs, the decoder panels as /panels, the SELECT/CAL directory
# as /data, Lux 1.0 (the K recipe's package) as /lux, <out-dir> as /out (rw) and the node's M9 Triton cache as
# /triton-cache (rw). A receipt with start / end UTC, exit status, GPU and the docker argv is written to
# <out-dir>.launch.json; GPU-hours are summed from these receipts (m9/gpuh.py).
set -euo pipefail

job=$1 src=$2 out=$3
shift 3
mode=${1:?--cpu or --gpu N}
gpu=""
if [[ $mode == --gpu ]]; then gpu=$2; shift 2; elif [[ $mode == --cpu ]]; then shift; else
  echo "need --cpu or --gpu N" >&2; exit 2
fi
[[ ${1:-} == -- ]] && shift

node=${M9_NODE:?set M9_NODE=c or a}
case $node in
  c) allowed=" 1 2 3 4 5 6 7 " lux=/data/dev2/models/Decision-1.0-Lux-9B/bd45a30aee8c84032791c245c70f86dee5389cc8 ;;
  a) allowed=" ${M9_FORMAL_GPUS:-6 7} " lux=/data/decision20-20260926/models/Decision-1.0-Lux-9B ;;
  *) echo "unknown node $node" >&2; exit 2 ;;
esac
image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
M=/data/dev2/runs/9b/m9
code=/data/dev2/src/$src/src/training/decision2
[[ -d $code ]] || { echo "missing exact mirror /data/dev2/src/$src" >&2; exit 2; }
receipt_json=/data/dev2/src/$src/.dev2-mirror.json
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$receipt_json")
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$receipt_json")
data=${M9_DATA:-$M/inputs/sel700-cal698}
cache=$M/triton-cache/f83b1d10
mkdir -p "$(dirname "$out")"
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }
mkdir -p "$out"

argv=(docker run --name "m9-$job" --rm --network none --shm-size 16g
  --security-opt seccomp=unconfined --group-add video --group-add render
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$code,dst=/code,readonly"
  --mount "type=bind,src=/data/dev2/models,dst=/models,readonly"
  --mount "type=bind,src=/data/dev2/runs/9b,dst=/runs,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly"
  --mount "type=bind,src=$data,dst=/data,readonly"
  --mount "type=bind,src=$lux,dst=/lux,readonly"
  --mount "type=bind,src=$out,dst=/out")
label="node ${node^^} CPU"
if [[ -n $gpu ]]; then
  [[ $allowed == *" $gpu "* ]] || { echo "GPU $gpu on node $node is not an M9 GPU" >&2; exit 2; }
  grep -qs "track=9b-m9" "/data/dev2/leases/gpu$gpu.lock/owner" \
    || { echo "lease gpu$gpu.lock/owner does not name 9b-m9" >&2; exit 2; }
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 == g {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  [[ -c $render ]] || { echo "no render node for GPU $gpu ($bdf)" >&2; exit 2; }
  [[ -d $cache ]] || { echo "missing M9 Triton cache $cache" >&2; exit 2; }
  argv+=(--device /dev/kfd --device "$render" -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0
    --mount "type=bind,src=$cache,dst=/triton-cache" -e TRITON_CACHE_DIR=/triton-cache
    -e TRITON_CACHE_AUTOTUNING=1)
  label="node ${node^^} GPU$gpu ($(basename "$render"))"
fi
argv+=(-w /code "$image" python3 "$@")

start=$(date -u +%FT%TZ)
set +e
"${argv[@]}" > "$out.stdout.log" 2> "$out.stderr.log"
status=$?
set -e
end=$(date -u +%FT%TZ)
python3 - "$receipt" "$job" "$sha" "$tree" "$image" "$start" "$end" "$status" "$label" "${argv[@]}" << 'EOF'
import json, sys
path, job, sha, tree, image, start, end, status, gpu, *argv = sys.argv[1:]
json.dump({"job": job, "source_commit": sha, "source_tree": tree, "image_id": image, "start_utc": start,
           "end_utc": end, "exit_status": int(status), "gpu": None if "CPU" in gpu else gpu,
           "docker_argv": argv}, open(path, "x"), indent=2)
EOF
exit $status
