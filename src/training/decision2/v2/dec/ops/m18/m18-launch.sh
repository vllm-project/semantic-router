#!/usr/bin/env bash
# Decoder M18 (M17's launcher): run one job in the pinned decoder image (dbe5f32b) on one M18 training GPU of node A
# or F, or on CPU.
#
# usage: M18_NODE=a|f [M18_CACHE=<tier>-train] m18-launch.sh <job> <mirror-dir> <out-dir> (--cpu | --gpu N) \
#          -- <python3 args...>
#
# GPU isolation (nodes C-F rule): the container gets /dev/kfd plus only its GPU's render node, resolved from the
# GPU's PCI address, with ROCR_VISIBLE_DEVICES=0. Only M18's training GPUs are accepted (node A GPU1-2 / 7, node F GPU4-5),
# and the GPU's lease owner file must name "dec-m18". Mounts (read-only unless noted): the exact mirror's
# src/training/decision2 as /code, the start checkpoints (node F: /data/dev2/models as /models; node A, as M14:
# /data/dev2/hf-cache as /hf), /data/dev2/runs/dec as /runs, the decoder panels as /panels, the SELECT700 / CAL698
# directory as /data, <out-dir> as /out (rw) and the M18 Triton cache named by M18_CACHE as /triton-cache (rw; copies
# made by m18-prep.sh / m18-prep-a.sh). A receipt with start / end UTC, exit status, GPU and the docker argv is written to <out-dir>.launch.json.
set -euo pipefail

job=$1 src=$2 out=$3
shift 3
mode=${1:?--cpu or --gpu N}
gpu=""
if [[ $mode == --gpu ]]; then gpu=$2; shift 2; elif [[ $mode == --cpu ]]; then shift; else
  echo "need --cpu or --gpu N" >&2; exit 2
fi
[[ ${1:-} == -- ]] && shift

node=${M18_NODE:?set M18_NODE=a or f}
case $node in
  a) allowed=" 1 2 7 " models="type=bind,src=/data/dev2/hf-cache,dst=/hf,readonly"
    data=${M18_DATA:-/data/dev2/runs/dec/m3/data-sel700-cal698} ;;
  f) allowed=" 4 5 " models="type=bind,src=/data/dev2/models,dst=/models,readonly"
    data=${M18_DATA:-/data/dev2/runs/dec/m10/inputs/sel700-cal698} ;;
  *) echo "unknown node $node" >&2; exit 2 ;;
esac
image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
M=/data/dev2/runs/dec/m18
code=/data/dev2/src/$src/src/training/decision2
[[ -d $code ]] || { echo "missing exact mirror /data/dev2/src/$src" >&2; exit 2; }
receipt_json=/data/dev2/src/$src/.dev2-mirror.json
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$receipt_json")
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$receipt_json")
cache=$M/triton-cache/${M18_CACHE:-none}
mkdir -p "$(dirname "$out")"
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }
mkdir -p "$out"

argv=(docker run --name "m18-$job" --rm --network none --shm-size 16g
  --security-opt seccomp=unconfined --group-add video --group-add render
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$code,dst=/code,readonly"
  --mount "$models"
  --mount "type=bind,src=/data/dev2/runs/dec,dst=/runs,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly"
  --mount "type=bind,src=$data,dst=/data,readonly"
  --mount "type=bind,src=$out,dst=/out")
label="node ${node^^} CPU"
if [[ -n $gpu ]]; then
  [[ $allowed == *" $gpu "* ]] || { echo "GPU $gpu on node $node is not an M18 GPU" >&2; exit 2; }
  grep -qs "dec-m18" "/data/dev2/leases/gpu$gpu.lock/owner" \
    || { echo "lease gpu$gpu.lock/owner does not name dec-m18" >&2; exit 2; }
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  [[ -c $render ]] || { echo "no render node for GPU $gpu ($bdf)" >&2; exit 2; }
  [[ -n ${M18_CACHE:-} && -d $cache ]] || { echo "missing M18 Triton cache (M18_CACHE=${M18_CACHE:-unset})" >&2; exit 2; }
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
