#!/usr/bin/env bash
# Decoder M11: run one job in the pinned decoder image (dbe5f32b) on one M11 GPU of node E or F, or on CPU.
#
# usage: M11_NODE=e|f [M11_CACHE=<tier>-train|<tier>-read] m11-launch.sh <job> <mirror-dir> <out-dir> (--cpu | --gpu N) \
#          -- <python3 args...>
#
# GPU isolation (nodes C-F rule): the container gets /dev/kfd plus only its GPU's render node, resolved from the
# GPU's PCI address as /root/d2-gpu-smoke.sh does, with ROCR_VISIBLE_DEVICES=0. Only M11's GPUs are accepted (node E
# GPU0-3, node F GPU2-7), and the GPU's lease owner file must name "dec-m11". Mounts (read-only unless noted): the
# exact mirror's src/training/decision2 as /code, /data/dev2/models as /models, /data/dev2/runs/dec as /runs, the
# decoder panels as /panels, the SELECT/CAL directory as /data, <out-dir> as /out (rw) and the M11 Triton cache named
# by M11_CACHE as /triton-cache (rw; prereg: per node and tier, training and readouts use separate caches, each a copy
# of the node's M10 decoder cache made by m11-prep.sh). A receipt with start / end UTC, exit status, GPU and the docker
# argv is written to <out-dir>.launch.json.
set -euo pipefail

job=$1 src=$2 out=$3
shift 3
mode=${1:?--cpu or --gpu N}
gpu=""
if [[ $mode == --gpu ]]; then gpu=$2; shift 2; elif [[ $mode == --cpu ]]; then shift; else
  echo "need --cpu or --gpu N" >&2; exit 2
fi
[[ ${1:-} == -- ]] && shift

node=${M11_NODE:?set M11_NODE=e or f}
case $node in
  e) allowed=" 0 1 2 3 " ;;
  f) allowed=" 2 3 4 5 6 7 " ;;
  *) echo "unknown node $node" >&2; exit 2 ;;
esac
image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
M=/data/dev2/runs/dec/m11
code=/data/dev2/src/$src/src/training/decision2
[[ -d $code ]] || { echo "missing exact mirror /data/dev2/src/$src" >&2; exit 2; }
receipt_json=/data/dev2/src/$src/.dev2-mirror.json
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$receipt_json")
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$receipt_json")
data=${M11_DATA:-/data/dev2/runs/dec/m10/inputs/sel700-cal698}
cache=$M/triton-cache/${M11_CACHE:-none}
mkdir -p "$(dirname "$out")"
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }
mkdir -p "$out"

argv=(docker run --name "m11-$job" --rm --network none --shm-size 16g
  --security-opt seccomp=unconfined --group-add video --group-add render
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$code,dst=/code,readonly"
  --mount "type=bind,src=/data/dev2/models,dst=/models,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec,dst=/runs,readonly"
  --mount "type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly"
  --mount "type=bind,src=$data,dst=/data,readonly"
  --mount "type=bind,src=$out,dst=/out")
label="node ${node^^} CPU"
if [[ -n $gpu ]]; then
  [[ $allowed == *" $gpu "* ]] || { echo "GPU $gpu on node $node is not an M11 GPU" >&2; exit 2; }
  grep -qs "dec-m11" "/data/dev2/leases/gpu$gpu.lock/owner" \
    || { echo "lease gpu$gpu.lock/owner does not name dec-m11" >&2; exit 2; }
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  [[ -c $render ]] || { echo "no render node for GPU $gpu ($bdf)" >&2; exit 2; }
  [[ -n ${M11_CACHE:-} && -d $cache ]] || { echo "missing M11 Triton cache (M11_CACHE=${M11_CACHE:-unset})" >&2; exit 2; }
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
