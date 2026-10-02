#!/usr/bin/env bash
# Arm factory: run one job in the pinned image of its size on one arm-factory GPU, or on CPU.
#
# usage: AF_NODE=a|b|c|f [AF_CACHE=<name>] af-launch.sh <job> <mirror-dir> <out-dir> (--cpu | --gpu N) -- <python3 args...>
#
# Sizes and images: node A trains 9B (M10's image f83b1d10, Lux 1.0 mounted as /lux); nodes B, C and F train 4B
# (M17's image dbe5f32b). GPU isolation (the nodes C-F rule, used on every node here): the container gets /dev/kfd
# plus only its GPU's render node, resolved from the GPU's PCI address, with ROCR_VISIBLE_DEVICES=0 and --network none.
# Only the arm factory's GPUs are accepted (COORDINATION 2026-10-02 22:00): node A GPU1-7, node C GPU3-7, node F
# GPU2-7, node B GPU2 / 4 / 6 / 7, and the GPU's lease owner file must name track=arm-factory. Mounts (read-only
# unless noted): the exact mirror's src/training/decision2 as /code, /data/dev2/models as /models, /data/dev2/runs/af
# as /runs, the decoder panels as /panels, the owners' run trees /data/dev2/runs/dec and /data/dev2/runs/9b as /dec and
# /r9b (each when present), the SELECT/CAL directory as /data, <out-dir> as /out (rw) and
# the Triton cache /data/dev2/runs/af/<size>/triton-cache/$AF_CACHE as /triton-cache (rw). A receipt with start / end
# UTC, exit status, GPU and the docker argv is written to <out-dir>.launch.json; GPU-hours are summed from these.
set -euo pipefail

job=$1 src=$2 out=$3
shift 3
mode=${1:?--cpu or --gpu N}
gpu=""
if [[ $mode == --gpu ]]; then gpu=$2; shift 2; elif [[ $mode == --cpu ]]; then shift; else
  echo "need --cpu or --gpu N" >&2; exit 2
fi
[[ ${1:-} == -- ]] && shift

node=${AF_NODE:?set AF_NODE=a, b, c or f}
lux=""
case $node in
  a) size=9b allowed=" 1 2 3 4 5 6 7 " lux=/data/decision20-20260926/models/Decision-1.0-Lux-9B
     sel=/data/dev2/runs/9b/m9/inputs/sel700-cal698 ;;
  c) size=4b allowed=" 3 4 5 6 7 " sel=/data/dev2/runs/af/4b/inputs/dec/m10/inputs/sel700-cal698 ;;
  f) size=4b allowed=" 2 3 4 5 6 7 " sel=/data/dev2/runs/af/4b/inputs/dec/m10/inputs/sel700-cal698 ;;
  b) size=4b allowed=" 2 4 6 7 " sel=/data/dev2/runs/af/4b/inputs/dec/m10/inputs/sel700-cal698 ;;
  *) echo "unknown node $node" >&2; exit 2 ;;
esac
case $size in
  9b) image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54 ;;
  4b) image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 ;;
esac
code=/data/dev2/src/$src/src/training/decision2
[[ -d $code ]] || { echo "missing exact mirror /data/dev2/src/$src" >&2; exit 2; }
receipt_json=/data/dev2/src/$src/.dev2-mirror.json
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$receipt_json")
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$receipt_json")
data=${AF_DATA:-$sel}
cache=/data/dev2/runs/af/$size/triton-cache/${AF_CACHE:-none}
mkdir -p "$(dirname "$out")"
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }
mkdir -p "$out"

argv=(docker run --name "af-$job" --rm --network none --shm-size 16g
  --security-opt seccomp=unconfined --group-add video --group-add render
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$code,dst=/code,readonly"
  --mount "type=bind,src=/data/dev2/models,dst=/models,readonly"
  --mount "type=bind,src=/data/dev2/runs/af,dst=/runs,readonly"
  --mount "type=bind,src=$data,dst=/data,readonly"
  --mount "type=bind,src=$out,dst=/out")
[[ -d /data/dev2/runs/dec/panels ]] && argv+=(--mount "type=bind,src=/data/dev2/runs/dec/panels,dst=/panels,readonly")
[[ -d /data/dev2/runs/dec ]] && argv+=(--mount "type=bind,src=/data/dev2/runs/dec,dst=/dec,readonly")
[[ -d /data/dev2/runs/9b ]] && argv+=(--mount "type=bind,src=/data/dev2/runs/9b,dst=/r9b,readonly")
[[ -n $lux ]] && argv+=(--mount "type=bind,src=$lux,dst=/lux,readonly")
label="node ${node^^} CPU"
if [[ -n $gpu ]]; then
  [[ $allowed == *" $gpu "* ]] || { echo "GPU $gpu on node $node is not an arm-factory GPU" >&2; exit 2; }
  grep -qs "^track=arm-factory" "/data/dev2/leases/gpu$gpu.lock/owner" \
    || { echo "lease gpu$gpu.lock/owner does not name track=arm-factory" >&2; exit 2; }
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 == g {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  [[ -c $render ]] || { echo "no render node for GPU $gpu ($bdf)" >&2; exit 2; }
  [[ -n ${AF_CACHE:-} && -d $cache ]] || { echo "missing Triton cache $cache (AF_CACHE=${AF_CACHE:-unset})" >&2; exit 2; }
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
