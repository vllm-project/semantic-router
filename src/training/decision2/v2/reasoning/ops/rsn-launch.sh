#!/usr/bin/env bash
# Reasoning track: run one job of an exact mirror in a pinned training image on one GPU (or CPU) of this node.
#
# usage: rsn-launch.sh <job> <mirror-dir> <out-dir> (--cpu | --gpu N) [--cpus RANGE] [--image 4b|9b] -- <python3 args...>
#
# GPU isolation as in the program's launchers: /dev/kfd plus only the GPU's render node (resolved from its PCI
# address), ROCR_VISIBLE_DEVICES=0, --network none; the GPU's lease must name track=reasoning*. CPU work is pinned
# with --cpuset-cpus and thread caps. Mounts (read-only unless noted): the mirror's src/training/decision2 as /code,
# /data/dev2/models as /models, /data/dev2/runs/af as /af, /data/dev2/runs/reasoning as /rsn, <out-dir> as /out (rw),
# the Triton cache /data/dev2/runs/reasoning/triton-cache/<image> as /triton-cache (rw). A receipt with start / end
# UTC, exit status, GPU, commit / tree and the docker argv is written to <out-dir>.launch.json.
set -euo pipefail

job=$1 src=$2 out=$3
shift 3
gpu="" cpus="" size=4b
while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in
    --gpu) gpu=$2; shift 2 ;;
    --cpu) shift ;;
    --cpus) cpus=$2; shift 2 ;;
    --image) size=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ ${1:-} == -- ]] && shift
[[ -n $cpus ]] || { echo "--cpus RANGE is required (pin every job)" >&2; exit 2; }
case $size in
  9b) image=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54 ;;
  4b) image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 ;;
  *) echo "unknown image $size" >&2; exit 2 ;;
esac
code=/data/dev2/src/$src/src/training/decision2
[[ -d $code ]] || { echo "missing exact mirror /data/dev2/src/$src" >&2; exit 2; }
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "/data/dev2/src/$src/.dev2-mirror.json")
tree=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "/data/dev2/src/$src/.dev2-mirror.json")
receipt="$out.launch.json"
[[ -e $receipt ]] && { echo "receipt exists: $receipt" >&2; exit 2; }
mkdir -p "$out" /data/dev2/runs/reasoning
threads=$(python3 -c 'import sys; n=0
for part in sys.argv[1].split(","):
    a, _, b = part.partition("-"); n += int(b or a) - int(a) + 1
print(n)' "$cpus")
cache=/data/dev2/runs/reasoning/triton-cache/$size
mkdir -p "$cache"
argv=(docker run --name "rsn-$job" --rm --network none --shm-size 16g --cpuset-cpus "$cpus"
  --security-opt seccomp=unconfined --group-add video --group-add render
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e OMP_NUM_THREADS="$threads" -e MKL_NUM_THREADS="$threads" -e TOKENIZERS_PARALLELISM=false
  -e DEC_SOURCE_COMMIT="$sha" -e DEC_SOURCE_TREE="$tree" -e DEC_IMAGE_ID="$image"
  --mount "type=bind,src=$code,dst=/code,readonly"
  --mount "type=bind,src=/data/dev2/models,dst=/models,readonly"
  --mount "type=bind,src=/data/dev2/runs/reasoning,dst=/rsn,readonly"
  --mount "type=bind,src=$out,dst=/out"
  --mount "type=bind,src=$cache,dst=/triton-cache" -e TRITON_CACHE_DIR=/triton-cache -e TRITON_CACHE_AUTOTUNING=1)
[[ -d /data/dev2/runs/af ]] && argv+=(--mount "type=bind,src=/data/dev2/runs/af,dst=/af,readonly")
label="CPU"
if [[ -n $gpu ]]; then
  grep -qs "^track=reasoning" "/data/dev2/leases/gpu$gpu.lock/owner" \
    || { echo "lease gpu$gpu.lock/owner does not name track=reasoning*" >&2; exit 2; }
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 == g {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  [[ -c $render ]] || { echo "no render node for GPU $gpu ($bdf)" >&2; exit 2; }
  argv+=(--device /dev/kfd --device "$render" -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0)
  label="GPU$gpu ($(basename "$render"))"
else
  argv+=(-e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=)
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
           "end_utc": end, "exit_status": int(status), "gpu": None if gpu == "CPU" else gpu,
           "docker_argv": argv}, open(path, "x"), indent=2)
EOF
exit $status
