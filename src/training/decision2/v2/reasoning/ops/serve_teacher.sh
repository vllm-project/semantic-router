#!/usr/bin/env bash
# Serve the teacher on one GPU of this node: one vLLM replica per GPU, bound to 127.0.0.1:<18100 + GPU>, host
# network (the node's Docker has no bridge), only that GPU's render node mounted, CPUs pinned to the given range.
#
# usage: serve_teacher.sh <gpu> <cpuset> [model-snapshot] [served-name]
set -euo pipefail
gpu=$1 cpus=$2
snap=${3:-/data/dev2/hf-cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a}
name=${4:-gpt-oss-120b}
image=vllm/vllm-openai-rocm:v0.31.0
bus=$(rocm-smi --showbus --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["card"+sys.argv[1]]["PCI Bus"].lower())' "$gpu")
render="" card=""
for r in /sys/class/drm/renderD*; do
  dev=$(readlink -f "$r/device")
  if [[ $(basename "$dev") == "$bus" ]]; then
    render=/dev/dri/$(basename "$r")
    card=/dev/dri/$(basename "$(ls -d "$dev"/drm/card* | head -1)")
  fi
done
[[ -n $render ]] || { echo "no render node for GPU $gpu" >&2; exit 2; }
grep -qs '^track=reasoning' "/data/dev2/leases/gpu$gpu.lock/owner" || { echo "GPU $gpu is not leased to reasoning" >&2; exit 2; }
docker rm -f "rsn-teacher-g$gpu" > /dev/null 2>&1 || true
docker run -d --name "rsn-teacher-g$gpu" --network host --ipc private --shm-size 16g --cpuset-cpus "$cpus" \
  --memory 200g --device /dev/kfd --device "$render" --device "$card" --group-add video \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /data/dev2/hf-cache:/data/dev2/hf-cache:ro "$image" --model "$snap" --served-model-name "$name" \
  --host 127.0.0.1 --port $((18100 + gpu)) --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 256
