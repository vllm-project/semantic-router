#!/usr/bin/env bash
# Run one 0.6B-track command on exactly one leased GPU from an exact source mirror.
# Usage: run_container.sh <gpu-index> <commit-sha> <run-name> -- <python module args...>
# Only that GPU's render node is mapped; the container has no network. Wall-clock
# start/end and exit status are written next to the log for GPU-hour accounting.
set -euo pipefail

gpu="$1"
sha="$2"
name="$3"
shift 3
[ "${1:-}" = "--" ] && shift

root=/data/dev2
src="$root/src/$sha"
image=decision20-train-fast:host2
logs="$root/logs/06b"
[ -d "$src/src/training/decision2/v2/06b" ] || { echo "missing exact mirror $src" >&2; exit 2; }
if ! grep -qs '^track=06b-encoder$' "$root/leases/gpu$gpu.lock/owner"; then
  echo "gpu$gpu is not leased to the 0.6B track" >&2
  exit 2
fi

bus=$(rocm-smi --showbus | awk -v g="GPU[$gpu]" '$1 == g {print tolower($NF)}')
render=$(readlink -f "/dev/dri/by-path/pci-$bus-render")
[ -c "$render" ] || { echo "no render node for gpu$gpu ($bus)" >&2; exit 2; }

mkdir -p "$logs" "$root/runs/06b"
start=$(date -u +%FT%TZ)
t0=$(date +%s)
set +e
# shellcheck disable=SC2086 # DEV2_DOCKER_EXTRA is an intentionally word-split argument list
docker run --rm --name "dev2-06b-$name" --network none --ipc=host --shm-size 32g \
  --device /dev/kfd --device "$render" --group-add video --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
  -e PYTHONPATH=/src/src/training/decision2 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
  -v /data/decision20-20260926:/work:ro -v "$src":/src:ro -v "$root/runs/06b":/runs \
  ${DEV2_DOCKER_EXTRA:-} -w /src/src/training/decision2 "$image" "${DEV2_PYTHON:-/work/envs/kai-lex/bin/python}" "$@" \
  > "$logs/$name.log" 2>&1
status=$?
set -e
t1=$(date +%s)
printf '{"run":"%s","gpu":%s,"render":"%s","commit":"%s","image":"%s","start_utc":"%s","end_utc":"%s","wall_seconds":%s,"gpu_hours":%s,"exit":%s}\n' \
  "$name" "$gpu" "$render" "$sha" "$image" "$start" "$(date -u +%FT%TZ)" "$((t1 - t0))" \
  "$(awk -v s=$((t1 - t0)) 'BEGIN{printf "%.5f", s/3600}')" "$status" > "$logs/$name.timing.json"
exit "$status"
