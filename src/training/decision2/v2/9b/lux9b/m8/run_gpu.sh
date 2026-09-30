#!/usr/bin/env bash
# usage: run_gpu.sh GPU RUN_DIR PURPOSE EXPECTED_MIN -- docker-run-args... IMAGE CMD...
# Milestone 8 copy of m7/run_gpu.sh for GPUs lent by the ~27B track (container
# d2-9b-m8-NAME-gGPU): refuses a GPU outside M8's allocation (node A GPU2-4; D2_9B_GPUS adds GPU6-7
# once M7 ends), a GPU whose ~27B owner entry is not idle, our lease entry marked running, a GPU
# with allocated VRAM, a reused run directory or an existing container name. It writes only
# gpuN.lock/owner.9b-m8 (never the owner's gpuN.lock/owner) while the job runs, and start / end /
# exit files into RUN_DIR.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
gpu=$1; run=$2; purpose=$3; expected=$4; shift 4; [ "$1" = "--" ] && shift
allowed="${D2_9B_GPUS:-2 3 4}"
[[ " $allowed " == *" $gpu "* ]] || { echo "GPU $gpu is not allocated to 9B M8" >&2; exit 64; }
render=renderD-dry
if [ "${DRY_RUN:-0}" != 1 ]; then
  bus=$(rocm-smi --showbus 2>/dev/null | awk -v g="GPU[$gpu]" '$1 == g {print tolower($NF)}' || true)
  render=""
  for r in /sys/class/drm/renderD*; do
    slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$r/device/uevent" 2>/dev/null || true)
    if [ -n "$bus" ] && [ "$slot" = "$bus" ]; then render=$(basename "$r"); break; fi
  done
  [ -n "$render" ] || { echo "no render node for GPU $gpu" >&2; exit 64; }
fi
lock=$LEASES/gpu$gpu.lock
mkdir -p "$lock"
exec 9>"$lock/.flock-9b-m8"
flock -n 9 || { echo "GPU $gpu lease is held by another 9B M8 launcher" >&2; exit 65; }
lent_ok "$gpu" || exit 65
if [ "${DRY_RUN:-0}" != 1 ]; then
  for _ in $(seq 1 36); do
    vram=$(rocm-smi -d "$gpu" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}')
    [ "$vram" = 0 ] && break
    sleep 5
  done
  [ "$vram" = 0 ] || { echo "GPU $gpu is not idle (VRAM%=${vram:-unknown})" >&2; exit 65; }
fi
if [ -e "$run/start-utc.txt" ]; then echo "$run was already used; pick a new run directory" >&2; exit 66; fi
name=d2-9b-m8-$(basename "$run")-g$gpu
if [ "${DRY_RUN:-0}" != 1 ] && docker container inspect "$name" >/dev/null 2>&1; then
  echo "container $name exists" >&2; exit 67
fi
mkdir -p "$run"
start=$(date -u +%FT%TZ)
printf "track=%s\nstatus=running\npurpose=%s\nrun=%s\nstart_utc=%s\nexpected_minutes=%s\ncontainer=%s\nlent_by=27b (owner entry untouched)\n" \
  "$TRACK" "$purpose" "$(basename "$run")" "$start" "$expected" "$name" > "$lock/$LEASE_ENTRY"
echo "$start" > "$run/start-utc.txt"
echo "$gpu" > "$run/gpu.txt"
set +e
dry docker run --rm --cidfile "$run/container.id" --name "$name" --network none --device=/dev/kfd \
  --device="/dev/dri/$render" --shm-size=16g --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 "$@" > "$run/console.log" 2>&1
status=$?
set -e
end=$(date -u +%FT%TZ)
echo "$end" > "$run/end-utc.txt"; echo "$status" > "$run/exit-code.txt"
if [ -s "$run/container.id" ]; then docker rm -f "$(cat "$run/container.id")" >/dev/null 2>&1 || true; fi
printf "track=%s\nstatus=idle\nlast_run=%s\nlast_start_utc=%s\nlast_end_utc=%s\nlast_exit=%s\nlent_by=27b (owner entry untouched)\n" \
  "$TRACK" "$(basename "$run")" "$start" "$end" "$status" > "$lock/$LEASE_ENTRY"
exit $status
