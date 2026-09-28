#!/usr/bin/env bash
# usage: run_gpu.sh GPU RUN_DIR PURPOSE EXPECTED_MIN -- docker-run-args... IMAGE CMD...
# Node-side GPU job wrapper for the 9B track: refuses a GPU outside the track's current
# allocation, a lease marked running, a reused run directory or an existing container name;
# writes the lease owner file while the job runs and start/end/exit files into RUN_DIR.
set -euo pipefail
gpu=$1; run=$2; purpose=$3; expected=$4; shift 4; [ "$1" = "--" ] && shift
allowed="${D2_9B_GPUS:-2 3 4 6 7}"
[[ " $allowed " == *" $gpu "* ]] || { echo "GPU $gpu is not allocated to the 9B track" >&2; exit 64; }
bus=$(rocm-smi --showbus 2>/dev/null | awk -v g="GPU[$gpu]" '$1 == g {print tolower($NF)}' || true)
render=""
for r in /sys/class/drm/renderD*; do
  slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$r/device/uevent" 2>/dev/null || true)
  if [ -n "$bus" ] && [ "$slot" = "$bus" ]; then render=$(basename "$r"); break; fi
done
[ -n "$render" ] || { echo "no render node for GPU $gpu" >&2; exit 64; }
lock=/data/dev2/leases/gpu$gpu.lock
mkdir -p "$lock"
exec 9>"$lock/.flock-9b"
flock -n 9 || { echo "GPU $gpu lease is held by another 9B launcher" >&2; exit 65; }
if grep -q "^status=running" "$lock/owner" 2>/dev/null; then echo "GPU $gpu lease says running" >&2; exit 65; fi
if [ -e "$run/start-utc.txt" ]; then echo "$run was already used; pick a new run directory" >&2; exit 66; fi
name=d2-9b-$(basename "$run")-g$gpu
if docker container inspect "$name" >/dev/null 2>&1; then echo "container $name exists" >&2; exit 67; fi
mkdir -p "$run"
start=$(date -u +%FT%TZ)
printf "track=9b-clm\nstatus=running\npurpose=%s\nrun=%s\nstart_utc=%s\nexpected_minutes=%s\ncontainer=%s\n" \
  "$purpose" "$(basename "$run")" "$start" "$expected" "$name" > "$lock/owner"
echo "$start" > "$run/start-utc.txt"
echo "$gpu" > "$run/gpu.txt"
set +e
docker run --rm --cidfile "$run/container.id" --name "$name" --network none --device=/dev/kfd \
  --device="/dev/dri/$render" --shm-size=16g --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 "$@" > "$run/console.log" 2>&1
status=$?
set -e
end=$(date -u +%FT%TZ)
echo "$end" > "$run/end-utc.txt"; echo "$status" > "$run/exit-code.txt"
if [ -s "$run/container.id" ]; then docker rm -f "$(cat "$run/container.id")" >/dev/null 2>&1 || true; fi
printf "track=9b-clm\nstatus=idle\nlast_run=%s\nlast_start_utc=%s\nlast_end_utc=%s\nlast_exit=%s\n" \
  "$(basename "$run")" "$start" "$end" "$status" > "$lock/owner"
exit $status
