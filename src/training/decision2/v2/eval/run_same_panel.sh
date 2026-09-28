#!/usr/bin/env bash
# Launch one native collection inside the pinned image on one leased GPU (run on the node).
#
# Usage (SRC is a mirror directory name under /data/dev2/src: <sha> or <sha>-src_training_decision2):
#   run_same_panel.sh --gpu N --track TRACK --src SRC --run-dir DIR --model-dir DIR \
#       [--image IMAGE] [--mount HOST_PATH]... [--mount-rw HOST_PATH]... [--env KEY=VALUE]... \
#       [--purpose TEXT] [--expected-end UTC] [--lease-name NAME [--shared]] -- <same_panel collect arguments except --run-dir>
# --lease-name writes this track's entry as gpuN.lock/NAME (for a GPU shared with its owner track,
# e.g. owner.eval) and leaves the owner track's gpuN.lock/owner untouched. --shared (only with a
# named entry) skips the idle-VRAM check for an approved co-tenancy and records it in GPU-TIME.json.
# --env is for non-secret runtime settings only (for example TRITON_CACHE_DIR).
#
# Mounts (same path inside and outside): the exact mirror /data/dev2/src/SHA (ro), the
# gold-free panels only (ro), the model directory and extra mounts (ro), the run directory
# (rw). Only ROCR_VISIBLE_DEVICES selects the GPU, so the model sees it as cuda:0. The GPU
# lease /data/dev2/leases/gpuN.lock/owner must be free or already owned by TRACK, and the
# GPU must show no allocated VRAM. Wall time lands in <run-dir>/GPU-TIME.json.
set -euo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

gpu="" track="" sha="" run_dir="" model_dir="" image="decision20-train-fast:host2"
purpose="same-panel native collection" expected_end="" mounts=() rw_mounts=() envs=() lease_name="owner" shared=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --track) track="$2"; shift 2 ;;
    --src) sha="$2"; shift 2 ;;
    --run-dir) run_dir="$2"; shift 2 ;;
    --model-dir) model_dir="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --mount) mounts+=("$2"); shift 2 ;;
    --mount-rw) rw_mounts+=("$2"); shift 2 ;;
    --env) envs+=(-e "$2"); shift 2 ;;
    --purpose) purpose="$2"; shift 2 ;;
    --expected-end) expected_end="$2"; shift 2 ;;
    --lease-name) lease_name="$2"; shift 2 ;;
    --shared) shared=1; shift ;;
    --) shift; break ;;
    *) usage ;;
  esac
done
[[ -n "$gpu" && -n "$track" && -n "$sha" && -n "$run_dir" && -n "$model_dir" ]] || usage
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "gpu must be 0-7" >&2; exit 2; }

src="/data/dev2/src/$sha"
panel_root="/data/dev2/private/panels"
[[ -f "$src/.dev2-mirror.json" ]] || { echo "no verified mirror at $src" >&2; exit 1; }
[[ -d "$panel_root/goldfree" ]] || { echo "gold-free panels missing under $panel_root" >&2; exit 1; }
[[ -d "$model_dir" ]] || { echo "model dir missing: $model_dir" >&2; exit 1; }
mkdir -p "$run_dir"
[[ ! -e "$run_dir/GPU-TIME.json" ]] || { echo "$run_dir already used" >&2; exit 1; }

lease="/data/dev2/leases/gpu$gpu.lock"
[[ "$lease_name" =~ ^owner(\.[a-z0-9-]+)?$ ]] || { echo "bad --lease-name" >&2; exit 2; }
[[ "$shared" == 0 || "$lease_name" != "owner" ]] || { echo "--shared needs a named --lease-name" >&2; exit 2; }
if [[ "$lease_name" == "owner" && -f "$lease/owner" ]] && ! grep -qx "track=$track" "$lease/owner"; then
  echo "gpu$gpu is leased by another track:" >&2
  cat "$lease/owner" >&2
  exit 1
fi
vram="$(rocm-smi -d "$gpu" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}')"
if [[ "$shared" == 0 && ( -z "$vram" || "$vram" != "0" ) ]]; then
  echo "gpu$gpu is not idle (VRAM%=${vram:-unknown})" >&2
  exit 1
fi
mkdir -p "$lease"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'track=%s\npurpose=%s\nstart_utc=%s\nexpected_end_utc=%s\nrun_dir=%s\n' \
  "$track" "$purpose" "$start_utc" "${expected_end:-unknown}" "$run_dir" > "$lease/$lease_name"

image_id="$(docker image inspect --format '{{.Id}}' "$image")"
volumes=(-v "$src:$src:ro" -v "$panel_root/goldfree:$panel_root/goldfree:ro"
         -v "$model_dir:$model_dir:ro" -v "$run_dir:$run_dir")
for mount in "${mounts[@]}"; do
  volumes+=(-v "$mount:$mount:ro")
done
for mount in "${rw_mounts[@]}"; do
  volumes+=(-v "$mount:$mount")
done

name="dev2-${track}-gpu${gpu}-$(date -u +%H%M%S)"
started="$(date +%s.%N)"
set +e
docker run --rm --name "$name" --network none \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES="$gpu" -e DEV2_IMAGE_ID="$image_id" -e DEV2_GPU_LABEL="gpu$gpu" \
  "${envs[@]}" "${volumes[@]}" -w "$src/src/training/decision2" --entrypoint python3 "$image" \
  -m v2.eval.same_panel collect --run-dir "$run_dir" --panel-root "$panel_root" "$@"
code=$?
set -e
ended="$(date +%s.%N)"
end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 - "$run_dir/GPU-TIME.json" "$gpu" "$image" "$image_id" "$sha" "$start_utc" "$end_utc" "$started" "$ended" "$code" "$shared" "${vram:-unknown}" <<'EOF'
import json, sys
path, gpu, image, image_id, sha, start, end, t0, t1, code, shared, vram = sys.argv[1:]
wall = float(t1) - float(t0)
record = {"schema": "dev2-gpu-time/1", "gpu": int(gpu), "gpus": 1, "image": image,
          "image_id": image_id, "source_commit": sha, "start_utc": start, "end_utc": end,
          "wall_seconds": wall, "gpu_hours": wall / 3600, "exit_code": int(code)}
if shared == "1":
    record.update(shared=True, vram_pct_at_start=vram)
json.dump(record, open(path, "x"), indent=2)
EOF
printf 'last_job_end_utc=%s\nlast_job_exit=%s\n' "$end_utc" "$code" >> "$lease/$lease_name"
exit "$code"
