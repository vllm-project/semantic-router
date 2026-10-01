#!/usr/bin/env bash
# Serving-track launcher (node side): one session container on one GPU under a shared lease.
#
# Usage: run.sh --src <decision2 mirror> --plugin <plugin mirror> --gpu N --run NAME
#               [--image IMAGE] [--mount DIR]... [--frozen-cache DIR:DIGEST] -- <session> [session args]
#
# <decision2 mirror> is a mirror_to_node.sh --path src/training/decision2 directory and <plugin
# mirror> one of --path src/vllm-sr-plugins, both of the same pushed commit. The session runs as
# `python3 -m v2.serving.session <session> --out /data/dev2/runs/serving/NAME ...` in the scored
# image with no network (vLLM listens on the container's loopback only), the mirrors, packages
# and panels read-only, and only the run directory writable. The GPU must be idle (< 5% use,
# < 2 GiB VRAM); the lease file owner.serving records the job and is marked released on exit.
# --frozen-cache copies a persisted Triton autotune cache into the run (digest checked first)
# for the shipped-runtime session, as the release parity does.
set -euo pipefail
src="" plugin="" gpu="" name="" image="decision20-train-fast:host2" frozen=""
mounts=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) src="$2"; shift 2 ;;
    --plugin) plugin="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) name="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --mount) mounts+=("$2"); shift 2 ;;
    --frozen-cache) frozen="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$src" && -n "$plugin" && "$gpu" =~ ^[0-7]$ && "$name" =~ ^[A-Za-z0-9._-]+$ && $# -ge 1 ]] \
  || { sed -n '2,15p' "$0" >&2; exit 2; }
for mirror in "$src" "$plugin"; do
  [[ -f "$mirror/.dev2-mirror.json" ]] || { echo "no verified mirror at $mirror" >&2; exit 1; }
done
commit() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$1/.dev2-mirror.json"; }
[[ "$(commit "$src")" == "$(commit "$plugin")" ]] || { echo "mirrors are of different commits" >&2; exit 1; }
session="$1"; shift
S="$src/src/training/decision2"
P="$plugin/src/vllm-sr-plugins"
run="/data/dev2/runs/serving/$name"
[[ ! -e "$run" ]] || { echo "$run exists; use a new run name" >&2; exit 1; }

rocm-smi --showuse --showmeminfo vram --json | python3 -c '
import json, sys
card = json.load(sys.stdin)["card" + sys.argv[1]]
use = float(card["GPU use (%)"])
used = int(card["VRAM Total Used Memory (B)"])
if use > 5 or used > 2 * 2**30:
    sys.exit(f"GPU{sys.argv[1]} is busy: use {use}%, VRAM used {used / 2**30:.1f} GiB")
' "$gpu"

mkdir -p "$run"
lease="/data/dev2/leases/gpu$gpu.lock"
mkdir -p "$lease"
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
start_s="$(date +%s)"
printf 'track=serving\npurpose=vllm-sr-plugins %s session (shared lease, recorded co-tenant)\nstart_utc=%s\nexpected_end_utc=%s\nrun_dir=%s\n' \
  "$session" "$start_utc" "$(date -u -d '+45 min' +%Y-%m-%dT%H:%M:%SZ)" "$run" > "$lease/owner.serving"
# shellcheck disable=SC2329  # invoked by the EXIT trap
finish() {
  local code=$?
  local wall=$(( $(date +%s) - start_s ))
  printf 'track=serving\nstatus=released (%s session, exit %s)\nstart_utc=%s\nlast_job_end_utc=%s\nrun_dir=%s\ngpu_hours=%s\n' \
    "$session" "$code" "$start_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run" \
    "$(python3 -c "print(round($wall / 3600, 4))")" > "$lease/owner.serving"
  python3 - "$run/launcher.json" "$code" "$wall" <<'EOF'
import json, sys
path, code, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
record = json.load(open(path))
record.update({"exit_code": code, "wall_seconds": wall, "gpu_hours": round(wall / 3600, 4)})
json.dump(record, open(path, "w"), indent=2, sort_keys=True)
EOF
}
trap finish EXIT

envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$S" -e DEV2_MIRROR="$(commit "$src")"
      -e HOME="$run/home")
volumes=(-v "$src:$src:ro" -v "$plugin:$plugin:ro" -v "$run:$run")
for m in "${mounts[@]}"; do volumes+=(-v "$m:$m:ro"); done
if [[ -n "$frozen" ]]; then
  dir="${frozen%%:*}" want="${frozen##*:}"
  digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
  [[ "$(digest "$dir")" == "$want" ]] || { echo "frozen cache $dir changed" >&2; exit 1; }
  cp -a "$dir" "$run/triton-frozen"
  envs+=(-e TRITON_CACHE_AUTOTUNING=1 -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_DIR="$run/triton-frozen")
fi
mkdir -p "$run/home"
python3 - "$run/launcher.json" "$image" "$(docker image inspect --format '{{.Id}}' "$image")" "$gpu" \
  "$session" "$src" "$plugin" "$frozen" "${mounts[@]}" <<'EOF'
import json, sys
path, image, image_id, gpu, session, src, plugin, frozen, *mounts = sys.argv[1:]
mirror = lambda d: json.load(open(d + "/.dev2-mirror.json"))
json.dump({"schema": "dev2-serving-launcher/1", "image": image, "image_id": image_id, "gpu": int(gpu),
           "session": session, "decision2_mirror": mirror(src), "plugin_mirror": mirror(plugin),
           "frozen_cache": frozen or None, "mounts": mounts}, open(path, "x"), indent=2, sort_keys=True)
EOF
echo "run=$run gpu=$gpu session=$session commit=$(commit "$src")"
docker run --rm --name "serving-$name" --network none --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES="$gpu" -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  "${envs[@]}" "${volumes[@]}" -w "$S" --entrypoint python3 "$image" \
  -m v2.serving.session "$session" --out "$run" "$@" 2>&1 | tee "$run/session.log"
exit "${PIPESTATUS[0]}"
