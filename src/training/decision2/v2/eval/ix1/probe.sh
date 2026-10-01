#!/usr/bin/env bash
# IX1 follow-up: run one command in the frozen image on one leased GPU, with a package mounted.
#
#   probe.sh --gpu G --package DIR --work DIR [--mount DIR]... [--name NAME] -- <bash script>
#
# Same container shape as launch.sh (render node of GPU G only, --network none, offline hub,
# the image's FLA / causal-conv1d kernels required, a fresh copy of the 27B frozen Triton cache
# when --cache is given). The work directory must be under /data/dev2/private/. The lease file is
# written with track=eval-ix1 and removed on exit.
set -euo pipefail

IMAGE="decision20-train-fast:host2"
IMAGE_PYTHONPATH="/opt/decision-fla"
HF_CACHE="/data/dev2/hf-cache"
gpu="" pkg="" work="" name="ixA-probe" cache="" mounts=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --package) pkg="$2"; shift 2 ;;
    --work) work="$2"; shift 2 ;;
    --mount) mounts+=("$2"); shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
script="$*"
[[ -n "$gpu" && -d "$pkg" && "$work" == /data/dev2/private/* && -n "$script" ]] || { sed -n '2,9p' "$0" >&2; exit 2; }

lease="/data/dev2/leases/gpu$gpu.lock"
mkdir -p "$lease"
if [[ -s "$lease/owner" ]] && ! grep -qx "track=eval-ix1" "$lease/owner"; then
  echo "gpu$gpu is leased by another owner; refusing" >&2; exit 1
fi
rocm-smi --showmeminfo vram --json | python3 -c '
import json, sys
used = int(json.load(sys.stdin)["card" + sys.argv[1]]["VRAM Total Used Memory (B)"])
sys.exit(f"GPU{sys.argv[1]} holds {used / 2**30:.1f} GiB; refusing" if used > 2 * 2**30 else 0)
' "$gpu"
printf 'track=eval-ix1\npurpose=%s\nstart_utc=%s\nrun_dir=%s\n' "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$work" > "$lease/owner"
trap 'rm -f "$lease/owner"' EXIT

bus="$(rocm-smi --showbus --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["card"+sys.argv[1]]["PCI Bus"].lower())' "$gpu")"
devs=()
for r in /sys/class/drm/renderD*; do
  dev="$(readlink -f "$r/device")"
  [[ "$(basename "$dev")" == "$bus" ]] || continue
  devs+=(--device "/dev/dri/$(basename "$r")")
  for c in "$dev"/drm/card*; do devs+=(--device "/dev/dri/$(basename "$c")"); done
done
(( ${#devs[@]} )) || { echo "no render node for gpu$gpu" >&2; exit 1; }

umask 077
mkdir -p "$work/home" "$work/triton"
if [[ -n "$cache" ]]; then rm -rf "$work/triton"; cp -a "$cache" "$work/triton"; fi
base_dir="$(python3 - "$pkg/MODEL_MANIFEST.json" "$HF_CACHE" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1])).get("base")
if base:
    print(f"{sys.argv[2]}/models--{base['repo_id'].replace('/', '--')}/snapshots/{base['revision']}")
EOF
)"
vols=(-v "$pkg:$pkg:ro" -v "$HF_CACHE:$HF_CACHE:ro" -v "$work:$work")
for m in "${mounts[@]}"; do vols+=(-v "$m:$m:ro"); done
docker run --rm --name "$name" --network none --shm-size 8g --memory "${IX1_MEMORY:-256g}" \
  --device /dev/kfd "${devs[@]}" --group-add video --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_CACHE="$HF_CACHE" -e TOKENIZERS_PARALLELISM=false -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH="$IMAGE_PYTHONPATH" -e HOME="$work/home" -e PKG="$pkg" -e BASE="$base_dir" \
  -e TRITON_CACHE_DIR="$work/triton" -e TRITON_CACHE_AUTOTUNING=1 -e HIP_FORCE_DEV_KERNARG=1 \
  "${vols[@]}" -w "$work" --entrypoint bash "$IMAGE" \
  -c "python3 -c 'import fla.ops.gated_delta_rule, causal_conv1d' || exit 97; $script"
