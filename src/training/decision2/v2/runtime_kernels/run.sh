#!/usr/bin/env bash
# ROCm kernel-track launcher (node side): one benchmark container on one leased GPU.
#
# Usage: run.sh --src <decision2 mirror> --gpu N --run NAME [--image IMAGE] [--mount DIR]...
#               [--env K=V]... [--profile] -- <module> [module args]
#
# <decision2 mirror> is a mirror_to_node.sh --path src/training/decision2 directory of a pushed
# commit. The module runs as `python3 -m v2.runtime_kernels.<module> --out <run dir> ...` with no
# network, no secrets, only the GPU's own /dev/dri/renderD* node plus /dev/kfd
# (ROCR_VISIBLE_DEVICES=0 inside), the mirror and every --mount read-only, and only the run
# directory (/data/dev2/runs/rocm-kernels/NAME) writable. The GPU's lease owner file must name
# track=rocm-kernels. --profile wraps the module in `rocprofv3 --kernel-trace --stats`.
set -euo pipefail
src="" gpu="" name="" image="decision20-train-fast:host2" profile=0
mounts=() extra_env=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) src="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --run) name="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --mount) mounts+=("$2"); shift 2 ;;
    --env) extra_env+=("$2"); shift 2 ;;
    --profile) profile=1; shift ;;
    --) shift; break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$src" && "$gpu" =~ ^[0-7]$ && "$name" =~ ^[A-Za-z0-9._-]+$ && $# -ge 1 ]] \
  || { sed -n '2,13p' "$0" >&2; exit 2; }
[[ -f "$src/.dev2-mirror.json" ]] || { echo "no verified mirror at $src" >&2; exit 1; }
owner="/data/dev2/leases/gpu$gpu.lock/owner"
grep -q '^track=rocm-kernels' "$owner" || { echo "GPU$gpu is not leased to rocm-kernels" >&2; exit 1; }
module="$1"; shift
S="$src/src/training/decision2"
run="/data/dev2/runs/rocm-kernels/$name"
[[ ! -e "$run" ]] || { echo "$run exists; use a new run name" >&2; exit 1; }

bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 ~ "^"g"$" {getline; print tolower($2)}')
render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
[[ -c "$render" ]] || { echo "no render node for GPU$gpu" >&2; exit 1; }

mkdir -p "$run/triton" "$run/home"
commit=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$src/.dev2-mirror.json")
start_s=$(date +%s)
python3 - "$run/launcher.json" "$commit" "$gpu" "$render" "$image" "$module" "$profile" "$@" <<'EOF'
import json, sys, time
path, commit, gpu, render, image, module, profile, *args = sys.argv[1:]
json.dump({"commit": commit, "gpu": int(gpu), "render": render.rsplit("/", 1)[-1], "image": image,
           "module": module, "profile": profile == "1", "args": args,
           "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(path, "w"), indent=2, sort_keys=True)
EOF
envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false
      -e PYTHONDONTWRITEBYTECODE=1 -e "PYTHONPATH=/opt/decision-fla:$S" -e ROCR_VISIBLE_DEVICES=0
      -e HIP_FORCE_DEV_KERNARG=1 -e "TRITON_CACHE_DIR=$run/triton" -e "HOME=$run/home"
      -e "DECISION2_CACHE=$run/d2cache" -e "DEV2_MIRROR=$commit")
for kv in "${extra_env[@]}"; do envs+=(-e "$kv"); done
volumes=(-v "$src:$src:ro" -v "$run:$run")
for m in "${mounts[@]}"; do volumes+=(-v "$m:$m:ro"); done
cmd=(python3 -m "v2.runtime_kernels.$module" --out "$run" "$@")
if [[ "$profile" == 1 ]]; then
  cmd=(rocprofv3 --kernel-trace --stats --output-format csv -d "$run/rocprof" -o trace -- "${cmd[@]}")
fi
rc=0
docker run --rm --name "rk-$name" --network none --ipc private --shm-size 16g \
  --device /dev/kfd --device "$render" --group-add video --group-add render \
  --security-opt seccomp=unconfined --memory 400g -w "$run" \
  "${envs[@]}" "${volumes[@]}" --entrypoint "${cmd[0]}" "$image" "${cmd[@]:1}" \
  > "$run/log.txt" 2>&1 || rc=$?
wall=$(( $(date +%s) - start_s ))
python3 - "$run/launcher.json" "$rc" "$wall" <<'EOF'
import json, sys
path, code, wall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
record = json.load(open(path))
record.update({"exit_code": code, "wall_seconds": wall, "gpu_hours": round(wall / 3600, 4)})
json.dump(record, open(path, "w"), indent=2, sort_keys=True)
EOF
echo "exit=$rc wall=${wall}s run=$run"
exit "$rc"
