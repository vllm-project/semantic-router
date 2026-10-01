#!/usr/bin/env bash
# One isolated container on node E: dock.sh <cpu|gpu6|gpu7> [--image IMAGE] [--net] [--mount DIR]... -- CMD...
#   gpuN passes only that GPU's render node plus /dev/kfd and ROCR_VISIBLE_DEVICES=0 (node E rule: GPU4-5
#   are foreign; never all of /dev/dri), and writes /data/dev2/leases/gpuN.lock/owner for the run.
#   --net uses the host network (Hub downloads); otherwise --network none. Mounts are read-only unless
#   given as --mount-rw. The HF token is never passed; --net runs read it from a mounted token file only.
set -euo pipefail
where="$1"; shift
image=decision20-train-fast:host2 net=(--network none) volumes=() envs=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) image="$2"; shift 2 ;;
    --net) net=(--network host); shift ;;
    --mount) volumes+=(-v "$2:$2:ro"); shift 2 ;;
    --mount-rw) volumes+=(-v "$2:$2"); shift 2 ;;
    --env)
      [[ "$2" =~ ^[A-Z][A-Z0-9_]*=.*$ && ! "${2%%=*}" =~ (TOKEN|SECRET|PASSWORD|_KEY$) ]] \
        || { echo "--env takes a non-secret KEY=VALUE" >&2; exit 2; }
      envs+=(-e "$2"); shift 2 ;;
    --) shift; break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$where" in
  cpu) gpu=(-e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=) ;;
  gpu6|gpu7)
    index=${where#gpu}
    bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $index" '$0 ~ "^"g"$" {getline; print tolower($2)}')
    render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
    [[ -c "$render" ]] || { echo "no render node for GPU$index" >&2; exit 1; }
    lease=/data/dev2/leases/gpu$index.lock
    mkdir -p "$lease"
    if [[ -f "$lease/owner" ]] && ! grep -qx "track=release-automap" "$lease/owner"; then
      echo "gpu$index is leased by another track" >&2; exit 1
    fi
    printf 'track=release-automap\npurpose=auto_map parity and smoke tests (worker 4c0a68cd)\nstart_utc=%s\nexpected_end_utc=+6h\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lease/owner"
    gpu=(--device /dev/kfd --device "$render" --group-add video --group-add render --security-opt seccomp=unconfined
         -e ROCR_VISIBLE_DEVICES=0) ;;
  *) echo "where is cpu, gpu6 or gpu7" >&2; exit 2 ;;
esac
exec docker run --rm "${net[@]}" --ipc host "${gpu[@]}" -e TOKENIZERS_PARALLELISM=false -e HF_HUB_DISABLE_TELEMETRY=1 \
  "${envs[@]}" "${volumes[@]}" --entrypoint "$1" "$image" "${@:2}"
