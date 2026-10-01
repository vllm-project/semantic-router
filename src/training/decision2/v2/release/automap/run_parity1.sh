#!/usr/bin/env bash
# Stage one Decision 1.0 repository from its downloaded head and run parity1.py
# in the pinned image, on the CPU or on one leased GPU of this node.
#
# Usage (on a node, from an exact mirror):
#   run_parity1.sh <Repo-Name> <head-sha> <cpu|gpuN> <work-dir> [--threads N] [--fla-profile DIR]
#                  [--python INTERPRETER] [--panel NAME:PROMPTS:PREDICTIONS[:N]]...
#                  [--reference-out DIR --kai-code DIR]
# With --reference-out, the native Kai runtime (reference_kai_native1.py) writes
# reference predictions for the repository into DIR instead of running parity.
# The GPU path takes /data/dev2/leases/gpuN.lock/owner only if it is free,
# passes just that GPU's render node plus /dev/kfd (ROCR_VISIBLE_DEVICES=0)
# and releases the lease on exit.
set -euo pipefail

repo="$1"; head="$2"; target="$3"; work="$4"; shift 4
threads=32
profile=""
python=python3
reference=""
kai_code=""
panels=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) threads="$2"; shift 2 ;;
    --fla-profile) profile="$2"; shift 2 ;;
    --python) python="$2"; shift 2 ;;
    --reference-out) reference="$2"; shift 2 ;;
    --kai-code) kai_code="$2"; shift 2 ;;
    --panel) panels+=("$2"); shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ ${#panels[@]} -gt 0 ]] || { echo "at least one --panel" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
decision2="$(cd "$here/../../.." && pwd)"
image="decision20-train-fast:host2"
snapshot="/data/dev2/hf-cache/models--llm-semantic-router--${repo}/snapshots/${head}"
[[ -d "$snapshot" ]] || { echo "missing snapshot $snapshot" >&2; exit 1; }
[[ -e "$work" ]] && { echo "work dir exists: $work" >&2; exit 1; }

docker_args=()
device=cpu
lease=""
release() { [[ -n "$lease" ]] && rm -f "$lease/owner"; }
trap release EXIT
if [[ "$target" == gpu* ]]; then
  index="${target#gpu}"
  lease="/data/dev2/leases/gpu${index}.lock"
  mkdir -p "$lease"
  if [[ -e "$lease/owner" ]]; then
    echo "gpu${index} is leased: $(head -c 200 "$lease/owner")" >&2; lease=""; exit 3
  fi
  printf 'track=release-dev1-automap purpose=Decision 1.0 auto_map parity %s start_utc=%s expected_end_utc=%s shared=no\n' \
    "$repo" "$(date -u +%FT%TZ)" "$(date -u -d '+90 min' +%FT%TZ)" > "$lease/owner"
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $index" '$0 ~ "^"g"$" {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  docker_args+=(--device /dev/kfd --device "$render" --group-add video --group-add render -e ROCR_VISIBLE_DEVICES=0)
  device="cuda:0"
fi
mkdir -p "$work/modules"
python3 "$here/stage1.py" --repo "llm-semantic-router/$repo" --head "$head" \
  --snapshot "$snapshot" --output "$work/stage" --staged "$work/staged" > "$work/stage.log"
docker_args+=(--rm --network none --ipc host --security-opt seccomp=unconfined
  -v /data/dev2/hf-cache:/data/dev2/hf-cache:ro -v /data/dev2/private:/data/dev2/private:ro
  -v /data/dev2/tools:/data/dev2/tools:ro
  -v "$decision2:$decision2:ro" -v "$work:$work"
  -e HF_HUB_OFFLINE=1 -e HF_MODULES_CACHE="$work/modules" -e PYTHONDONTWRITEBYTECODE=1)
real_cache="$(readlink -f /data/dev2/hf-cache)"
[[ "$real_cache" != /data/dev2/hf-cache ]] && docker_args+=(-v "$real_cache:$real_cache:ro")
if [[ -n "$profile" ]]; then
  docker_args+=(-v "$profile:/fla-profile:ro" -e FLA_CACHE_MODE=strict -e FLA_CONFIG_DIR=/fla-profile)
fi
panel_args=()
if [[ -n "$reference" ]]; then
  for panel in "${panels[@]}"; do
    IFS=: read -r name prompts _ <<< "$panel"
    panel_args+=(--panel "$name:$prompts")
  done
  docker_args+=(-v "$kai_code:$kai_code:ro" -v "$(dirname "$reference"):$(dirname "$reference")")
  docker run "${docker_args[@]}" "$image" "$python" -B "$here/reference_kai_native1.py" \
    --kai-code "$kai_code" --weights "$work/staged" --model-name "$repo" --work "$work/reference" \
    --device "$device" --output-dir "$reference" "${panel_args[@]}" 2>&1 | tail -n 40
  exit
fi
declare -A mounted=()
for panel in "${panels[@]}"; do
  panel_args+=(--panel "$panel")
  IFS=: read -r _ _ predictions _ <<< "$panel"
  where="$(dirname "$predictions")"
  if [[ "$where" != /data/dev2/private/* && -z "${mounted[$where]:-}" ]]; then
    docker_args+=(-v "$where:$where:ro"); mounted[$where]=1
  fi
done
docker run "${docker_args[@]}" "$image" "$python" -B "$here/parity1.py" \
  --model "$work/staged" --device "$device" --threads "$threads" \
  --output "$work/parity.json" --changes "$work/changes.jsonl" "${panel_args[@]}" 2>&1 | tail -n 40
