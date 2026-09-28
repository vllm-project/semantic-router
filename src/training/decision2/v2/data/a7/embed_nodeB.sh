#!/usr/bin/env bash
# Short GPU embedding scan of A7 files on node B GPU7, run from an exact mirror.
#
# Usage: embed_nodeB.sh <out-dir> <protected-manifest> <candidates.jsonl>...
#
# Runs `v2.data.embed_scan` with the research & data track's settings (pinned
# Qwen3-Embedding-0.6B snapshot, quarantine >= 0.93, 20-pair review sample in
# [0.85, 0.93), default seed and batch) in the runtime image with only GPU7
# visible. GPU7 belongs to the research & data track; A7 may run short jobs on
# it: the scan starts only if the owner file says "released" or "idle" and the
# GPU shows no load, writes an A7 owner file with an expected end at most 30
# minutes out, stops the container at the limit, and restores the previous
# owner bytes on exit unless another track rewrote the file meanwhile.
set -euo pipefail

out="${1:?out-dir}"
manifest="${2:?protected manifest}"
shift 2
[[ $# -gt 0 ]] || { echo "no candidate files" >&2; exit 2; }
S="$(cd "$(dirname "$0")/../../.." && pwd)"
mirror="$(cd "$S/../../.." && pwd)"
commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$mirror/.dev2-mirror.json")"
GPU=7
LEASE=/data/dev2/leases/gpu$GPU.lock
MODEL=/data/dev2/hf-cache/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
IMAGE="${A7_IMAGE:-decision20-lux-runtime:latest}"
LIMIT_S=1740
name="a7-embed-$(date -u +%Y%m%dT%H%M%SZ)"

[[ ! -e "$out" ]] || { echo "refusing to reuse $out" >&2; exit 2; }
umask 077
mkdir -p "$out"
prev="$out/lease.previous-owner"
cp "$LEASE/owner" "$prev"
if ! grep -q -E '"status" *: *"(released|idle)"|^status=(released|idle)' "$prev"; then
  echo "GPU$GPU owner file does not say released/idle; not starting" >&2
  exit 3
fi
load="$(rocm-smi -d $GPU --showuse --showmemuse --json | python3 -c '
import json, sys
card = next(iter(json.load(sys.stdin).values()))
print(card.get("GPU use (%)", "?"), card.get("GPU Memory Allocated (VRAM%)", "?"))')"
[[ "$load" == "0 0" ]] || { echo "GPU$GPU shows load ($load); not starting" >&2; exit 3; }

start_epoch=$(date -u +%s)
start_utc=$(date -u -d "@$start_epoch" +%FT%TZ)
end_utc=$(date -u -d "@$((start_epoch + 1800))" +%FT%TZ)
mine="{\"track\":\"a7\",\"purpose\":\"A7 embedding scan vs protected panels (short job; GPU7 owner is research & data, previous owner file restored afterwards)\",\"start_utc\":\"$start_utc\",\"expected_end_utc\":\"$end_utc\",\"status\":\"busy\",\"run_dir\":\"$out\",\"commit\":\"$commit\"}"
printf '%s\n' "$mine" > "$LEASE/owner"
# shellcheck disable=SC2329  # invoked by the EXIT trap
restore() {
  docker kill "$name" >/dev/null 2>&1 || true
  if [[ "$(cat "$LEASE/owner")" == "$mine" ]]; then
    cp "$prev" "$LEASE/owner"
    echo "lease owner restored" >> "$out/run.log"
  else
    echo "lease owner changed by another writer; left as is" >> "$out/run.log"
  fi
}
trap restore EXIT

args=()
for file in "$@"; do args+=(--candidates "$file"); done
set +e
timeout --kill-after=30 "$LIMIT_S" docker run --rm --name "$name" --network none \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e ROCR_VISIBLE_DEVICES=$GPU -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$S" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /data/dev2:/data/dev2 -w "$S" --entrypoint python3 "$IMAGE" \
  -m v2.data.embed_scan "${args[@]}" --protected-inventory "$manifest" --model-path "$MODEL" \
  --device cuda:0 --private-receipt "$out/embed.private.json" --public-receipt "$out/embed.public.json" \
  > "$out/embed.stdout" 2> "$out/embed.stderr"
rc=$?
set -e
end_epoch=$(date -u +%s)
python3 - "$out/run.json" "$commit" "$IMAGE" "$MODEL" "$manifest" "$start_epoch" "$end_epoch" "$rc" "$@" <<'EOF'
import hashlib, json, subprocess, sys
out, commit, image, model, manifest, start, end, rc, *files = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                          capture_output=True, text=True).stdout.strip()
wall = int(end) - int(start)
json.dump({"schema": "decision2.v2.a7.embed-run.v1", "commit": commit, "image": image_id,
           "model_snapshot": model.rsplit("/", 1)[-1], "protected_manifest_sha256": sha(manifest),
           "candidates": [{"name": f.rsplit("/", 1)[-1], "sha256": sha(f)} for f in files],
           "gpu": "node B GPU7", "wall_s": wall, "gpu_hours": round(wall / 3600, 4), "rc": int(rc)},
          open(out, "w"), indent=1, sort_keys=True)
EOF
echo "rc=$rc wall=$((end_epoch - start_epoch))s" >> "$out/run.log"
exit "$rc"
