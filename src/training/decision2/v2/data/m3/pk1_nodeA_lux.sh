#!/usr/bin/env bash
# Node A GPU2: own-Lux1 on the positional-key A0 re-derivation prompts (M3a prereg section 6).
#
# Usage: pk1_nodeA_lux.sh MIRROR_DIR PROMPTS OUTPUT CACHE_COPY
#
# Same runtime as the canonical own-Lux A0 TRAIN predictions (813b06aa, node A): image f83b1d10,
# Decision-1.0-Lux-9B at the runtime-bearing revision bd45a30a, `inference.run --backend lux
# --over-budget-invalid`, the eval track's persisted-autotune settings, and a private copy
# (CACHE_COPY, created here) of that run's post-run autotune cache (tree c94e66c5). Writes OUTPUT,
# OUTPUT.log and OUTPUT.manifest.json.
set -euo pipefail
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
MODEL=/data/decision20-20260926/models/Decision-1.0-Lux-9B
REVISION=bd45a30aee8c84032791c245c70f86dee5389cc8
SOURCE_CACHE=/data/dev2/runs/06b/m2/lux1-triton-cache
EXPECTED_CACHE=c94e66c5fabb512229283a0470cf71dfc4cee4c4554001f43039b4e0322413df
GPU=2
src="$1"; input="$2"; output="$3"; cache="$4"
[[ -f "$src/.dev2-mirror.json" && -f "$input" && ! -e "$output" && ! -e "$cache" ]]
tree() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1); }
[[ "$(tree "$SOURCE_CACHE")" == "$EXPECTED_CACHE" ]] || { echo "source cache moved" >&2; exit 2; }
cp -a "$SOURCE_CACHE" "$cache"
before="$(tree "$cache")"
[[ "$before" == "$EXPECTED_CACHE" ]]
for _ in $(seq 60); do
  vram="$(rocm-smi -d "$GPU" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}')"
  [[ "${vram:-100}" -le 2 ]] && break
  sleep 5
done
[[ "${vram:-100}" -le 2 ]] || { echo "GPU$GPU busy" >&2; exit 2; }
out_dir="$(dirname "$output")"; in_dir="$(dirname "$input")"
mkdir -p "$out_dir"
start="$(date -u +%FT%TZ)"; t0="$(date +%s)"
set +e
docker run --rm --name dev2-data-m3a-pk1-lux --network none \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -e ROCR_VISIBLE_DEVICES="$GPU" -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR="$cache" \
  -v "$src:$src:ro" -v "$MODEL:$MODEL:ro" -v "$in_dir:$in_dir:ro" -v "$out_dir:$out_dir" \
  -v "$cache:$cache" -w "$src/src/training/decision2" --entrypoint python3 "$IMAGE_ID" \
  -m inference.run --backend lux --model-path "$MODEL" --model-revision "$REVISION" \
  --input "$input" --output "$output" --device cuda:0 --over-budget-invalid > "$output.log" 2>&1
code=$?
set -e
wall=$(( $(date +%s) - t0 ))
printf '{"schema":"decision2-m3a-pk1-lux/1","node":"node A","gpu":%s,"image_id":"%s","model_revision":"%s","mirror_commit":"%s","input_sha256":"%s","output_sha256":"%s","exit_code":%s,"start_utc":"%s","wall_seconds":%s,"cache_before":"%s","cache_after":"%s"}\n' \
  "$GPU" "$IMAGE_ID" "$REVISION" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$src/.dev2-mirror.json")" \
  "$(sha256sum "$input" | cut -d' ' -f1)" "$( [[ $code -eq 0 ]] && sha256sum "$output" | cut -d' ' -f1)" \
  "$code" "$start" "$wall" "$before" "$(tree "$cache")" > "$output.manifest.json"
exit "$code"
