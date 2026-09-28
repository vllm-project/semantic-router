#!/usr/bin/env bash
# Node A: one AutoJev-27B native collector process on one leased GPU (M3a preregistration).
#
# Usage: autojev_job.sh --gpu N --src MIRROR_DIR --input PROMPTS --output OUT --cache DIR --label L
#
# Runs `inference.autojev27` from the exact mirror MIRROR_DIR (/data/dev2/src/<sha>-...) in the
# pinned image, with only ROCR_VISIBLE_DEVICES selecting the GPU, the FLA kernels of the image
# on the import path, and the shared persisted Triton autotune cache DIR mounted read-write
# (TRITON_CACHE_AUTOTUNING=1). Writes OUT, OUT.log and OUT.manifest.json (image, GPU, times,
# autotune-entry digest before and after, FLA path check, collector summary). Exit code = the
# collector's; 4 when the reference gated-delta fallback was used.
set -euo pipefail

IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
MODEL=/data/decision20-20260926/references/autojev27-6f5b
SOURCE=/data/decision20-20260926/runs/autojev27-native-smoke-r1/external/autojev-source
REVISION=6f5b557e037f5edb25c7dc92dbc6553e5a19c015

gpu="" src="" input="" output="" cache="" label=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --src) src="$2"; shift 2 ;;
    --input) input="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --cache) cache="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[2-4]$ ]] || { echo "research & data M3a uses node A GPU2-4 only" >&2; exit 2; }
[[ -f "$src/.dev2-mirror.json" ]] || { echo "no verified mirror at $src" >&2; exit 2; }
[[ -f "$input" && -d "$cache" && -n "$label" ]] || { echo "missing input, cache or label" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE_ID")" == "$IMAGE_ID" ]]
for _ in $(seq 24); do
  vram="$(rocm-smi -d "$gpu" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}')"
  [[ "${vram:-100}" -le 2 ]] && break
  sleep 5
done
[[ "${vram:-100}" -le 2 ]] || { echo "GPU$gpu is in use (VRAM ${vram}%)" >&2; exit 2; }

autotune_digest() {
  (cd "$cache" && find . -type f -name '*.autotune.json' -print0 | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)
}
autotune_count() { find "$cache" -type f -name '*.autotune.json' | wc -l | tr -d ' '; }

[[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "label must be a container-name token" >&2; exit 2; }
code_dir="$src/src/training/decision2"
out_dir="$(dirname "$output")"
in_dir="$(dirname "$input")"
mkdir -p "$out_dir"
[[ "$in_dir" != "$out_dir" ]] || { echo "input and output must be in different directories" >&2; exit 2; }
before="$(autotune_digest)"; before_n="$(autotune_count)"
start_utc="$(date -u +%FT%TZ)"; t0="$(date +%s.%N)"
set +e
docker run --rm --name "dev2-data-m3a-$label" --network none \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host \
  -e ROCR_VISIBLE_DEVICES="$gpu" -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR="$cache" \
  -v "$src:$src:ro" -v "$MODEL:$MODEL:ro" -v "$SOURCE:$SOURCE:ro" \
  -v "$in_dir:$in_dir:ro" -v "$out_dir:$out_dir" -v "$cache:$cache" \
  -w "$code_dir" --entrypoint python3 "$IMAGE_ID" \
  -m inference.autojev27 --model-path "$MODEL" --source-path "$SOURCE" \
  --model-revision "$REVISION" --input "$input" --output "$output" --device cuda:0 \
  > "$output.log" 2>&1
code=$?
set -e
t1="$(date +%s.%N)"; end_utc="$(date -u +%FT%TZ)"
after="$(autotune_digest)"; after_n="$(autotune_count)"
fallback=false
grep -q 'chunk_gated_delta_rule` is falling back' "$output.log" && fallback=true
python3 - "$output" "$gpu" "$src" "$input" "$cache" "$label" "$start_utc" "$end_utc" "$t0" "$t1" \
  "$code" "$before" "$before_n" "$after" "$after_n" "$fallback" "$IMAGE_ID" <<'EOF'
import hashlib, json, sys
(output, gpu, src, input_path, cache, label, start, end, t0, t1, code, before, before_n,
 after, after_n, fallback, image_id) = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as s:
        for block in iter(lambda: s.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()
summary = None
for line in open(output + ".log", encoding="utf-8", errors="replace"):
    line = line.strip()
    if line.startswith("{") and '"loaded_parameters"' in line:
        summary = json.loads(line)
rows = sum(1 for _ in open(output, "rb")) if code == "0" else None
wall = float(t1) - float(t0)
manifest = {
    "schema": "decision2-m3a-autojev-job/1", "label": label, "node": "node A", "gpu": int(gpu),
    "gpus": 1, "image_id": image_id, "mirror": json.load(open(src + "/.dev2-mirror.json")),
    "input": input_path, "input_sha256": sha(input_path), "output": output,
    "output_sha256": sha(output) if code == "0" else None, "output_rows": rows,
    "log_sha256": sha(output + ".log"), "exit_code": int(code), "start_utc": start,
    "end_utc": end, "wall_seconds": wall, "gpu_hours": wall / 3600,
    "env": {"TRITON_CACHE_AUTOTUNING": "1", "TRITON_CACHE_DIR": cache, "HF_HUB_OFFLINE": "1"},
    "autotune_before": {"entries": int(before_n), "sha256": before},
    "autotune_after": {"entries": int(after_n), "sha256": after},
    "fla_reference_fallback": fallback == "true", "collector": summary,
}
with open(output + ".manifest.json", "x") as stream:
    json.dump(manifest, stream, indent=1, sort_keys=True)
EOF
[[ "$fallback" == false ]] || exit 4
exit "$code"
