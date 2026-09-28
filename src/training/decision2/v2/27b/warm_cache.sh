#!/usr/bin/env bash
# Milestone 3 warm-up / memory smoke (node B host side): builds the candidate frozen 27B inference
# autotune cache and checks that the input limit fits.
#   1. cp -a the frozen comparator cache to /data/dev2/runs/27b/NAME/triton-cache; the copy's tree
#      hash must equal CACHE_SHA.
#   2. Kernel-adapter smoke through the eval runner, 8 items per panel (PANELS, default the formal
#      and development panels), at LIMIT, writing into that cache.
#   3. One kernel-path forward (launch.py) on the CSS15 prompt whose question is the longest one
#      admitted at LIMIT (typed_collect_kernel --longest; tokens of the gold-free prompts only),
#      same cache, with rocm-smi VRAM sampling on the host.
#   4. WARMUP.json: wall times, GPU-hours, peak memory (torch allocated/reserved, rocm-smi used
#      VRAM), the selected input's token counts, the cache tree hash before/after and the added files.
# The warmed cache stays in place as the candidate frozen cache; freeze and record it by amendment.
# Usage: warm_cache.sh NAME GPU SRC CHECKPOINT CALIBRATION LIMIT FROZEN_CACHE CACHE_SHA
set -euo pipefail

NAME=$1 GPU=$2 SRC=$3 CKPT=$4 CAL=$5 LIMIT=$6 FROZEN=$7 CACHE_SHA=$8
S=/data/dev2/src/$SRC/src/training/decision2
OUT=/data/dev2/runs/27b/$NAME
TC=$OUT/triton-cache
PANELS=${PANELS:-typed-final,css15,public231,typed-dev,css-pilot}
case "$GPU" in 5 | 6 | 7) ;; *) echo "GPU$GPU: launch.py maps only node B GPU5-7" >&2; exit 2 ;; esac
[[ "$CACHE_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "CACHE_SHA must be a full SHA-256" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "$OUT exists; use a new NAME" >&2; exit 1; }
cd "$S"
export PYTHONPATH=$S
source "$S/v2/27b/kernel_common.sh"
CSS15=$PANEL_ROOT/goldfree/css15.prompts.jsonl
python3 - "$CSS15" <<'EOF'
import importlib, pathlib, sys
panels = importlib.import_module("v2.eval.panels")
if panels.sha_file(pathlib.Path(sys.argv[1])) != panels.ALL["css15"]["prompts_sha256"]:
    raise SystemExit("gold-free CSS15 prompts differ from the frozen panel")
EOF
REVISION="checkpoint-sha256:$(model_sha "$CAL")"
mkdir -p "$OUT/receipts" "$OUT/longest"
cache_copy "$FROZEN" "$CACHE_SHA" "$TC"

runner "$OUT/smoke" "$CKPT" "$TC" "27b $NAME kernel warm-up smoke at $LIMIT" \
  --mount "$(dirname "$CAL")" -- --revision "$REVISION" --extra "source=$BASE" \
  --extra "calibration=$CAL" --extra "max_length=$LIMIT" --panels "$PANELS" --max-items 8 ||
  { cache_finish "$TC"; echo "warm-up smoke failed" >&2; exit 1; }

VRAM_LOG=$OUT/longest/rocm-smi-vram-used.txt
(
  while :; do
    rocm-smi -d "$GPU" --showmeminfo vram 2>/dev/null | awk '/Used Memory/ {print $NF}' >> "$VRAM_LOG"
    sleep 2
  done
) &
SAMPLER=$!
status=0
launcher "d2-27b-$NAME-longest" 1.0 "M3 warm-up longest CSS15 forward at $LIMIT" \
  "$OUT/receipts/longest.json" "$TC" -- --mount "$CKPT:$CKPT" --mount "$(dirname "$CAL"):$(dirname "$CAL")" \
  --mount "$CSS15:/data/css15.prompts.jsonl" --mount "$TC:$TC:rw" --mount "$OUT/longest:$OUT/longest:rw" -- \
  python3 -m v2.27b.typed_collect_kernel --checkpoint "$CKPT" --source-path "$BASE" \
  --model-id llm-semantic-router/DEV2.0-27B --model-revision "$REVISION" --max-length "$LIMIT" \
  --calibration "$CAL" --input /data/css15.prompts.jsonl \
  --output "$OUT/longest/css15-longest.predictions.jsonl" --longest || status=$?
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true
cache_finish "$TC"

python3 - "$OUT" "$TC" "$LIMIT" "$CKPT" "$status" <<'EOF'
import json, pathlib, sys
out, tc, limit, ckpt, status = sys.argv[1:]
out = pathlib.Path(out)
def load(path):
    path = out / path
    return json.loads(path.read_text()) if path.is_file() else None
longest = load("longest/css15-longest.predictions.jsonl.runtime.json") or {}
manifest = load("longest/css15-longest.predictions.jsonl.manifest.json") or {}
vram = [int(v) for v in (out / "longest/rocm-smi-vram-used.txt").read_text().split() if v.isdigit()] \
    if (out / "longest/rocm-smi-vram-used.txt").is_file() else []
post = load("triton-cache.post.json")
smoke_time, longest_receipt = load("smoke/GPU-TIME.json"), load("receipts/longest.json")
report = {
    "schema": "decision2-27b-warmup/1",
    "limit": int(limit),
    "checkpoint": ckpt,
    "cache": tc,
    "cache_pre_sha256": post["pre_sha256"],
    "cache_post_sha256": post["post_sha256"],
    "cache_files_after": post["files_after"],
    "cache_added": post["added"],
    "cache_changed": post["changed"],
    "cache_removed": post["removed"],
    "smoke_wall_seconds": smoke_time["wall_seconds"],
    "smoke_gpu_hours": smoke_time["gpu_hours"],
    "longest_exit": int(status),
    "longest_wall_seconds": (longest_receipt or {}).get("elapsed_seconds"),
    "longest_gpu_hours": (longest_receipt or {}).get("gpu_hours"),
    "longest_input": longest.get("longest"),
    "longest_counts": manifest.get("counts"),
    "longest_infer_wall_seconds": longest.get("infer_wall_seconds"),
    "torch_memory": longest.get("memory"),
    "rocm_smi_peak_vram_used_bytes": max(vram) if vram else None,
    "fits": int(status) == 0
    and (manifest.get("counts") or {}).get("valid_questions", 0) >= 1
    and (manifest.get("counts") or {}).get("invalid_questions", 1) == 0,
}
(out / "WARMUP.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: report[k] for k in ("limit", "fits", "cache_post_sha256", "rocm_smi_peak_vram_used_bytes")}))
EOF
echo "candidate frozen cache: $TC"
exit "$status"
