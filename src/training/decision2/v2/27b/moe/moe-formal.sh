#!/usr/bin/env bash
# 27B MoE milestone: formal post-key same-panel run of one frozen LoRA package on an official MoE base (node host
# side, one of the milestone's GPUs; formal runs go on node B, where the 27B comparators were collected).
# Usage: moe-formal.sh NODE GPU NAME CKPT BASE_DIR MIRROR LOADED_PARAMETERS ACTIVE_PARAMETERS [STAGES]
#   STAGES from smoke,collect,score,mlx (default smoke,collect,score):
#     smoke    the runner's 8-item smoke per formal panel -> formal/NAME-smoke
#     collect  typed FINAL + CSS15 + public 231 at 32,768 tokens -> formal/NAME
#     score    gold-free seal, report (tier 27B, loaded and active parameters), paired compares vs DEV2.0-27B (A20r),
#              AutoJev-27B, Eikos-27B and Jebadiah-27B (node B run dirs)
#     mlx      mlx-diag collection -> formal/NAME-mlx (scored on node A, where its gold lives)
# CALIBRATION=<calibration.json> (an adopted CAL698 fit under the 23:15 rule) switches to adapters/moe-lora-cal.json;
# unset = T = 1. Every collection starts from a fresh copy of DEV2.0-27B's scored cache 03b172f1 and records the
# post-run cache (MoE gated-delta shapes add autotune entries; a release run reuses the formal run's post-run cache).
set -euo pipefail
NODE=$1 GPU=$2 NAME=$3 CKPT=$4 BASE=$5 MIRROR=$6 LOADED=$7 ACTIVE=$8 STAGES=${9:-smoke,collect,score}
S=/data/dev2/src/$MIRROR/src/training/decision2
OUT=/data/dev2/runs/27b-moe/formal/$NAME
LIMIT=32768
FROZEN=/data/dev2/runs/27b/m3-f2/f1-scored-cache
CACHE_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
CALIBRATION=${CALIBRATION:-}
declare -A COMPARATORS=(
  [DEV2.0-27B]=/data/dev2/runs/27b/M4-A20r-soup/formal
  [AutoJev-27B]=/data/dev2/runs/eval/m4/nodeB-kernel/autojev27
  [Eikos-27B]=/data/dev2/runs/eval/m4/nodeB-kernel/eikos27b
  [Jebadiah-27B]=/data/dev2/runs/eval/m4/nodeB-kernel/jebadiah27b
)
case "$NODE:$GPU" in a:3 | a:4 | a:5 | b:6 | b:7) ;; *) echo "node $NODE GPU$GPU is outside the MoE allocation" >&2; exit 2 ;; esac
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[[ "$LOADED" =~ ^[0-9]+$ && "$ACTIVE" =~ ^[0-9]+$ ]] || { echo "parameter counts must be integers" >&2; exit 2; }
[ -f "$CKPT/decision_config.json" ] && [ -d "$BASE" ] || { echo "missing checkpoint or base" >&2; exit 2; }
SPEC=$S/v2/27b/moe/adapters/moe-lora.json
CAL_ARGS=() CAL_MOUNT=()
if [ -n "$CALIBRATION" ]; then
  SPEC=$S/v2/27b/moe/adapters/moe-lora-cal.json
  CAL_ARGS=(--extra "calibration=$CALIBRATION")
  CAL_MOUNT=(--mount "$(dirname "$CALIBRATION")")
fi
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
REVISION="checkpoint-sha256:$(python3 - "$CKPT" <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("trainer_state.pt",))
rows = [[str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in files]
print(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
EOF
)"
take_lease() {
  python3 -c "import importlib, sys; m = importlib.import_module('v2.27b.moe.launch'); m.configure(); m.base.wait_until_idle(int(sys.argv[1]))" "$GPU"
  python3 - "$LEASE" "$1" <<'EOF'
import importlib, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
if lease.get("track") != "27b-moe" or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
owner.write_text(f"track=27b-moe\npurpose={sys.argv[2]}\n", encoding="utf-8")
EOF
}
release_lease() {
  python3 - "$LEASE" "$1" <<'EOF'
import importlib, json, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
lease.update({"track": "27b-moe", "status": "reserved-idle", "container": None, "last_run": sys.argv[2]})
owner.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
EOF
}
collect() {  # RUN_DIR PURPOSE [collect options]...
  local dir=$1 purpose=$2 status=0
  shift 2
  [ ! -e "$dir/GPU-TIME.json" ] || { echo "$dir already collected" >&2; return 66; }
  mkdir -p "$dir"
  python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$CACHE_SHA" --dest "$dir/triton-cache"
  take_lease "$purpose"
  "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b-moe --src "$MIRROR" --run-dir "$dir" --model-dir "$CKPT" \
    --mount "$BASE" "${CAL_MOUNT[@]}" --image "$IMAGE" --env TRITON_CACHE_AUTOTUNING=1 \
    --env "TRITON_CACHE_DIR=$dir/triton-cache" --env HIP_FORCE_DEV_KERNARG=1 --purpose "$purpose" -- \
    --adapter-spec "$SPEC" --model-path "$CKPT" --revision "$REVISION" --extra "source=$BASE" \
    --extra "max_length=$LIMIT" "${CAL_ARGS[@]}" "$@" || status=$?
  python3 -m v2.27b.triton_cache finish --dest "$dir/triton-cache" > /dev/null || true
  release_lease "$dir"
  return "$status"
}
has smoke && collect "$OUT-smoke" "27b-moe $NAME formal smoke" --max-items 8
has collect && collect "$OUT" "27b-moe $NAME formal post-key same-panel collection"
if has score; then
  [ -f "$OUT/SEAL.json" ] || python3 -m v2.eval.same_panel seal --run-dir "$OUT"
  python3 -m v2.eval.same_panel report --run-dir "$OUT" --label "$NAME" --tier 27B --family decision2 \
    --model-id "llm-semantic-router/DEV2.0-27B-MoE-candidate" --revision "$REVISION" \
    --loaded-parameters "$LOADED" --active-parameters "$ACTIVE" \
    --parameter-source "pinned official MoE base text decoder + LoRA adapter + head (safetensors headers); active = non-expert + top-k experts per token"
  for right in "${!COMPARATORS[@]}"; do
    python3 -m v2.eval.same_panel compare --run-dir "$OUT" --comparator-run-dir "${COMPARATORS[$right]}" \
      --left-name "$NAME" --right-name "$right"
  done
fi
has mlx && collect "$OUT-mlx" "27b-moe $NAME mlx-diag collection" --panels mlx-diag
echo "moe formal $NAME stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
