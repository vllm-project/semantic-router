#!/usr/bin/env bash
# 27B MoE milestone: T = 1 development readout of one LoRA checkpoint (an MoE cell, a soup, or the dense matched
# reference), node host side, on one of the milestone's GPUs. Never a release score.
# Typed DEV, CSS pilot and HT-DEV v2 go through the eval runner at 32,768 tokens with the MoE collector
# (gated-delta models on the image FLA kernel path, Gemma on SDPA), on a fresh copy of the frozen readout cache
# 583241fb, then v2.eval.dev_readout gives P_dev = 100*sqrt(T_dev*H_pilot) and H_dev2 (vs REFERENCE if given).
# Usage: moe-readout.sh NODE GPU NAME CKPT BASE_DIR MIRROR [HTDEV2_REFERENCE_PREDICTIONS]
#   NAME      output /data/dev2/runs/27b-moe/readouts/NAME (one collection per name)
#   CKPT      a LoRA checkpoint directory (decision_config.json + adapter/ + head)
#   BASE_DIR  the checkpoint's pinned source (its fingerprint is verified by the loader)
set -euo pipefail
NODE=$1 GPU=$2 NAME=$3 CKPT=$4 BASE=$5 MIRROR=$6 REF=${7:-}
S=/data/dev2/src/$MIRROR/src/training/decision2
OUT=/data/dev2/runs/27b-moe/readouts/$NAME
LIMIT=32768
FROZEN=/data/dev2/runs/27b/m3-warm-32768/triton-cache
CACHE_SHA=583241fbc3bc89e22be51a49722996eab742162356100400d64fb4208cb20daf
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
PANELS=${PANELS:-typed-dev,css-pilot,ht-dev2}
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
case "$NODE:$GPU" in a:3 | a:4 | a:5 | b:6 | b:7) ;; *) echo "node $NODE GPU$GPU is outside the MoE allocation" >&2; exit 2 ;; esac
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
[ -f "$CKPT/decision_config.json" ] && [ -d "$BASE" ] && [ -d "$FROZEN" ] || { echo "missing checkpoint, base or cache" >&2; exit 2; }
[ -z "$REF" ] || [ -f "$REF" ] || { echo "missing HT-DEV v2 reference $REF" >&2; exit 2; }
[ ! -e "$OUT/GPU-TIME.json" ] || { echo "readout $NAME already collected" >&2; exit 66; }
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp DEV2_NODE=$NODE
mkdir -p "$OUT"
REVISION="checkpoint-sha256:$(python3 - "$CKPT" <<'EOF'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("trainer_state.pt",))
rows = [[str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest()] for p in files]
print(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest())
EOF
)"
python3 -c "import importlib, sys; m = importlib.import_module('v2.27b.moe.launch'); m.configure(); m.base.wait_until_idle(int(sys.argv[1]))" "$GPU"
python3 - "$LEASE" "27b-moe $NAME T=1 development readout" <<'EOF'
import importlib, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
if lease.get("track") != "27b-moe" or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
owner.write_text(f"track=27b-moe\npurpose={sys.argv[2]}\n", encoding="utf-8")
EOF
python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$CACHE_SHA" --dest "$OUT/triton-cache"
status=0
"$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b-moe --src "$MIRROR" --run-dir "$OUT" --model-dir "$CKPT" \
  --mount "$BASE" --image "$IMAGE" --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$OUT/triton-cache" \
  --env HIP_FORCE_DEV_KERNARG=1 --purpose "27b-moe $NAME T=1 development readout" -- \
  --adapter-spec "$S/v2/27b/moe/adapters/moe-lora.json" --model-path "$CKPT" --revision "$REVISION" \
  --extra "source=$BASE" --extra "max_length=$LIMIT" --panels "$PANELS" || status=$?
python3 -m v2.27b.triton_cache finish --dest "$OUT/triton-cache" > /dev/null || true
python3 - "$LEASE" "$OUT" <<'EOF'
import importlib, json, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
lease.update({"track": "27b-moe", "status": "reserved-idle", "container": None, "last_run": sys.argv[2]})
owner.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
EOF
[ "$status" = 0 ] || exit "$status"
ref=()
[ -z "$REF" ] || ref=(--htdev2-reference "$REF")
python3 -m v2.eval.dev_readout --run-dir "$OUT" --label "$NAME" --output "$OUT/READOUT.json" "${ref[@]}"
echo "moe readout $NAME complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
