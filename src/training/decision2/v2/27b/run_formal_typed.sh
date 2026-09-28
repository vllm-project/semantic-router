#!/usr/bin/env bash
# Formal post-key same-panel run of one ~27B LoRA checkpoint through the eval
# track's frozen runner (node B host side): native collection on one leased GPU,
# gold-free seal, report, paired comparisons.
# Usage: run_formal_typed.sh RUN GPU SRC CHECKPOINT CALIBRATION MAX_LENGTH LABEL STAGES [COMPARATOR=DIR]...
#   SRC         mirror directory name under /data/dev2/src (<sha>[-src_training_decision2])
#   CHECKPOINT  selected checkpoint directory; CALIBRATION its CAL report at MAX_LENGTH
#   STAGES      comma list of smoke (8 items per formal panel into RUN-smoke), collect, score
#   COMPARATOR  NAME=RUN_DIR of a sealed same-panel run (paired bootstrap, 5,000 draws)
set -euo pipefail

RUN=$1 GPU=$2 SRC=$3 CKPT=$4 CAL=$5 MAXLEN=$6 LABEL=$7 STAGES=$8
shift 8
S=/data/dev2/src/$SRC/src/training/decision2
BASE=/data/decision20-20260926/models/Qwen3.8-27B
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
OUT=/data/dev2/runs/27b/$RUN
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
case "$GPU" in 5 | 6) ;; *) echo "GPU$GPU is outside the ~27B allocation" >&2; exit 2 ;; esac
[ -f "$S/v2/27b/adapters/typed-lora.json" ] || { echo "missing mirror $SRC" >&2; exit 2; }
cd "$S"
export PYTHONPATH=$S
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

MODEL_SHA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_sha256'])" "$CAL")
REVISION="checkpoint-sha256:$MODEL_SHA"

take_lease() {  # the lease must be ours and idle; the runner rewrites it as KEY=VALUE lines
  python3 - "$LEASE" <<'EOF'
import importlib, pathlib, sys
lease = importlib.import_module("v2.27b.launch").read_lease(pathlib.Path(sys.argv[1]))
if lease.get("track") != "27b" or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
EOF
  printf 'track=27b\npurpose=%s\n' "$1" > "$LEASE"
}
release_lease() {
  python3 - "$LEASE" "$1" <<'EOF'
import importlib, json, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
lease.update({"track": "27b", "status": "reserved-idle", "container": None, "last_run": sys.argv[2]})
owner.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
EOF
}
collect() {  # run-dir [--max-items N]
  local dir=$1
  shift
  python3 -c "import importlib, sys; importlib.import_module('v2.27b.launch').wait_until_idle(int(sys.argv[1]))" "$GPU"
  take_lease "27b formal same-panel collection $RUN"
  local status=0
  "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b --src "$SRC" --run-dir "$dir" \
    --model-dir "$CKPT" --mount "$BASE" --mount "$(dirname "$CAL")" --image "$IMAGE" \
    --purpose "27b $RUN post-key same-panel collection" \
    -- --adapter-spec "$S/v2/27b/adapters/typed-lora.json" --model-path "$CKPT" \
    --revision "$REVISION" --extra "source=$BASE" --extra "calibration=$CAL" \
    --extra "max_length=$MAXLEN" "$@" || status=$?
  release_lease "$dir"
  return "$status"
}

if has smoke; then
  collect "$OUT-smoke" --max-items 8
fi
if has collect; then
  collect "$OUT"
fi
if has score; then
  python3 -m v2.eval.same_panel seal --run-dir "$OUT"
  python3 -m v2.eval.same_panel report --run-dir "$OUT" --label "$LABEL" --tier 27B \
    --family decision2 --model-id llm-semantic-router/DEV2.0-27B --revision "$REVISION" \
    --loaded-parameters 25688227840 \
    --parameter-source "pinned base text backbone 25,624,600,064 + LoRA 58,363,904 + head 5,263,872 (safetensors headers)"
  for spec in "$@"; do
    python3 -m v2.eval.same_panel compare --run-dir "$OUT" --comparator-run-dir "${spec#*=}" \
      --left-name "$LABEL" --right-name "${spec%%=*}"
  done
fi
echo "formal driver $RUN stages $STAGES complete"
