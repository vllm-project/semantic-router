#!/usr/bin/env bash
# ~27B Milestone 5: one HT-DEV v2 (`ht-dev2`) collection and readout of a 27B model on its kernel path
# (node-side host script; a development screen, never a release score).
# Usage: m5-htdev2.sh SPEC_JSON KEY GPU MIRROR_SHA [REFERENCE_KEY]
#   SPEC_JSON      model specs ({"models": {KEY: {...}}}; v2/27b/m5/htdev2-refs-nodeA.json for the references)
#   KEY            one entry of SPEC_JSON's "models"
#   GPU            a GPU of this node whose lease belongs to track 27b
#   MIRROR_SHA     code commit; the mirror /data/dev2/src/<sha>[-src_training_decision2]
#   REFERENCE_KEY  optional: a key already collected here; its ht-dev2 predictions are the paired reference
# Spec fields: kind (own | peer), model_path, revision, cache {frozen, sha256}, mounts [...], extra {...},
# own: calibration (the package's calibration.json, mounted by its directory); peer: adapter (eval adapter name).
# Output: /data/dev2/runs/27b/m5/htdev2/<KEY>/ with the runner's output, triton-cache (a fresh verified copy of
# the frozen cache) + triton-cache.post.json, and READOUT.json from v2.eval.dev_readout. STAGES (default
# collect,readout) selects the steps; `readout` alone re-scores against a new REFERENCE_KEY. PANELS (default ht-dev2)
# adds development panels to the same collection (e.g. typed-dev,css-pilot,ht-dev2); ROOT overrides the output root.
set -euo pipefail
echo "m5 htdev2 $*: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

SPEC=${1:?SPEC_JSON} KEY=${2:?KEY} GPU=${3:?GPU} MIRROR_SHA=${4:?MIRROR_SHA} REF=${5:-}
[[ "$KEY" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "KEY must be one directory name" >&2; exit 2; }
[[ -z "$REF" || "$REF" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "REFERENCE_KEY must be one directory name" >&2; exit 2; }
[[ "$MIRROR_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0-7" >&2; exit 2; }
SRC=$MIRROR_SHA
[ -d "/data/dev2/src/$SRC" ] || SRC=$MIRROR_SHA-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
SPEC=$(readlink -f "$SPEC")
STAGES=${STAGES:-collect,readout}
ROOT=${ROOT:-/data/dev2/runs/27b/m5/htdev2}
PANELS=${PANELS:-ht-dev2}
OUT=$ROOT/$KEY
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/dev2/tmp
source "$S/v2/27b/kernel_common.sh"
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

field() {  # NAME -> the spec field (JSON for objects and lists)
  python3 - "$SPEC" "$KEY" "$1" <<'EOF'
import json, sys
spec, key, name = sys.argv[1:]
model = json.load(open(spec))["models"][key]
value = model.get(name, "")
print(json.dumps(value) if isinstance(value, (dict, list)) else value)
EOF
}
KIND=$(field kind) MODEL=$(field model_path) REVISION=$(field revision)
FROZEN=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['frozen'])" "$(field cache)")
FROZEN_SHA=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['sha256'])" "$(field cache)")
mapfile -t MOUNTS < <(python3 -c "import json,sys; [print(m) for m in json.loads(sys.argv[1] or '[]')]" "$(field mounts)")
mapfile -t EXTRA < <(python3 -c "import json,sys; [print(f'{k}={v}') for k, v in json.loads(sys.argv[1] or '{}').items()]" "$(field extra)")
case "$KIND" in own | peer) ;; *) echo "kind must be own or peer, got '$KIND'" >&2; exit 2 ;; esac

if has collect; then
  [ ! -e "$OUT/GPU-TIME.json" ] || { echo "$OUT already collected" >&2; exit 1; }
  need "$MODEL" "$FROZEN" "${MOUNTS[@]}"
  opts=() args=(--revision "$REVISION" --panels "$PANELS")
  for m in "${MOUNTS[@]}"; do opts+=(--mount "$m"); done
  for e in "${EXTRA[@]}"; do args+=(--extra "$e"); done
  if [ "$KIND" = own ]; then
    CAL=$(field calibration)
    need "$CAL"
    python3 - "$CAL" <<'EOF'
import json, sys
cal = json.load(open(sys.argv[1]))
print(f"calibration {sys.argv[1]}: model_sha256 {cal.get('model_sha256')} temperatures {cal.get('temperature_by_type')}")
EOF
    args+=(--extra "source=$BASE" --extra "calibration=$CAL" --extra "max_length=32768")
  fi
  mkdir -p "$OUT"
  cache_copy "$FROZEN" "$FROZEN_SHA" "$OUT/triton-cache"
  status=0
  if [ "$KIND" = own ]; then
    runner "$OUT" "$MODEL" "$OUT/triton-cache" "27b M5 ht-dev2 $KEY (development screen)" "${opts[@]}" -- \
      "${args[@]}" || status=$?
  else
    ADAPTER=$(field adapter)
    kernel_env "$OUT/triton-cache"
    python3 -c "import importlib, sys; importlib.import_module('v2.27b.launch').wait_until_idle(int(sys.argv[1]))" "$GPU"
    take_lease "27b M5 ht-dev2 $KEY (development screen)"
    "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b --src "$SRC" --run-dir "$OUT" --model-dir "$MODEL" \
      --image "$IMAGE" "${KENV[@]}" "${opts[@]}" --purpose "27b M5 ht-dev2 $KEY (development screen)" -- \
      --adapter "$ADAPTER" --model-path "$MODEL" "${args[@]}" || status=$?
    release_lease "$OUT"
  fi
  cache_finish "$OUT/triton-cache"
  [ "$status" = 0 ] || exit "$status"
fi
if has readout; then
  ref=()
  if [ -n "$REF" ]; then
    need "$ROOT/$REF/output/ht-dev2.predictions.jsonl"
    ref=(--htdev2-reference "$ROOT/$REF/output/ht-dev2.predictions.jsonl")
  fi
  python3 -m v2.eval.dev_readout --run-dir "$OUT" --label "27b M5 $KEY" "${ref[@]}" \
    --output "$OUT/READOUT${REF:+-vs-$REF}.json"
fi
echo "m5 htdev2 $KEY stages $STAGES complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
