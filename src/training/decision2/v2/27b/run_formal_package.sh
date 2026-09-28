#!/usr/bin/env bash
# Package-native parity and formal post-key same-panel collection (node B host side).
# Usage: run_formal_package.sh RUN GPU MIRROR_SHA PACKAGE_DIR PACKAGE_SHA256 SOURCE_DIR \
#          MODEL_ID SOURCE_DEV_PREDICTIONS SOURCE_CSS_PREDICTIONS STAGES
# STAGES: comma list of parity (DEV + CSS pilot vs the source run, fixed
# panel_parity rule) and formal (typed FINAL, CSS15, public231). Inference uses
# the deterministic reference gated-delta path (no FLA on PYTHONPATH).
set -euo pipefail

RUN=$1 GPU=$2 SHA=$3 PACKAGE=$4 PACKAGE_SHA=$5 SOURCE=$6 MODEL_ID=$7
SOURCE_DEV=$8 SOURCE_CSS=$9 STAGES=${10}
CODE=/data/dev2/src/$SHA/src/training/decision2
OUT=/data/dev2/runs/27b/$RUN
D=/data/decision20-20260926
DEV=$D/runs/dev.prompts.jsonl
CSS=$D/runs/css-transfer-v1/css-pilot.prompts.jsonl
FINAL=$D/peer-decider4b-v3-20260927/panels/typed-final.prompts.jsonl
CSS15=$D/runs/css-transfer-v1/css-evaluation.prompts.jsonl
PUBLIC=$D/bench/jevbench-public-231/prompts.jsonl
cd "$CODE"
mkdir -p "$OUT"

python3 - "$DEV" "$CSS" "$FINAL" "$CSS15" "$PUBLIC" <<'EOF'
import hashlib, sys
expected = [
    "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
]
for path, digest in zip(sys.argv[1:], expected):
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != digest:
        raise SystemExit(f"{path}: prompt bytes differ from the frozen panel")
print("gold-free prompts verified")
EOF

collect() {  # stage-name -- NAME=PATH...
  local stage=$1
  shift 2
  local panels=()
  for spec in "$@"; do
    panels+=(--panel "${spec%%=*}=/prompts/${spec%%=*}.jsonl")
  done
  local mounts=()
  for spec in "$@"; do
    mounts+=(--mount "${spec#*=}:/prompts/${spec%%=*}.jsonl")
  done
  mkdir -p "$OUT/$stage"
  chmod 700 "$OUT/$stage"
  python3 -m v2.27b.launch --name "d2-27b-$RUN-$stage" --gpu "$GPU" --cap-hours 1.5 \
    --purpose "$RUN $stage package-native collection" --receipt "$OUT/$stage.launch.json" \
    --mount "$CODE:/code" --mount "$PACKAGE:/package" --mount "$SOURCE:/source" \
    --mount "$OUT/$stage:/out:rw" "${mounts[@]}" --env PYTHONPATH=/code -- \
    python3 -m v2.27b.package_collect --package /package --source /source \
    --expected-package-sha256 "$PACKAGE_SHA" "${panels[@]}" --output-dir /out \
    --model-id "$MODEL_ID"
}

case ",$STAGES," in *,parity,*)
  collect parity -- dev="$DEV" css-pilot="$CSS"
  python3 -m publication.panel_parity --package-manifest "$PACKAGE/MODEL_MANIFEST.json" \
    --dev-prompts "$DEV" --dev-source "$SOURCE_DEV" --dev-package "$OUT/parity/dev.predictions.jsonl" \
    --css-prompts "$CSS" --css-source "$SOURCE_CSS" --css-package "$OUT/parity/css-pilot.predictions.jsonl" \
    --output-dir "$OUT/parity/receipt" | tee "$OUT/parity/panel-parity.stdout.json"
  status=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['status'])" "$OUT/parity/panel-parity.stdout.json")
  if [ "$status" != passed ]; then
    echo "package parity $status: formal collection is not allowed" >&2
    exit 1
  fi
  ;;
esac
case ",$STAGES," in *,formal,*)
  collect formal -- typed-final="$FINAL" css15="$CSS15" public231="$PUBLIC"
  ;;
esac
echo "formal driver $RUN stages $STAGES complete"
