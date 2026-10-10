#!/usr/bin/env bash
# Evidence chain for the image-capable runtime on one GPU. Resumable: a step with a .done marker is skipped;
# the chain stops at the first failing step. Usage: run_all.sh <code tag> [steps...]
set -uo pipefail
TAG=${1:?code tag}
shift
ROOT=/data/d25/omni/runtime
CODE=$ROOT/src/$TAG/src/training/decision2
OUT=$ROOT/runs/$TAG
PKG=$OUT/package
ORIG=${D25_RUNTIME_ORIGINAL:-/data/d25/omni/inits/vega25-s5135}
GRAFT=${D25_RUNTIME_ENGINE_CKPT:-/data/d25/omni/inits/omni-graft-s5135}
SUITE=${D25_RUNTIME_SUITE:-/data/d25/omni/suite/vision-0.3.1}
ZS=${D25_RUNTIME_ZS:-/data/d25/omni/results/zs/omni-graft-s5135/public}
STEPS=${*:-package env example parity-text parity-image latency smoke}
mkdir -p "$OUT"
cd "$CODE" || exit 2

run() {
  local name=$1
  shift
  if [ -f "$OUT/$name.done" ]; then
    echo "skip $name (done $(cat "$OUT/$name.done"))"
    return 0
  fi
  echo "=== $(date -u +%FT%TZ) $name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  tail -n 4 "$OUT/$name.log"
  if [ $rc -ne 0 ]; then
    echo "FAILED $name (exit $rc)"
    exit $rc
  fi
  date -u +%FT%TZ > "$OUT/$name.done"
}

for step in $STEPS; do
  case $step in
    package) run package python -m d25.omni.runtime.pkg --original "$ORIG" --code "$CODE/d25/omni/runtime/package" --out "$PKG" ;;
    env) run env python -m d25.omni.runtime.env --package "$PKG" --engine-ckpt "$GRAFT" --suite "$SUITE" --out "$OUT/env.json" ;;
    example) run example python -m d25.omni.runtime.example_image --out "$OUT/assets" ;;
    parity-text) run parity-text python -m d25.omni.runtime.parity_text --original "$ORIG" --package "$PKG" \
      --rows "$ROOT/samples/parity-600.jsonl.gz" --out "$OUT/parity-text.json" ;;
    parity-image) run parity-image python -m d25.omni.runtime.parity_image --engine-ckpt "$GRAFT" --package "$PKG" \
      --suite "$SUITE" --zs-results "$ZS" --save-probs "$OUT/suite-probs.json" --out "$OUT/parity-image.json" ;;
    pack) run pack python -m d25.omni.runtime.pack --out "$OUT/pack" ;;
    parity-pack) run parity-pack python -m d25.omni.runtime.parity_image --engine-ckpt "$PKG" --package "$PKG" \
      --suite "$OUT/pack" --rows 64 --min-multi 16 --kinds-rows 8 --save-probs "$OUT/pack-reference.json" \
      --out "$OUT/parity-pack.json" ;;
    parity-pack-seq) run parity-pack-seq python -m d25.omni.runtime.parity_image --engine-ckpt "$PKG" \
      --package "$PKG" --suite "$OUT/pack" --rows 64 --min-multi 16 --kinds-rows 8 --sequential \
      --reference "$OUT/pack-reference.json" --out "$OUT/parity-pack-seq.json" ;;
    latency-pack) run latency-pack python -m d25.omni.runtime.latency --package "$PKG" --suite "$OUT/pack" \
      --timed 100 --timed-four 50 --out "$OUT/latency-pack.json" ;;
    smoke-pack) run smoke-pack python -m d25.omni.runtime.smoke --package "$PKG" --suite "$OUT/pack" \
      --example "$OUT/assets/example-receipt.png" --out "$OUT/smoke-pack.json" ;;
    latency) run latency python -m d25.omni.runtime.latency --package "$PKG" --suite "$SUITE" --out "$OUT/latency.json" ;;
    smoke) run smoke python -m d25.omni.runtime.smoke --package "$PKG" --suite "$SUITE" \
      --example "$OUT/assets/example-receipt.png" --out "$OUT/smoke.json" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
echo "ALL-DONE $(date -u +%FT%TZ)"
