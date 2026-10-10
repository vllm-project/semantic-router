#!/usr/bin/env bash
# Permutation-average arm of d3 on one NVIDIA RTX PRO 6000 (a Hugging Face Job; see launch.py).
#
# The package is MODEL at REVISION (vllm-sr/d3 v3.0.2) seen through a view whose d3_runtime.py, d3_engine.py and
# d3_format.py come from this code snapshot (MODEL_MANIFEST.json re-hashed for them); weights are the download.
# Results go to the PRIVATE work dataset (runs/$RUN/), uploaded also when a step fails. MODE:
#   lat     kit runner on the release latency sample (10 warm-up + 750 timed), OFF then ON; latency per arm
#   par     text parity v3.0.2 runtime vs this runtime with the flag OFF (parity-600), then tta_run OFF/ON there
#   proxy   tta_run OFF/ON on the pv1 O- and S-proxy rows, shard $SHARD (i/n)
#   vision  tta_run OFF/ON on a vision proxy staged in the work dataset ($VISION_DIR: rows.jsonl.gz + images)
set -uo pipefail
: "${WORK:?private work dataset}" "${RUN:?run name}" "${MODEL:?Hub repo id}" "${MODE:?lat|par|proxy|vision}"
REVISION=${REVISION:-main}
SUITE_REPO=${SUITE_REPO:-vllm-sr/d25-index-suite-0.3}
SUITE_REVISION=${SUITE_REVISION:-50d4d82cf6e7f01d0240a58d9c4fba7a16e286b8}
PROXY_REPO=${PROXY_REPO:-vllm-sr/d25-vega-proxy}
LATENCY_SHA=${LATENCY_SHA:-}
PARITY_SHA=${PARITY_SHA:-}
SHARD=${SHARD:-0/1}
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SRC=/tmp/src
OUT=/tmp/out/$RUN
ROWS=/tmp/rows
PKG=/tmp/pkg
TTA=/tmp/pkg-tta
mkdir -p "$OUT" "$ROWS"
exec > >(tee -a "$OUT/job.log") 2>&1
export PYTHONPATH=$SRC:$SRC/decision-index-kit

finish() {
  code=$?
  echo "exit $code" > "$OUT/EXIT"
  for f in "$OUT"/*/results.jsonl "$OUT"/*/*/results.jsonl; do [ -f "$f" ] && gzip -f "$f"; done
  for i in 1 2 3; do hf upload "$WORK" "$OUT" "runs/$RUN" --repo-type dataset --commit-message "tta job $RUN (exit $code)" && break; sleep 30; done
}
trap finish EXIT
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $1"; sleep 30; done; return 1; }
fail() { echo "FAILED: $*"; exit 1; }

step environment
if ! command -v gcc >/dev/null; then
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev > /dev/null; } || fail "gcc"
fi
pip install -q "transformers==5.17.0" "accelerate>=1.10" "safetensors>=0.6" "flash-linear-attention==0.5.2" einops \
  "pillow>=10" || fail "pip install"
python -c "import torchvision" 2>/dev/null || pip install -q --no-deps "${TORCHVISION_SPEC:-torchvision==0.23.0}" || fail "torchvision"
pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -2 || true
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python - <<'PY' | tee "$OUT/environment.json"
import json, platform, torch, transformers
info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)}
for name in ("fla", "causal_conv1d", "triton"):
    try:
        info[name] = __import__(name).__version__
    except Exception as exc:
        info[name] = f"missing ({type(exc).__name__})"
print(json.dumps(info))
PY

step package
retry hf download "$MODEL" --revision "$REVISION" --local-dir $PKG > /dev/null || fail "download $MODEL"
python - "$PKG" "$SRC/d25/vega/release/package_d3" "$TTA" <<'PY' | tee "$OUT/overlay.json" || fail "overlay view"
import hashlib, json, os, sys
from pathlib import Path
pkg, code, view = (Path(p) for p in sys.argv[1:])
view.mkdir(parents=True, exist_ok=True)
changed = ("d3_runtime.py", "d3_engine.py", "d3_format.py")
manifest = json.loads((pkg / "MODEL_MANIFEST.json").read_text())
for path in pkg.iterdir():
    if path.name not in changed and path.name != "MODEL_MANIFEST.json" and not path.name.startswith("."):
        os.symlink(path, view / path.name)
record = {}
for name in changed:
    data = (code / name).read_bytes()
    (view / name).write_bytes(data)
    record[name] = {"v3.0.2": manifest["files_sha256"][name], "view": hashlib.sha256(data).hexdigest()}
    manifest["files_sha256"][name] = record[name]["view"]
    manifest["files_bytes"][name] = len(data)
(view / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(record))
PY

rows_public() {
  retry python -m d25.vega.eval.suite03 fetch --kit "$SRC/decision-index-kit" --repo "$SUITE_REPO" --revision "$SUITE_REVISION" \
    --out /tmp/suite-0.3 > "$OUT/suite.json" || fail "suite fetch"
  python -m d25.vega.release.sample --kit "$SRC/decision-index-kit" --suite-dir /tmp/suite-0.3 --n 760 --warmup 10 \
    --seed 20260926 --out $ROWS/latency-760.jsonl.gz || fail "latency sample"
  python -m d25.vega.release.sample --kit "$SRC/decision-index-kit" --suite-dir /tmp/suite-0.3 --n 600 --warmup 0 \
    --seed 20261010 --out $ROWS/parity-600.jsonl.gz || fail "parity sample"
  for pair in "latency-760:$LATENCY_SHA" "parity-600:$PARITY_SHA"; do
    got=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_ids_sha256'])" "$ROWS/${pair%%:*}.jsonl.gz.json")
    [ -z "${pair#*:}" ] || [ "$got" = "${pair#*:}" ] || fail "rows ${pair%%:*} differ from the release sample ($got)"
  done
  cp $ROWS/*.json "$OUT/"
}
kit_run() {  # kit_run <name> <rows> [engine options...]
  local name=$1 rows=$2; shift 2
  PYTHONPATH=$TTA:$PYTHONPATH python -m decision_index run --engine d3_engine:D3Engine \
    --option model=$TTA --option device=cuda:0 "$@" --rows "$rows" --out "$OUT/$name" --compact | tail -1
}

case $MODE in
lat)
  step rows; rows_public
  step latency-off
  kit_run kit-760-off $ROWS/latency-760.jsonl.gz || fail "kit run off"
  python -m d25.vega.release.compare latency --results "$OUT/kit-760-off/results.jsonl" \
    --design $ROWS/latency-760.jsonl.gz.json --out "$OUT/latency-off.json"
  step latency-on
  kit_run kit-760-on $ROWS/latency-760.jsonl.gz --option permutation_average=1 || fail "kit run on"
  python -m d25.vega.release.compare latency --results "$OUT/kit-760-on/results.jsonl" \
    --design $ROWS/latency-760.jsonl.gz.json --out "$OUT/latency-on.json"
  ;;
par)
  step rows; rows_public
  step parity-v3.0.2-vs-off
  python -m d25.vega.release.text_parity_rev --old $PKG --new $TTA --rows $ROWS/parity-600.jsonl.gz --device cuda:0 \
    --out "$OUT/text-parity-rev.json" || echo "TEXT PARITY FAILED (see text-parity-rev.json)"
  step paired-parity-600
  python -m d25.vega.tta.tta_run --package $TTA --rows $ROWS/parity-600.jsonl.gz --out "$OUT/parity-600" || fail "tta_run"
  ;;
proxy)
  step proxy-rows
  retry hf download "$PROXY_REPO" --repo-type dataset --include "pv1/*" --local-dir /tmp/proxy > /dev/null || fail "proxy download"
  step "paired-proxy $SHARD"
  python -m d25.vega.tta.tta_run --package $TTA --rows /tmp/proxy/pv1/o/*.jsonl.gz /tmp/proxy/pv1/s/*.jsonl.gz \
    --shard "$SHARD" --out "$OUT/proxy" || fail "tta_run"
  ;;
vision)
  step vision-rows
  retry hf download "$WORK" --repo-type dataset --include "$VISION_DIR/*" --local-dir /tmp/vision > /dev/null || fail "vision download"
  step "paired-vision $SHARD"
  python -m d25.vega.tta.tta_run --package $TTA --rows "/tmp/vision/$VISION_DIR/rows.jsonl.gz" --shard "$SHARD" \
    --out "$OUT/vision" || fail "tta_run"
  ;;
*) fail "unknown MODE $MODE" ;;
esac
echo ALL-DONE
