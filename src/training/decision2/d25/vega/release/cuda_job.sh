#!/usr/bin/env bash
# Decision 2.5 CUDA readiness on one NVIDIA RTX PRO 6000 (a Hugging Face Job; see hf_job.py).
#
# Self-contained: inputs come from the Hub, results go to the PRIVATE work dataset (runs/$RUN/), uploaded
# also when a step fails. Steps:
#   1. environment: pinned transformers, flash-linear-attention (+ causal-conv1d when a wheel exists)
#   2. the package: MODEL=<repo> at REVISION, or MODEL=standin (pplx-decider-v1.1-27b, public, built into
#      a package with our runtime by build.py; same layout and architecture as ours); OVERLAY=<dir in the work
#      dataset> turns the download into a new code revision of the same weights before it is uploaded
#      (step revparity: the downloaded revision's text answers are identical)
#   3. the rows: the latency-v1-style samples drawn from the private suite dataset (checked by run-id hash)
#   4. smoke: AutoModel remote code, pipeline, /v1/systemone server (and the card Quickstart if any)
#   5. latency protocol: the kit runner, one request at a time, 10 warm-up + 750 timed rows
#   6. parity: engine.py vs the package on 600 rows (same process), and the 760 answers vs a stored
#      MI325X kit run (REFERENCE, a path in the work dataset) when given
#   7. variants on the first VARIANT_ROWS timed rows: batch sizes, then without causal-conv1d / fla
set -uo pipefail
: "${WORK:?private work dataset}" "${RUN:?run name}" "${MODEL:?Hub repo id or standin}"
REVISION=${REVISION:-main}
REFERENCE=${REFERENCE:-}
SUITE_REPO=${SUITE_REPO:-vllm-sr/d25-index-suite-0.3}
SUITE_REVISION=${SUITE_REVISION:-50d4d82cf6e7f01d0240a58d9c4fba7a16e286b8}
LATENCY_SHA=${LATENCY_SHA:-}
PARITY_SHA=${PARITY_SHA:-}
VARIANTS=${VARIANTS:-"batch16 batch64 noconv nofla"}
VARIANT_ROWS=${VARIANT_ROWS:-150}
STEPS=${STEPS:-"smoke latency parity variants"}
PROBE_IDS=${PROBE_IDS:-}
PROBE_RUNTIME=${PROBE_RUNTIME:-package}
PPLX_REPO=perplexity-ai/pplx-decider-v1.1-27b
PPLX_REVISION=6195aa55a72ae21e503c8a60198f7b390d42adb5
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
SRC=/tmp/src
OUT=/tmp/out/$RUN
ROWS=/tmp/rows
PKG=/tmp/pkg
mkdir -p "$OUT" "$ROWS"
exec > >(tee -a "$OUT/job.log") 2>&1
export PYTHONPATH=$SRC:$SRC/decision-index-kit

finish() {
  code=$?
  echo "exit $code" > "$OUT/EXIT"
  for f in "$OUT"/*/results.jsonl; do [ -f "$f" ] && gzip -f "$f"; done
  for i in 1 2 3; do hf upload "$WORK" "$OUT" "runs/$RUN" --repo-type dataset --commit-message "cuda job $RUN (exit $code)" && break; sleep 30; done
}
trap finish EXIT
step() { echo; echo "=== $(date -u +%H:%M:%S) $*"; }
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $1"; sleep 30; done; return 1; }
fail() { echo "FAILED: $*"; exit 1; }

step environment
# Optional: a specific torch build first, e.g. TORCH_SPEC="torch==2.14.0" TORCH_INDEX=https://download.pytorch.org/whl/cu130
if [ -n "${TORCH_SPEC:-}" ]; then
  # The image's torchaudio is built for its own torch and breaks transformers' import under another one.
  pip uninstall -q -y torchaudio 2>/dev/null
  pip install -q "$TORCH_SPEC" ${TORCH_INDEX:+--index-url "$TORCH_INDEX"} || fail "torch $TORCH_SPEC"
fi
# Triton (flash-linear-attention's kernels) compiles its launchers with the system C compiler.
if ! command -v gcc >/dev/null; then
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc libc6-dev > /dev/null; } || fail "gcc"
fi
pip install -q "transformers==5.17.0" "accelerate>=1.10" "safetensors>=0.6" "flash-linear-attention==0.5.2" einops \
  fastapi uvicorn httpx "pillow>=10" || fail "pip install"
python -c "import torchvision" 2>/dev/null || pip install -q --no-deps "${TORCHVISION_SPEC:-torchvision==0.23.0}" || fail "torchvision"
pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -2 || true
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
python - <<'PY' | tee "$OUT/environment.json"
import json, platform, torch, transformers
info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0))}
for name in ("fla", "causal_conv1d", "triton"):
    try:
        info[name] = __import__(name).__version__
    except Exception as exc:
        info[name] = f"missing ({type(exc).__name__})"
print(json.dumps(info))
PY

step sdpa-probe
python -m d25.vega.release.sdpa_probe --out "$OUT/sdpa.json" > /dev/null || echo "SDPA PROBE FAILED (continuing)"

step rows
retry python -m d25.vega.eval.suite03 fetch --kit "$SRC/decision-index-kit" --repo "$SUITE_REPO" --revision "$SUITE_REVISION" \
  --out /tmp/suite-0.3 > "$OUT/suite.json" || fail "suite fetch"
python -m d25.vega.release.sample --kit "$SRC/decision-index-kit" --suite-dir /tmp/suite-0.3 --n 760 --warmup 10 \
  --seed 20260926 --out $ROWS/latency-760.jsonl.gz || fail "latency sample"
python -m d25.vega.release.sample --kit "$SRC/decision-index-kit" --suite-dir /tmp/suite-0.3 --n 600 --warmup 0 \
  --seed 20261010 --out $ROWS/parity-600.jsonl.gz || fail "parity sample"
check_rows() {
  got=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['run_ids_sha256'])" "$1")
  [ -z "$2" ] || [ "$got" = "$2" ] || fail "rows $1 differ from the reference sample ($got)"
}
check_rows $ROWS/latency-760.jsonl.gz.json "$LATENCY_SHA"
check_rows $ROWS/parity-600.jsonl.gz.json "$PARITY_SHA"
cp $ROWS/*.json "$OUT/"

step package
if [ "$MODEL" = standin ]; then
  retry hf download $PPLX_REPO --revision $PPLX_REVISION --local-dir /tmp/base > /dev/null || fail "download $PPLX_REPO"
  python -m d25.vega.release.build --export /tmp/base --out $PKG --model-name d25-pplx-standin \
    --repo-id $PPLX_REPO > "$OUT/build.json" || fail "build"
else
  hf download "$MODEL" --revision "$REVISION" --local-dir $PKG > /dev/null \
    || retry python -m d25.vega.release.hub fetch --repo "$MODEL" --revision "$REVISION" --out $PKG --receipt "$OUT/fetch.json" \
    || fail "download $MODEL"
fi
# OVERLAY: <dir>/package/ holds the files that differ from the download, <dir>/SHA256SUMS every file of the built
# package. $PKG becomes that package (links into the download plus the overlay files); the download stays $PKG_OLD.
PKG_OLD=
if [ -n "${OVERLAY:-}" ]; then
  retry hf download "$WORK" --repo-type dataset --include "$OVERLAY/*" --local-dir /tmp/overlay > /dev/null \
    || fail "overlay download"
  PKG_OLD=$PKG; PKG=/tmp/pkg-new
  python - "$PKG_OLD" "/tmp/overlay/$OVERLAY" "$PKG" <<'PY' || fail "overlay view"
import os, sys
from pathlib import Path
old, overlay, new = (Path(p) for p in sys.argv[1:])
for line in (overlay / "SHA256SUMS").read_text().splitlines():
    name = line.split(None, 1)[1].lstrip("*").strip()
    source = overlay / "package" / name
    (new / name).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source if source.exists() else old / name, new / name)
PY
  (cd $PKG && sha256sum -c --quiet "/tmp/overlay/$OVERLAY/SHA256SUMS") || fail "overlay view differs from the built package"
  echo "overlay $OVERLAY over $REVISION: $(cd "/tmp/overlay/$OVERLAY/package" && find . -type f | sort | tr '\n' ' ')"
fi
du -sh $PKG
# A d3-family package names its runtime d3_* and uses public format and prompt ids; the reference engines (engine.py,
# the image engine) read the same files with the internal ids, through a view that differs only in decision_config.json.
ENGINE=decision25_engine:Decision25Engine; REF=$PKG
if [ -f $PKG/d3_runtime.py ]; then
  ENGINE=d3_engine:D3Engine; REF=/tmp/pkg-reference
  python - "$PKG" "$REF" <<'PY' || fail "reference view"
import json, os, sys
from pathlib import Path
pkg, ref = Path(sys.argv[1]), Path(sys.argv[2])
ref.mkdir(parents=True, exist_ok=True)
for path in pkg.iterdir():
    if path.name != "decision_config.json" and not (ref / path.name).exists():
        os.symlink(path, ref / path.name)
config = json.loads((pkg / "decision_config.json").read_text())
assert config["prompt"] == "d3" and config["format_id"] == "d3-code-readout-v1"
config.update(format_id="d25-vega-code-readout-v1", prompt="d25-vega")
(ref / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n")
PY
fi
# The card's Quickstart runs verbatim except for the model load, which points at the downloaded revision (a second
# 52 GB download inside the job is slow); other Hub calls of the Quickstart (the example image) stay real.
CARD=(); [ -f $PKG/README.md ] && grep -q '```python' $PKG/README.md \
  && CARD=(--card "$PKG/README.md" --replace-repo "from_pretrained(\"$MODEL\"=from_pretrained(\"$PKG\"")

has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }

if has smoke; then
step smoke
python -m d25.vega.release.smoke --model $PKG --device cuda:0 "${CARD[@]}" --out "$OUT/smoke.json" \
  || echo "SMOKE FAILED (continuing)"
python -m d25.vega.release.smoke --model $PKG --device cuda:0 --server-only --expected "$OUT/smoke.json" \
  --out "$OUT/smoke-server.json" || echo "SERVER SMOKE FAILED (continuing)"
fi

kit_run() {  # kit_run <name> <rows> [engine options...]
  local name=$1 rows=$2; shift 2
  PYTHONPATH=$PKG:$PYTHONPATH python -m decision_index run --engine $ENGINE \
    --option model=$PKG --option device=cuda:0 "$@" --rows "$rows" --out "$OUT/$name" --compact | tail -1
}
if has latency; then
step latency-protocol
kit_run kit-760 $ROWS/latency-760.jsonl.gz || fail "kit run"
python -m d25.vega.release.compare latency --results "$OUT/kit-760/results.jsonl" \
  --design $ROWS/latency-760.jsonl.gz.json --out "$OUT/latency.json"
fi
if has long; then
step long-inputs
python -m d25.vega.release.long_check --package $PKG --kit "$SRC/decision-index-kit" --suite-dir /tmp/suite-0.3 \
  --top "${LONG_TOP:-20}" --out "$OUT/long.json" || echo "long-input check: not every row answered (see long.json)"
fi

if [ -n "$PROBE_IDS" ]; then
step latency-probe
read -r -a PROBE_ID_LIST <<< "$PROBE_IDS"
python -m d25.vega.release.latency_probe --package $PKG --rows $ROWS/latency-760.jsonl.gz --run-ids "${PROBE_ID_LIST[@]}" \
  --runtime-source "${PROBE_RUNTIME:-package}" --modes default --out "$OUT/latency-probe.json" || echo "PROBE FAILED (continuing)"
fi

if has parity; then
step parity
python -m d25.vega.release.parity --package $PKG --engine-ckpt $REF --rows $ROWS/parity-600.jsonl.gz --device cuda:0 \
  --sequential --out "$OUT/parity-engine.json" || echo "PARITY vs engine.py FAILED (see parity-engine.json)"
if [ -n "$REFERENCE" ]; then
  mkdir -p /tmp/work && retry curl -sfL -H "Authorization: Bearer $HF_TOKEN" -o /tmp/work/reference.jsonl.gz \
      "https://huggingface.co/datasets/$WORK/resolve/main/$REFERENCE" \
    && python -m d25.vega.release.compare answers --results "$OUT/kit-760/results.jsonl" \
      --reference /tmp/work/reference.jsonl.gz --out "$OUT/answers-vs-reference.json" | head -14
fi
fi
if has revparity && [ -n "$PKG_OLD" ]; then
step "text-parity-vs-$REVISION"
python -m d25.vega.release.text_parity_rev --old $PKG_OLD --new $PKG --rows $ROWS/parity-600.jsonl.gz --device cuda:0 \
  --out "$OUT/text-parity-rev.json" || echo "TEXT PARITY vs $REVISION FAILED (see text-parity-rev.json)"
fi

if has images; then
step images
PACK_DIR=/tmp/packdl
retry hf download vllm-sr/d25-vega-release-work --repo-type dataset --include "packs/${PACK:-vision-rt04}/*" \
  --local-dir $PACK_DIR > /dev/null || fail "pack download"
PACKP=$PACK_DIR/packs/${PACK:-vision-rt04}
IMAGE_CHECKS=${IMAGE_CHECKS:-"smoke parity latency"}
case " $IMAGE_CHECKS " in *" smoke "*)
python -m d25.omni.runtime.smoke --package "$PKG" --suite "$PACKP" --example "$PKG/assets/example-receipt.png" \
  --out "$OUT/smoke-images.json" || echo "image smoke: FAILED (see smoke-images.json)";; esac
case " $IMAGE_CHECKS " in *" parity "*)
python -m d25.omni.runtime.parity_image --engine-ckpt "$REF" --package "$PKG" --suite "$PACKP" --rows 64 --min-multi 16 \
  --kinds-rows 8 --sequential --reference "$PACKP/pack-reference.json" --out "$OUT/parity-images.json" \
  || echo "image parity: FAILED (see parity-images.json)";; esac
case " $IMAGE_CHECKS " in *" latency "*)
python -m d25.omni.runtime.latency --package "$PKG" --suite "$PACKP" --timed 100 --timed-four 50 \
  --out "$OUT/latency-images.json" || echo "image latency: FAILED";; esac
fi

has variants || { echo ALL-DONE; exit 0; }
step variants
python - "$ROWS/latency-760.jsonl.gz" "$VARIANT_ROWS" <<'PY'
import gzip, sys
# The 10 warm-up rows, then every k-th timed row: the timed rows are sorted by benchmark, so this spans all of them.
rows, n = sys.argv[1], int(sys.argv[2])
with gzip.open(rows, "rt") as src:
    lines = src.readlines()
step = max(1, (len(lines) - 10) // n)
with gzip.open(rows.replace("760", "variant"), "wt") as dst:
    dst.writelines(lines[:10] + lines[10::step][:n])
PY
for v in $VARIANTS; do
  case $v in
    batch*) kit_run "variant-$v" "$ROWS/latency-variant.jsonl.gz" --option "batch_size=${v#batch}" ;;
    cudnn) DECISION25_CUDNN_SDPA=1 kit_run "variant-$v" $ROWS/latency-variant.jsonl.gz ;;
    noconv) pip uninstall -q -y causal-conv1d 2>/dev/null; kit_run "variant-$v" $ROWS/latency-variant.jsonl.gz ;;
    nofla) pip uninstall -q -y flash-linear-attention fla-core 2>/dev/null; kit_run "variant-$v" $ROWS/latency-variant.jsonl.gz ;;
  esac
  python -m d25.vega.release.compare latency --results "$OUT/variant-$v/results.jsonl" \
    --design $ROWS/latency-760.jsonl.gz.json --out "$OUT/latency-variant-$v.json" | head -9
done
echo ALL-DONE
