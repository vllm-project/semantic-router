#!/usr/bin/env bash
# Prepare what the runs need inside a vLLM image: LMCache, WEKA's kv-cache-tester, Qwen3-14B and the
# SemiAnalysis traces, each at the version the reported runs used. On ROCm, LMCache is built from source with
# HIP. Everything goes under WORK.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$PWD/agent-kv-tier-work}
LMCACHE_REF=${LMCACHE_REF:-v0.5.6rc5}
TESTER_REF=${TESTER_REF:-94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf}
MODEL_DIR=${MODEL_DIR:-$WORK/models/Qwen3-14B}
MODEL_REV=${MODEL_REV:-40c069824f4251a91eefaf281ebe4c544efd3e18}
mkdir -p "$WORK"
cd "$WORK"

[ -d LMCache ] || git clone --depth 1 --branch "$LMCACHE_REF" https://github.com/LMCache/LMCache.git
# LMCache's kernels build as lmcache.cuda_ops, also when hipified for ROCm.
if ! python3 -c "import lmcache.cuda_ops" 2>/dev/null; then
  if command -v hipcc >/dev/null; then
    # The reported runs removed these CUDA-only packages from the ROCm vLLM image before building LMCache.
    pip uninstall -y nixl nixl-cu12 cupy-cuda12x cufile-python cuda-pathfinder || true
    (cd LMCache && BUILD_WITH_HIP=1 PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}" CXX=hipcc \
      pip install -e . --no-build-isolation)
  else
    (cd LMCache && pip install -e . --no-build-isolation)
  fi
  python3 -c "import lmcache.cuda_ops"
fi

[ -d kv-cache-tester ] || git clone https://github.com/callanjfox/kv-cache-tester.git
git -C kv-cache-tester checkout -q "$TESTER_REF"

[ -f "$MODEL_DIR/config.json" ] || python3 -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-14B', revision='$MODEL_REV', local_dir='$MODEL_DIR')"
if [ ! -d traces ] || [ -z "$(ls traces)" ]; then
  python3 "$HERE/prepare_traces.py" --out "$WORK/traces"
fi
echo "setup done in $WORK"
