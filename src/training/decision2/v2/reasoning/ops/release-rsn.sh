#!/usr/bin/env bash
# Reasoning release, node side, one stage per call (work dir W holds every receipt):
#   card     W/card-inputs.json -> W/card/{README.md, assets/, card-receipt.json} (card-render venv, CPU)
#   build    W/card + scored BF16 copy + the base's released package -> W/package/<name> (verify_bundle inside)
#   checks   on GPU N: native System One examples in two processes (bit-identical), the card's Transformers block,
#            AutoModel / pipeline("decision") with trust_remote_code vs the native run, single-request latency of
#            the new package and of the base package over the first 400 typed-final prompts (same GPU, in turn)
# The receipts hold hashes, counts and timings only.
#
# usage: release-rsn.sh <mirror-dir> <stage> <W> <name> <base-template> <checkpoint> <model-sha256> <base-rev> [GPU] [CPUS]
set -euo pipefail
src=$1 stage=$2 W=$3 name=$4 template=$5 ckpt=$6 model=$7 base_rev=$8 gpu=${9:-} cpus=${10:-0-15}
S=/data/dev2/src/$src/src/training/decision2
IMG=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
PKG=$W/package/$name
TOOLS=/data/dev2/tools/card-render
mkdir -p "$W"
cpu_run() {
  docker run --rm --network none --cpuset-cpus "$cpus" -e PYTHONDONTWRITEBYTECODE=1 -e HIP_VISIBLE_DEVICES= \
    -e PYTHONPATH="$S" -v "$S:$S:ro" -v "$template:$template:ro" -v "$ckpt:$ckpt:ro" -v "$W:$W" -w "$S" \
    --entrypoint python3 "$IMG" -B "$@"
}
gpu_run() {  # isolated interpreter, the package's own runtime, one GPU, the image's kernels, a persisted cache
  local bdf render
  bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 == g {getline; print tolower($2)}')
  render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
  mkdir -p "$W/triton"
  docker run --rm --network none --cpuset-cpus "$cpus" --device /dev/kfd --device "$render" --group-add video \
    --group-add render --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TRITON_CACHE_DIR="$W/triton" -e TRITON_CACHE_AUTOTUNING=1 \
    -e HIP_FORCE_DEV_KERNARG=1 -e HOME="$W/home" -v "$S:$S:ro" -v "$template:$template:ro" -v "$W:$W" -w "$S" \
    --entrypoint python "$IMG" -I -B "$@"
}
case $stage in
  card)
    (cd "$S" && PYTHONPATH=$S taskset -c "$cpus" "$TOOLS/venv/bin/python" -B -m v2.reasoning.card \
      --inputs "$W/card-inputs.json" --logo "$TOOLS/logo.png" --fonts "$TOOLS/fonts" --output "$W/card") ;;
  build)
    [[ -e $PKG ]] && { echo "$PKG exists" >&2; exit 2; }
    cpu_run -m v2.reasoning.release_pkg --template "$template" --checkpoint "$ckpt" --model-sha256 "$model" \
      --repo-id "vllm-sr/$name" --origin-revision "$base_rev" --card "$W/card/README.md" --assets "$W/card/assets" \
      --scored "$W/scored.json" --out "$PKG" | tee "$W/build.json" ;;
  checks)
    [[ -n $gpu ]] || { echo "checks need a GPU" >&2; exit 2; }
    E=$S/v2/release/examples.py B=$S/v2/release/runtime_bench.py
    P=/data/dev2/private/panels/goldfree/typed-final.prompts.jsonl
    mkdir -p "$W/checks"
    gpu_run "$E" run --package "$PKG" --output "$W/checks/run-a.json" --site /opt/decision-fla --require-kernels
    gpu_run "$E" run --package "$PKG" --output "$W/checks/run-b.json" --site /opt/decision-fla --require-kernels
    gpu_run "$E" compare "$W/checks/run-a.json" "$W/checks/run-b.json" --output "$W/checks/repeat.json"
    gpu_run "$E" automap --package "$PKG" --output "$W/checks/automap.json" --site /opt/decision-fla \
      --reference "$W/checks/run-a.json" --require-kernels
    gpu_run "$E" automap-card --package "$PKG" --output "$W/checks/automap-card.json" --site /opt/decision-fla \
      --require-kernels
    for side in new base; do
      pkg=$PKG
      [[ $side == base ]] && pkg=$template
      docker run --rm --network none --cpuset-cpus "$cpus" --device /dev/kfd \
        --device "$(readlink -f "/dev/dri/by-path/pci-$(amd-smi list 2>/dev/null | awk -v g="GPU: $gpu" '$0 == g {getline; print tolower($2)}')-render")" \
        --group-add video --group-add render --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=0 \
        -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRITON_CACHE_DIR="$W/triton" -e TRITON_CACHE_AUTOTUNING=1 \
        -e HIP_FORCE_DEV_KERNARG=1 -e HOME="$W/home" -v "$S:$S:ro" -v "$template:$template:ro" -v "$W:$W" \
        -v "$P:$P:ro" -w "$S" --entrypoint python "$IMG" -I -B "$B" run --package "$pkg" --prompts "$P" \
        --count 400 --warmup 20 --warmup-passes 2 --output "$W/checks/bench-$side.json" --site /opt/decision-fla \
        --require-kernels
    done
    python3 - "$W/checks" << 'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
out = {}
for side in ("new", "base"):
    b = json.loads((d / f"bench-{side}.json").read_text())
    out[side] = {k: b.get(k) for k in ("latency_ms", "items_per_second", "memory") if k in b}
for name in ("repeat", "automap", "automap-card"):
    r = json.loads((d / f"{name}.json").read_text())
    out[name] = {k: r.get(k) for k in ("status", "identical", "max_drift", "answers", "passed") if k in r}
(d / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
EOF
    ;;
  *) echo "unknown stage $stage" >&2; exit 2 ;;
esac
