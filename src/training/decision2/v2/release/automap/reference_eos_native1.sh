#!/usr/bin/env bash
# Native Decision-1.0-Eos reference predictions on one leased GPU of this node: the
# published runtime of revision 3c2d6326 (decision/, MODEL_MANIFEST.json, runtime.json)
# run by the eval collector inference.run exactly as the scored run did, on the
# byte-identical weights of the current head. Optional --fla-profile DIR pins FLA's
# l2norm launch configurations (FLA_CACHE_MODE=strict, FLA_CONFIG_DIR=DIR).
#
# Usage: reference_eos_native1.sh <gpuN> <native-code-dir> <head-snapshot> <out-dir> [--fla-profile DIR]
set -euo pipefail
target="$1"; code="$2"; snapshot="$3"; out="$4"; shift 4
profile=""
[[ "${1:-}" == "--fla-profile" ]] && profile="$2"
here="$(cd "$(dirname "$0")" && pwd)"
decision2="$(cd "$here/../../.." && pwd)"
goldfree=/data/dev2/private/dev1-automap/private/panels/goldfree
[[ -e "$out" ]] && { echo "exists: $out" >&2; exit 1; }
index="${target#gpu}"
lease="/data/dev2/leases/gpu${index}.lock"
mkdir -p "$lease"
[[ -e "$lease/owner" ]] && { echo "gpu${index} is leased" >&2; exit 3; }
printf 'track=release-dev1-automap purpose=Decision-1.0-Eos native reference start_utc=%s expected_end_utc=%s shared=no\n' \
  "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$lease/owner"
trap 'rm -f "$lease/owner"' EXIT
mkdir -p "$out/package" "$out/predictions"
cp -r "$code"/. "$out/package/"
for name in backbone/model.safetensors decision_head.safetensors tokenizer.json LICENSE NOTICE; do
  mkdir -p "$(dirname "$out/package/$name")"
  cp "$(readlink -f "$snapshot/$name")" "$out/package/$name"
done
rm -rf "$out/package/.cache"
bdf=$(amd-smi list 2>/dev/null | awk -v g="GPU: $index" '$0 ~ "^"g"$" {getline; print tolower($2)}')
render=$(readlink -f "/dev/dri/by-path/pci-${bdf}-render")
args=(--rm --network none --ipc host --security-opt seccomp=unconfined --device /dev/kfd --device "$render"
  --group-add video --group-add render -e ROCR_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1
  -v /data/dev2/private:/data/dev2/private:ro -v "$decision2:$decision2:ro" -v "$out:$out" -w "$decision2")
[[ -n "$profile" ]] && args+=(-v "$profile:/fla-profile:ro" -e FLA_CACHE_MODE=strict -e FLA_CONFIG_DIR=/fla-profile)
for panel in typed-final css15 public231; do
  docker run "${args[@]}" decision20-train-fast:host2 python3 -B -m inference.run --backend eos \
    --model-path "$out/package" --model-revision 3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd \
    --input "$goldfree/$panel.prompts.jsonl" --output "$out/predictions/$panel.predictions.jsonl" \
    --device cuda:0 --over-budget-invalid 2>&1 | tail -n 2
done
