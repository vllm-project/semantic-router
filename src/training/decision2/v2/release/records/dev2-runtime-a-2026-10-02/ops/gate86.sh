#!/usr/bin/env bash
# The Index harness's 86-request parity gate on one downloaded Decision 2.0 package (COORDINATION 2026-10-03 02:38
# UTC+8: runtime-only revisions pass it). The same three steps as v2/eval/ix1/launch.sh parity, with the package's own
# repo_id as the engine's model_id (launch.sh still names the former organization): on one GPU of node A, in IX1's
# image with its kernel path, (1) the package's documented entry point (v2.eval.ix1.native_ref) over the gold-free
# compatibility rows with a fresh autotune cache, (2) the kit runner with the release engine over the same rows with
# a copy of that cache, (3) v2.eval.ix1.parity. Answers stay under /data/dev2/private/runtime-a/gate86/; the summary
# (counts and verdict, no answers) goes to --summary.
# Usage: gate86.sh --package DIR --repo ORG/NAME --revision SHA --gpu N --summary FILE [--base-path DIR]
set -euo pipefail
pkg="" repo="" revision="" gpu="" summary="" base_path=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --package) pkg=$2; shift 2 ;;
    --repo) repo=$2; shift 2 ;;
    --revision) revision=$2; shift 2 ;;
    --gpu) gpu=$2; shift 2 ;;
    --summary) summary=$2; shift 2 ;;
    --base-path) base_path=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -f "$pkg/MODEL_MANIFEST.json" && "$repo" == */* && "$revision" =~ ^[0-9a-f]{40}$ && "$gpu" =~ ^[0-7]$ && -n "$summary" ]] \
  || { sed -n '2,/^set -euo/p' "$0" | sed '$d' >&2; exit 2; }
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(cd "$S/../../.." && pwd)
IMAGE=decision20-train-fast:host2 IMAGE_ID_PREFIX=sha256:f83b1d10
KIT=/data/dev2/private/eval/index021/kit-87d4650b KIT_REVISION=87d4650b42b377c0291a89c1f1a879f9b31082bf
ROWS=/data/dev2/private/eval/index021/compat-86.jsonl.gz
HFC=/data/dev2/hf-cache
ENGINE=publication.decision_index_release_engine:ReleasedDecisionIndexEngine
FALLBACK="falling back to its reference PyTorch implementation"
[[ "$(docker image inspect --format '{{.Id}}' "$IMAGE")" == "$IMAGE_ID_PREFIX"* ]] || { echo "image $IMAGE is not IX1's" >&2; exit 1; }
[[ "$(git -C "$KIT" rev-parse HEAD)" == "$KIT_REVISION" ]] || { echo "kit is not at $KIT_REVISION" >&2; exit 1; }
[[ -f "$ROWS" ]] || { echo "no compatibility rows" >&2; exit 1; }
manifest_sha=$(sha256sum "$pkg/MODEL_MANIFEST.json" | cut -c1-64)
name=$(basename "$pkg")
W=/data/dev2/private/runtime-a/gate86/$name-${revision:0:8}-$(date -u +%Y%m%dT%H%M%SZ)
umask 077
mkdir -p "$W/ref/home" "$W/ref/triton" "$W/kit/home" "$W/kit/triton"

run() {  # workdir script
  local work=$1 envs=(-e ROCR_VISIBLE_DEVICES="$gpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e HF_HUB_CACHE="$HFC" -e TOKENIZERS_PARALLELISM=false -e PYTHONDONTWRITEBYTECODE=1
    -e PYTHONPATH="$S:$KIT:/opt/decision-fla" -e HOME="$1/home" -e DECISION2_PACKAGE_DIR="$pkg"
    -e TRITON_CACHE_DIR="$1/triton" -e TRITON_CACHE_AUTOTUNING=1 -e HIP_FORCE_DEV_KERNARG=1)
  [[ -z "$base_path" ]] || envs+=(-e DECISION2_BASE_DIR="$base_path")
  docker run --rm --network none --ipc host --shm-size 8g --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined "${envs[@]}" -v "$SRC:$SRC:ro" -v "$KIT:$KIT:ro" -v "$pkg:$pkg:ro" \
    -v "$HFC:$HFC:ro" -v "$(dirname "$ROWS"):$(dirname "$ROWS"):ro" -v "$work:$work" -w "$S" --entrypoint bash "$IMAGE" \
    -c "python3 -c 'import fla.ops.gated_delta_rule, causal_conv1d' || exit 97; $2"
}

run "$W/ref" "python3 -m v2.eval.ix1.native_ref --package $pkg --rows $ROWS --out $W/ref/ref.jsonl ${base_path:+--base-path $base_path}" \
  > "$W/ref/native_ref.log" 2>&1 || { echo "reference pass failed ($W)" >&2; exit 1; }
! grep -q "$FALLBACK" "$W/ref/native_ref.log" || { echo "the reference pass used the reference kernel path" >&2; exit 1; }
cp -a "$W/ref/triton/." "$W/kit/triton/"
run "$W/kit" "python3 -m decision_index run --engine $ENGINE --option model_id=$repo --option revision=$revision --option package_manifest_sha256=$manifest_sha --option device=cuda:0 --rows $ROWS --out $W/kit --compact" \
  > "$W/kit/runner.log" 2>&1 || { echo "kit pass failed ($W)" >&2; exit 1; }
! grep -q "$FALLBACK" "$W/kit/runner.log" || { echo "the kit pass used the reference kernel path" >&2; exit 1; }
status=0
PYTHONPATH="$S" python3 -m v2.eval.ix1.parity --kit "$W/kit/results.jsonl" --ref "$W/ref/ref.jsonl" --out "$W/parity.json" \
  > /dev/null || status=1
python3 - "$W/parity.json" "$summary" "$repo" "$revision" "$manifest_sha" <<'PY'
import json, sys
parity, out, repo, revision, manifest = sys.argv[1:]
p = json.load(open(parity))
keep = {k: p[k] for k in ("schema", "requests", "statuses", "questions_compared", "max_abs_dp", "tolerance", "pass")}
keep["mismatched_requests"] = len(p["mismatched_run_ids"])
json.dump({"schema": "dev2-runtime-a-gate86/1", "repo": repo, "revision": revision,
           "package_manifest_sha256": manifest, "rows": "compat-86", "kit_revision": "87d4650b", "parity": keep},
          open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({"gate86": keep}))
PY
exit "$status"
