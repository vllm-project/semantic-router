#!/usr/bin/env bash
# Speed-up phase A (user approval 2026-10-02 18:00 UTC+8, COORDINATION; worker 2d541b40, track=runtime-a): the exact
# GPU fast path of runtime/fast.py against the released runtime, on the released weights of one Decision 2.0 tier.
#   --preview          downloads the repository's current main (the released package: old side) and builds the same
#                      package with the new runtime (make_fast.py preview: staging spec, runtime_source = this
#                      mirror); the manifest diff may change only decision2/*.py outside _vendor and card files
#   --parity --gpu N   every prompt of the four scored panels (typed-final 1,600, css15 6,547, public231 231,
#                      mlx-diag 2,275) as single requests through the old and then the new package, one isolated
#                      container each, each with a fresh copy of the tier's frozen autotune cache; then
#                      examples.py compare-answers old new at tolerance 0 (0 answer changes, max drift 0.0)
#   --bench --gpu N    runtime_bench.py on the first 400 typed-final prompts, old then new, two untimed passes
#                      (the fast path captures a shape's HIP graph on its second use) then one timed; compare
# Usage: bash <mirror>/v2/release/records/dev2-runtime-a-2026-10-02/ops/fast.sh <tier> <mode> [--gpu N]
# Work: /data/dev2/runs/runtime-a/<tier>/ (the newest preview is used by --parity and --bench).
# Lease: /data/dev2/leases/gpuN.lock/owner.runtime-a (a co-tenant entry; status set to idle on exit).
set -euo pipefail
tier="${1:-}" mode="${2:-}"
shift 2 || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(preview|parity|bench)$ ]] || { echo "mode: --preview|--parity|--bench" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
M=/data/dev2/src/$SRC
OPS=$S/v2/release/records/dev2-runtime-a-2026-10-02/ops
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
image=decision20-train-fast:host2 kernels=1 frozen="" frozen_digest="" cache_tool=digest PM="" base_mounts=() base_args=()
case "$tier" in
  0.6B)
    key=0p6b codename=Kai kernels=0 base_spec=$S/v2/release/specs/dev2-0p6b-card4.json
    P=/data/dev2/runs/06b/m8/formal/m8-s5-b05/output PM=/data/dev2/runs/06b/m8/formal/m8-s5-b05-mlx/output ;;
  *) echo "tier must be one of: 0.6B" >&2; exit 2 ;;
esac
name=Decision-2.0-$codename-$tier REPO=vllm-sr/$name STAGE=dev2-release-staging-ra$key
T=/data/dev2/runs/runtime-a/$tier
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" "$T" /data/dev2/runs/runtime-a/triton
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
copy_cache() {
  if [[ -z "$frozen" ]]; then mkdir -p "$1"; return; fi
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$1")
  else
    [[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
    cp -a "$frozen" "$1"
    chmod -R u+w "$1"
  fi
}

if [[ "$mode" == --preview ]]; then
  W=$T/preview-$TS
  mkdir -p "$W/old" "$W/receipts"
  "$HFPY" - "$REPO" "$W/old/$name" "$W/receipts/download.json" <<'PY'
import json, sys
from huggingface_hub import HfApi, snapshot_download
repo, target, receipt = sys.argv[1:]
info = HfApi().model_info(repo)
path = snapshot_download(repo, revision=info.sha, local_dir=target)
json.dump({"repo_id": repo, "revision": info.sha, "private": info.private, "path": path}, open(receipt, "x"), indent=1)
print(f"downloaded {repo}@{info.sha}")
PY
  python3 "$OPS/make_fast.py" preview --base "$base_spec" --runtime-source "$S" --key "$key" \
    --output "$W/preview.spec.json"
  python3 -m v2.release.build --spec "$W/preview.spec.json" --output "$W/new/$STAGE" > "$W/build.log"
  python3 - "$W/new/$STAGE/MODEL_MANIFEST.json" "$W/old/$name/MODEL_MANIFEST.json" > "$W/receipts/preview-diff.json" <<'PY' || status=$?
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
card = {"README.md", "config.json"}
other = [n for n in changed if n not in card and not (n.startswith("decision2/") and not n.startswith("decision2/_vendor/"))
         and not n.startswith("assets/")]
same = all(new[k] == old[k] for k in ("identity", "parameters", "profile", "max_input_tokens"))
print(json.dumps({"changed": changed, "changed_other_files": other, "identity_parameters_equal": same,
                  "ok": not other and same}, indent=1))
sys.exit(0 if not other and same else 1)
PY
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2])); print("old", r["repo_id"], r["revision"], "| changed", d["changed"], "| ok", d["ok"])' \
    "$W/receipts/preview-diff.json" "$W/receipts/download.json"
  echo "work=$W"
  exit "${status:-0}"
fi

[[ "$gpu" =~ ^[0-7]$ ]] || { echo "--gpu N" >&2; exit 2; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 40 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
W0=$(ls -d "$T"/preview-* | tail -1)
OLD=$W0/old/$name NEW=$W0/new/$STAGE
[[ -f "$OLD/MODEL_MANIFEST.json" && -f "$NEW/MODEL_MANIFEST.json" ]] || { echo "run --preview first" >&2; exit 1; }
W=$T/${mode#--}-$TS
mkdir -p "$W/logs" "$W/receipts"
lease=/data/dev2/leases/gpu$gpu.lock/owner.runtime-a
printf 'track=runtime-a\nstatus=busy\npurpose=speed-up phase A %s %s (worker 2d541b40; release-worker co-tenant)\nstart_utc=%s\nrun_dir=%s\n' \
  "$name" "${mode#--}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$W" > "$lease"
trap 'printf "track=runtime-a\nstatus=idle (last run %s ended %s)\n" "$W" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$lease"' EXIT
echo "mirror $SRC tier $tier gpu $gpu old $OLD new $NEW work $W"
panels=(
  "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  "mlx-diag:$G/mlx-diag.prompts.jsonl:${PM:-$P}/mlx-diag.predictions.jsonl:2275"
)
kernel_args=()
[[ "$kernels" != 1 ]] || kernel_args=(--site /opt/decision-fla --require-kernels)
for side in old new; do
  pkg=$OLD
  [[ "$side" == new ]] && pkg=$NEW
  TC=/data/dev2/runs/runtime-a/triton/$tier-${mode#--}-$side-$TS
  copy_cache "$TC"
  volumes=(-v "$M:$M:ro" -v "$W0:$W0:ro" -v "$W:$W" -v "$G:$G:ro" -v "$P:$P:ro" -v "$TC:$TC" "${base_mounts[@]}")
  [[ -z "$PM" ]] || volumes+=(-v "$PM:$PM:ro")
  envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1
        -e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$TC" -e "HF_HUB_CACHE=$HFC")
  if [[ "$mode" == --parity ]]; then
    args=(examples.py parity --package "$pkg" --device cuda:0 --threads 4 --tolerance 1
          --output "$W/receipts/parity-$side.json" --answers "$W/answers-$side.jsonl")
    for panel in "${panels[@]}"; do args+=(--panel "$panel"); done
  else
    args=(runtime_bench.py run --package "$pkg" --prompts "$G/typed-final.prompts.jsonl" --count 400 --warmup 400
          --warmup-passes 2 --threads 4 --output "$W/receipts/bench-$side.json")
  fi
  started=$(date +%s.%N)
  status=0
  docker run --rm --network none --ipc host --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$gpu" "${envs[@]}" "${volumes[@]}" \
    --entrypoint python3 "$image" -I -B "$S/v2/release/${args[0]}" "${args[@]:1}" "${kernel_args[@]}" "${base_args[@]}" \
    > "$W/logs/$side.log" 2>&1 || status=$?
  echo "$side exit=$status wall_seconds=$(python3 -c "print(round($(date +%s.%N) - $started, 1))") cache_files=$(find "$TC" -type f | wc -l)" \
    | tee -a "$W/logs/wall.txt"
done
if [[ "$mode" == --parity ]]; then
  python3 "$S/v2/release/examples.py" compare-answers "$W/answers-old.jsonl" "$W/answers-new.jsonl" --tolerance 0 \
    --output "$W/receipts/answers-compare.json" > /dev/null || true
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({"passed": c["passed"], "max_abs_drift": c["max_abs_drift"], "panels": {k: [v["prompts"], v["identical_prompts"], v["category_changes"], v["missing"], v["max_abs_drift"]] for k, v in c["panels"].items()}}))' \
    "$W/receipts/answers-compare.json"
else
  python3 "$S/v2/release/runtime_bench.py" compare "$W/receipts/bench-old.json" "$W/receipts/bench-new.json" \
    --output "$W/receipts/compare.json" > /dev/null || true
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("items","bit_identical_items","totals","latency_ms","memory_gib","passed")}))' \
    "$W/receipts/compare.json"
fi
echo "work=$W"
