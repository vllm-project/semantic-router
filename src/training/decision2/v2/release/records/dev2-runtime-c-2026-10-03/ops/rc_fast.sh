#!/usr/bin/env bash
# Runtime C (inference owner 885d85cc, track=runtime-c; COORDINATION 2026-10-03 11:35 UTC+8): this mirror's runtime and
# Transformers remote code against each repository's current main, on the released weights (from phase A's fast.sh).
#   --preview          downloads main (the old side) and builds the same package with this mirror's runtime and remote
#                      code (make_rc.py preview); the manifest diff may change only decision2/*.py outside _vendor, the
#                      root remote code and card files; identity, parameters, profile and max_input_tokens are equal
#   --parity --gpu N   every prompt of the four scored panels (typed-final 1,600, css15 6,547, public231 231, mlx-diag
#                      2,275) as single requests through the old and then the new package, one isolated container each,
#                      each from a copy of the autotune cache main's release used; then compare-answers at tolerance 0
#   --bench --gpu N    runtime_bench.py run on the first 400 typed-final prompts (two untimed passes, then one timed)
#   --shared --gpu N   runtime_bench.py shared: the public many-question request at 16, 64 and 128 questions, switch
#                      off and on, old then new
#   --tf518            (with --parity / --bench / --shared) Transformers 5.18 first on the path, as the Hub smoke runs it
# Usage: bash <mirror>/v2/release/records/dev2-runtime-c-2026-10-03/ops/rc_fast.sh <tier> <mode> [--gpu N] [--tf518]
# Work: /data/dev2/runs/runtime-c/<tier>/ (the newest preview is used by the other modes).
# Lease: /data/dev2/leases/gpuN.lock/owner.runtime-c (status set back to reserved on exit).
set -euo pipefail
tier="${1:-}" mode="${2:-}"
shift 2 || true
gpu="" tf518=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --tf518) tf518=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(preview|parity|bench|shared)$ ]] || { echo "mode: --preview|--parity|--bench|--shared" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
M=/data/dev2/src/$SRC
OPS=$S/v2/release/records/dev2-runtime-c-2026-10-03/ops
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
REL=/data/dev2/runs/release
HOST2=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
FORMAL=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
image=$FORMAL kernels=1 vram=40 base_mounts=() base_args=()
# Each tier: the release work directory that built and sealed the current main, and that release's image.
case "$tier" in
  0.6B) key=0p6b codename=Kai kernels=0 image=$HOST2 work=dev2-ras-0.6B-20261002T235739Z ;;
  0.8B) key=0p8b codename=Eos work=dev2-ras-0.8B-20261003T000237Z ;;
  2B) key=2b codename=Sol work=dev2-ras-2B-20261003T000946Z ;;
  4B) key=4b codename=Nox work=dev2-4b-lrhxall-release-20261003T003955Z ;;
  9B) key=9b codename=Lux vram=60 image=$HOST2 work=dev2-ras-9B-20261003T022358Z ;;
  27B)
    key=27b codename=Vega vram=130 work=dev2-ras-27B-20261002T213625Z
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    base_mounts=(-v "$base_repo:$base_repo:ro")
    [[ ! -d "$HFC/blobs" ]] || base_mounts+=(-v "$HFC/blobs:$HFC/blobs:ro") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
base_spec=$REL/$work/receipts/spec.json old_cache=$REL/triton/$work
name=Decision-2.0-$codename-$tier REPO=vllm-sr/$name STAGE=dev2-release-staging-rc$key
T=/data/dev2/runs/runtime-c/$tier
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$T" /data/dev2/runs/runtime-c/triton
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
copy_cache() {
  cp -a "$1" "$2"
  chmod -R u+w "$2"
  echo "cache $2 from $1 files=$(find "$2" -type f | wc -l) digest=$(digest "$2")" | tee -a "$W/logs/wall.txt"
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
  python3 "$OPS/make_rc.py" preview --base "$base_spec" --runtime-source "$S" --key "$key" \
    --output "$W/preview.spec.json"
  python3 -m v2.release.build --spec "$W/preview.spec.json" --output "$W/new/$STAGE" > "$W/build.log"
  python3 - "$W/new/$STAGE/MODEL_MANIFEST.json" "$W/old/$name/MODEL_MANIFEST.json" > "$W/receipts/preview-diff.json" <<'PY' || status=$?
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
allowed = {"README.md", "config.json", "modeling_decision2.py", "pipeline_decision2.py"}
other = [n for n in changed if n not in allowed and not (n.startswith("decision2/") and not n.startswith("decision2/_vendor/"))
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
lease=/data/dev2/leases/gpu$gpu.lock
grep -q '^track=runtime-c' "$lease/owner.runtime-c" || { echo "GPU$gpu is not leased to runtime-c" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" "$vram" "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
W0=$(ls -d "$T"/preview-* | tail -1)
OLD=$W0/old/$name NEW=$W0/new/$STAGE
[[ -f "$OLD/MODEL_MANIFEST.json" && -f "$NEW/MODEL_MANIFEST.json" ]] || { echo "run --preview first" >&2; exit 1; }
W=$T/${mode#--}${tf518:+-tf518}-$TS
mkdir -p "$W/logs" "$W/receipts"
sed -i "s/^status=.*/status=busy/; s|^purpose=.*|purpose=runtime C $name ${mode#--}${tf518:+ tf518} ($W)|" "$lease/owner.runtime-c"
trap 'sed -i "s/^status=.*/status=reserved/" "$lease/owner.runtime-c"' EXIT
echo "mirror $SRC tier $tier gpu $gpu old $OLD new $NEW work $W"
P=$W/seal-none
mkdir -p "$P"
for panel in typed-final css15 public231 mlx-diag; do : > "$P/$panel.predictions.jsonl"; done
panels=(
  "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275"
)
site_args=()
[[ -z "$tf518" ]] || site_args=(--site "$TF518")
[[ "$kernels" != 1 ]] || site_args+=(--site /opt/decision-fla --require-kernels)
source_cache=$old_cache
for side in old new; do
  pkg=$OLD
  [[ "$side" == new ]] && pkg=$NEW
  TC=/data/dev2/runs/runtime-c/triton/$tier-${mode#--}${tf518:+-tf518}-$side-$TS
  # The new side starts from the old side's cache after its run: both use the same autotuned configurations.
  copy_cache "$source_cache" "$TC"
  source_cache=$TC
  volumes=(-v "$M:$M:ro" -v "$W0:$W0:ro" -v "$W:$W" -v "$G:$G:ro" -v "$TC:$TC" -v "$TF518:$TF518:ro" "${base_mounts[@]}")
  envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1
        -e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$TC" -e "HF_HUB_CACHE=$HFC")
  if [[ "$mode" == --parity ]]; then
    args=(examples.py parity --package "$pkg" --device cuda:0 --threads 4 --tolerance 1
          --output "$W/receipts/parity-$side.json" --answers "$W/answers-$side.jsonl")
    for panel in "${panels[@]}"; do args+=(--panel "$panel"); done
  elif [[ "$mode" == --shared ]]; then
    args=(runtime_bench.py shared --package "$pkg" --ns "16,64,128" --warmup 5 --runs 20 --threads 4
          --output "$W/receipts/shared-$side.json")
  else
    args=(runtime_bench.py run --package "$pkg" --prompts "$G/typed-final.prompts.jsonl" --count 400 --warmup 400
          --warmup-passes 2 --threads 4 --output "$W/receipts/bench-$side.json")
  fi
  started=$(date +%s.%N)
  status=0
  docker run --rm --network none --ipc host --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$gpu" "${envs[@]}" "${volumes[@]}" \
    --entrypoint python3 "$image" -I -B "$S/v2/release/${args[0]}" "${args[@]:1}" "${site_args[@]}" "${base_args[@]}" \
    > "$W/logs/$side.log" 2>&1 || status=$?
  echo "$side exit=$status wall_seconds=$(python3 -c "print(round($(date +%s.%N) - $started, 1))") cache_files=$(find "$TC" -type f | wc -l)" \
    | tee -a "$W/logs/wall.txt"
done
if [[ "$mode" == --parity ]]; then
  python3 "$S/v2/release/examples.py" compare-answers "$W/answers-old.jsonl" "$W/answers-new.jsonl" --tolerance 0 \
    --output "$W/receipts/answers-compare.json" > /dev/null || true
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({"passed": c["passed"], "max_abs_drift": c["max_abs_drift"], "panels": {k: [v["prompts"], v["identical_prompts"], v["category_changes"], v["missing"], v["max_abs_drift"]] for k, v in c["panels"].items()}}))' \
    "$W/receipts/answers-compare.json"
elif [[ "$mode" == --shared ]]; then
  python3 - "$W/receipts/shared-old.json" "$W/receipts/shared-new.json" <<'PY'
import json, sys
old, new = (json.load(open(p)) for p in sys.argv[1:3])
for n in new["per_n"]:
    o, w = old["per_n"][n], new["per_n"][n]
    print(json.dumps({"n": n, **{m: [round(o[m]["latency_ms"]["p50"], 1), round(w[m]["latency_ms"]["p50"], 1),
                                      o[m]["answers_sha256"] == w[m]["answers_sha256"]] for m in ("off", "on")},
                      "on_vs_off_changes": w["on_vs_off"]["category_changes"]}))
PY
else
  python3 "$S/v2/release/runtime_bench.py" compare "$W/receipts/bench-old.json" "$W/receipts/bench-new.json" \
    --output "$W/receipts/compare.json" > /dev/null || true
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("items","bit_identical_items","totals","latency_ms","memory_gib","passed")}))' \
    "$W/receipts/compare.json"
fi
echo "work=$W"
