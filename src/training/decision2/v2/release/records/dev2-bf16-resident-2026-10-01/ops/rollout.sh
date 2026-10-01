#!/usr/bin/env bash
# BF16-resident runtime rollout (coordinator note 2026-10-01 10:30 UTC+8, b5f60b33): one runtime-only revision of
# one DEV2.0 repository, whose package runtime comes from the BF16-resident commit (spec runtime_source).
#   --preview           CPU build of the draft spec (weights kept, for --bench) and the manifest diff against the
#                       released revision's download: only decision2/*.py outside _vendor and card files may change;
#                       identity and parameter counts equal
#   --bench --gpu N     old runtime (the verified download of the released revision) vs new runtime (the newest
#                       preview build): the first 400 typed-final prompts as single requests, once untimed (first
#                       use of each input shape) and then timed, one isolated container process each (old first),
#                       each with a fresh copy of the frozen autotune cache, the scored image and kernels; then the
#                       answers and latency comparison
#   --mlx-only --gpu N  no-upload release.sh run with mlx-diag parity only, with the mlx run's own cache (4B)
#   --release --gpu N   release.sh --upload --collect --already-collected with the final spec and decision and full
#                       parity before and after the real download (typed-final 1,600, css15 6,547, public231 231 and
#                       mlx-diag 2,275 unless --mlx-only covers it); then the runtime diff against the replaced
#                       revision (weights byte-identical), the example answers against the replaced revision's
#                       pre-upload receipt (bit-identical), collection order, card HTTP, links, gate evaluate, storage
# Usage: bash <mirror>/v2/release/records/dev2-bf16-resident-2026-10-01/ops/rollout.sh <tier> <mode> [--gpu N]
#   node A GPU0-1 for 0.6B-9B (the 9B's scored image and inputs exist only on node A); node B GPU2-4 for 27B.
#   Shared lease /data/dev2/leases/gpuN.lock/owner.release-bf16r.
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
[[ "$mode" =~ ^--(preview|bench|mlx-only|release)$ ]] || { echo "mode: --preview|--bench|--mlx-only|--release" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
M=/data/dev2/src/$SRC
R=$S/v2/release/records/dev2-bf16-resident-2026-10-01
RECORDS=$S/v2/release/records
RENAME_OPS=$RECORDS/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-bf16r
image=decision20-train-fast:host2 kernels=1 gpus="0 1" vram=60
base_args=() bench_base=() tolerance_args=() frozen="" frozen_digest="" cache_tool=digest mlx_cache="" mlx_digest="" PM=""
case "$tier" in
  0.6B)
    key=0p6b released=476fe984a2316519f3e583b7f31b1670295b2477
    manifest=5a3317a2034bf7fe9cc75ac228d6434907730ef97e0be7446c639851ad721dde
    OLD=/data/dev2/runs/release/dev2-c1pk-0.6B-20260929T174324Z/download/DEV2.0-0.6B
    OLD_PRE=$RECORDS/dev2-0p6b-c1postkey-card-2026-09-30/0p6b/receipts/pre-a.json
    P=/data/dev2/runs/06b/m8/formal/m8-s5-b05/output PM=/data/dev2/runs/06b/m8/formal/m8-s5-b05-mlx/output
    IN=/data/dev2/runs/release/inputs/dev2-0p6b-m8-bf16 kernels=0 tolerance_args=(--parity-tolerance 0) ;;
  0.8B)
    key=0p8b released=bede7938a8c209c09f27400b79eed57948d6b75e
    manifest=a653dde2f40bd0f67300f64343bcc0537a486fcf116c2de0994984ad1399fcc2
    OLD=/data/dev2/runs/release/dev2-bf16-0.8B-20260929T224530Z/download/DEV2.0-0.8B
    OLD_PRE=$RECORDS/dev2-bf16-storage-2026-09-30/0p8b/release/receipts/pre-a.json
    P=/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-0p8b-bf16
    frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b released=a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
    manifest=637a8af08bb95c2a91c17dc4899f7406b85f9541661cb447425228f23b45846a
    OLD=/data/dev2/runs/release/dev2-bf16-2B-20260929T223257Z/download/DEV2.0-2B
    OLD_PRE=$RECORDS/dev2-bf16-storage-2026-09-30/2b/release/receipts/pre-a.json
    P=/data/dev2/runs/release/inputs/dev2-2b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-2b-bf16
    frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  4B)
    key=4b released=fadbba4ff671b4948fb7530fa6748f522b1ac9e4
    manifest=d683a54182fb46718cf55db2c567248950e13d55fac972317bffad9a5f8096ab
    OLD=/data/dev2/runs/release/dev2-bf16-4B-20260929T223705Z/download/DEV2.0-4B
    OLD_PRE=$RECORDS/dev2-bf16-storage-2026-09-30/4b/release/receipts/pre-a.json
    P=/data/dev2/runs/release/inputs/dev2-4b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-4b-bf16
    frozen=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
    frozen_digest=438618a6e3beb39407dabb96e689b5740d021a9c93b9df9769453c3d94deb119
    mlx_cache=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-triton
    mlx_digest=ba8f21322724f57beafac7dbce5348af0a9c148368d89394e459287226c481fc ;;
  9B)
    key=9b released=e51f9881b92f646cb0bd62b2876d4878cc8d16ec
    manifest=e01993466799d45dbada6e88189f05f164bc405227d166fc32eeacedc6466cc4
    OLD=/data/dev2/runs/release/dev2-c1card-9B-20260929T165656Z/download/DEV2.0-9B
    OLD_PRE=$RECORDS/dev2-c1-card-pass-2026-09-29/9b/receipts/pre-a.json
    P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived IN=/data/dev2/runs/release/inputs/dev2-8b-bf16
    frozen=/data/dev2/runs/9b/formal-m4/triton-cache
    frozen_digest=5604ffdc5f1916068c0b8df0526efc455f7bbec708081533020b834aba52586d ;;
  27B)
    key=27b released=5323310327e52d4eadd119cd10accac9b106c97d
    manifest=82c71c2e232be71e842784cf8f305b1de7d81ff67a58fe11b4464f88b2e62d37
    OLD=/data/dev2/runs/release/inputs/dev2-27b-a20r-download/DEV2.0-27B
    OLD_PRE=$RECORDS/dev2-27b-a20r-release-2026-09-30/release/receipts/pre-a.json
    P=/data/dev2/runs/27b/M4-A20r-soup/formal/output PM=/data/dev2/runs/27b/m4-mlx/M4-A20r-soup/output
    IN=/data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint
    frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache cache_tool=27b
    frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
    image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 gpus="2 3 4" vram=130
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_snapshot=$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
    base_args=(--base-path "$base_snapshot" --env "HF_HUB_CACHE=$HFC" --mount "$base_repo")
    # node B's cache links model blobs into the shared store
    [[ ! -d "$HFC/blobs" ]] || base_args+=(--mount "$HFC/blobs")
    bench_base=(--base-path "$base_snapshot") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
install_decision() {
  if [[ -e "$2" ]]; then cmp "$1" "$2"; else cp "$1" "$2"; chmod 444 "$2"; fi
}
copy_cache() {
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$1" --expect "$2" --dest "$3")
  else
    [[ "$(digest "$1")" == "$2" ]] || { echo "frozen cache $1 changed" >&2; exit 1; }
    cp -a "$1" "$3"
    echo "cache copy $3 files=$(find "$3" -type f | wc -l) digest=$(digest "$3")"
  fi
}
finish_cache() {
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$1") || true
  else
    echo "cache after run $1 files=$(find "$1" -type f | wc -l) digest=$(digest "$1")"
  fi
}
check_gpu() {
  [[ " $gpus " == *" $gpu "* ]] || { echo "$tier uses GPU $gpus only (--gpu)" >&2; exit 2; }
  rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" "$vram" "$gpu" >/dev/null \
    || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
}
check_old() {
  [[ "$(sha256sum < "$OLD/MODEL_MANIFEST.json" | cut -c1-64)" == "$manifest" ]] \
    || { echo "$OLD is not the released revision $released" >&2; exit 1; }
}
kernel_args=() env_kernel=()
if [[ "$kernels" == 1 ]]; then
  kernel_args=(--site /opt/decision-fla --require-kernels)
  env_kernel=(--env TRITON_CACHE_AUTOTUNING=1)
fi

if [[ "$mode" == --preview ]]; then
  check_old
  install_decision "$R/$name.decision.bf16r.draft.json" "$D/$name.decision.bf16r.draft.json"
  W=/data/dev2/runs/release/dev2-bf16r-preview-$tier-$TS
  mkdir -p "$W/logs"
  (cd "$S" && python3 -m v2.release.build --spec "$S/v2/release/specs/dev2-$key-bf16r.draft.json" \
    --output "$W/package/$name") > "$W/logs/build.log"
  python3 - "$W/package/$name/MODEL_MANIFEST.json" "$OLD/MODEL_MANIFEST.json" > "$W/preview-diff.json" <<'PY' || status=$?
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
card = {"README.md", "config.json", "LICENSE", "NOTICE", "ATTRIBUTIONS.md", "LICENSING.md"}
runtime = {"decision2/__init__.py", "decision2/api.py", "decision2/qwen.py"}
other = [n for n in changed if n not in card | runtime and not n.startswith(("assets/", "evaluation/", "LICENSES/"))]
same = all(new[k] == old[k] for k in ("identity", "parameters", "profile", "max_input_tokens"))
print(json.dumps({"changed": changed, "changed_other_files": other, "identity_parameters_equal": same,
                  "ok": not other and same}, indent=1))
sys.exit(0 if not other and same else 1)
PY
  echo "preview=$W/package/$name diff=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("ok" if d["ok"] else "NOT RUNTIME-ONLY " + ",".join(d["changed_other_files"]))' "$W/preview-diff.json")"
  exit "${status:-0}"
fi

check_gpu
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT

if [[ "$mode" == --bench ]]; then
  check_old
  PKG=$(ls -d "/data/dev2/runs/release/dev2-bf16r-preview-$tier-"*"/package/$name" | tail -1)
  W=/data/dev2/runs/release/dev2-bf16r-bench-$tier-$TS
  mkdir -p "$W/logs" "$W/receipts" "/data/dev2/leases/gpu$gpu.lock"
  printf 'track=release-bf16r\npurpose=BF16-resident runtime bench %s (old vs new)\nstart_utc=%s\nexpected_end_utc=+1h\nrun_dir=%s\n' \
    "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$W" > "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"
  echo "mirror $SRC tier $tier gpu $gpu old $OLD new $PKG work $W"
  for side in old new; do
    pkg=$OLD
    [[ "$side" == new ]] && pkg=$PKG
    volumes=(-v "$M:$M:ro" -v "$pkg:$pkg:ro" -v "$W/receipts:$W/receipts" -v "$G:$G:ro")
    envs=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1)
    if [[ -n "$frozen" ]]; then
      TC=/data/dev2/runs/release/triton/dev2-bf16r-bench-$tier-$side-$TS
      copy_cache "$frozen" "$frozen_digest" "$TC"
      volumes+=(-v "$TC:$TC")
      envs+=(-e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$TC")
    fi
    if [[ ${#bench_base[@]} -gt 0 ]]; then
      volumes+=(-v "$base_repo:$base_repo:ro")
      [[ ! -d "$HFC/blobs" ]] || volumes+=(-v "$HFC/blobs:$HFC/blobs:ro")
      envs+=(-e "HF_HUB_CACHE=$HFC")
    fi
    started=$(date +%s.%N)
    docker run --rm --network none --ipc host --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$gpu" "${envs[@]}" "${volumes[@]}" \
      --entrypoint python3 "$image" -I -B "$S/v2/release/runtime_bench.py" run --package "$pkg" \
      --prompts "$G/typed-final.prompts.jsonl" --count 400 --warmup 400 --threads 4 \
      --output "$W/receipts/bench-$side.json" "${kernel_args[@]}" "${bench_base[@]}" > "$W/logs/bench-$side.log" 2>&1
    echo "$side wall_seconds=$(python3 -c "print($(date +%s.%N) - $started)")" | tee -a "$W/logs/wall.txt"
    [[ -z "$frozen" ]] || finish_cache "$TC"
  done
  python3 "$S/v2/release/runtime_bench.py" compare "$W/receipts/bench-old.json" "$W/receipts/bench-new.json" \
    --output "$W/receipts/compare.json"
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({k: c[k] for k in ("items","bit_identical_items","totals","latency_ms","memory_gib","residency","passed")}))' \
    "$W/receipts/compare.json"
  echo "work=$W"
  exit 0
fi

SPEC=$S/v2/release/specs/dev2-$key-bf16r.json
install_decision "$R/$name.decision.bf16r.json" "$D/$name.decision.bf16r.json"
TC=/data/dev2/runs/release/triton/dev2-bf16r-$tier-$TS
mounts=(--mount "$G" --mount "$P" --mount "$IN")
[[ -z "$PM" ]] || mounts+=(--mount "$PM")
if [[ "$mode" == --mlx-only ]]; then
  [[ -n "$mlx_cache" ]] || { echo "--mlx-only is for a tier whose mlx-diag run had its own cache" >&2; exit 2; }
  W=/data/dev2/runs/release/dev2-bf16r-$tier-mlx-$TS
  copy_cache "$mlx_cache" "$mlx_digest" "$TC"
  parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275")
  hub_args=()
else
  W=/data/dev2/runs/release/dev2-bf16r-$tier-$TS
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
  [[ -z "$frozen" ]] || copy_cache "$frozen" "$frozen_digest" "$TC"
  parity_args=(
    --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
    --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
    --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  )
  [[ -n "$mlx_cache" ]] || parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:${PM:-$P}/mlx-diag.predictions.jsonl:2275")
  hub_args=(--upload --collect --already-collected)
fi
cache_args=()
[[ ! -d "$TC" ]] || cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
echo "mirror $SRC tier $tier gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" --track release-bf16r --shared-lease "$LEASE" --threads 4 \
  "${kernel_args[@]}" "${env_kernel[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  "${mounts[@]}" "${tolerance_args[@]}" "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
[[ ! -d "$TC" ]] || finish_cache "$TC"
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
if [[ "$mode" == --mlx-only ]]; then
  echo "work=$W mlx-only parity passed (no upload)"
  exit 0
fi
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
cd "$S"
"$HFPY" "$R/ops/runtime_diff.py" "$REPO" "$released" "$REV" "$W/extra/runtime-diff.json" || status=1
python3 "$S/v2/release/examples.py" compare "$OLD_PRE" "$W/receipts/pre-a.json" --tolerance 0 \
  --output "$W/extra/examples-vs-released.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
