#!/usr/bin/env bash
# DEV2.0-4B successor m10-4b-LH (user release job 2026-10-01): node A GPU0 or GPU1, shared lease
# owner.release-4b-lh, the scored run's image dbe5f32b with its kernels, HIP_FORCE_DEV_KERNARG=1 and a fresh copy of
# the persisted autotune cache of the run that scored each panel (checked against its manifest before the copy).
#   --prerelease  release.sh without upload on the draft spec and decision: build, examples, card, parity of
#                 typed-final 1,600 / css15 6,547 / public231 231 (formal run's cache); then verify_bundle in a CPU
#                 container. The first one (7517d321, runtime 5dc962b00) is the frozen package C1 (item 8)
#                 scored; later ones carry the auto_map remote code and the merged runtime (same weights).
#   --mlx         the same without upload, parity of mlx-diag 2,275 only (the mlx-diag run's cache)
#   --bench       runtime_bench of the current revision's verified download (old) and the newest --prerelease
#                 package (new): the first 400 typed-final prompts as single requests, untimed then timed, one
#                 isolated container each, each with its own scored cache
#   --release     only if the Hub main is the superseded revision (SUPERSEDED, default the auto_map revision
#                 3785b7b9) and the Transformers 5.18 site matches TF518_DIGEST:
#                 release.sh --upload --collect --already-collected --hub-site tf518=... with the final spec and decision (successor
#                 R1-R8) and full parity of the formal panels before and after the real download (mlx-diag parity
#                 comes from --mlx on the same weights and runtime); then the frozen C1 package vs the released
#                 package (weights and identity equal), the revision diff, collection order, card HTTP, links, gate
#                 evaluate and storage
# Usage (node A): bash <mirror>/v2/release/records/dev2-4b-m10lh-2026-10-01/ops/release_lh.sh <mode> --gpu <0|1>
set -euo pipefail
mode="${1:-}"
shift || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(prerelease|mlx|bench|release)$ ]] || { echo "mode: --prerelease|--mlx|--bench|--release" >&2; exit 2; }
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this release uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
M=/data/dev2/src/$SRC
R=$S/v2/release/records/dev2-4b-m10lh-2026-10-01
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-4b-lh
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
name=DEV2.0-4B REPO=llm-semantic-router/DEV2.0-4B
IN=/data/dev2/runs/release/inputs/dev2-4b-lh
P=$IN/t1
FORMAL_CACHE=/data/dev2/runs/dec/formal/m10/m10-4b-LH-cache-frozen
FORMAL_MANIFEST=569e86f58a4718b794b0e25983cea48f94db9097836e16057890b5fd27ef024b
MLX_CACHE=$IN/mlx-cache-frozen
MLX_MANIFEST=0bfda30514ff28379b064956829871d8ab7c019a32d7326f20dfaaee34e9e130
OLD=/data/dev2/runs/release/dev2-bf16r-4B-20261001T041149Z/download/DEV2.0-4B
OLD_MANIFEST=bbf9456946a0f3a9d742c38fffcca243618cf78989457f7390cede98013910e2
OLD_CACHE=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
OLD_CACHE_DIGEST=438618a6e3beb39407dabb96e689b5740d021a9c93b9df9769453c3d94deb119
superseded=${SUPERSEDED:-3785b7b963d2f56de0e44f9ec638c814c5ee6499}
C1PKG=/data/dev2/runs/release/dev2-4b-lh-prerelease-20261001T071009Z/package/$name
C1PKG_MANIFEST=7517d321a88f946020e53ae1369e5f2d2301adf564e7fc1b61f04bb7b53c2947
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
install_decision() {
  if [[ -e "$2" ]]; then cmp "$1" "$2"; else cp "$1" "$2"; chmod 444 "$2"; fi
}
copy_cache() { # frozen manifest-file expected-manifest-sha dest
  [[ "$(sha256sum < "$2" | cut -c1-64)" == "$3" ]] || { echo "cache manifest $2 is not $3" >&2; exit 1; }
  (cd "$1" && sha256sum -c --quiet "$2") || { echo "frozen cache $1 differs from its manifest" >&2; exit 1; }
  cp -a "$1" "$4"
  chmod -R u+w "$4"
  echo "cache copy $4 files=$(find "$4" -type f | wc -l) manifest=$3"
}
finish_cache() { # dest manifest-file
  local changed
  changed=$( (cd "$1" && sha256sum -c --quiet "$2" 2>/dev/null) | wc -l || true)
  echo "cache after run $1 files=$(find "$1" -type f | wc -l) entries_differing_from_frozen=$changed"
}
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)

if [[ "$mode" == --bench ]]; then
  [[ "$(sha256sum < "$OLD/MODEL_MANIFEST.json" | cut -c1-64)" == "$OLD_MANIFEST" ]] || { echo "old download changed" >&2; exit 1; }
  NEW=$(ls -d /data/dev2/runs/release/dev2-4b-lh-prerelease-*/package/$name | tail -1)
  W=/data/dev2/runs/release/dev2-4b-lh-bench-$TS
  mkdir -p "$W/logs" "$W/receipts" "/data/dev2/leases/gpu$gpu.lock"
  printf 'track=release-4b-lh\npurpose=DEV2.0-4B LH runtime bench (old vs new)\nstart_utc=%s\nexpected_end_utc=+1h\nrun_dir=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$W" > "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"
  echo "mirror $SRC gpu $gpu old $OLD new $NEW work $W"
  for side in old new; do
    TC=/data/dev2/runs/release/triton/dev2-4b-lh-bench-$side-$TS
    if [[ "$side" == old ]]; then
      pkg=$OLD
      [[ "$(digest "$OLD_CACHE")" == "$OLD_CACHE_DIGEST" ]] || { echo "old cache changed" >&2; exit 1; }
      cp -a "$OLD_CACHE" "$TC"
      image=decision20-train-fast:host2
    else
      pkg=$NEW
      copy_cache "$FORMAL_CACHE" "$FORMAL_CACHE.sha256" "$FORMAL_MANIFEST" "$TC"
      image=$IMAGE
    fi
    started=$(date +%s.%N)
    docker run --rm --network none --ipc host --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$gpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$TC" \
      -v "$M:$M:ro" -v "$pkg:$pkg:ro" -v "$W/receipts:$W/receipts" -v "$G:$G:ro" -v "$TC:$TC" \
      --entrypoint python3 "$image" -I -B "$S/v2/release/runtime_bench.py" run --package "$pkg" \
      --prompts "$G/typed-final.prompts.jsonl" --count 400 --warmup 400 --threads 4 \
      --output "$W/receipts/bench-$side.json" --site /opt/decision-fla --require-kernels > "$W/logs/bench-$side.log" 2>&1
    echo "$side image=$image wall_seconds=$(python3 -c "print($(date +%s.%N) - $started)")" | tee -a "$W/logs/wall.txt"
  done
  python3 "$S/v2/release/runtime_bench.py" compare "$W/receipts/bench-old.json" "$W/receipts/bench-new.json" \
    --output "$W/receipts/compare.json" || true
  python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({k: c.get(k) for k in ("items","totals","latency_ms","memory_gib","residency")}))' \
    "$W/receipts/compare.json"
  echo "work=$W"
  exit 0
fi

if [[ "$mode" == --release ]]; then
  SPEC=$S/v2/release/specs/dev2-4b-m10lh.json
  install_decision "$R/$name.decision.m10lh.json" "$D/$name.decision.m10lh.json"
  W=/data/dev2/runs/release/dev2-4b-lh-release-$TS
else
  SPEC=$S/v2/release/specs/dev2-4b-m10lh.draft2.json
  install_decision "$R/$name.decision.m10lh.draft2.json" "$D/$name.decision.m10lh.draft2.json"
  W=/data/dev2/runs/release/dev2-4b-lh-${mode#--}-$TS
fi
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-4b-lh-profile-$TS.json" \
  || { echo "successor profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-4b-lh-${mode#--}-$TS
formal_parity=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
)
case "$mode" in
  --prerelease) copy_cache "$FORMAL_CACHE" "$FORMAL_CACHE.sha256" "$FORMAL_MANIFEST" "$TC"; manifest_file=$FORMAL_CACHE.sha256
    parity_args=("${formal_parity[@]}") hub_args=() ;;
  --mlx) copy_cache "$MLX_CACHE" "$MLX_CACHE.sha256" "$MLX_MANIFEST" "$TC"; manifest_file=$MLX_CACHE.sha256
    parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275") hub_args=() ;;
  --release)
    # Publish only on top of the revision the final decision supersedes (never concurrently with another worker).
    main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
    [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
    [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
      || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
    bash "$S/v2/common/hf_headroom.sh" --min-free-gb 10
    copy_cache "$FORMAL_CACHE" "$FORMAL_CACHE.sha256" "$FORMAL_MANIFEST" "$TC"; manifest_file=$FORMAL_CACHE.sha256
    parity_args=("${formal_parity[@]}") hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518") ;;
esac
echo "mirror $SRC mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track release-4b-lh --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$P" --mount "$IN" \
  "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
finish_cache "$TC" "$manifest_file"
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
mkdir -p "$W/extra"
PKG=$W/package/$name
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -v "$PKG:$PKG:ro" -w /tmp \
  --entrypoint python3 "$IMAGE" -I -B -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from decision2 import verify_bundle; m = verify_bundle(sys.argv[1]); print(json.dumps({"verify_bundle": "ok", "files": len(m.get("files_sha256") or {}), "identity": (m.get("identity") or {}).get("model_sha256")}))' \
  "$PKG" | tee "$W/extra/verify-bundle.json"
if [[ "$mode" != --release ]]; then
  echo "work=$W package=$PKG manifest=$(sha256sum < "$PKG/MODEL_MANIFEST.json" | cut -c1-64) (no upload)"
  exit 0
fi
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
cd "$S"
[[ "$(sha256sum < "$C1PKG/MODEL_MANIFEST.json" | cut -c1-64)" == "$C1PKG_MANIFEST" ]] || status=1
python3 - "$PKG/MODEL_MANIFEST.json" "$C1PKG/MODEL_MANIFEST.json" > "$W/extra/c1-package-diff.json" <<'PY' || status=1
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
weights = [n for n in changed if n.endswith(".safetensors") or n.startswith("backbone/") or n == "decision_config.json"]
same = new["identity"]["model_sha256"] == old["identity"]["model_sha256"]
print(json.dumps({"changed": changed, "changed_weight_files": weights, "identity_equal": same,
                  "ok": not weights and same}, indent=1))
sys.exit(0 if not weights and same else 1)
PY
"$HFPY" "$RENAME_OPS/revision_diff.py" "$REPO" "$superseded" "$REV" "$W/extra/revision-diff-vs-superseded.json" \
  || echo "revision diff vs $superseded lists weight changes (expected for new weights; see the file)"
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
