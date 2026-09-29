#!/usr/bin/env bash
# DEV2.0-0.6B card-only revision with the approved post-key C1 line (coordinator 2026-09-30 00:30 UTC+8); the C1
# card pass runner (dev2-c1-card-pass-2026-09-29/ops/card.sh) with this revision's spec, decision and replaced revision.
# release.sh --upload --collect --already-collected, subset parity, node A GPU0 or GPU1 under the shared lease
# owner.release-c1pk, with subset parity against the scored predictions. Afterwards: every
# weight file byte-identical to the released revision, collection order and title, card HTTP, links, gate
# evaluate, storage.
# Usage (node A): bash <mirror>/v2/release/records/dev2-0p6b-c1postkey-card-2026-09-30/ops/card.sh <tier> --gpu <0|1>
#                 bash <mirror>/v2/release/records/dev2-0p6b-c1postkey-card-2026-09-30/ops/card.sh <tier> --preview
#   <tier>: 0.6B 0.8B 2B 4B 9B 27B. --preview: CPU build only, then the released card text is fetched for the
#   review (no GPU, no Hub writes).
set -euo pipefail
tier="${1:-}"
shift || true
preview=0 gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview) preview=1; shift ;;
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-0p6b-c1postkey-card-2026-09-30
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-c1pk
image_args=() kernel_args=() base_args=() extra_mounts=() tolerance_args=()
typed=200 css=300 public=100 mlx=0 frozen="" frozen_digest="" cache_tool=digest
[[ "$tier" == 0.6B ]] || { echo "this revision is DEV2.0-0.6B only" >&2; exit 2; }
case "$tier" in
  0.6B)
    spec=dev2-0p6b-card-c1pk.json released=188eb4c822e8643034f8ca87d865a16a86f69952
    P=/data/dev2/runs/06b/m8/formal/m8-s5-b05/output PM=/data/dev2/runs/06b/m8/formal/m8-s5-b05-mlx/output
    mlx=100 extra_mounts=(--mount /data/dev2/runs/release/inputs/dev2-0p6b-m8-bf16) tolerance_args=(--parity-tolerance 0) ;;
  0.8B)
    spec=dev2-0p8b-card-c1.json released=f458c34ccfb4a5d4d32babeda1570919adb1a3c8
    P=/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived PM=$P typed=150 css=200 mlx=150
    frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    spec=dev2-2b-card-c1.json released=5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38
    P=/data/dev2/runs/release/inputs/dev2-2b-t1/derived PM=$P typed=150 css=200 mlx=150
    frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  4B)
    spec=dev2-4b-card-c1.json released=452f133211de292a87bc29ab7e24a3bd0704e40d
    P=/data/dev2/runs/release/inputs/dev2-4b-t1/derived PM=$P
    frozen=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
    frozen_digest=438618a6e3beb39407dabb96e689b5740d021a9c93b9df9769453c3d94deb119 ;;
  9B)
    spec=dev2-9b-card-c1.json released=ae6831960dd1114296cb15a59248b79832c42959
    P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived PM=$P mlx=100
    frozen=/data/dev2/runs/9b/formal-m4/triton-cache
    frozen_digest=5604ffdc5f1916068c0b8df0526efc455f7bbec708081533020b834aba52586d ;;
  27B)
    spec=dev2-27b-card-c1.json released=32e7e8b1960fa1e6af3438cd11395cac2849a8c1
    P=/data/dev2/runs/27b/M3-A-soup/formal/output PM=$P cache_tool=27b
    frozen=/data/dev2/runs/27b/M3-A-soup/formal/triton-cache
    frozen_digest=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    image_args=(--image sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1)
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
      --env "HF_HUB_CACHE=$HFC" --mount "$base_repo") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name SPEC=$S/v2/release/specs/$spec
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
if [[ -e "$D/$name.decision.card-c1pk.json" ]]; then
  cmp "$R/$name.decision.json" "$D/$name.decision.card-c1pk.json"
else
  cp "$R/$name.decision.json" "$D/$name.decision.card-c1pk.json"
fi
if [[ "$preview" == 1 ]]; then
  W=/data/dev2/runs/release/dev2-c1pk-preview-$tier-$TS
  mkdir -p "$W/logs" "$W/released"
  (cd "$S" && python3 -m v2.release.build --spec "$SPEC" --output "$W/package/$name") > "$W/logs/build.log"
  find "$W/package" -name '*.safetensors' -delete
  for f in README.md evaluation/EVALUATION.md MODEL_MANIFEST.json config.json NOTICE ATTRIBUTIONS.md; do
    "$HFPY" -c 'import shutil,sys; from huggingface_hub import hf_hub_download; shutil.copyfile(hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3]), sys.argv[4])' \
      "$REPO" "$f" "$released" "$W/released/${f//\//_}" 2>/dev/null || echo "not in $released: $f"
  done
  # Card-only means: against the released manifest, only card files may change.
  python3 - "$W/package/$name/MODEL_MANIFEST.json" "$W/released/MODEL_MANIFEST.json" > "$W/preview-diff.json" <<'PY'
import json, sys
new, old = (json.load(open(p))["files_sha256"] for p in sys.argv[1:3])
card = {"README.md", "config.json", "LICENSE", "NOTICE", "ATTRIBUTIONS.md", "LICENSING.md"}
changed = sorted(n for n in set(new) | set(old) if new.get(n) != old.get(n))
other = [n for n in changed if n not in card and not n.startswith(("assets/", "evaluation/", "LICENSES/"))]
print(json.dumps({"changed_card_files": [n for n in changed if n not in other], "changed_other_files": other, "ok": not other}, indent=1))
PY
  echo "preview=$W/package/$name released=$W/released diff=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("ok" if d["ok"] else "NON-CARD " + ",".join(d["changed_other_files"]))' "$W/preview-diff.json") (weights removed after the build)"
  exit 0
fi
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this pass uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" "$([[ $tier == 27B ]] && echo 130 || echo 60)" "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ -n "$frozen" ]]; then
  kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1)
fi
TC=/data/dev2/runs/release/triton/dev2-c1pk-$tier-copy-$TS
W=/data/dev2/runs/release/dev2-c1pk-$tier-$TS
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 1
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
if [[ "$cache_tool" == 27b ]]; then
  (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$TC")
elif [[ -n "$frozen" ]]; then
  [[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
  cp -a "$frozen" "$TC"
  echo "cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$frozen_digest"
fi
cache_args=()
[[ -z "$frozen" ]] || cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
parity_args=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:$typed"
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:$css"
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:$public"
)
[[ "$mlx" == 0 ]] || parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$PM/mlx-diag.predictions.jsonl:$mlx")
mounts=(--mount "$G" --mount "$P")
[[ "$PM" == "$P" ]] || mounts+=(--mount "$PM")
echo "mirror $SRC tier $tier gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" "${image_args[@]}" \
  --gpu "$gpu" --track release-c1pk --shared-lease "$LEASE" --threads 4 \
  "${kernel_args[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  "${mounts[@]}" "${extra_mounts[@]}" "${tolerance_args[@]}" "${parity_args[@]}" \
  --upload --collect --already-collected || status=$?
set +x
if [[ "$cache_tool" == 27b ]]; then
  (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
elif [[ -n "$frozen" ]]; then
  echo "cache after run files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$frozen")"
fi
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
cd "$S"
"$HFPY" "$RENAME_OPS/revision_diff.py" "$REPO" "$released" "$REV" "$W/extra/revision-diff.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
