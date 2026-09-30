#!/usr/bin/env bash
# BF16 storage revisions (coordinator note 2026-09-30 06:10, 27B release path step 1), step 2: one storage-only
# revision of DEV2.0-0.8B / 2B / 4B. release.sh --upload --collect --already-collected with the BF16 spec and its
# final decision; full-panel parity (typed-final 1,600, css15 6,547, public231 231 and, where the scored mlx-diag run
# shared the formal cache, mlx-diag 2,275) against the T = 1 bindings before the upload and again on the real
# download, each with a fresh copy of the scoring run's frozen autotune cache (digest checked first). --mlx-only (4B,
# whose mlx-diag run had its own cache): a no-upload run with mlx-diag parity only and that cache. Node A GPU0 or
# GPU1 under the shared lease owner.release-bf16. Afterwards: the file-level diff against the replaced revision
# (only the BF16 backbone files, the card files and MODEL_MANIFEST.json may change), collection order, card HTTP,
# links, gate evaluate and storage.
# Usage (node A): bash <mirror>/v2/release/records/dev2-bf16-storage-2026-09-30/ops/bf16.sh <0.8B|2B|4B> --gpu <0|1> [--mlx-only]
#                 bash <mirror>/v2/release/records/dev2-bf16-storage-2026-09-30/ops/bf16.sh <0.8B|2B|4B> --preview
set -euo pipefail
tier="${1:-}"
shift || true
preview=0 gpu="" mlx_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preview) preview=1; shift ;;
    --gpu) gpu=$2; shift 2 ;;
    --mlx-only) mlx_only=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-bf16-storage-2026-09-30
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-bf16
mlx_cache="" mlx_digest=""
case "$tier" in
  0.8B)
    key=0p8b released=d4812ac6bb07333ec60d66091f08e1557933aa30
    frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b released=a47bdf895d982896b0e65c2bdc61330fe1ad2d7c
    frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  4B)
    key=4b released=197b70ca90759aca48410a9f94bb534e8ae2919f
    frozen=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA-triton
    frozen_digest=438618a6e3beb39407dabb96e689b5740d021a9c93b9df9769453c3d94deb119
    mlx_cache=/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-triton
    mlx_digest=ba8f21322724f57beafac7dbce5348af0a9c148368d89394e459287226c481fc ;;
  *) echo "tier must be one of 0.8B 2B 4B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name SPEC=$S/v2/release/specs/dev2-$key-bf16.json
P=/data/dev2/runs/release/inputs/dev2-$key-t1/derived
IN=/data/dev2/runs/release/inputs/dev2-$key-bf16
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
if [[ -e "$D/$name.decision.bf16.json" ]]; then
  cmp "$R/$name.decision.json" "$D/$name.decision.bf16.json"
else
  cp "$R/$name.decision.json" "$D/$name.decision.bf16.json"
  chmod 444 "$D/$name.decision.bf16.json"
fi
if [[ "$preview" == 1 ]]; then
  W=/data/dev2/runs/release/dev2-bf16-preview-$tier-$TS
  mkdir -p "$W/logs"
  (cd "$S" && python3 -m v2.release.build --spec "$SPEC" --output "$W/package/$name") > "$W/logs/build.log"
  find "$W/package" -name '*.safetensors' -delete
  echo "preview=$W/package/$name (weights removed after the build)"
  exit 0
fi
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this pass uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
[[ "$mlx_only" == 0 || -n "$mlx_cache" ]] || { echo "--mlx-only is for a tier whose mlx-diag run had its own cache" >&2; exit 2; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ "$mlx_only" == 1 ]]; then
  frozen=$mlx_cache frozen_digest=$mlx_digest
fi
TC=/data/dev2/runs/release/triton/dev2-bf16-$tier-copy-$TS
W=/data/dev2/runs/release/dev2-bf16-$tier-$([[ $mlx_only == 1 ]] && echo mlx- || true)$TS
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 11
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
[[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
cp -a "$frozen" "$TC"
echo "cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$frozen_digest"
if [[ "$mlx_only" == 1 ]]; then
  parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275")
  hub_args=()
else
  parity_args=(
    --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
    --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
    --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  )
  [[ -n "$mlx_cache" ]] || parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275")
  hub_args=(--upload --collect --already-collected)
fi
echo "mirror $SRC tier $tier gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" \
  --gpu "$gpu" --track release-bf16 --shared-lease "$LEASE" --threads 4 \
  --site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1 \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$P" --mount "$IN" \
  "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
echo "cache after run files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$frozen")"
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
if [[ "$mlx_only" == 1 ]]; then
  echo "work=$W mlx-only parity passed (no upload)"
  exit 0
fi
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
cd "$S"
"$HFPY" "$R/ops/bf16_diff.py" "$REPO" "$released" "$REV" "$IN/bf16-copy.json" "$W/extra/revision-diff.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
