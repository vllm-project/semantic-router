#!/usr/bin/env bash
# 0.8B / 2B Index-first successors (user decision 2026-10-02 09:55; M16 08b-RA-a75 -> Decision-2.0-Eos-0.8B,
# 2b-RA-a75 -> Decision-2.0-Sol-2B), node A GPU0 or GPU1 as a recorded co-tenant (shared lease
# owner.release-<key>-ixf; the 0.6B allocation allows release work), the scored image dbe5f32b with its kernels,
# HIP_FORCE_DEV_KERNARG=1 and a fresh copy of the persisted autotune cache of the formal run that scored each panel
# (relayed from node B by relay_cache.sh, checked against its manifest before the copy).
#   --stage    CPU: the current revision's committed gate receipt and decision (card round 3) into the release
#              inputs (identical bytes), and the final decision into the decisions directory
#   --mlx      release.sh without upload: build, examples, card, AutoModel, parity of mlx-diag 2,275 (the mlx-diag
#              run's cache); then verify_bundle
#   --release  only if the Hub main is the superseded revision (never concurrently with another worker) and the
#              Transformers 5.18 site matches TF518_DIGEST: hf_headroom.sh, then release.sh --upload --collect
#              --already-collected --hub-site tf518=... with exact parity of typed-final 1,600 / css15 6,547 /
#              public231 231 before and after the real download (AutoModel against native on every scored
#              prompt); then the revision diff, collection order, card HTTP, links and gate evaluate; only if all
#              pass, purge_superseded.py plan and apply (the superseded weight blobs, rewrite_history=False; node
#              copy = the verified BF16 checkpoint of the superseded weights)
# Usage (node A): bash <mirror>/.../ops/release_ixf.sh 0p8b|2b --stage|--mlx|--release [--gpu 0|1]
set -euo pipefail
KEY=${1:-} mode=${2:-}
shift 2 || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(stage|mlx|release)$ ]] || { echo "mode: --stage|--mlx|--release" >&2; exit 2; }
case "$KEY" in
  0p8b) name=Decision-2.0-Eos-0.8B POINT=08b-RA-a75 superseded=9c7f3ea09a2b04a0647e5919af23c20ed982f246 ;;
  2b) name=Decision-2.0-Sol-2B POINT=2b-RA-a75 superseded=b42b6ff3efeedcd6534a3169a5db60247b363fb8 ;;
  *) echo "tier key 0p8b or 2b" >&2; exit 2 ;;
esac
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/dec/records/dec-08bfast-indexpath-2026-10-02
CARD3=$S/v2/release/records/dev2-card3-2026-10-02
PURGE=$S/v2/dec/records/dec-08bfast-indexpath-2026-10-02/ops/purge_superseded.py
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-$KEY-ixf
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
REPO=llm-semantic-router/$name
IN=/data/dev2/runs/release/inputs/dev2-$KEY-ixf
RUN=/data/dev2/runs/release/dev2-$KEY-ixf-t1
MLXRUN=$RUN-mlx
F=/data/dev2/runs/dec/formal/m16/m16-$POINT
FORMAL_CACHE=$F-cache
MLX_CACHE=$F-mlx-cache
PURGE_NODE_COPY=/data/dev2/runs/release/inputs/dev2-$KEY-bf16/checkpoint
SPEC=$S/v2/release/specs/dev2-$KEY-ixf.json
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
install_file() {
  if [[ -e "$2" ]]; then cmp "$1" "$2"; else mkdir -p "$(dirname "$2")"; cp "$1" "$2"; chmod 444 "$2"; fi
}
copy_cache() { # cache-dir run-receipt dest
  local want
  want=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache"]["after_manifest_sha256"])' "$2")
  [[ "$(sha256sum < "$1.sha256" | cut -c1-64)" == "$want" ]] || { echo "cache manifest $1.sha256 is not $want" >&2; exit 1; }
  (cd "$1" && sha256sum -c --quiet "$1.sha256") || { echo "cache $1 differs from its manifest" >&2; exit 1; }
  cp -a "$1" "$3"
  chmod -R u+w "$3"
  echo "cache copy $3 files=$(find "$3" -type f | wc -l) manifest=$want"
}

if [[ "$mode" == --stage ]]; then
  install_file "$CARD3/$KEY/release/receipts/gate.json" "$IN/current/gate.json"
  install_file "$CARD3/$name.decision.card3.json" "$IN/current/$name.decision.card3.json"
  echo "current gate $(sha256sum < "$IN/current/gate.json" | cut -c1-12) decision $(sha256sum < "$IN/current/$name.decision.card3.json" | cut -c1-12)"
  exit 0
fi

[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this release uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE" ]] || { echo "image $IMAGE is missing" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)
install_file "$R/$name.decision.ixf.json" "$D/$name.decision.ixf.json"
W=/data/dev2/runs/release/dev2-$KEY-ixf-${mode#--}-$TS
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-$KEY-ixf-profile-$TS.json" \
  || { echo "index_first profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-$KEY-ixf-${mode#--}-$TS
parity_args=() hub_args=()
if [[ "$mode" == --mlx ]]; then
  copy_cache "$MLX_CACHE" "$F-mlx/M6-RECEIPT.json" "$TC"
  parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$MLXRUN/output/mlx-diag.predictions.jsonl:2275")
else
  main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
  [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
  [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
    || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
  if pgrep -f "v2/release/release.sh" >/dev/null; then echo "another release.sh runs on this node" >&2; exit 1; fi
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 10
  copy_cache "$FORMAL_CACHE" "$F/M6-RECEIPT.json" "$TC"
  parity_args=(
    --parity "typed-final:$G/typed-final.prompts.jsonl:$RUN/output/typed-final.predictions.jsonl:1600"
    --parity "css15:$G/css15.prompts.jsonl:$RUN/output/css15.predictions.jsonl:6547"
    --parity "public231:$G/public231.prompts.jsonl:$RUN/output/public231.predictions.jsonl:231"
  )
  hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518")
fi
echo "mirror $SRC tier $KEY mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track "$LEASE" --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$RUN" --mount "$MLXRUN" --mount "$IN" \
  "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
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
"$HFPY" "$RENAME_OPS/revision_diff.py" "$REPO" "$superseded" "$REV" "$W/extra/revision-diff-vs-superseded.json" \
  || echo "revision diff vs $superseded lists weight changes (expected for new weights; see the file)"
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
if [[ "$status" == 0 ]]; then
  # Only a verified new revision lets the superseded weight blobs go; node copies stay the durable store.
  for step in plan apply; do
    "$HFPY" "$PURGE" "$step" "$W/extra/purge-$step.json" --repo "$REPO" \
      --old-revision "$superseded" --new-revision "$REV" --node-copy "$PURGE_NODE_COPY" || { status=1; break; }
  done
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
fi
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
