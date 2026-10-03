#!/usr/bin/env bash
# Decision-2.0-Lux-9B Index-first successor of KIB4-a40 (9B M10 amendment 7; user rule 2026-10-02 09:55; the Lux-9B
# publisher): one node A GPU under the shared lease owner.release-9b-m10c, the scored image host2 (f83b1d10) with its kernels,
# HIP_FORCE_DEV_KERNARG=1 and a fresh copy of the current revision's frozen pre-warmed autotune cache
# (release/triton/runtime-a-9b: the formal runs' persisted cache plus the phase A fused kernels; digest-checked).
#   --prerelease  release.sh without upload on the final spec and decision: build, native and card examples, AutoModel
#                 and pipeline before upload, exact parity of typed-final 1,600 / css15 6,547 / public231 231 /
#                 mlx-diag 2,275 against the T = 1 derivation of the formal run, then verify_bundle
#   --release     only if the Hub main is the current revision 214ffa43 (checked right before the upload; never
#                 concurrently) and no other release.sh runs on node A: hf_headroom.sh, then release.sh --upload
#                 --collect --already-collected --hub-site tf518=... with the same parity before upload and after the
#                 real download, AutoModel against native on every scored prompt and the Hub trust_remote_code smoke
#                 under Transformers 5.17 and 5.18; then the revision diff, collection order, card HTTP, links and
#                 gate evaluate; only if all pass, purge_superseded.py plan and apply (the KIB4-a40 weight blobs of 214ffa43,
#                 rewrite_history=False; node copy = KIB4-a40's verified release package)
# Usage (node A): bash <mirror>/v2/release/records/dev2-9b-m10c-2026-10-03/ops/release_m10c.sh CAND <mode> --gpu N
set -euo pipefail
CAND="${1:-}" mode="${2:-}"
shift 2 || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(prerelease|release)$ ]] || { echo "mode: --prerelease|--release" >&2; exit 2; }
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "--gpu 0-7 (node A; 0 is the release GPU)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-9b-m10c-2026-10-03
PURGE=$S/v2/release/records/dev2-9b-m10-2026-10-02/ops/purge_superseded.py
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-9b-m10c
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
name=Decision-2.0-Lux-9B REPO=vllm-sr/Decision-2.0-Lux-9B
superseded=214ffa4322bc1bce3215c1bd5de6168402c76969  # the Lux switch revision (KIB4-a40's weights)
SPEC=$S/v2/release/specs/dev2-9b-m10c-$CAND.json
DECISION=$R/$name.decision.m10c-$CAND.json
[[ -f "$SPEC" && -f "$DECISION" ]] || { echo "no spec / decision for $CAND in this mirror" >&2; exit 2; }
IN=/data/dev2/runs/release/inputs/dev2-9b-m10c-$CAND
P=$IN/t1
FORMAL_CACHE=/data/dev2/runs/release/triton/runtime-a-9b
FORMAL_CACHE_DIGEST=${FORMAL_CACHE_DIGEST:-}
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
PURGE_NODE_COPY=/data/dev2/runs/release/dev2-9b-m10-KIB4-a40-release-20261002T151423Z/package/Decision-2.0-Lux-9B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]] || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 1; }
[[ -n "$FORMAL_CACHE_DIGEST" && "$(digest "$FORMAL_CACHE")" == "$FORMAL_CACHE_DIGEST" ]] \
  || { echo "formal cache $FORMAL_CACHE does not match FORMAL_CACHE_DIGEST" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ -e "$D/$name.decision.m10c-$CAND.json" ]]; then cmp "$DECISION" "$D/$name.decision.m10c-$CAND.json"; else
  cp "$DECISION" "$D/$name.decision.m10c-$CAND.json"; chmod 444 "$D/$name.decision.m10c-$CAND.json"; fi
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)
W=/data/dev2/runs/release/dev2-9b-m10c-$CAND-${mode#--}-$TS
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-9b-m10c-$CAND-profile-$TS.json" \
  || { echo "the Index-first profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-9b-m10c-$CAND-${mode#--}-$TS
cp -a "$FORMAL_CACHE" "$TC"
chmod -R u+w "$TC"
parity_args=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275"
)
hub_args=()
if [[ "$mode" == --release ]]; then
  [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
    || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
  if pgrep -f "v2/release/release[.]sh" >/dev/null; then echo "another release.sh runs on this node" >&2; exit 1; fi
  # Publish only on top of the revision the final decision supersedes (never concurrently with another worker).
  main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
  [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 20
  hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518")
fi
echo "mirror $SRC candidate $CAND mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track release-9b-m10c --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$P" --mount "$IN" \
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
