#!/usr/bin/env bash
# Decision-2.0-Vega-27B successor over M6-IBxIB2-m50 (27B owner, continuation #7): release27bx.sh of
# dev2-27b-xarm-2026-10-02 with this record's spec and decision (make_27bs.py), on top of the current revision
# 9b067a95 (runtime-a's runtime-only revision of the M6-IBxIB2-m50 weights: phase A and the opt-in shared-context
# switch, which the successor keeps through the spec's runtime_source / automap_source), purging the superseded weight
# blobs against 9b067a95's verified download (dev2-ras-27B-20261002T213625Z; 5c85c127 and 9b067a95 serve the same
# weight files). One node-A GPU with 130 GB free: GPU0-1 as a shared-lease co-tenant (owner.release-27b-27bs), any
# other GPU only if its lease has no owner, this track's or a released one (set aside as owner.prev-27bs-<UTC>;
# release.sh then writes owner), the formal runs' image dbe5f32b with its kernels, HIP_FORCE_DEV_KERNARG=1, the
# Qwen3.8-27B base snapshot and a fresh copy of the ARM's frozen formal autotune cache.
#   --prerelease  release.sh without upload on the final spec and decision: build, native and card examples, AutoModel
#                 and pipeline before upload, exact parity of typed-final 1,600 / css15 6,547 / public231 231 against
#                 the formal run's and of mlx-diag 2,275 against the mlx-diag run's predictions, then verify_bundle
#   --release     only if the Hub main is the current revision 9b067a95 (never concurrently: checked right before the
#                 upload) and no other release.sh for Vega-27B runs on node A: hf_headroom.sh, then release.sh --upload
#                 --collect --already-collected --hub-site tf518=... with the formal panels' parity before upload and
#                 after the real download, AutoModel against native on every scored prompt and the Hub
#                 trust_remote_code smoke under Transformers 5.17 and 5.18; then the revision diff, collection order,
#                 card HTTP, links and gate evaluate; only if all pass, purge_superseded.py plan and apply (the
#                 superseded weight blobs, rewrite_history=False; node copy = 9b067a95's verified download)
# Usage (node A): bash <mirror>/v2/release/records/dev2-27b-succ-2026-10-03/ops/release27bs.sh ARM <mode> --gpu N
set -euo pipefail
ARM="${1:-}" mode="${2:-}"
shift 2 || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$ARM" =~ ^(M7-(IB124ML|IB14ML)|M8-(IB14|IB124)|X7-(IBxIB2xIB14ML|4ARM)|X8-(IBxIB2-8|ML)|X9-(LRH2|LRH2xM50|LRH|LRHxM50|ML0|IBxIB2-10|IBLRH|IBLRHxM50|LRHALL))$ ]] ||
  { echo "bad ARM $ARM" >&2; exit 2; }
[[ "$mode" =~ ^--(prerelease|release)$ ]] || { echo "mode: --prerelease|--release" >&2; exit 2; }
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "--gpu 0..7 (node A)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-27b-succ-2026-10-03
PURGE=$S/v2/release/records/dev2-27b-xarm-2026-10-02/ops/purge_superseded.py
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
COLL=vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
TRACK=release-27b-27bs
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
name=Decision-2.0-Vega-27B REPO=vllm-sr/Decision-2.0-Vega-27B
superseded=9b067a95560284dac8c98ef4130fd5a2c5a92ff9
SPEC=$S/v2/release/specs/dev2-27b-27bs-$ARM.json
DECISION=$R/$name.decision.27bs-$ARM.json
[[ -f "$SPEC" && -f "$DECISION" ]] || { echo "no spec / decision for $ARM in this mirror" >&2; exit 2; }
read -r RUN MLX FROZEN FROZEN_TREE < <(python3 -c '
import json, sys
s = json.load(open(sys.argv[1])); c = s["frozen_autotune_cache"]; r = s["card"]["reports"][0]
assert r["role"] == "candidate" and r["mlx"].endswith("/mlx-diag.score.json")
print(s["gate_profile"]["run"], r["mlx"].rsplit("/", 1)[0], c["path"], c["tree_sha256"])' "$SPEC")
P=$RUN/output
base_repo=$HFC/models--Qwen--Qwen3.8-27B
BASE=$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
PURGE_NODE_COPY=/data/dev2/runs/release/dev2-ras-27B-20261002T213625Z/download/Decision-2.0-Vega-27B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE" ]] || { echo "image $IMAGE is missing" >&2; exit 1; }
[[ -d "$BASE" ]] || { echo "no base snapshot $BASE" >&2; exit 1; }
owner=/data/dev2/leases/gpu$gpu.lock/owner
lease_args=()
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 130 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ "$gpu" == 0 || "$gpu" == 1 ]]; then
  # COORDINATION: node A GPU0-1 take release workers as shared-lease co-tenants with their own lease entries.
  owner=$owner.$TRACK lease_args=(--shared-lease "$TRACK")
elif [[ -f "$owner" ]] && ! grep -qx "track=$TRACK" "$owner"; then
  grep -q '^status=released' "$owner" || { echo "gpu$gpu is leased ($(grep -m1 '^track=' "$owner"))" >&2; exit 1; }
  mv "$owner" "$owner.prev-27bs-$TS"
fi
if [[ -e "$D/$name.decision.27bs-$ARM.json" ]]; then cmp "$DECISION" "$D/$name.decision.27bs-$ARM.json"; else
  cp "$DECISION" "$D/$name.decision.27bs-$ARM.json"; chmod 444 "$D/$name.decision.27bs-$ARM.json"; fi
trap '[[ -f "$owner" ]] && grep -qx "track=$TRACK" "$owner" && rm -f "$owner"' EXIT
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-27b-27bs-$ARM-profile-$TS.json" \
  || { echo "the Index-first profile does not pass" >&2; exit 1; }
W=/data/dev2/runs/release/dev2-27b-27bs-$ARM-${mode#--}-$TS
TC=/data/dev2/runs/release/triton/dev2-27b-27bs-$ARM-${mode#--}-$TS
(cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_TREE" --dest "$TC")
parity_args=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
)
mounts=(--mount "$G" --mount "$P")
hub_args=()
if [[ "$mode" == --prerelease ]]; then
  parity_args+=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$MLX/output/mlx-diag.predictions.jsonl:2275")
  mounts+=(--mount "$MLX/output")
else
  [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
    || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
  # COORDINATION 2026-10-02 15:35: "never publish concurrently" is per repository; every Vega-27B spec is dev2-27b-*.
  if pgrep -af "v2/release/release[.]sh" | grep -q -- "/specs/dev2-27b-"; then
    echo "another release.sh for $REPO runs on this node" >&2
    exit 1
  fi
  # Publish only on top of the revision the final decision supersedes (never concurrently with another worker).
  main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
  [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 20
  hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518")
fi
echo "mirror $SRC arm $ARM mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track "$TRACK" "${lease_args[@]}" --threads 4 \
  --site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1 \
  --base-path "$BASE" --env "HF_HUB_CACHE=$HFC" --mount "$base_repo" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" "${mounts[@]}" \
  "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
(cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
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
