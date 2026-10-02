#!/usr/bin/env bash
# Decision-2.0-Nox-4B Index-first successor (user rule 2026-10-02 09:55; worker 5e7b8132): node A GPU0 or GPU1 under
# the shared lease owner.release-4b-4bif, the formal runs' image dbe5f32b with its kernels, HIP_FORCE_DEV_KERNARG=1
# and a fresh copy of the persisted autotune cache of the run that scored each panel (checked against its manifest).
#   --prerelease  release.sh without upload on the final spec and decision: build, native and card examples, AutoModel
#                 and pipeline before upload, exact parity of typed-final 1,600 / css15 6,547 / public231 231 against
#                 the formal run's predictions (formal cache), then verify_bundle
#   --mlx         the same without upload, parity of mlx-diag 2,275 only (the mlx-diag run's cache)
#   --release     only if the Hub main is the current revision 54b084f9 (never concurrently: checked right before the
#                 upload) and no other release.sh runs on node A: hf_headroom.sh, then release.sh --upload --collect
#                 --already-collected --hub-site tf518=... with the formal panels' parity before upload and after the
#                 real download, AutoModel against native on every scored prompt and the Hub trust_remote_code smoke
#                 under Transformers 5.17 and 5.18; then the revision diff, collection order, card HTTP, links and
#                 gate evaluate; only if all pass, purge_superseded.py plan and apply (the LH weight blobs,
#                 rewrite_history=False; node copy = the LH release's verified download)
# Usage (node A): bash <mirror>/v2/release/records/dev2-4b-indexfirst-2026-10-02/ops/release4b.sh CAND <mode> --gpu 0|1
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
[[ "$mode" =~ ^--(prerelease|mlx|release)$ ]] || { echo "mode: --prerelease|--mlx|--release" >&2; exit 2; }
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this release uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-4b-indexfirst-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-4b-4bif
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
name=Decision-2.0-Nox-4B REPO=llm-semantic-router/Decision-2.0-Nox-4B
superseded=54b084f98e1713480f7d5b83b11e37171cba2270
SPEC=$S/v2/release/specs/dev2-4b-4bif-$CAND.json
DECISION=$R/$name.decision.4bif-$CAND.json
[[ -f "$SPEC" && -f "$DECISION" ]] || { echo "no spec / decision for $CAND in this mirror" >&2; exit 2; }
read -r RUN IN < <(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["gate_profile"]["run"], s["checkpoint"].rsplit("/bf16/", 1)[0])' "$SPEC")
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
PURGE_NODE_COPY=/data/dev2/runs/release/dev2-4b-lh-release-20261001T081057Z/download/DEV2.0-4B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
copy_cache() { # copy_cache <cache> <copy>: the cache must still match its manifest
  [[ "$(digest "$1")" == "$(sha256sum < "$1.sha256" | cut -c1-64)" ]] || { echo "$1 differs from its manifest" >&2; exit 1; }
  cp -a "$1" "$2"
  chmod -R u+w "$2"
}
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE" ]] || { echo "image $IMAGE is missing" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ -e "$D/$name.decision.4bif-$CAND.json" ]]; then cmp "$DECISION" "$D/$name.decision.4bif-$CAND.json"; else
  cp "$DECISION" "$D/$name.decision.4bif-$CAND.json"; chmod 444 "$D/$name.decision.4bif-$CAND.json"; fi
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)
W=/data/dev2/runs/release/dev2-4b-4bif-$CAND-${mode#--}-$TS
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-4b-4bif-$CAND-profile-$TS.json" \
  || { echo "the Index-first profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-4b-4bif-$CAND-${mode#--}-$TS
parity_args=() hub_args=()
case "$mode" in
  --mlx)
    copy_cache "$IN/mlx-cache" "$TC"
    parity_args=(--parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$RUN-mlx/output/mlx-diag.predictions.jsonl:2275") ;;
  *)
    copy_cache "$IN/formal-cache" "$TC"
    parity_args=(
      --parity "typed-final:$G/typed-final.prompts.jsonl:$RUN/output/typed-final.predictions.jsonl:1600"
      --parity "css15:$G/css15.prompts.jsonl:$RUN/output/css15.predictions.jsonl:6547"
      --parity "public231:$G/public231.prompts.jsonl:$RUN/output/public231.predictions.jsonl:231"
    ) ;;
esac
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
  --gpu "$gpu" --track release-4b-4bif --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$RUN" --mount "$RUN-mlx" --mount "$IN" \
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
    "$HFPY" "$R/ops/purge_superseded.py" "$step" "$W/extra/purge-$step.json" --repo "$REPO" \
      --old-revision "$superseded" --new-revision "$REV" --node-copy "$PURGE_NODE_COPY" || { status=1; break; }
  done
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
fi
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
