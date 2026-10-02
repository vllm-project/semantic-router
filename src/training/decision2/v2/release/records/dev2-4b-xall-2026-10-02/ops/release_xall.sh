#!/usr/bin/env bash
# Decision-2.0-Nox-4B M17 4b-SDMLxALL release (user decision 2026-10-02 23:02 UTC+8; decoder M17, the Nox-4B
# publisher), as the 9B M10 release's release_m10.sh: one node A GPU under the shared lease owner.release-4b-xall,
# the formal image dbe5f32b with its kernels, HIP_FORCE_DEV_KERNARG=1 and a fresh copy of the current revision's frozen
# pre-warmed autotune cache (release/triton/runtime-a-4b: the phase A parity cache; digest-checked).
#   --prerelease  release.sh without upload on the final spec and decision: build, native and card examples, AutoModel
#                 and pipeline before upload, exact parity of typed-final 1,600 / css15 6,547 / public231 231 (the
#                 formal run's T = 1 predictions) and mlx-diag 2,275 (the mlx-diag run's), then verify_bundle
#   --release     the fast path (COORDINATION 2026-10-02 23:20: the four-panel answer re-run after upload is waived;
#                 the pre-upload checks are R3, the row-level audit and the 86-request package parity gate of the
#                 Index run): only if the Hub main is the current revision 137e28ce (checked right before the upload;
#                 never concurrently) and no other release.sh runs on node A: hf_headroom.sh, then release.sh --upload
#                 --collect --already-collected --hub-site tf518=... without panel parity (the real download is
#                 re-hashed file by file against the package manifest; the Hub trust_remote_code smoke under
#                 Transformers 5.17 and 5.18); then the revision diff, collection order, card HTTP, links, gate
#                 evaluate and the IX1 86-request parity gate (v2/eval/ix1/launch.sh parity) on a hard-linked copy of
#                 the download (DEV2.0-4B-SDMLxALL-hub); only if all pass, purge_superseded.py plan and apply (the
#                 137e28ce weight blobs, rewrite_history=False; node copy = the phase A release's verified package)
#   --post WORK   for WORK's upload (still main): only the checks after release.sh, the IX1 gate and the purge
#                 (the earlier extra/ receipts are kept as extra.before-<time>)
# Never concurrently with another release.sh for this repository (other repositories' releases may share the node).
# Usage (node A): bash <mirror>/v2/release/records/dev2-4b-xall-2026-10-02/ops/release_xall.sh <mode> --gpu N
set -euo pipefail
mode="" post=""
[[ "${1:-}" == --post ]] || { mode="${1:-}"; shift || true; }
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --post) mode=--post post=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(prerelease|release|post)$ ]] || { echo "mode: --prerelease|--release|--post WORK" >&2; exit 2; }
[[ "$gpu" =~ ^[0-7]$ ]] || { echo "--gpu 0-7 (node A)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-4b-xall-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-4b-xall
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
IMAGE_ID=$IMAGE
name=Decision-2.0-Nox-4B REPO=vllm-sr/Decision-2.0-Nox-4B
superseded=137e28ceebd52b9611bb72e3645536867e0ec1ec
SPEC=$S/v2/release/specs/dev2-4b-xall.json
DECISION=$R/$name.decision.xall.json
[[ -f "$SPEC" && -f "$DECISION" ]] || { echo "no spec / decision in this mirror" >&2; exit 2; }
IN=/data/dev2/runs/release/inputs/dev2-4b-xall
RUN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate_profile"]["run"])' "$SPEC")
FORMAL_CACHE=/data/dev2/runs/release/triton/runtime-a-4b
FORMAL_CACHE_DIGEST=98750b7b891ea53fd4e56cc3bb54027384ae47de42ce957fc60a9aa184b59e70
TF518=/data/dev2/tools/tf518
TF518_DIGEST=93df9002544f8394b572497d97dd2df778e79b63adbcc5624173bcf84de7fffe
PURGE_NODE_COPY=/data/dev2/runs/release/dev2-ra-4B-20261002T115340Z/package/Decision-2.0-Nox-4B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]] || { echo "image $IMAGE is missing" >&2; exit 1; }
[[ -f "$PURGE_NODE_COPY/MODEL_MANIFEST.json" ]] || { echo "no node copy of the superseded package" >&2; exit 1; }
[[ -n "$FORMAL_CACHE_DIGEST" && "$(digest "$FORMAL_CACHE")" == "$FORMAL_CACHE_DIGEST" ]] \
  || { echo "formal cache $FORMAL_CACHE does not match FORMAL_CACHE_DIGEST" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
if [[ -e "$D/$name.decision.xall.json" ]]; then cmp "$DECISION" "$D/$name.decision.xall.json"; else
  cp "$DECISION" "$D/$name.decision.xall.json"; chmod 444 "$D/$name.decision.xall.json"; fi
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)
if [[ "$mode" == --post ]]; then
W=$post
[[ -f "$W/receipts/upload.json" ]] || { echo "$W has no upload receipt" >&2; exit 2; }
main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
[[ "$main" == "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")" ]] \
  || { echo "$REPO main $main is not $W's upload" >&2; exit 1; }
PKG=$W/package/$name
[[ ! -d "$W/extra" ]] || mv "$W/extra" "$W/extra.before-$TS"
mkdir -p "$W/extra"
status=0
else
W=/data/dev2/runs/release/dev2-4b-xall-${mode#--}-$TS
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-4b-xall-profile-$TS.json" \
  || { echo "the Index-first profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-4b-xall-${mode#--}-$TS
cp -a "$FORMAL_CACHE" "$TC"
chmod -R u+w "$TC"
parity_args=(
  --parity "typed-final:$G/typed-final.prompts.jsonl:$RUN/output/typed-final.predictions.jsonl:1600"
  --parity "css15:$G/css15.prompts.jsonl:$RUN/output/css15.predictions.jsonl:6547"
  --parity "public231:$G/public231.prompts.jsonl:$RUN/output/public231.predictions.jsonl:231"
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$RUN-mlx/output/mlx-diag.predictions.jsonl:2275"
)
hub_args=()
if [[ "$mode" == --release ]]; then
  [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
    || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
  if pgrep -af "v2/release/release[.]sh" | grep -q -- "specs/dev2-4b-"; then echo "another release.sh for $REPO runs" >&2; exit 1; fi
  # Publish only on top of the revision the final decision supersedes (never concurrently with another worker).
  main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
  [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 20
  hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518")
  parity_args=()
fi
echo "mirror $SRC candidate 4b-SDMLxALL mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track release-4b-xall --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$RUN" --mount "$RUN-mlx" --mount "$IN" \
  "${parity_args[@]}" "${hub_args[@]}" || status=$?
set +x
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
mkdir -p "$W/extra"
PKG=$W/package/$name
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -v "$PKG:$PKG:ro" -w /tmp \
  --entrypoint python3 "$IMAGE" -I -B -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from decision2 import verify_bundle; m = verify_bundle(sys.argv[1]); print(json.dumps({"verify_bundle": "ok", "files": len(m.get("files_sha256") or {}), "identity": (m.get("identity") or {}).get("model_sha256")}))' \
  "$PKG" | tee "$W/extra/verify-bundle.json"
fi
if [[ "$mode" == --prerelease ]]; then
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
HUB=/data/dev2/models/ix1/dec-m17/DEV2.0-4B-SDMLxALL-hub
[[ ! -e "$HUB" ]] || mv "$HUB" "$HUB.before-$TS"
mkdir -p "$(dirname "$HUB")" && cp -al "$W/download/$name" "$HUB"
PAR=/data/dev2/private/eval/index021/ix1/parity/DEV2.0-4B-SDMLxALL-hub-$TS
if bash "$S/v2/eval/ix1/launch.sh" parity --src "/data/dev2/src/$SRC" --model DEV2.0-4B-SDMLxALL-hub --gpu "$gpu" \
  --run "$PAR" --rows /data/dev2/private/eval/index021/compat-86.jsonl.gz > "$W/extra/ix1-parity.log" 2>&1; then
  python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(json.dumps({k: p[k] for k in ("pass", "requests", "questions_compared", "max_abs_dp", "statuses")})); sys.exit(0 if p["pass"] else 1)' \
    "$PAR/parity.json" | tee "$W/extra/ix1-parity.json" || status=1
else
  echo "IX1 86-request parity gate failed (see $W/extra/ix1-parity.log)" >&2; status=1
fi
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
