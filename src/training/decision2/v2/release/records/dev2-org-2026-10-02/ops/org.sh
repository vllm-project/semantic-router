#!/usr/bin/env bash
# Organization card-only revisions of the Decision 2.0 repositories (user instruction 2026-10-02 17:35 UTC+8: the
# Hugging Face organization llm-semantic-router is now vllm-sr; org worker (2)), one run per repository and never
# while that repository's release worker publishes. Adapted from round 4's card4.sh.
#   1. refuses unless main is the revision the org decision supersedes (or --resume REV after an interrupted upload
#      of this package) and make_org.py --check re-derives the committed spec and decision from the release's work
#      directory on this node
#   2. builds the package from the org spec; org_check.py compare: only README.md changes, it is the released README
#      with the organization renamed, MODEL_MANIFEST.json differs only in its Hub references, README digests and
#      builder provenance, and no package text names the former organization; then gate profile on the spec (the
#      superseded release's items) must pass
#   3. Hub pre-check under the new ID: it resolves to itself at main, privately, and the former ID redirects to it;
#      AutoConfig and AutoTokenizer with trust_remote_code load it from the Hub in a fresh cache (release image)
#   4. refuses while another release.sh for this repository runs on the node or when main moved since step 1;
#      hf_headroom.sh, then release.sh --upload --collect --already-collected (card-only: no parity) with the native
#      examples, the card's Transformers example on the package, on the download and from the Hub in fresh caches
#      under Transformers 5.17 and 5.18, readback, gate and collection; the released package must equal step 2's
#   5. org_check.py verify (main, private, redirect, exactly README.md and MODEL_MANIFEST.json changed, no text file
#      names the former organization), card HTTP, links, the read-only collection check and gate evaluate
# Node A (Eos 0.8B, Sol 2B, Nox 4B, Lux 9B, Vega 27B, where their release inputs are): a GPU whose owner file reads
# released, or the 0.6B track's idle allocation, under the shared lease owner.release-org-<key> (removed on exit).
# Node E (Kai 0.6B): GPU6 or GPU7 without a lease entry, under this track's own lease (track=release-org).
# Kernel tiers autotune into a fresh cache (card-only, no scored-panel parity), as round 4 did for 4B. When no GPU of
# the node is free to it, a tier may run every check on CPU (--cpu: no GPU, no lease; the runtime's FP32 CPU path,
# without the GPU kernels). The card-only checks compare answers within the run, never with the scored predictions.
# Usage: bash <mirror>/v2/release/records/dev2-org-2026-10-02/ops/org.sh <tier> --gpu N|--cpu [--resume REV]
set -euo pipefail
tier="${1:-}"
shift || true
gpu="" resume="" cpu=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --cpu) cpu=1; shift ;;
    --resume) resume=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$cpu" == 1 && -z "$gpu" || "$cpu" == 0 && "$gpu" =~ ^[0-7]$ ]] || { echo "--gpu N or --cpu" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-org-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
COLL=vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
HOST2=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
FORMAL=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
image=$FORMAL kernels=1 node=A vram=60 base_args=()
case "$tier" in
  0.6B) key=0p6b codename=Kai node=E kernels=0 image=$HOST2 ;;
  0.8B) key=0p8b codename=Eos ;;
  2B) key=2b codename=Sol ;;
  4B) key=4b codename=Nox ;;
  9B) key=9b codename=Lux image=$HOST2 ;;
  27B)
    key=27b codename=Vega vram=130
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
               --env "HF_HUB_CACHE=$HFC" --mount "$base_repo") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
[[ "$(docker image inspect -f '{{.Id}}' "$image")" == "$image" ]] || { echo "image $image is missing" >&2; exit 1; }
lease=/data/dev2/leases/gpu$gpu.lock
if [[ "$cpu" == 1 ]]; then
  kernels=0
  device_args=(--cpu --threads 32)
elif [[ "$node" == E ]]; then
  [[ "$gpu" =~ ^[67]$ ]] || { echo "node E GPU6 or GPU7 only (--gpu)" >&2; exit 2; }
  if [[ -e "$lease" ]] && [[ -n "$(ls -A "$lease")" ]]; then
    echo "gpu$gpu already has a lease entry ($(ls "$lease")): refusing" >&2; exit 1
  fi
  lease_args=(--track release-org)
else
  { grep -q "^status=released" "$lease/owner" \
      || { grep -qx "track=06b-encoder" "$lease/owner" && grep -q "^status=idle" "$lease/owner"; }; } 2>/dev/null \
    || { echo "gpu$gpu is neither released by its owner nor the 0.6B track's idle allocation" >&2; exit 1; }
  ls "$lease"/owner.release-org-* >/dev/null 2>&1 && { echo "gpu$gpu already runs an org revision" >&2; exit 1; }
  lease_args=(--track release-org --shared-lease "release-org-$key")
  trap 'rm -f "$lease/owner.release-org-$key"' EXIT
fi
if [[ "$cpu" == 0 ]]; then
  device_args=(--gpu "$gpu" "${lease_args[@]}" --threads 4)
  rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" "$vram" "$gpu" >/dev/null \
    || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
fi
name=Decision-2.0-$codename-$tier REPO=vllm-sr/$name
SPEC=$S/v2/release/specs/dev2-$key-org.json
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D" /data/dev2/runs/release/org20
exec 9> "/data/dev2/runs/release/org20/$key.lock"
flock -n 9 || { echo "another org.sh for $tier runs on this node" >&2; exit 1; }
X=/data/dev2/runs/release/org20/dev2-org-$tier-$TS
mkdir -p "$X"
manifest_sha() { sha256sum "$1/MODEL_MANIFEST.json" | cut -c1-64; }
hub_main() { "$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO"; }
decision=$D/$name.decision.org.json
(cd "$S" && python3 "$R/ops/make_org.py" --tiers "$tier" --check) > "$X/derivation.txt" \
  || { echo "the committed org spec / decision differ from their derivation: refusing" >&2; exit 1; }
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.org.json" "$decision"; else
  cp "$R/$name.decision.org.json" "$decision"; chmod 444 "$decision"; fi

expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["supersedes"]["released_as"].split("@")[1])' \
  "$R/$name.decision.org.json")
main=$(hub_main)
if [[ -n "$resume" && "$main" == "$resume" ]]; then
  echo "$REPO main $main = the revision an interrupted run of this package uploaded (resume)"
else
  [[ "$main" == "$expected" ]] || { echo "$REPO main is $main, not the superseded revision $expected: refusing" >&2; exit 1; }
  echo "$REPO main $main = superseded revision"
fi

check=$TMPDIR/dev2-org-$tier-check-$TS
python3 -m v2.release.build --spec "$SPEC" --output "$check/$name" > "$X/build-check.log"
built=$(manifest_sha "$check/$name")
"$HFPY" "$R/ops/org_check.py" compare --repo "$REPO" --revision "$expected" --package "$check/$name" \
  --output "$X/compare.json" || { echo "the fresh build is not a card-only org revision of $expected" >&2; exit 1; }
rm -rf "$check"
echo "fresh build $built: only the organization differs from $expected"
python3 -m v2.release.gate profile --spec "$SPEC" > "$X/profile.json" \
  || { echo "gate profile of $SPEC does not pass: refusing" >&2; exit 1; }
echo "gate profile passes"

"$HFPY" "$R/ops/org_check.py" precheck --repo "$REPO" --revision "$main" --output "$X/precheck.json" \
  || { echo "Hub pre-check of $REPO failed" >&2; exit 1; }
home=$X/hub-pre
mkdir -p "$home/hf-home"
docker run --rm --network host -e HF_HOME="$home/hf-home" -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e HF_TOKEN_PATH=/run/decision2-hf/token -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
  -v /root/.cache/huggingface/token:/run/decision2-hf/token:ro -v "$S:$S:ro" -v "$X:$X" \
  --entrypoint python3 "$image" -I -B "$R/ops/org_check.py" remote-load --repo "$REPO" --revision "$main" \
  --output "$X/remote-load-pre.json" || { rm -rf "$home"; echo "trust_remote_code pre-check failed" >&2; exit 1; }
rm -rf "$home"

if pgrep -af "[v]2/release/release[.]sh" | grep -q -- "specs/dev2-$key-"; then
  echo "another release.sh for $REPO runs on this node: refusing" >&2; exit 1
fi
[[ "$(hub_main)" == "$main" ]] || { echo "$REPO main moved since the check: refusing" >&2; exit 1; }
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
W=/data/dev2/runs/release/dev2-org-$tier-$TS
TC=/data/dev2/runs/release/triton/dev2-org-$tier-$TS
kernel_args=() cache_args=()
if [[ "$kernels" == 1 ]]; then
  mkdir -p "$TC"
  kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1)
  cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
fi
echo "mirror $SRC tier $tier node $node gpu ${gpu:-cpu} work $W tf518 $(cd "$TF518" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" "${device_args[@]}" \
  "${kernel_args[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  --upload --collect --already-collected --hub-site "tf518=$TF518" || status=$?
set +x
if [[ "$cpu" == 0 && "$node" == E ]] && grep -qx "track=release-org" "$lease/owner" 2>/dev/null \
  && grep -qx "run_dir=$W" "$lease/owner"; then
  rm -f "$lease/owner" && rmdir "$lease" 2>/dev/null || true
fi
python3 - "$W/receipts" <<'PY' || true
import json, sys
from pathlib import Path
receipts = Path(sys.argv[1])
for name in ("card-pre", "automap-pre", "automap-card-pre", "card-post", "automap-post", "automap-hub",
             "automap-hub-tf518", "readback", "gate"):
    path = receipts / f"{name}.json"
    if path.is_file():
        r = json.loads(path.read_text())
        print(f"{name}: passed={r.get('passed', 'n/a')} example={r.get('example', '-')}")
PY
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
[[ "$(manifest_sha "$W/package/$name")" == "$built" ]] || { echo "released package differs from the checked build" >&2; status=1; }
mkdir -p "$W/extra"
cp "$X"/*.json "$X/derivation.txt" "$W/extra/"
cd "$S"
"$HFPY" "$R/ops/org_check.py" verify --repo "$REPO" --revision "$REV" --superseded "$expected" \
  --output "$W/extra/org-verify.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
