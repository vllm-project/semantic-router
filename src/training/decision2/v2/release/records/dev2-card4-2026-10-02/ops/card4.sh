#!/usr/bin/env bash
# Banner revisions (round 4, concept A) of the Decision 2.0 repositories, one repository at a time: Kai 0.6B by the
# banner worker (user job 2026-10-02 09:44 UTC+8; release worker 4c0a68cd), the other sizes by the roll-out (user
# request 2026-10-02 10:35 UTC+8, banner A on every card now; card worker fb5dd490), each only while no release of
# that size runs or is about to publish.
#   1. refuses unless main is the superseded revision of the card4 decision (or --resume REV after an interrupted
#      upload of this package)
#   2. builds the package from the card4 spec (private Index input and regenerated assets pinned by SHA-256) and
#      compares MODEL_MANIFEST.json files_sha256 with the released revision's: only assets/banner.png may change;
#      nothing is added or removed; then gate profile on the spec (the superseded release's items) must pass
#   3. refuses while another release.sh runs on this node or when main moved since step 1; hf_headroom.sh, then
#      release.sh --upload --collect --already-collected (no parity: card-only) with the native examples, the card's
#      Transformers example on the package, on the download and from the Hub in fresh caches under Transformers 5.17
#      and 5.18, readback, gate and collection; the released package must equal step 2's
#   4. card HTTP, links, the read-only collection check and gate evaluate
# GPU: node E GPU6 or GPU7 only, under this track's own lease (track=release-automap), removed afterwards if still ours.
# Lux 9B runs on node A, where its K-a13IB release inputs are: GPU6 or GPU7 when the owner file reads released and the
# GPU is idle, under the shared lease owner.release-card4 (the K-a13IB release's convention), removed on exit.
# Usage: bash <mirror>/v2/release/records/dev2-card4-2026-10-02/ops/card4.sh <tier> --gpu 6|7 [--resume REV]
set -euo pipefail
tier="${1:-}"
shift || true
gpu="" resume=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[67]$ ]] || { echo "GPU6 or GPU7 only (--gpu)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-card4-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
image=decision20-train-fast:host2 image_id="" kernels=1 codename="" frozen="" frozen_digest="" cache_tool=digest
base_args=() node=E
case "$tier" in
  0.6B) key=0p6b kernels=0 codename=Kai ;;
  0.8B)
    key=0p8b codename=Eos frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b codename=Sol frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  # The formal run's persisted cache stays on the node that scored it; the kernels autotune into a fresh one.
  4B) key=4b codename=Nox image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 ;;
  # K-a13IB: the scored image host2 and a fresh copy of its formal run's persisted cache, as its release.
  9B)
    key=9b codename=Lux node=A frozen=/data/dev2/runs/9b/formal-m9/triton-cache
    frozen_digest=48a2611aa852efff9348c1e3f3e0247d5e2a6915bec8ed8aeca488b53563e1ca
    image_id=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54 ;;
  27B)
    key=27b codename=Vega frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache cache_tool=27b
    frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
    image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
               --env "HF_HUB_CACHE=$HFC" --mount "$base_repo") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
lease=/data/dev2/leases/gpu$gpu.lock
lease_args=(--track release-automap)
if [[ "$node" == E ]]; then
  if [[ -e "$lease" ]] && [[ -n "$(ls -A "$lease")" ]]; then
    echo "gpu$gpu already has a lease entry ($(ls "$lease")): refusing" >&2; exit 1
  fi
else
  grep -qx "status=released" "$lease/owner" 2>/dev/null || { echo "gpu$gpu is not released by its owner" >&2; exit 1; }
  [[ ! -e "$lease/owner.release-card4" ]] || { echo "gpu$gpu already has owner.release-card4" >&2; exit 1; }
  rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
    || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
  [[ "$(docker image inspect -f '{{.Id}}' "$image")" == "$image_id" ]] || { echo "image $image is not $image_id" >&2; exit 1; }
  lease_args=(--track release-card4 --shared-lease release-card4)
  trap 'rm -f "$lease/owner.release-card4"' EXIT
fi
name=Decision-2.0-$codename-$tier REPO=llm-semantic-router/$name
SPEC=$S/v2/release/specs/dev2-$key-card4.json
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
manifest_sha() { sha256sum "$1/MODEL_MANIFEST.json" | cut -c1-64; }
hub_main() { "$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO"; }
decision=$D/$name.decision.card4.json
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.card4.json" "$decision"; else
  cp "$R/$name.decision.card4.json" "$decision"; chmod 444 "$decision"; fi

expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["supersedes"]["released_as"].split("@")[1])' \
  "$R/$name.decision.card4.json")
main=$(hub_main)
if [[ -n "$resume" && "$main" == "$resume" ]]; then
  echo "$REPO main $main = the revision an interrupted run of this package uploaded (resume)"
else
  [[ "$main" == "$expected" ]] || { echo "$REPO main is $main, not the superseded revision $expected: refusing" >&2; exit 1; }
  echo "$REPO main $main = superseded revision"
fi

check=$TMPDIR/dev2-card4-$tier-check-$TS
python3 -m v2.release.build --spec "$SPEC" --output "$check/$name" > /dev/null
built=$(manifest_sha "$check/$name")
"$HFPY" - "$REPO" "$expected" "$check/$name/MODEL_MANIFEST.json" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
repo, revision, new_path = sys.argv[1:]
old = json.load(open(hf_hub_download(repo, "MODEL_MANIFEST.json", revision=revision)))["files_sha256"]
new = json.load(open(new_path))["files_sha256"]
card = {"assets/banner.png"}
changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
other = [n for n in changed if not (n in card and n in old and n in new)]
print(json.dumps({"changed": changed, "non_card_changes": other, "files": [len(old), len(new)]}))
sys.exit(1 if other else 0)
PY
rm -rf "$check"
echo "fresh build $built: only the banner differs from $expected"
python3 -m v2.release.gate profile --spec "$SPEC" > "$TMPDIR/dev2-card4-$tier-profile-$TS.json" \
  || { echo "gate profile of $SPEC does not pass: refusing" >&2; exit 1; }
echo "gate profile passes ($TMPDIR/dev2-card4-$tier-profile-$TS.json)"

if pgrep -f "v2/release/release.sh" > /dev/null; then echo "another release.sh runs on this node: refusing" >&2; exit 1; fi
[[ "$(hub_main)" == "$main" ]] || { echo "$REPO main moved since the check: refusing" >&2; exit 1; }
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
W=/data/dev2/runs/release/dev2-card4-$tier-$TS
TC=/data/dev2/runs/release/triton/dev2-card4-$tier-$TS
kernel_args=() cache_args=()
if [[ "$kernels" == 1 ]]; then
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$TC")
  elif [[ -n "$frozen" ]]; then
    [[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
    cp -a "$frozen" "$TC"
    chmod -R u+w "$TC"
  else
    mkdir -p "$TC"
  fi
  kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1)
  cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
fi
echo "mirror $SRC tier $tier node $node gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" "${lease_args[@]}" --threads 4 \
  "${kernel_args[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  --upload --collect --already-collected --hub-site "tf518=$TF518" || status=$?
set +x
[[ "$cache_tool" != 27b || ! -d "$TC" ]] || (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
if [[ "$node" == E ]] && grep -qx "track=release-automap" "$lease/owner" 2>/dev/null \
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
cd "$S"
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
