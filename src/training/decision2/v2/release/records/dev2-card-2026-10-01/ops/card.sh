#!/usr/bin/env bash
# Card-redesign revisions of the six DEV2.0 repositories (user card job 2026-10-01 21:58 UTC+8; release worker
# 4c0a68cd), on node E, one repository at a time.
#   1. refuses unless main is the superseded revision (or --resume REV after an interrupted upload of this package)
#   2. builds the package from the card spec and compares MODEL_MANIFEST.json files_sha256 with the released
#      revision's: only card files (README.md, assets/, evaluation/, NOTICE, ATTRIBUTIONS.md) may differ
#   3. hf_headroom.sh, then release.sh --upload --collect --already-collected (no parity: card-only) with the native
#      examples, the card's Transformers example on the package, on the download and from the Hub in fresh caches
#      under Transformers 5.17 and 5.18, readback, gate and collection; the released package must equal step 2's
#   4. card HTTP, links and gate evaluate
# GPU: node E GPU6 or GPU7 only, as a co-tenant entry (--shared-lease) where the owner's lease allows release smokes.
# Usage: bash <mirror>/v2/release/records/dev2-card-2026-10-01/ops/card.sh <tier> --gpu 6|7 [--resume REV]
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
[[ "$gpu" =~ ^[67]$ ]] || { echo "node E GPU6 or GPU7 only (--gpu)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-card-2026-10-01
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
image=decision20-train-fast:host2 kernels=1 frozen="" frozen_digest="" cache_tool=digest base_args=()
case "$tier" in
  0.6B) key=0p6b kernels=0 ;;
  0.8B)
    key=0p8b frozen=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
    frozen_digest=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2 ;;
  2B)
    key=2b frozen=/data/dev2/runs/dec/formal/m3/m3-S2T-soup-nodeA-triton
    frozen_digest=abdfd6872ccee3efc2b8e67358e9726e83eec882b1b0de3059d2ad408b2e26f0 ;;
  # The formal run's persisted cache stays on the node that scored it; the kernels autotune into a fresh one.
  4B) key=4b image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 ;;
  9B)
    key=9b frozen=/data/dev2/runs/9b/formal-m4/triton-cache
    frozen_digest=5604ffdc5f1916068c0b8df0526efc455f7bbec708081533020b834aba52586d ;;
  27B)
    key=27b frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache cache_tool=27b
    frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
    image=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
               --env "HF_HUB_CACHE=$HFC" --mount "$base_repo") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
name=DEV2.0-$tier REPO=llm-semantic-router/$name
SPEC=$S/v2/release/specs/dev2-$key-card.json
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
manifest_sha() { sha256sum "$1/MODEL_MANIFEST.json" | cut -c1-64; }
decision=$D/$name.decision.card.json
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.card.json" "$decision"; else
  cp "$R/$name.decision.card.json" "$decision"; chmod 444 "$decision"; fi

expected=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["supersedes"]["released_as"].split("@")[1])' \
  "$R/$name.decision.card.json")
main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
if [[ -n "$resume" && "$main" == "$resume" ]]; then
  echo "$REPO main $main = the revision an interrupted run of this package uploaded (resume)"
else
  [[ "$main" == "$expected" ]] || { echo "$REPO main is $main, not the superseded revision $expected: refusing" >&2; exit 1; }
  echo "$REPO main $main = superseded revision"
fi

check=$TMPDIR/dev2-card-$tier-check-$TS
python3 -m v2.release.build --spec "$SPEC" --output "$check/$name" > /dev/null
built=$(manifest_sha "$check/$name")
"$HFPY" - "$REPO" "$expected" "$check/$name/MODEL_MANIFEST.json" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download
repo, revision, new_path = sys.argv[1:]
old = json.load(open(hf_hub_download(repo, "MODEL_MANIFEST.json", revision=revision)))["files_sha256"]
new = json.load(open(new_path))["files_sha256"]
card = lambda n: n in ("README.md", "NOTICE", "ATTRIBUTIONS.md") or n.startswith(("assets/", "evaluation/"))
changed = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
other = [n for n in changed if not card(n)]
print(json.dumps({"changed": changed, "non_card_changes": other, "files": [len(old), len(new)]}))
sys.exit(1 if other else 0)
PY
rm -rf "$check"
echo "fresh build $built: only card files differ from $expected"

bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
W=/data/dev2/runs/release/dev2-card-$tier-$TS
TC=/data/dev2/runs/release/triton/dev2-card-$tier-$TS
kernel_args=() cache_args=()
if [[ "$kernels" == 1 ]]; then
  if [[ "$cache_tool" == 27b ]]; then
    (cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$TC")
  elif [[ -n "$frozen" ]]; then
    [[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
    cp -a "$frozen" "$TC"
  else
    mkdir -p "$TC"
  fi
  kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1)
  cache_args=(--env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
fi
echo "mirror $SRC tier $tier gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" --track release-automap --shared-lease release-card --threads 4 \
  "${kernel_args[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  --upload --collect --already-collected --hub-site "tf518=$TF518" || status=$?
set +x
[[ "$cache_tool" != 27b || ! -d "$TC" ]] || (cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
rm -f "/data/dev2/leases/gpu$gpu.lock/owner.release-card"
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
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
