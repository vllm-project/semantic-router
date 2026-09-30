#!/usr/bin/env bash
# DEV2.0-27B successor release (coordinator note 2026-09-30 07:20 UTC+8): the M4-A20r soup as the new main of
# llm-semantic-router/DEV2.0-27B. release.sh --upload --collect --already-collected from this mirror with the final
# spec and decision (successor profile R1-R8), full-panel parity (typed-final 1,600, css15 6,547, public231 231)
# before and after the real download, against the scored run's predictions, with a fresh copy of its persisted
# autotune cache; node A GPU0 or GPU1 under the shared lease owner.release-27b-a20r. Afterwards: the released
# package against the frozen package that C1 scored (only card files may differ), the file diff against the
# superseded revision, collection order, card HTTP, links, gate evaluate and storage.
# Usage (node A): bash <mirror>/v2/release/records/dev2-27b-a20r-release-2026-09-30/ops/release_a20r.sh --gpu <0|1>
set -euo pipefail
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "this release uses node-A GPU0 or GPU1 only (--gpu 0|1)" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-27b-a20r-release-2026-09-30
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
HFC=/data/dev2/hf-cache
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-27b-a20r
name=DEV2.0-27B REPO=llm-semantic-router/DEV2.0-27B SPEC=$S/v2/release/specs/dev2-27b-a20r-release.json
superseded=c0dba600087f582a1033830a2276d22e96e6324c
P=/data/dev2/runs/27b/M4-A20r-soup/formal/output
frozen=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache
frozen_digest=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
C1PKG=/data/dev2/runs/release/dev2-27b-a20r-20260930/c1pkg/package/DEV2.0-27B
base_repo=$HFC/models--Qwen--Qwen3.8-27B
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
if [[ -e "$D/$name.decision.a20r.json" ]]; then
  cmp "$R/$name.decision.a20r.json" "$D/$name.decision.a20r.json"
else
  cp "$R/$name.decision.a20r.json" "$D/$name.decision.a20r.json"
fi
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 130 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-27b-a20r-profile-$TS.json" \
  || { echo "successor profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-27b-a20r-copy-$TS
W=/data/dev2/runs/release/dev2-27b-a20r-release-$TS
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 5
(cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$frozen" --expect "$frozen_digest" --dest "$TC")
echo "mirror $SRC gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" \
  --image sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 \
  --gpu "$gpu" --track release --shared-lease "$LEASE" --threads 4 \
  --site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 \
  --base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" \
  --env "HF_HUB_CACHE=$HFC" --mount "$base_repo" --env HIP_FORCE_DEV_KERNARG=1 \
  --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231" \
  --upload --collect --already-collected || status=$?
set +x
(cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$TC") || true
[[ "$status" == 0 ]] || { echo "release.sh failed ($status); work=$W" >&2; exit "$status"; }
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
mkdir -p "$W/extra"
cd "$S"
# The frozen package that C1 scored and the released package: only card files may differ.
python3 - "$W/package/$name/MODEL_MANIFEST.json" "$C1PKG/MODEL_MANIFEST.json" > "$W/extra/c1-package-diff.json" <<'PY' || status=1
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
card = {"README.md", "config.json", "LICENSE", "NOTICE", "ATTRIBUTIONS.md", "LICENSING.md"}
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
other = [n for n in changed if n not in card and not n.startswith(("assets/", "evaluation/", "LICENSES/"))]
same_identity = new["identity"]["model_sha256"] == old["identity"]["model_sha256"]
print(json.dumps({"changed_card_files": [n for n in changed if n not in other], "changed_other_files": other,
                  "identity_equal": same_identity, "ok": not other and same_identity}, indent=1))
sys.exit(0 if not other and same_identity else 1)
PY
"$HFPY" "$RENAME_OPS/revision_diff.py" "$REPO" "$superseded" "$REV" "$W/extra/revision-diff-vs-superseded.json" \
  || echo "revision diff vs $superseded lists weight changes (expected for new weights; see the file)"
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
