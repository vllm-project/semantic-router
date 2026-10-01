#!/usr/bin/env bash
# Decision-2.0-Lux-9B successor K-a13IB (coordinator decision 2026-10-02 02:05, the Index path): node A GPU6 or GPU7,
# shared lease owner.release-9b-ka13ib, the scored image host2 (f83b1d10) with its kernels, HIP_FORCE_DEV_KERNARG=1 and
# a fresh copy of the formal run's persisted autotune cache (formal-m9/triton-cache, digest-checked; the formal panels
# and the mlx-diag run shared it).
#   --stage       CPU: the current revision's sealed gate receipt and decision (this record's current/ and the card
#                 round-2 record) into the release inputs, checked against make_ka13ib.py's digests
#   --prerelease  release.sh without upload on the draft spec and decision (build, native and card examples, AutoModel
#                 and pipeline before upload), then verify_bundle: the frozen package that C1 (item 8) scores
#   --release     only if the Hub main is the superseded card round-2 revision 586af779 and no other release.sh runs on
#                 node A: hf_headroom.sh, then release.sh --upload --collect --already-collected --hub-site tf518=...
#                 with the final spec and decision (R1', R2-R8) and exact parity of typed-final 1,600, css15 6,547,
#                 public231 231 and mlx-diag 2,275 before upload (AutoModel against native on every scored prompt);
#                 then the C1-scored package vs the released one (weights and identity equal), the revision diff,
#                 collection order, card HTTP, links and gate evaluate
# Usage (node A): bash <mirror>/v2/release/records/dev2-9b-ka13ib-2026-10-02/ops/release_ka13ib.sh <mode> [--gpu 6|7]
set -euo pipefail
mode="${1:-}"
shift || true
gpu=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$mode" =~ ^--(stage|prerelease|release)$ ]] || { echo "mode: --stage|--prerelease|--release" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-9b-ka13ib-2026-10-02
CARD2=$S/v2/release/records/dev2-card2-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
G=/data/dev2/private/panels/goldfree
D=/data/dev2/runs/release/decisions
HFPY=/data/dev2/tools/hf-cli/bin/python
COLL=llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00
LEASE=release-9b-ka13ib
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
name=Decision-2.0-Lux-9B REPO=llm-semantic-router/Decision-2.0-Lux-9B
IN=/data/dev2/runs/release/inputs/dev2-9b-ka13ib
P=$IN/t1
FORMAL_CACHE=/data/dev2/runs/9b/formal-m9/triton-cache
FORMAL_CACHE_DIGEST=${FORMAL_CACHE_DIGEST:-}
superseded=586af77916ee508320421bda6c22f7f0305a7279
TF518=/data/dev2/tools/tf518
TF518_DIGEST=${TF518_DIGEST:-}
C1PKG=${C1PKG:-}
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D"
digest() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
install_decision() {
  if [[ -e "$2" ]]; then cmp "$1" "$2"; else cp "$1" "$2"; chmod 444 "$2"; fi
}

if [[ "$mode" == --stage ]]; then
  mkdir -p "$IN/current"
  install_decision "$R/current/card2-9b-gate.json" "$IN/current/card2-9b-gate.json"
  install_decision "$CARD2/$name.decision.card2.json" "$IN/current/$name.decision.card2.json"
  python3 - "$IN/current" "$S/v2/release/records/dev2-9b-ka13ib-2026-10-02/ops/make_ka13ib.py" <<'PY'
import hashlib, re, sys
from pathlib import Path
text = Path(sys.argv[2]).read_text()
for name, key in (("card2-9b-gate.json", "gate_sha256"), ("Decision-2.0-Lux-9B.decision.card2.json", "decision_sha256")):
    want = re.search(rf'"{key}": "([0-9a-f]{{64}})"', text).group(1)
    got = hashlib.sha256((Path(sys.argv[1]) / name).read_bytes()).hexdigest()
    assert got == want, (name, got)
    print(name, got[:12])
PY
  echo "formal cache digest $(digest "$FORMAL_CACHE") files $(find "$FORMAL_CACHE" -type f | wc -l)"
  exit 0
fi

[[ "$gpu" == 6 || "$gpu" == 7 ]] || { echo "this release uses node-A GPU6 or GPU7 only (--gpu 6|7)" >&2; exit 2; }
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]] || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 1; }
[[ -n "$FORMAL_CACHE_DIGEST" && "$(digest "$FORMAL_CACHE")" == "$FORMAL_CACHE_DIGEST" ]] \
  || { echo "formal cache $FORMAL_CACHE does not match FORMAL_CACHE_DIGEST" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" 60 "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.$LEASE"' EXIT
kernel_args=(--site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1)

if [[ "$mode" == --release ]]; then
  SPEC=$S/v2/release/specs/dev2-9b-ka13ib.json
  install_decision "$R/$name.decision.ka13ib.json" "$D/$name.decision.ka13ib.json"
else
  SPEC=$S/v2/release/specs/dev2-9b-ka13ib.draft.json
  install_decision "$R/$name.decision.ka13ib.draft.json" "$D/$name.decision.ka13ib.draft.json"
fi
W=/data/dev2/runs/release/dev2-9b-ka13ib-${mode#--}-$TS
(cd "$S" && python3 -m v2.release.gate profile --spec "$SPEC") > "/data/dev2/runs/release/dev2-9b-ka13ib-profile-$TS.json" \
  || { echo "successor profile does not pass" >&2; exit 1; }
TC=/data/dev2/runs/release/triton/dev2-9b-ka13ib-${mode#--}-$TS
cp -a "$FORMAL_CACHE" "$TC"
chmod -R u+w "$TC"
parity_args=() hub_args=()
if [[ "$mode" == --release ]]; then
  # Publish only on top of the revision the final decision supersedes (never concurrently with another worker).
  main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO")
  [[ "$main" == "$superseded" ]] || { echo "$REPO main is $main, not the superseded revision $superseded" >&2; exit 1; }
  [[ -n "$TF518_DIGEST" && "$(digest "$TF518")" == "$TF518_DIGEST" ]] \
    || { echo "Transformers 5.18 site $TF518 does not match TF518_DIGEST" >&2; exit 1; }
  [[ -n "$C1PKG" && -f "$C1PKG/MODEL_MANIFEST.json" ]] || { echo "C1PKG (the C1-scored package) is required" >&2; exit 2; }
  if pgrep -f "v2/release/release.sh" >/dev/null; then echo "another release.sh runs on this node" >&2; exit 1; fi
  bash "$S/v2/common/hf_headroom.sh" --min-free-gb 20
  parity_args=(
    --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600"
    --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547"
    --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231"
    --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275"
  )
  hub_args=(--upload --collect --already-collected --hub-site "tf518=$TF518")
fi
echo "mirror $SRC mode $mode gpu $gpu work $W"
set -x
status=0
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$IMAGE" \
  --gpu "$gpu" --track release-9b-ka13ib --shared-lease "$LEASE" --threads 4 "${kernel_args[@]}" \
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
python3 - "$PKG/MODEL_MANIFEST.json" "$C1PKG/MODEL_MANIFEST.json" > "$W/extra/c1-package-diff.json" <<'PY' || status=1
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
fn, fo = new["files_sha256"], old["files_sha256"]
changed = sorted(n for n in set(fn) | set(fo) if fn.get(n) != fo.get(n))
weights = [n for n in changed if n.endswith(".safetensors") or n.startswith(("backbone/", "decision2/")) or n == "decision_config.json"]
same = new["identity"]["model_sha256"] == old["identity"]["model_sha256"]
print(json.dumps({"changed": changed, "changed_weight_or_runtime_files": weights, "identity_equal": same,
                  "ok": not weights and same}, indent=1))
sys.exit(0 if not weights and same else 1)
PY
"$HFPY" "$RENAME_OPS/revision_diff.py" "$REPO" "$superseded" "$REV" "$W/extra/revision-diff-vs-superseded.json" \
  || echo "revision diff vs $superseded lists weight changes (expected for new weights; see the file)"
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$PKG" --output "$W/extra/hub-links.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV gpu=$gpu post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
