#!/usr/bin/env bash
# Speed-up phase A runtime-only revisions of the Decision 2.0 repositories (user approval 2026-10-02 18:00 UTC+8,
# COORDINATION 18:00; worker 2d541b40, track=runtime-a), one repository at a time, never while another release.sh
# for it runs. Adapted from the org worker's org.sh and round 4's card4.sh.
#   1. refuses unless main is the revision the ra decision supersedes (or --resume REV after an interrupted upload of
#      this package) and make_fast.py release --check re-derives the committed spec and decision on this node
#   2. builds the package from the ra spec and compares MODEL_MANIFEST.json files_sha256 with main's: only README.md
#      and the runtime may change (decision2/__init__.py, api.py, qwen.py changed; fast.py, fast_kernels.py added),
#      nothing else is added or removed, identity, parameters, profile and max_input_tokens are equal, and the
#      README differs from main's only in its Speed line; then gate profile on the spec must pass
#   3. refuses when main moved since step 1; hf_headroom.sh, then release.sh --upload --collect --already-collected
#      (no panel parity here: fast.sh --parity answered every prompt of the four panels through main's package and
#      this one, committed as <key>/parity) with the native examples, the card's Transformers example on the
#      package, on the download and from the Hub in fresh caches under Transformers 5.17 and 5.18, readback, gate
#      and collection; the tier's frozen pre-warmed autotune cache (<key>/triton.json) is copied for the run
#   4. ra_diff.py (weights byte-identical, only runtime and card files changed), the native examples against the
#      superseded package's run on this GPU, image and frozen cache (bit-identical), card HTTP, links, the
#      read-only collection check and gate evaluate; --post-only WORK repeats only this step for WORK's upload
# Node A, GPU0 or GPU1 (the 0.6B track's allocation, where release workers run as recorded co-tenants), under the
# shared lease owner.runtime-a-release (removed on exit).
# Usage: bash <mirror>/v2/release/records/dev2-runtime-a-2026-10-02/ops/ra.sh <tier> --gpu N [--resume REV | --post-only WORK]
set -euo pipefail
tier="${1:-}"
shift || true
gpu="" resume="" post_only=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --resume) resume=$2; shift 2 ;;
    --post-only) post_only=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[01]$ ]] || { echo "--gpu 0 or 1" >&2; exit 2; }
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
R=$S/v2/release/records/dev2-runtime-a-2026-10-02
RENAME_OPS=$S/v2/release/records/dev2-rename-9b-27b-2026-09-29/ops
COLL=vllm-sr/decision-20-6ab7cf7bdfb506bf8269cb00
D=/data/dev2/runs/release/decisions
HFC=/data/dev2/hf-cache
TF518=/data/dev2/tools/tf518
HFPY=/data/dev2/tools/hf-cli/bin/python
HOST2=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
FORMAL=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
image=$FORMAL kernels=1 vram=60 base_args=()
case "$tier" in
  0.6B) key=0p6b codename=Kai kernels=0 image=$HOST2 ;;
  0.8B) key=0p8b codename=Eos ;;
  2B) key=2b codename=Sol ;;
  4B) key=4b codename=Nox ;;
  9B) key=9b codename=Lux image=$HOST2 ;;
  27B)
    key=27b codename=Vega vram=130
    base_repo=$HFC/models--Qwen--Qwen3.8-27B
    base_args=(--base-path "$base_repo/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
               --env "HF_HUB_CACHE=$HFC" --mount "$base_repo")
    [[ ! -d "$HFC/blobs" ]] || base_args+=(--mount "$HFC/blobs") ;;
  *) echo "tier must be one of 0.6B 0.8B 2B 4B 9B 27B" >&2; exit 2 ;;
esac
[[ "$(docker image inspect -f '{{.Id}}' "$image")" == "$image" ]] || { echo "image $image is missing" >&2; exit 1; }
lease=/data/dev2/leases/gpu$gpu.lock
[[ ! -e "$lease/owner.runtime-a-release" ]] || { echo "gpu$gpu already runs a phase A release" >&2; exit 1; }
rocm-smi --showuse --showmeminfo vram --json | python3 "$RENAME_OPS/pick_gpu.py" "$vram" "$gpu" >/dev/null \
  || { echo "GPU$gpu is busy or lacks free VRAM" >&2; exit 1; }
trap 'rm -f "$lease/owner.runtime-a-release"' EXIT
name=Decision-2.0-$codename-$tier REPO=vllm-sr/$name
SPEC=$S/v2/release/specs/dev2-$key-ra.json
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" /data/dev2/runs/release/triton "$D" /data/dev2/runs/release/runtime-a
exec 9> "/data/dev2/runs/release/runtime-a/$key.lock"
flock -n 9 || { echo "another ra.sh for $tier runs on this node" >&2; exit 1; }
X=/data/dev2/runs/release/runtime-a/dev2-ra-$tier-$TS
mkdir -p "$X"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
manifest_sha() { sha256sum "$1/MODEL_MANIFEST.json" | cut -c1-64; }
hub_main() { "$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' "$REPO"; }
decision=$D/$name.decision.ra.json
(cd "$S" && python3 "$R/ops/make_fast.py" release --tiers "$tier" --check) > "$X/derivation.txt" \
  || { echo "the committed ra spec / decision differ from their derivation: refusing" >&2; exit 1; }
if [[ -e "$decision" ]]; then cmp "$R/$name.decision.ra.json" "$decision"; else
  cp "$R/$name.decision.ra.json" "$decision"; chmod 444 "$decision"; fi
read -r expected superseded_work < <(python3 - "$R/$name.decision.ra.json" "$S/v2/release/specs/dev2-$key-ra.json" <<'PY'
import json, sys
from pathlib import Path
d, s = (json.load(open(p)) for p in sys.argv[1:3])
print(d["supersedes"]["released_as"].split("@")[1], Path(s["_release"]["replaces_spec"]["spec"]).parents[1])
PY
)
main=$(hub_main)
if [[ -n "$post_only" ]]; then
  [[ -f "$post_only/receipts/upload.json" ]] || { echo "$post_only has no upload receipt" >&2; exit 1; }
  uploaded=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$post_only/receipts/upload.json")
  [[ "$main" == "$uploaded" ]] || { echo "$REPO main is $main, not $uploaded uploaded by $post_only: refusing" >&2; exit 1; }
  echo "$REPO main $main = the revision $post_only uploaded (post-checks only)"
elif [[ -n "$resume" && "$main" == "$resume" ]]; then
  echo "$REPO main $main = the revision an interrupted run of this package uploaded (resume)"
else
  [[ "$main" == "$expected" ]] || { echo "$REPO main is $main, not the superseded revision $expected: refusing" >&2; exit 1; }
  echo "$REPO main $main = superseded revision"
fi

check=$TMPDIR/dev2-ra-$tier-check-$TS
python3 -m v2.release.build --spec "$SPEC" --output "$check/$name" > "$X/build-check.log"
built=$(manifest_sha "$check/$name")
"$HFPY" - "$REPO" "$expected" "$check/$name" "$X/compare.json" <<'PY' || { echo "the fresh build is not a runtime-only revision of $expected" >&2; exit 1; }
import difflib, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, revision, package, out = sys.argv[1:]
old = json.load(open(hf_hub_download(repo, "MODEL_MANIFEST.json", revision=revision)))
new = json.load(open(Path(package) / "MODEL_MANIFEST.json"))
fo, fn = old["files_sha256"], new["files_sha256"]
changed = sorted(n for n in set(fo) | set(fn) if fo.get(n) != fn.get(n))
runtime = {"decision2/__init__.py", "decision2/api.py", "decision2/qwen.py", "decision2/kai_native.py"}
added = {"decision2/fast.py", "decision2/fast_kernels.py"}
other = [n for n in changed if not (n == "README.md" or (n in runtime and n in fo and n in fn)
                                    or (n in added and n not in fo))]
same = {k: old[k] == new[k] for k in ("identity", "parameters", "profile", "max_input_tokens")}
readme_old = open(hf_hub_download(repo, "README.md", revision=revision), encoding="utf-8").read().splitlines()
readme_new = (Path(package) / "README.md").read_text(encoding="utf-8").splitlines()
lines = [l for l in difflib.unified_diff(readme_old, readme_new, lineterm="", n=0)
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
readme_ok = bool(lines) and all("**Speed:**" in l for l in lines)
ok = not other and all(same.values()) and readme_ok and added <= set(changed)
json.dump({"changed": changed, "other": other, "equal": same, "readme_diff": lines, "ok": ok},
          open(out, "w"), indent=1)
print(json.dumps({"changed": changed, "other": other, "readme_diff": lines, "ok": ok}))
sys.exit(0 if ok else 1)
PY
cp "$check/$name/MODEL_MANIFEST.json" "$X/build-manifest.json"
rm -rf "$check"
echo "fresh build $built: only the runtime and the Speed line differ from $expected"
python3 -m v2.release.gate profile --spec "$SPEC" > "$X/profile.json" \
  || { echo "gate profile of $SPEC does not pass: refusing" >&2; exit 1; }
echo "gate profile passes"

read -r frozen frozen_digest < <(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); print(t["path"], t["digest"])' "$R/$key/triton.json")
[[ "$(digest "$frozen")" == "$frozen_digest" ]] || { echo "frozen cache $frozen changed" >&2; exit 1; }
kernel_args=()
[[ "$kernels" != 1 ]] || kernel_args=(--site /opt/decision-fla --require-kernels)
status=0
if [[ -n "$post_only" ]]; then
W=$post_only
[[ ! -d "$W/extra" ]] || mv "$W/extra" "$W/extra.before-$TS"
else
if pgrep -af "[v]2/release/release[.]sh" | grep -q -- "specs/dev2-$key-"; then
  echo "another release.sh for $REPO runs on this node: refusing" >&2; exit 1
fi
[[ "$(hub_main)" == "$main" ]] || { echo "$REPO main moved since the check: refusing" >&2; exit 1; }
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 3
W=/data/dev2/runs/release/dev2-ra-$tier-$TS
TC=/data/dev2/runs/release/triton/dev2-ra-$tier-$TS
cp -a "$frozen" "$TC"
chmod -R u+w "$TC"
cache_args=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC")
echo "mirror $SRC tier $tier gpu $gpu work $W cache $TC from $frozen tf518 $(cd "$TF518" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)"
set -x
"$S/v2/release/release.sh" --spec "$SPEC" --src "$SRC" --work "$W" --image "$image" \
  --gpu "$gpu" --track runtime-a --shared-lease runtime-a-release --threads 4 \
  "${kernel_args[@]}" "${base_args[@]}" --env HIP_FORCE_DEV_KERNARG=1 "${cache_args[@]}" \
  --upload --collect --already-collected --hub-site "tf518=$TF518" || status=$?
set +x
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
fi
REV=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['revision'])" "$W/receipts/upload.json")
if [[ -n "$post_only" ]]; then
  # Rebuilt from this mirror: builder.source_commit names it instead of the mirror that built the upload.
  python3 - "$X/build-manifest.json" "$W/package/$name/MODEL_MANIFEST.json" <<'PY' \
    || { echo "released package differs from the checked build" >&2; status=1; }
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for m in (a, b):
    m["builder"].pop("source_commit")
sys.exit(0 if a == b else 1)
PY
else
  [[ "$(manifest_sha "$W/package/$name")" == "$built" ]] || { echo "released package differs from the checked build" >&2; status=1; }
fi
mkdir -p "$W/extra"
cp "$X"/*.json "$X/derivation.txt" "$W/extra/"
cd "$S"
"$HFPY" "$R/ops/ra_diff.py" "$REPO" "$expected" "$REV" "$W/extra/runtime-diff.json" || status=1
# The superseded package runs its examples here, on this GPU and image with a copy of the same frozen autotune
# cache as the new package's pre-upload examples: the superseded release's own receipt came from another GPU,
# device or autotune cache (Kai's org revision: CPU; Nox's: another GPU and cache), whose kernel configurations
# differ. That cross-environment comparison is kept for the record only.
old=$X/superseded/$name
OTC=/data/dev2/runs/release/triton/dev2-ra-$tier-$TS-superseded
cp -a "$frozen" "$OTC"
chmod -R u+w "$OTC"
"$HFPY" -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' \
  "$REPO" "$expected" "$old" > /dev/null
printf 'track=runtime-a\npurpose=superseded examples %s\nstart_utc=%s\nexpected_end_utc=%s\nrun_dir=%s\n' "$REPO" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date -u -d '+30 min' +%Y-%m-%dT%H:%M:%SZ)" "$W" > "$lease/owner.runtime-a-release"
old_mounts=() old_args=()
for ((i = 0; i < ${#base_args[@]}; i += 2)); do
  case "${base_args[i]}" in
    --base-path) old_args+=(--base-path "${base_args[i+1]}") ;;
    --env) old_mounts+=(-e "${base_args[i+1]}") ;;
    --mount) old_mounts+=(-v "${base_args[i+1]}:${base_args[i+1]}:ro") ;;
  esac
done
docker run --rm --network none --ipc host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES="$gpu" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_AUTOTUNING=1 -e "TRITON_CACHE_DIR=$OTC" \
  -v "/data/dev2/src/$SRC:/data/dev2/src/$SRC:ro" -v "$old:$old:ro" -v "$W/extra:$W/extra" -v "$OTC:$OTC" \
  "${old_mounts[@]}" --entrypoint python3 "$image" -I -B \
  "$S/v2/release/examples.py" run --package "$old" --device cuda:0 --threads 4 "${kernel_args[@]}" "${old_args[@]}" \
  --output "$W/extra/examples-superseded-gpu.json" > "$X/examples-superseded-gpu.log" 2>&1 || status=1
python3 "$S/v2/release/examples.py" compare "$W/extra/examples-superseded-gpu.json" "$W/receipts/pre-a.json" \
  --tolerance 0 --output "$W/extra/examples-vs-superseded.json" || status=1
python3 "$S/v2/release/examples.py" compare "$superseded_work/receipts/pre-a.json" "$W/receipts/pre-a.json" \
  --tolerance 0 --output "$W/extra/examples-vs-superseded-receipt.json" > /dev/null || true
"$HFPY" -m v2.release.tests.hub_card_http_check --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/card-http.json" || status=1
"$HFPY" -m v2.release.hub_links --repo "$REPO" --revision "$REV" \
  --package "$W/package/$name" --output "$W/extra/hub-links.json" || status=1
"$HFPY" "$RENAME_OPS/collection_order.py" "$COLL" "$W/extra/collection-order.json" || status=1
python3 -m v2.release.gate evaluate --work "$W" > "$W/extra/gate-evaluate.json" || status=1
bash "$S/v2/common/hf_headroom.sh" --min-free-gb 0
echo "work=$W revision=$REV post_checks=$([[ $status == 0 ]] && echo ok || echo FAILED)"
exit "$status"
