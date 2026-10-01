#!/usr/bin/env bash
# ~27B M6 stage-1 mixture build (node B host; CPU only, pinned image, --network none), run from its own mirror
# (preregistration "Data"). Inputs: M5's a20 inputs, plus the release-safe IB1 files of private
# llm-semantic-router/decision-2.0-training-data at REVISION (m6/ib1/ib1.train.jsonl, ib1.dev.jsonl), fetched into
# the node's HF cache with the HF CLI and checked against TRAIN_SHA / DEV_SHA (the data record's values).
# Steps:
#   1. IBX = IB1 TRAIN minus the in-distribution families w2c, isarc (m6_data drop-families) -> m6-data/ibx.train.jsonl
#   2. a20, a20ib1, a20ib1x built twice into /data/dev2/private/27b/m6-data/mixtures-m6-{1,2}; the builds must be
#      byte-identical; a20 must be M4's file 4aa0dc96; every a20 row must be in both M6 mixtures; every IB family
#      must be exhausted (all its non-duplicate groups taken)
#   3. the source-level C1 registry check of the rows new against a20 (m4_c1_sources.py; counts only)
#   4. BUILD.json: rows, tokens, updates (16 rows each), SAVE_EVERY = ceil(updates / 8), the GPU-hour projection
#      (0.1 s per row plus 0.0012048 s per token: L128's 9.96 GPU-h on a20 with a per-row allowance) and the per-seed
#      cap check (projection x 1.15 <= the cap: 16 for stage 1, 20 for stage 2)
# With the optional IB2 arguments (stage 2, preregistration "Stage 2: IB2") the same build also writes a20ib12 = a20's
# terms + IB1 + IB2, each block whole after the dedup (IB2 must add <= 10M tokens), and the stage-2 IB DEV slice
# ib12.dev.jsonl = IB1 DEV + IB2 DEV (m6_data concat).
# Usage: m6-build.sh REVISION TRAIN_SHA DEV_SHA [IB2_REVISION IB2_TRAIN_SHA IB2_DEV_SHA]
#   (from /data/dev2/src/<sha>-src_training_decision2/...)
set -uo pipefail
export TMPDIR=/data/dev2/tmp HF_HUB_CACHE=/data/dev2/hf-cache
REV=${1:?REVISION} TRAIN_SHA=${2:?TRAIN_SHA} DEV_SHA=${3:?DEV_SHA}
REV2=${4:-} TRAIN2_SHA=${5:-} DEV2_SHA=${6:-}
for v in "$REV" "$TRAIN_SHA" "$DEV_SHA" ${REV2:+"$REV2" "$TRAIN2_SHA" "$DEV2_SHA"}; do
  [[ "$v" =~ ^[0-9a-f]{40}([0-9a-f]{24})?$ ]] || { echo "bad hash $v" >&2; exit 2; }
done
M=$(cd "$(dirname "$0")/../../.." && pwd)
R=/data/dev2/private/27b/m2-data D=/data/dev2/private/27b/m3-data O=/data/dev2/private/27b/m6-data
SNAP=$HF_HUB_CACHE/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/$REV/m6/ib1
A20=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
[ -f "$M/v2/27b/build_mixtures.py" ] || { echo "not inside a mirror: $M" >&2; exit 2; }
for k in 1 2; do [ ! -e "$O/mixtures-m6-$k" ] || { echo "$O/mixtures-m6-$k exists" >&2; exit 66; }; done
mkdir -p "$O" && chmod 700 "$O"
echo "=== $(date -u +%FT%TZ) m6-build from $M (IB1 revision $REV)"
hf download llm-semantic-router/decision-2.0-training-data --repo-type dataset --revision "$REV" \
  --include "m6/ib1/ib1.train.jsonl" --include "m6/ib1/ib1.dev.jsonl" --include "m6/ib1/status.json" > /dev/null ||
  { echo "IB1 download failed"; exit 1; }
IB1=$(readlink -f "$SNAP/ib1.train.jsonl") IBDEV=$(readlink -f "$SNAP/ib1.dev.jsonl")
[ "$(sha256sum < "$IB1" | cut -c1-64)" = "$TRAIN_SHA" ] || { echo "ib1.train.jsonl is not $TRAIN_SHA"; exit 1; }
[ "$(sha256sum < "$IBDEV" | cut -c1-64)" = "$DEV_SHA" ] || { echo "ib1.dev.jsonl is not $DEV_SHA"; exit 1; }
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get('release_safe') is True else 1)" \
  "$SNAP/status.json" || { echo "status.json at $REV is not release_safe"; exit 1; }
[ -e "$O/ibx.train.jsonl" ] || (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m6.m6_data drop-families --input "$IB1" \
  --drop w2c --drop isarc --output "$O/ibx.train.jsonl") | tee "$O/ibx.json" || { echo "IBX FAILED"; exit 1; }
IBX_SHA=$(sha256sum < "$O/ibx.train.jsonl" | cut -c1-64)
IB2DEV='' IB12DEV_SHA='' stage2=() mount2=()
if [ -n "$REV2" ]; then
  SNAP2=$HF_HUB_CACHE/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/$REV2/m6/ib2
  hf download llm-semantic-router/decision-2.0-training-data --repo-type dataset --revision "$REV2" \
    --include "m6/ib2/ib2.train.jsonl" --include "m6/ib2/ib2.dev.jsonl" --include "m6/ib2/status.json" > /dev/null ||
    { echo "IB2 download failed"; exit 1; }
  IB2=$(readlink -f "$SNAP2/ib2.train.jsonl") IB2DEV=$(readlink -f "$SNAP2/ib2.dev.jsonl")
  [ "$(sha256sum < "$IB2" | cut -c1-64)" = "$TRAIN2_SHA" ] || { echo "ib2.train.jsonl is not $TRAIN2_SHA"; exit 1; }
  [ "$(sha256sum < "$IB2DEV" | cut -c1-64)" = "$DEV2_SHA" ] || { echo "ib2.dev.jsonl is not $DEV2_SHA"; exit 1; }
  python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get('release_safe') is True else 1)" \
    "$SNAP2/status.json" || { echo "status.json at $REV2 is not release_safe"; exit 1; }
  [ -e "$O/ib12.dev.jsonl" ] || (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m6.m6_data concat --input "$IBDEV" \
    --input "$IB2DEV" --output "$O/ib12.dev.jsonl") | tee "$O/ib12.json" || { echo "IB12 DEV FAILED"; exit 1; }
  IB12DEV_SHA=$(sha256sum < "$O/ib12.dev.jsonl" | cut -c1-64)
  mount2=(--mount "type=bind,src=$IB2,dst=/ib/ib2.train.jsonl,readonly")
  spec12="a20ib12=base:full,A6:rho,A7:tokens=20000000:family-equal,IB1:tokens=1000000000:family-equal"
  spec12+=",IB2:tokens=1000000000:family-equal"
  stage2=(--arm IB2=/ib/ib2.train.jsonl --expect "/ib/ib2.train.jsonl=$TRAIN2_SHA" --mixture "$spec12")
fi
for k in 1 2; do docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/pipeline:/code -e PYTHONDONTWRITEBYTECODE=1 \
 --mount "type=bind,src=$M,dst=/code,readonly" --mount "type=bind,src=$R/pipeline,dst=/pipeline,readonly" \
 --mount type=bind,src=/data/decision20-20260926/models/Qwen3.8-27B,dst=/source,readonly \
 --mount "type=bind,src=$R,dst=/m2,readonly" --mount "type=bind,src=$D/a7,dst=/a7,readonly" \
 --mount "type=bind,src=$D/hf-ed87a03a,dst=/cal,readonly" \
 --mount "type=bind,src=$IB1,dst=/ib/ib1.train.jsonl,readonly" --mount "type=bind,src=$O/ibx.train.jsonl,dst=/ib/ibx.train.jsonl,readonly" \
 "${mount2[@]}" \
 --mount "type=bind,src=$O,dst=/outroot" \
 -w /code --entrypoint python3 "$IMAGE" \
 -m v2.27b.build_mixtures --source /source --limit 4096 \
 --base A0s-strict=/m2/hf-12912429/m3/pk1/A0s-strict/train.jsonl \
 --arm A6=/m2/hf-5c0255ed/v2/arms/A6g/train.jsonl,/m2/hf-5c0255ed/v2/arms/A6h/train.jsonl \
 --arm A7=/a7/A7g/train.jsonl,/a7/A7o/train.jsonl,/a7/A7p/train.jsonl,/a7/A7i/train.jsonl \
 --arm IB1=/ib/ib1.train.jsonl --arm IB1X=/ib/ibx.train.jsonl \
 --aho A6g=/m2/mixtures-v1/aho-A6g.jsonl --aho A6h=/m2/mixtures-v1/aho-A6h.jsonl \
 --aho-sample A7=/a7/A7g/aho.jsonl,/a7/A7o/aho.jsonl,/a7/A7p/aho.jsonl,/a7/A7i/aho.jsonl:800 \
 --select /m2/hf-5c0255ed/select.jsonl --cal /cal/m2/cal/CAL698/cal.jsonl \
 --expect /m2/hf-12912429/m3/pk1/A0s-strict/train.jsonl=df5c76888e4397068ada2444815b50baa7eb6e36fe6c9a1c991ef069b01a7853 \
 --expect /m2/hf-5c0255ed/v2/arms/A6g/train.jsonl=ac94b7f4f1e7d8b73ddbffebe15035aa3b47cca9c825f622d0ec8d04bd7fc02f \
 --expect /m2/hf-5c0255ed/v2/arms/A6h/train.jsonl=23440f0c5561077780973259777fca4aa80400f5b30e9df6e1376f6d9aa31927 \
 --expect /a7/A7g/train.jsonl=cc87e0b7a441774aaacbd34b59425294bf8fd5f1b678f92c9a15fd008dcdfddd \
 --expect /a7/A7o/train.jsonl=dbba002225188fb4dca8f45d15a4801c41a8964d313ee49eba335e2af6355625 \
 --expect /a7/A7p/train.jsonl=cc9217e6128da1910784cb1839127a3edf7e5b90d5c795cd104d78f35f2f137e \
 --expect /a7/A7i/train.jsonl=0b8284b1e08229698e9516be7104fe2b6f7dad79ec25eddc3cd3c1ad48e72e12 \
 --expect "/ib/ib1.train.jsonl=$TRAIN_SHA" --expect "/ib/ibx.train.jsonl=$IBX_SHA" \
 --expect /m2/mixtures-v1/aho-A6g.jsonl=c6b2a0b4bf55e82a8176cb5871106d383900a76934630b4543b7bc285f7d5304 \
 --expect /m2/mixtures-v1/aho-A6h.jsonl=5b6c3fd090464ff33d5a60dbbbc7f0cbfd1b2cdc9df12089549ac24f4b1c0fb4 \
 --expect /a7/A7g/aho.jsonl=25f31a2edad5a6768104a5de5bab993bc41388cc97dd0a254b750737baba2580 \
 --expect /a7/A7o/aho.jsonl=00986778604daf87906b29eef9d4e48622868fdc9a9fcfad1987eb0f0897a4a0 \
 --expect /a7/A7p/aho.jsonl=19ab727c9ec76c9ed262813c01f0e651a3fa23ea24a0666adb4827e17e6db740 \
 --expect /a7/A7i/aho.jsonl=78f580f62b7784a12fb2841774f85d5e6b4a7bc9aa22847c90fc939cbefea2d6 \
 --expect /m2/hf-5c0255ed/select.jsonl=32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6 \
 --expect /cal/m2/cal/CAL698/cal.jsonl=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f \
 --mixture a20=base:full,A6:rho,A7:tokens=20000000:family-equal \
 --mixture a20ib1=base:full,A6:rho,A7:tokens=20000000:family-equal,IB1:tokens=1000000000:family-equal \
 --mixture a20ib1x=base:full,A6:rho,A7:tokens=20000000:family-equal,IB1X:tokens=1000000000:family-equal \
 "${stage2[@]}" --seed decision2-27b-m3 --output-dir "/outroot/mixtures-m6-$k" || { echo "BUILD $k FAILED"; exit 1; }; done
diff -r "$O/mixtures-m6-1" "$O/mixtures-m6-2" || { echo "BUILDS DIFFER"; exit 1; }
echo IDENTICAL
(cd "$O/mixtures-m6-1" && sha256sum -- * > ../mixtures-m6-1.sha256 && cat ../mixtures-m6-1.sha256)
python3 - "$O/mixtures-m6-1" "$A20" > "$O/mixtures-m6-1.checks.json" <<'EOF' || { echo "CHECKS FAILED"; exit 1; }
import hashlib, json, math, sys
from collections import Counter
from pathlib import Path
root, a20_sha = Path(sys.argv[1]), sys.argv[2]
a20 = (root / "a20.train.jsonl").read_bytes()
assert hashlib.sha256(a20).hexdigest() == a20_sha, "a20 is not M4's file"
a20_lines = set(a20.splitlines(keepends=True))
manifest = json.loads((root / "MIXTURES.json").read_text())
out = {"a20_is_m4": True}
plan = [("a20ib1", ["IB1"], 16.0), ("a20ib1x", ["IB1X"], 16.0)]
if "a20ib12" in manifest["mixtures"]:
    plan.append(("a20ib12", ["IB1", "IB2"], 20.0))
for name, arms, cap in plan:
    lines = (root / f"{name}.train.jsonl").read_bytes().splitlines(keepends=True)
    assert a20_lines <= set(lines), f"{name} misses a20 rows"
    parts = {arm: manifest["mixtures"][name]["parts"][arm] for arm in arms}
    for arm, part in parts.items():
        assert all(f["exhausted"] for f in part["families"].values()), f"{name}: an {arm} family was not exhausted"
    if "IB2" in parts:
        assert parts["IB2"]["tokens"] <= 10_000_000, f"{name}: IB2 adds more than 10M tokens (prereg: family-equal 10M)"
    rows = manifest["mixtures"][name]["rows"]
    tokens = manifest["mixtures"][name]["tokens"]
    updates = math.ceil(rows / 16)
    projection = (0.1 * rows + 0.0012048 * tokens) / 3600
    added = [json.loads(line) for line in lines if line not in a20_lines]
    out[name] = {"rows": rows, "tokens": tokens, "updates": updates, "save_every": math.ceil(updates / 8),
                 "added_rows": len(added), "added_tokens": {a: p["tokens"] for a, p in parts.items()},
                 "dropped_duplicate_groups": {a: p["dropped_duplicate_groups"] for a, p in parts.items()},
                 "added_by_family": dict(sorted(Counter(r["family"] for r in added).items())),
                 "projection_gpu_h": round(projection, 2), "cap_gpu_h": cap,
                 "cap_ok": projection * 1.15 <= cap}
    assert out[name]["cap_ok"], f"{name}: projection {projection:.2f} x 1.15 passes the {cap} cap"
print(json.dumps(out, indent=1, sort_keys=True))
EOF
cat "$O/mixtures-m6-1.checks.json"
for name in a20ib1 a20ib1x ${REV2:+a20ib12}; do
  python3 "$M/v2/27b/m4/m4_c1_sources.py" --train "$O/mixtures-m6-1/$name.train.jsonl" \
    --m3a "$O/mixtures-m6-1/a20.train.jsonl" --output "$O/c1-sources-$name.json" || { echo "C1 SOURCES $name FAILED"; exit 1; }
done
python3 - "$O" "$REV" "$TRAIN_SHA" "$DEV_SHA" "$IBX_SHA" "$IBDEV" "$REV2" "$TRAIN2_SHA" "$DEV2_SHA" "$IB12DEV_SHA" \
  > "$O/BUILD.json" <<'EOF'
import json, sys
from pathlib import Path
o, rev, train, dev, ibx, ibdev, rev2, train2, dev2, ib12 = sys.argv[1:]
o = Path(o)
names = ["a20ib1", "a20ib1x"] + (["a20ib12"] if rev2 else [])
c1 = {n: json.loads((o / f"c1-sources-{n}.json").read_text()) for n in names}
stage2 = {"ib2_revision": rev2, "ib2_train_sha256": train2, "ib2_dev_sha256": dev2,
          "ib12_dev_path": str(o / "ib12.dev.jsonl"), "ib12_dev_sha256": ib12} if rev2 else {}
print(json.dumps({"schema": "decision2-27b-m6-build/1", "ib1_revision": rev, "ib1_train_sha256": train,
                  "ib1_dev_sha256": dev, "ib1_dev_path": ibdev, "ibx_train_sha256": ibx, **stage2,
                  "files_sha256": dict(line.split()[::-1] for line in (o / "mixtures-m6-1.sha256").read_text().splitlines()),
                  "checks": json.loads((o / "mixtures-m6-1.checks.json").read_text()), "c1_sources": c1},
                 indent=1, sort_keys=True))
EOF
echo "=== $(date -u +%FT%TZ) BUILD-DONE"
