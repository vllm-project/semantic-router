#!/usr/bin/env bash
# ~27B M6 amendment 3 build (node B host; CPU only, pinned image, --network none), run from its own mirror, after
# m6-build.sh: the hedge mixture a20ib12pn = a20ib12's terms + PN1H, where PN1H is PN1-r2 TRAIN (private
# llm-semantic-router/decision-2.0-training-data at PN1_REVISION, m4/pn1/arms/pn1.train.jsonl = PN1_SHA) minus the groups
# of the drop lists (the G0 Index-row scan and the PN1 short-text scans; m6_data drop-groups).
# Steps:
#   1. PN1H -> m6-data/pn1h.train.jsonl (the drop lists' SHA-256 recorded);
#   2. a20, a20ib1, a20ib1x, a20ib12 (amendment 1's terms, same order) and a20ib12pn built twice into
#      /data/dev2/private/27b/m6-data/mixtures-m6pn-{1,2}; the builds must be byte-identical, the first four must equal
#      amendment 1's files (BUILD.json), every a20ib12 row must be in a20ib12pn, and every IB1, IB2 and PN1H family must
#      be exhausted;
#   3. the source-level C1 registry check of a20ib12pn's rows new against a20 (counts only);
#   4. BUILD-pn.json: rows, tokens, updates, SAVE_EVERY, the projection from M6's node D receipts (9.85 s per update
#      plus 900 s of SELECT700 evaluations and saves) and the cap check (projection x 1.15 <= 22), plus BUILD.json's
#      IB DEV slice fields.
# Usage: m6-build-pn.sh PN1_REVISION PN1_SHA DROP_LIST...   (from /data/dev2/src/<sha>-src_training_decision2/...)
set -uo pipefail
export TMPDIR=/data/dev2/tmp HF_HUB_CACHE=/data/dev2/hf-cache
PREV=${1:?PN1_REVISION} PN1_SHA=${2:?PN1_SHA}
shift 2
[ $# -ge 1 ] || { echo "at least one drop list" >&2; exit 2; }
[[ "$PREV" =~ ^[0-9a-f]{40}$ ]] && [[ "$PN1_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "bad PN1 revision / hash" >&2; exit 2; }
M=$(cd "$(dirname "$0")/../../.." && pwd)
R=/data/dev2/private/27b/m2-data D=/data/dev2/private/27b/m3-data O=/data/dev2/private/27b/m6-data
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
[ -f "$M/v2/27b/build_mixtures.py" ] || { echo "not inside a mirror: $M" >&2; exit 2; }
[ -f "$O/BUILD.json" ] || { echo "no $O/BUILD.json (m6-build.sh first)" >&2; exit 2; }
for k in 1 2; do [ ! -e "$O/mixtures-m6pn-$k" ] || { echo "$O/mixtures-m6pn-$k exists" >&2; exit 66; }; done
[ ! -e "$O/pn1h.train.jsonl" ] || { echo "$O/pn1h.train.jsonl exists" >&2; exit 66; }
val() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d}))" "$O/BUILD.json" "$1"; }
REV1=$(val "d['ib1_revision']") TRAIN1_SHA=$(val "d['ib1_train_sha256']") IBX_SHA=$(val "d['ibx_train_sha256']")
REV2=$(val "d['ib2_revision']") TRAIN2_SHA=$(val "d['ib2_train_sha256']")
SNAPS=$HF_HUB_CACHE/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
IB1=$(readlink -f "$SNAPS/$REV1/m6/ib1/ib1.train.jsonl") IB2=$(readlink -f "$SNAPS/$REV2/m6/ib2/ib2.train.jsonl")
echo "=== $(date -u +%FT%TZ) m6-build-pn from $M (PN1 revision $PREV)"
hf download llm-semantic-router/decision-2.0-training-data --repo-type dataset --revision "$PREV" \
  --include "m4/pn1/arms/pn1.train.jsonl" > /dev/null || { echo "PN1 download failed"; exit 1; }
PN1=$(readlink -f "$SNAPS/$PREV/m4/pn1/arms/pn1.train.jsonl")
[ "$(sha256sum < "$PN1" | cut -c1-64)" = "$PN1_SHA" ] || { echo "pn1.train.jsonl is not $PN1_SHA"; exit 1; }
[ "$(sha256sum < "$IB1" | cut -c1-64)" = "$TRAIN1_SHA" ] || { echo "ib1.train.jsonl is not $TRAIN1_SHA"; exit 1; }
[ "$(sha256sum < "$IB2" | cut -c1-64)" = "$TRAIN2_SHA" ] || { echo "ib2.train.jsonl is not $TRAIN2_SHA"; exit 1; }
[ "$(sha256sum < "$O/ibx.train.jsonl" | cut -c1-64)" = "$IBX_SHA" ] || { echo "ibx.train.jsonl is not $IBX_SHA"; exit 1; }
groups=()
for list in "$@"; do [ -f "$list" ] || { echo "no drop list $list"; exit 1; }; groups+=(--groups "$list"); done
(cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m6.m6_data drop-groups --input "$PN1" "${groups[@]}" \
  --output "$O/pn1h.train.jsonl") | tee "$O/pn1h.json" || { echo "PN1H FAILED"; exit 1; }
PN1H_SHA=$(sha256sum < "$O/pn1h.train.jsonl" | cut -c1-64)
spec12="base:full,A6:rho,A7:tokens=20000000:family-equal,IB1:tokens=1000000000:family-equal"
spec12+=",IB2:tokens=1000000000:family-equal"
for k in 1 2; do docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/pipeline:/code -e PYTHONDONTWRITEBYTECODE=1 \
 --mount "type=bind,src=$M,dst=/code,readonly" --mount "type=bind,src=$R/pipeline,dst=/pipeline,readonly" \
 --mount type=bind,src=/data/decision20-20260926/models/Qwen3.8-27B,dst=/source,readonly \
 --mount "type=bind,src=$R,dst=/m2,readonly" --mount "type=bind,src=$D/a7,dst=/a7,readonly" \
 --mount "type=bind,src=$D/hf-ed87a03a,dst=/cal,readonly" \
 --mount "type=bind,src=$IB1,dst=/ib/ib1.train.jsonl,readonly" --mount "type=bind,src=$O/ibx.train.jsonl,dst=/ib/ibx.train.jsonl,readonly" \
 --mount "type=bind,src=$IB2,dst=/ib/ib2.train.jsonl,readonly" --mount "type=bind,src=$O/pn1h.train.jsonl,dst=/pn/pn1h.train.jsonl,readonly" \
 --mount "type=bind,src=$O,dst=/outroot" \
 -w /code --entrypoint python3 "$IMAGE" \
 -m v2.27b.build_mixtures --source /source --limit 4096 \
 --base A0s-strict=/m2/hf-12912429/m3/pk1/A0s-strict/train.jsonl \
 --arm A6=/m2/hf-5c0255ed/v2/arms/A6g/train.jsonl,/m2/hf-5c0255ed/v2/arms/A6h/train.jsonl \
 --arm A7=/a7/A7g/train.jsonl,/a7/A7o/train.jsonl,/a7/A7p/train.jsonl,/a7/A7i/train.jsonl \
 --arm IB1=/ib/ib1.train.jsonl --arm IB1X=/ib/ibx.train.jsonl --arm IB2=/ib/ib2.train.jsonl --arm PN1H=/pn/pn1h.train.jsonl \
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
 --expect "/ib/ib1.train.jsonl=$TRAIN1_SHA" --expect "/ib/ibx.train.jsonl=$IBX_SHA" \
 --expect "/ib/ib2.train.jsonl=$TRAIN2_SHA" --expect "/pn/pn1h.train.jsonl=$PN1H_SHA" \
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
 --mixture "a20ib12=$spec12" --mixture "a20ib12pn=$spec12,PN1H:tokens=1000000000:family-equal" \
 --seed decision2-27b-m3 --output-dir "/outroot/mixtures-m6pn-$k" || { echo "BUILD $k FAILED"; exit 1; }; done
diff -r "$O/mixtures-m6pn-1" "$O/mixtures-m6pn-2" || { echo "BUILDS DIFFER"; exit 1; }
echo IDENTICAL
(cd "$O/mixtures-m6pn-1" && sha256sum -- * > ../mixtures-m6pn-1.sha256 && cat ../mixtures-m6pn-1.sha256)
python3 - "$O/mixtures-m6pn-1" "$O/BUILD.json" > "$O/mixtures-m6pn-1.checks.json" <<'EOF' || { echo "CHECKS FAILED"; exit 1; }
import hashlib, json, math, sys
from collections import Counter
from pathlib import Path
root, build = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
locked = build["files_sha256"]
out = {}
for name in ("a20", "a20ib1", "a20ib1x", "a20ib12"):
    got = hashlib.sha256((root / f"{name}.train.jsonl").read_bytes()).hexdigest()
    assert got == locked[f"{name}.train.jsonl"], f"{name} is not amendment 1's file"
    out[f"{name}_equals_amendment_1"] = True
manifest = json.loads((root / "MIXTURES.json").read_text())
base = set((root / "a20ib12.train.jsonl").read_bytes().splitlines(keepends=True))
lines = (root / "a20ib12pn.train.jsonl").read_bytes().splitlines(keepends=True)
assert base <= set(lines), "a20ib12pn misses a20ib12 rows"
parts = manifest["mixtures"]["a20ib12pn"]["parts"]
for arm in ("IB1", "IB2", "PN1H"):
    assert all(f["exhausted"] for f in parts[arm]["families"].values()), f"a PN1H-build {arm} family was not exhausted"
rows = manifest["mixtures"]["a20ib12pn"]["rows"]
tokens = manifest["mixtures"]["a20ib12pn"]["tokens"]
updates = math.ceil(rows / 16)
projection = (9.85 * updates + 900) / 3600
added = [json.loads(line) for line in lines if line not in base]
out["a20ib12pn"] = {"rows": rows, "tokens": tokens, "updates": updates, "save_every": math.ceil(updates / 8),
                    "added_rows_vs_a20ib12": len(added), "pn1h_tokens": parts["PN1H"]["tokens"],
                    "dropped_duplicate_groups": {a: parts[a]["dropped_duplicate_groups"] for a in ("IB1", "IB2", "PN1H")},
                    "added_by_family": dict(sorted(Counter(r["family"] for r in added).items())),
                    "added_by_language": dict(sorted(Counter(r["language"] for r in added).items())),
                    "projection_gpu_h": round(projection, 2), "cap_gpu_h": 22.0, "cap_ok": projection * 1.15 <= 22.0}
assert out["a20ib12pn"]["cap_ok"], f"projection {projection:.2f} x 1.15 passes the 22 cap"
print(json.dumps(out, indent=1, sort_keys=True))
EOF
cat "$O/mixtures-m6pn-1.checks.json"
python3 "$M/v2/27b/m4/m4_c1_sources.py" --train "$O/mixtures-m6pn-1/a20ib12pn.train.jsonl" \
  --m3a "$O/mixtures-m6pn-1/a20.train.jsonl" --output "$O/c1-sources-a20ib12pn.json" || { echo "C1 SOURCES FAILED"; exit 1; }
python3 - "$O" "$PREV" "$PN1_SHA" "$PN1H_SHA" "$@" > "$O/BUILD-pn.json" <<'EOF'
import hashlib, json, sys
from pathlib import Path
o, prev, pn1, pn1h, lists = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
build = json.loads((o / "BUILD.json").read_text())
print(json.dumps({"schema": "decision2-27b-m6-build-pn/1", "mixtures_dir": "mixtures-m6pn-1",
                  "pn1_revision": prev, "pn1_train_sha256": pn1, "pn1h_train_sha256": pn1h,
                  "drop_lists_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in lists},
                  "pn1h": json.loads((o / "pn1h.json").read_text()),
                  **{k: build[k] for k in ("ib1_revision", "ib1_train_sha256", "ib1_dev_sha256", "ib1_dev_path",
                                           "ibx_train_sha256", "ib2_revision", "ib2_train_sha256", "ib2_dev_sha256",
                                           "ib12_dev_path", "ib12_dev_sha256")},
                  "files_sha256": dict(line.split()[::-1] for line in (o / "mixtures-m6pn-1.sha256").read_text().splitlines()),
                  "checks": json.loads((o / "mixtures-m6pn-1.checks.json").read_text()),
                  "c1_sources": {"a20ib12pn": json.loads((o / "c1-sources-a20ib12pn.json").read_text())}},
                 indent=1, sort_keys=True))
EOF
echo "=== $(date -u +%FT%TZ) BUILD-PN-DONE"
