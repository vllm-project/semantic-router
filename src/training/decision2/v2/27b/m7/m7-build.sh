#!/usr/bin/env bash
# ~27B M7 mixture build (node B host; CPU only, pinned image, --network none), run from its own mirror (M7 plan,
# records/m7-plan-2026-10-02.md; COORDINATOR UPDATE 15:50 UTC+8). M6's a20 inputs and recipe (m6-build.sh) with:
#   IB1S  IB1 TRAIN (revision 31b200a3, M6's file) without `sentfin` (IB4's `sentfin3` replaces it; m6_data drop-families)
#   IB4   IB4 phase 1 TRAIN (m6/ib4/p1 @ 76cea510; C1 recheck r3 PASS as a block), checked against its SHA-256
#   IB2   M6's IB2 TRAIN (c5dbdd0a)
# Mixtures (built twice into /data/dev2/private/27b/m7-data/mixtures-m7-{1,2}; byte-identical; a20 must be M4's
# 4aa0dc96 and every a20 row in every mixture; every IB family exhausted):
#   a20ib14   a20's terms + IB1S + IB4            a20ib124  a20's terms + IB1S + IB2 + IB4
# then the ML block on each (m7_data ml-upsample, seed decision2-27b-m7; run once per build, the outputs equal):
#   a20ib14ml, a20ib124ml = the mixture + one copy of whole non-English a20 groups restoring a20's multilingual share.
# BUILD.json: rows, tokens (an ML copy counted at a20's mean tokens per row), updates (16 rows each), SAVE_EVERY =
# ceil(updates / 8) and the per-seed projection (M6's cost model scaled by M6-IB's measured 13.5 h on a20ib1) against
# the 22 GPU-h seed cap; the source-level C1 registry check of each mixture's rows new against a20 (counts only).
# Usage: m7-build.sh   (from /data/dev2/src/<sha>-src_training_decision2/...)
set -uo pipefail
export TMPDIR=/data/dev2/tmp HF_HUB_CACHE=/data/dev2/hf-cache
M=$(cd "$(dirname "$0")/../../.." && pwd)
R=/data/dev2/private/27b/m2-data D=/data/dev2/private/27b/m3-data O=/data/dev2/private/27b/m7-data
SNAPS=$HF_HUB_CACHE/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
IB1=$SNAPS/31b200a34759d6b2ca197737603dfaa253c471cd/m6/ib1/ib1.train.jsonl
IB1_SHA=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
IB2=$SNAPS/c5dbdd0a88efe58059c6ece8ae2b181f9132619f/m6/ib2/ib2.train.jsonl
IB2_SHA=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
IB4=$SNAPS/76cea51098907acf3e68e1cc7db4b5d771beb49e/m6/ib4/p1/ib4.train.jsonl
IB4_SHA=6045b456d4b3032db4b76d70d792e3e9e30aad13e16a8f86fb3e99bc325806fb
A20=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
[ -f "$M/v2/27b/build_mixtures.py" ] || { echo "not inside a mirror: $M" >&2; exit 2; }
for k in 1 2; do [ ! -e "$O/mixtures-m7-$k" ] && [ ! -e "$O/ml-$k-a20ib14.json" ] || { echo "$O/mixtures-m7-$k exists" >&2; exit 66; }; done
mkdir -p "$O" && chmod 700 "$O"
echo "=== $(date -u +%FT%TZ) m7-build from $M"
IB1=$(readlink -f "$IB1") IB2=$(readlink -f "$IB2") IB4=$(readlink -f "$IB4")
for pair in "$IB1=$IB1_SHA" "$IB2=$IB2_SHA" "$IB4=$IB4_SHA"; do
  [ "$(sha256sum < "${pair%=*}" | cut -c1-64)" = "${pair#*=}" ] || { echo "${pair%=*} is not ${pair#*=}"; exit 1; }
done
[ -e "$O/ib1s.train.jsonl" ] || (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m6.m6_data drop-families --input "$IB1" \
  --drop sentfin --output "$O/ib1s.train.jsonl") | tee "$O/ib1s.json" || { echo "IB1S FAILED"; exit 1; }
IB1S_SHA=$(sha256sum < "$O/ib1s.train.jsonl" | cut -c1-64)
base="base:full,A6:rho,A7:tokens=20000000:family-equal,IB1S:tokens=1000000000:family-equal"
for k in 1 2; do docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/pipeline:/code -e PYTHONDONTWRITEBYTECODE=1 \
 --mount "type=bind,src=$M,dst=/code,readonly" --mount "type=bind,src=$R/pipeline,dst=/pipeline,readonly" \
 --mount type=bind,src=/data/decision20-20260926/models/Qwen3.8-27B,dst=/source,readonly \
 --mount "type=bind,src=$R,dst=/m2,readonly" --mount "type=bind,src=$D/a7,dst=/a7,readonly" \
 --mount "type=bind,src=$D/hf-ed87a03a,dst=/cal,readonly" \
 --mount "type=bind,src=$O/ib1s.train.jsonl,dst=/ib/ib1s.train.jsonl,readonly" \
 --mount "type=bind,src=$IB2,dst=/ib/ib2.train.jsonl,readonly" --mount "type=bind,src=$IB4,dst=/ib/ib4.train.jsonl,readonly" \
 --mount "type=bind,src=$O,dst=/outroot" \
 -w /code --entrypoint python3 "$IMAGE" \
 -m v2.27b.build_mixtures --source /source --limit 4096 \
 --base A0s-strict=/m2/hf-12912429/m3/pk1/A0s-strict/train.jsonl \
 --arm A6=/m2/hf-5c0255ed/v2/arms/A6g/train.jsonl,/m2/hf-5c0255ed/v2/arms/A6h/train.jsonl \
 --arm A7=/a7/A7g/train.jsonl,/a7/A7o/train.jsonl,/a7/A7p/train.jsonl,/a7/A7i/train.jsonl \
 --arm IB1S=/ib/ib1s.train.jsonl --arm IB2=/ib/ib2.train.jsonl --arm IB4=/ib/ib4.train.jsonl \
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
 --expect "/ib/ib1s.train.jsonl=$IB1S_SHA" --expect "/ib/ib2.train.jsonl=$IB2_SHA" --expect "/ib/ib4.train.jsonl=$IB4_SHA" \
 --expect /m2/mixtures-v1/aho-A6g.jsonl=c6b2a0b4bf55e82a8176cb5871106d383900a76934630b4543b7bc285f7d5304 \
 --expect /m2/mixtures-v1/aho-A6h.jsonl=5b6c3fd090464ff33d5a60dbbbc7f0cbfd1b2cdc9df12089549ac24f4b1c0fb4 \
 --expect /a7/A7g/aho.jsonl=25f31a2edad5a6768104a5de5bab993bc41388cc97dd0a254b750737baba2580 \
 --expect /a7/A7o/aho.jsonl=00986778604daf87906b29eef9d4e48622868fdc9a9fcfad1987eb0f0897a4a0 \
 --expect /a7/A7p/aho.jsonl=19ab727c9ec76c9ed262813c01f0e651a3fa23ea24a0666adb4827e17e6db740 \
 --expect /a7/A7i/aho.jsonl=78f580f62b7784a12fb2841774f85d5e6b4a7bc9aa22847c90fc939cbefea2d6 \
 --expect /m2/hf-5c0255ed/select.jsonl=32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6 \
 --expect /cal/m2/cal/CAL698/cal.jsonl=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f \
 --mixture a20=base:full,A6:rho,A7:tokens=20000000:family-equal \
 --mixture "a20ib14=$base,IB4:tokens=1000000000:family-equal" \
 --mixture "a20ib124=$base,IB2:tokens=1000000000:family-equal,IB4:tokens=1000000000:family-equal" \
 --seed decision2-27b-m3 --output-dir "/outroot/mixtures-m7-$k" || { echo "BUILD $k FAILED"; exit 1; }
 for mix in a20ib14 a20ib124; do
   (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m7.m7_data ml-upsample --base "$O/mixtures-m7-$k/a20.train.jsonl" \
     --input "$O/mixtures-m7-$k/$mix.train.jsonl" --seed decision2-27b-m7 --output "$O/mixtures-m7-$k/${mix}ml.train.jsonl") \
     > "$O/ml-$k-$mix.json" || { echo "ML $mix $k FAILED"; exit 1; }
 done
done
diff -r "$O/mixtures-m7-1" "$O/mixtures-m7-2" || { echo "BUILDS DIFFER"; exit 1; }
echo IDENTICAL
(cd "$O/mixtures-m7-1" && sha256sum -- * > ../mixtures-m7-1.sha256 && cat ../mixtures-m7-1.sha256)
python3 - "$O/mixtures-m7-1" "$A20" > "$O/BUILD.json" <<'EOF' || { echo "CHECKS FAILED"; exit 1; }
import hashlib, json, math, sys
from collections import Counter
from pathlib import Path
root, a20_sha = Path(sys.argv[1]), sys.argv[2]
a20 = (root / "a20.train.jsonl").read_bytes()
assert hashlib.sha256(a20).hexdigest() == a20_sha, "a20 is not M4's file"
a20_lines = set(a20.splitlines(keepends=True))
manifest = json.loads((root / "MIXTURES.json").read_text())
# M6-IB measured 13.5 h per seed on a20ib1 (81,294 rows, 29,792,014 tokens); M6's model gives 12.2 h there.
scale = 13.5 / ((0.1 * 81294 + 0.0012048 * 29792014) / 3600)
out = {"schema": "decision2-27b-m7-build/1", "a20_is_m4": True, "cost_scale": round(scale, 4), "mixtures": {}}
for name, arms in (("a20ib14", ["IB1S", "IB4"]), ("a20ib124", ["IB1S", "IB2", "IB4"])):
    lines = (root / f"{name}.train.jsonl").read_bytes().splitlines(keepends=True)
    assert a20_lines <= set(lines), f"{name} misses a20 rows"
    parts = {arm: manifest["mixtures"][name]["parts"][arm] for arm in arms}
    for arm, part in parts.items():
        assert all(f["exhausted"] for f in part["families"].values()), f"{name}: an {arm} family was not exhausted"
    rows, tokens = manifest["mixtures"][name]["rows"], manifest["mixtures"][name]["tokens"]
    ml = json.loads((root.parent / f"ml-1-{name}.json").read_text())
    added = [json.loads(line) for line in lines if line not in a20_lines]
    a20_mix = manifest["mixtures"]["a20"]
    ml_tokens = tokens + round(ml["added_rows"] * a20_mix["tokens"] / a20_mix["rows"])  # copies cost like a20 rows
    for final, r, t in ((name, rows, tokens), (f"{name}ml", ml["rows_out"], ml_tokens)):
        updates = math.ceil(r / 16)
        projection = scale * (0.1 * r + 0.0012048 * t) / 3600
        out["mixtures"][final] = {
            "rows": r, "tokens": t, "updates": updates, "save_every": math.ceil(updates / 8),
            "projection_gpu_h": round(projection, 2), "cap_gpu_h": 22.0, "cap_ok": projection * 1.15 <= 22.0,
            "sha256": hashlib.sha256((root / f"{final}.train.jsonl").read_bytes()).hexdigest(),
        }
    out["mixtures"][name]["added_tokens"] = {a: p["tokens"] for a, p in parts.items()}
    out["mixtures"][name]["added_by_family"] = dict(sorted(Counter(r["family"] for r in added).items()))
    out["mixtures"][f"{name}ml"]["ml"] = {k: ml[k] for k in ("a20_ml_share", "input_ml_share", "output_ml_share",
                                                              "added_rows", "added_groups", "added_rows_by_language")}
print(json.dumps(out, indent=1, sort_keys=True))
EOF
cat "$O/BUILD.json"
for name in a20ib14ml a20ib124ml; do
  python3 "$M/v2/27b/m4/m4_c1_sources.py" --train "$O/mixtures-m7-1/$name.train.jsonl" \
    --m3a "$O/mixtures-m7-1/a20.train.jsonl" --output "$O/c1-sources-$name.json" || { echo "C1 SOURCES $name FAILED"; exit 1; }
done
echo "=== $(date -u +%FT%TZ) m7-build done"
