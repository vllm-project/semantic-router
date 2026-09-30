#!/usr/bin/env bash
# ~27B M5 mixture build (node B host; CPU only, pinned image, --network none), run from its own mirror.
# Builds a20 (M4's file, reproduction check) and a20h (a20's terms + HS1 + PN1-r2 at dataset revision 27b1d2f1)
# twice into /data/dev2/private/27b/m5-data/mixtures-m5-{1,2}, requires the two builds to be byte-identical, then
# checks:
#   - a20.train.jsonl is M4's frozen file 4aa0dc96 (same builder, inputs and seed);
#   - every a20 row is in a20h (the round-2 block is added on top of a20's rows).
# Usage: m5-build.sh   (from /data/dev2/src/<sha>-src_training_decision2/src/training/decision2/v2/27b/m5)
set -uo pipefail
export TMPDIR=/data/dev2/tmp
M=$(cd "$(dirname "$0")/../../.." && pwd)
R=/data/dev2/private/27b/m2-data D=/data/dev2/private/27b/m3-data O=/data/dev2/private/27b/m5-data
R2=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4
A20=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
[ -f "$M/v2/27b/build_mixtures.py" ] || { echo "not inside a mirror: $M" >&2; exit 2; }
for k in 1 2; do
  [ ! -e "$O/mixtures-m5-$k" ] || { echo "$O/mixtures-m5-$k exists" >&2; exit 66; }
done
mkdir -p "$O"
echo "=== $(date -u +%FT%TZ) m5-build from $M"
for k in 1 2; do docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=/pipeline:/code -e PYTHONDONTWRITEBYTECODE=1 \
 --mount "type=bind,src=$M,dst=/code,readonly" --mount "type=bind,src=$R/pipeline,dst=/pipeline,readonly" \
 --mount type=bind,src=/data/decision20-20260926/models/Qwen3.8-27B,dst=/source,readonly \
 --mount "type=bind,src=$R,dst=/m2,readonly" --mount "type=bind,src=$D/a7,dst=/a7,readonly" \
 --mount "type=bind,src=$D/hf-ed87a03a,dst=/cal,readonly" --mount "type=bind,src=$(readlink -f "$R2/hs1/train.jsonl"),dst=/r2/hs1.train.jsonl,readonly" \
 --mount "type=bind,src=$(readlink -f "$R2/pn1/arms/pn1.train.jsonl"),dst=/r2/pn1.train.jsonl,readonly" \
 --mount "type=bind,src=$O,dst=/outroot" \
 -w /code --entrypoint python3 "$IMAGE" \
 -m v2.27b.build_mixtures --source /source --limit 4096 \
 --base A0s-strict=/m2/hf-12912429/m3/pk1/A0s-strict/train.jsonl \
 --arm A6=/m2/hf-5c0255ed/v2/arms/A6g/train.jsonl,/m2/hf-5c0255ed/v2/arms/A6h/train.jsonl \
 --arm A7=/a7/A7g/train.jsonl,/a7/A7o/train.jsonl,/a7/A7p/train.jsonl,/a7/A7i/train.jsonl \
 --arm HS1=/r2/hs1.train.jsonl --arm PN1=/r2/pn1.train.jsonl \
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
 --expect /r2/hs1.train.jsonl=0dfaa6ebff1b614e5f1c432f2c7432c3fd217f7c344d402c7a7b37fe507de5e7 \
 --expect /r2/pn1.train.jsonl=c1cec06bd5f1caadb7631338796054aae120cbf8b17dcb0bbf4c694470c52cfb \
 --expect /m2/mixtures-v1/aho-A6g.jsonl=c6b2a0b4bf55e82a8176cb5871106d383900a76934630b4543b7bc285f7d5304 \
 --expect /m2/mixtures-v1/aho-A6h.jsonl=5b6c3fd090464ff33d5a60dbbbc7f0cbfd1b2cdc9df12089549ac24f4b1c0fb4 \
 --expect /a7/A7g/aho.jsonl=25f31a2edad5a6768104a5de5bab993bc41388cc97dd0a254b750737baba2580 \
 --expect /a7/A7o/aho.jsonl=00986778604daf87906b29eef9d4e48622868fdc9a9fcfad1987eb0f0897a4a0 \
 --expect /a7/A7p/aho.jsonl=19ab727c9ec76c9ed262813c01f0e651a3fa23ea24a0666adb4827e17e6db740 \
 --expect /a7/A7i/aho.jsonl=78f580f62b7784a12fb2841774f85d5e6b4a7bc9aa22847c90fc939cbefea2d6 \
 --expect /m2/hf-5c0255ed/select.jsonl=32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6 \
 --expect /cal/m2/cal/CAL698/cal.jsonl=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f \
 --mixture a20=base:full,A6:rho,A7:tokens=20000000:family-equal \
 --mixture a20h=base:full,A6:rho,A7:tokens=20000000:family-equal,HS1:tokens=7500000:family-equal,PN1:full \
 --seed decision2-27b-m3 --output-dir "/outroot/mixtures-m5-$k" || { echo "BUILD $k FAILED"; exit 1; }; done
diff -r "$O/mixtures-m5-1" "$O/mixtures-m5-2" || { echo "BUILDS DIFFER"; exit 1; }
echo IDENTICAL
cd "$O/mixtures-m5-1" && sha256sum -- * > ../mixtures-m5-1.sha256 && cat ../mixtures-m5-1.sha256
python3 - "$O/mixtures-m5-1" "$A20" <<'EOF' || { echo "REPRODUCTION CHECK FAILED"; exit 1; }
import hashlib, json, sys
from collections import Counter
from pathlib import Path
root, a20_sha = Path(sys.argv[1]), sys.argv[2]
a20 = (root / "a20.train.jsonl").read_bytes()
assert hashlib.sha256(a20).hexdigest() == a20_sha, "a20 is not M4's file"
a20_lines = a20.splitlines(keepends=True)
a20h_lines = (root / "a20h.train.jsonl").read_bytes().splitlines(keepends=True)
a20h_set = set(a20h_lines)
missing = sum(line not in a20h_set for line in a20_lines)
assert missing == 0, f"{missing} a20 rows are not in a20h"
extra = [json.loads(line) for line in a20h_lines if line not in set(a20_lines)]
print(json.dumps({"a20_is_m4": True, "a20_rows_in_a20h": len(a20_lines), "a20h_rows": len(a20h_lines),
                  "added_rows": len(extra), "added_by_source": dict(Counter(r.get("source") for r in extra)),
                  "added_by_family": dict(Counter(r.get("family") for r in extra))}, sort_keys=True))
EOF
echo "=== $(date -u +%FT%TZ) BUILD-DONE"
