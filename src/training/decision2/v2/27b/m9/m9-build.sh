#!/usr/bin/env bash
# ~27B M9 mixture build (node B host; CPU only), run from its own mirror (prereg records/m9-prereg-2026-10-03.md):
# the M15 ML block (m7_data ml-upsample, seed decision2-27b-m7, as M7) on the released soup's two mixtures, without IB4:
#   a20ib1ml   M6-IB's a20ib1 (a2ccf844...) + one copy of whole non-English a20 groups restoring a20's multilingual share
#   a20ib12ml  M6-IB2's a20ib12 (16cb5bbb...) + the same kind of copies
# Each output starts with its input byte for byte (the copies carry the id suffix ~m2). Built twice into
# /data/dev2/private/27b/m9-data/mixtures-m9-{1,2} (byte-identical), BUILD.json (rows, updates of 16 rows,
# SAVE_EVERY = ceil(updates / 8), SHA-256) and the source-level C1 registry check of the rows new against a20.
# Usage: m9-build.sh   (from /data/dev2/src/<sha>-src_training_decision2/...)
set -euo pipefail
export TMPDIR=/data/dev2/tmp
M=$(cd "$(dirname "$0")/../../.." && pwd)
I=/data/dev2/private/27b/m6-data/mixtures-m6-1 O=/data/dev2/private/27b/m9-data
declare -A WANT=([a20]=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
  [a20ib1]=a2ccf844c612dc61cfc76ab9c53e4f23fddd8311ae451b9c9cdd352dac42491a
  [a20ib12]=16cb5bbbcc40623426c00163e920fa1a1149348e00a724aed4ab9b4a29f7200c)
[ -f "$M/v2/27b/m7/m7_data.py" ] || { echo "not inside a mirror: $M" >&2; exit 2; }
[ ! -e "$O/mixtures-m9-1" ] || { echo "$O/mixtures-m9-1 exists" >&2; exit 66; }
for f in a20 a20ib1 a20ib12; do
  [ "$(sha256sum < "$I/$f.train.jsonl" | cut -c1-64)" = "${WANT[$f]}" ] || { echo "$I/$f.train.jsonl is not ${WANT[$f]}" >&2; exit 1; }
done
mkdir -p "$O" && chmod 700 "$O"
echo "=== $(date -u +%FT%TZ) m9-build from $M"
for k in 1 2; do
  mkdir -p "$O/mixtures-m9-$k"
  for mix in a20ib1 a20ib12; do
    (cd "$M" && PYTHONPATH=$M python3 -m v2.27b.m7.m7_data ml-upsample --base "$I/a20.train.jsonl" \
      --input "$I/$mix.train.jsonl" --seed decision2-27b-m7 --output "$O/mixtures-m9-$k/${mix}ml.train.jsonl") \
      > "$O/ml-$k-$mix.json"
  done
done
diff -r "$O/mixtures-m9-1" "$O/mixtures-m9-2" && cmp "$O/ml-1-a20ib1.json" "$O/ml-2-a20ib1.json" && echo IDENTICAL
(cd "$O/mixtures-m9-1" && sha256sum -- * > ../mixtures-m9-1.sha256 && cat ../mixtures-m9-1.sha256)
python3 - "$O" "$I" > "$O/BUILD.json" <<'EOF'
import hashlib, json, math, sys
from pathlib import Path
out, inputs = Path(sys.argv[1]), Path(sys.argv[2])
build = {"schema": "decision2-27b-m9-build/1", "inputs": str(inputs), "mixtures": {}}
for mix in ("a20ib1", "a20ib12"):
    name = f"{mix}ml"
    data = (out / "mixtures-m9-1" / f"{name}.train.jsonl").read_bytes()
    base = (inputs / f"{mix}.train.jsonl").read_bytes()
    assert data.startswith(base), f"{name} does not start with {mix} byte for byte"
    rows = data.count(b"\n")
    updates = math.ceil(rows / 16)
    ml = json.loads((out / f"ml-1-{mix}.json").read_text())
    build["mixtures"][name] = {
        "rows": rows, "updates": updates, "save_every": math.ceil(updates / 8),
        "sha256": hashlib.sha256(data).hexdigest(), "input": mix, "input_sha256": hashlib.sha256(base).hexdigest(),
        "ml": {k: ml.get(k) for k in ("a20_ml_share", "input_ml_share", "output_ml_share", "added_rows", "added_groups")},
    }
print(json.dumps(build, indent=1, sort_keys=True))
EOF
cat "$O/BUILD.json"
for name in a20ib1ml a20ib12ml; do
  python3 "$M/v2/27b/m4/m4_c1_sources.py" --train "$O/mixtures-m9-1/$name.train.jsonl" \
    --m3a "$I/a20.train.jsonl" --output "$O/c1-sources-$name.json" || { echo "C1 SOURCES $name FAILED"; exit 1; }
done
echo "=== $(date -u +%FT%TZ) m9-build done"
