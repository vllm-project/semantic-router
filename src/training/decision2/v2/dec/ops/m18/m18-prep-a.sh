#!/usr/bin/env bash
# Decoder M18 amendment-1 data on node A (prereg dec-m18-prereg-2026-10-02.md, "Amendment 1"), CPU only, in the
# pinned decoder image (re-runs itself in a container with /data/dev2/runs/dec mounted read-write). Idempotent; a
# failed build is not rerun.
#   1. inputs, each against its hash: M12 2b-RA / 08b-RA TRAIN and ids (M14's staged copies on node A), the IB3-r2
#      TRAIN and the own-Sol teacher (copied node F -> node A into m18/inputs/{ib3r2,own-sol});
#   2. 2b-RAM: 2b-RA TRAIN then the IB3-r2 TRAIN (byte for byte; the same file as 2b-RAUPM's TRAIN), teacher = the
#      own-Sol teacher, no weights;
#   3. 08b-RAM: 08b-RA TRAIN then the IB3-r2 TRAIN; weights 1.0 per row, 3.0 per IB3-r2 row (08b-RA holds its IB pool
#      three times), no teacher;
#   4. Triton caches: 08b-train = cp -a of node A's M14 08b-train cache; 2b-train must already be node F's M18 copy;
#   5. data/LOCK-candidate-a.json.
#
# usage: m18-prep-a.sh <mirror-dir>
set -euo pipefail
SRC=$1
R=/data/dev2/runs/dec
M=$R/m18
CODE=/data/dev2/src/$SRC/src/training/decision2
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
if [ -z "${M18_IN_IMAGE:-}" ]; then
  [ -d "$CODE" ] || { echo "missing exact mirror /data/dev2/src/$SRC" >&2; exit 2; }
  mkdir -p "$M/data" "$M/logs"
  exec docker run --name m18-prep-a --rm --network none -e M18_IN_IMAGE=1 \
    --mount "type=bind,src=/data/dev2/src/$SRC,dst=/data/dev2/src/$SRC,readonly" \
    --mount "type=bind,src=$R,dst=$R" -w / "$IMAGE" bash "$CODE/v2/dec/ops/m18/m18-prep-a.sh" "$SRC"
fi
log() { echo "$(date -u +%FT%TZ) prep-a $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
I=$R/m14/inputs/m12
RA2=$I/2b/2b-RA RA2_SHA=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592
RA2_IDS_SHA=22854cfca61cb61f48619d33a4d9e968b59b4b5d8bfea5c5ebceff882d2826f7
RA08=$I/08b/08b-RA RA08_SHA=12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3
OWN=$M/inputs/own-sol/teacher.jsonl OWN_SHA=3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1
IB3=$M/inputs/ib3r2/m6/ib3/ib3.train.jsonl IB3_SHA=9d92d92a207109a585ceead7fa5e1bdb0b2c61859de8227a2a6058428f852dea
check "$RA2/train.jsonl" $RA2_SHA
check "$RA2/train.ids.jsonl" $RA2_IDS_SHA
check "$RA08/train.jsonl" $RA08_SHA
check "$OWN" $OWN_SHA
check "$IB3" $IB3_SHA
[ -d "$M/triton-cache/2b-train" ] || { log "FAILED: no 2b-train cache (copy node F's m18/triton-cache/2b-train first)"; exit 1; }
log "M18 amendment-1 inputs verified"
[ -e "$M/data/a.FAILED" ] && { log "build failed earlier; not rerun"; exit 1; }
fail() { touch "$M/data/a.FAILED"; log "FAILED: $*"; exit 1; }

D=$M/data/2b/2b-RAM
if [ ! -f "$D/teacher.jsonl" ]; then
  mkdir -p "$D"
  cat "$RA2/train.jsonl" "$IB3" > "$D/train.jsonl"
  cp "$OWN" "$D/teacher.jsonl"
  log "2b-RAM: $(wc -l < "$D/train.jsonl") rows, TRAIN $(sha "$D/train.jsonl" | cut -c1-16)"
fi
D=$M/data/08b/08b-RAM
if [ ! -f "$D/weights.jsonl" ]; then
  mkdir -p "$D"
  cat "$RA08/train.jsonl" "$IB3" > "$D/train.jsonl.tmp"
  python3 - "$RA08/train.jsonl" "$IB3" "$D/weights.jsonl.tmp" << 'EOF' || fail "08b-RAM weights"
import json, sys
ra, ib3, out = sys.argv[1:]
seen = set()
with open(out, "x") as sink:
    for path, w in ((ra, 1.0), (ib3, 3.0)):
        for line in open(path, "rb"):
            rid = json.loads(line)["id"]
            if rid in seen:
                raise SystemExit(f"repeated id {rid}")
            seen.add(rid)
            sink.write(json.dumps({"id": rid, "weight": w}) + "\n")
EOF
  mv "$D/train.jsonl.tmp" "$D/train.jsonl" && mv "$D/weights.jsonl.tmp" "$D/weights.jsonl"
  log "08b-RAM: $(wc -l < "$D/train.jsonl") rows, TRAIN $(sha "$D/train.jsonl" | cut -c1-16)"
fi

if [ ! -d "$M/triton-cache/08b-train" ]; then
  cp -a "$R/m14/triton-cache/08b-train" "$M/triton-cache/08b-train.tmp"
  mv "$M/triton-cache/08b-train.tmp" "$M/triton-cache/08b-train"
  log "Triton cache 08b-train copied from m14 ($(find "$M/triton-cache/08b-train" -type f | wc -l) files)"
fi

python3 - "$M/data" "$M/data/LOCK-candidate-a.json" << 'EOF'
import hashlib, json, os, sys
d, out = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
arms = {}
for tier, arm in (("2b", "2b-RAM"), ("08b", "08b-RAM")):
    rec = {}
    for k in ("train", "weights", "teacher"):
        p = f"{d}/{tier}/{arm}/{k}.jsonl"
        if os.path.isfile(p):
            rec[k] = sha(p)
    rec["rows"] = sum(1 for _ in open(f"{d}/{tier}/{arm}/train.jsonl", "rb"))
    arms[arm] = rec
json.dump({"arms": arms}, open(out, "w"), indent=1)
print(json.dumps(arms))
EOF
log "prep finished"
