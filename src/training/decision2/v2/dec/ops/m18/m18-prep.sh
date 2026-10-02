#!/usr/bin/env bash
# Decoder M18 Part B data on node F (prereg dec-m18-prereg-2026-10-02.md, "Part B"), CPU only, in the pinned decoder
# image (the script re-runs itself in a container with /data/dev2/runs/dec mounted read-write). Idempotent; a failed
# build is not rerun.
#   1. inputs, each against its hash: M12 2b-RA TRAIN / ids, M13's 2B SD targets, the own-Sol teacher (M12's), the
#      IB3-r2 TRAIN (HF dataset revision 1c8452da, m6/ib3/ib3.train.jsonl, downloaded to m18/inputs/ib3r2);
#      Sol's released TRAIN = the first 56,141 lines of 2b-RA (M12's builder wrote it first, byte for byte);
#   2. 2b-RS17UP: m18_swap.py (M17's builder: sentfin dropped, share 0.17, seed 20261002); RAUP weights over its
#      blocks (m14_weights.py; the kept released rows are its released block); teacher = the swap's SD targets;
#   3. 2b-RAUPM: 2b-RA TRAIN followed by the IB3-r2 TRAIN, ids block `ib3`; RAUP weights with IB (IB3 included) 1.0;
#      teacher = the own-Sol teacher (released rows only, --teacher-partial);
#   4. the Triton cache 2b-train (cp -a of node F's M13 2B train cache);
#   5. data/LOCK-candidate.json: train / weights / teacher SHA-256 per arm (the data lock record commits these and
#      READY-m18.json is written from it).
#
# usage: m18-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
R=/data/dev2/runs/dec
M=$R/m18
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
if [ -z "${M18_IN_IMAGE:-}" ]; then
  [ -d "$CODE" ] || { echo "missing exact mirror /data/dev2/src/$SRC" >&2; exit 2; }
  mkdir -p "$M/data" "$M/logs"
  exec docker run --name m18-prep --rm --network none -e M18_IN_IMAGE=1 \
    --mount "type=bind,src=/data/dev2/src/$SRC,dst=/data/dev2/src/$SRC,readonly" \
    --mount "type=bind,src=$R,dst=$R" -w / "$IMAGE" bash "$OPS/m18/m18-prep.sh" "$SRC"
fi
log() { echo "$(date -u +%FT%TZ) prep-f $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
RA=$R/m12/data/2b/2b-RA RA_SHA=08140409a6afc439e35bc4e0aefc9b5812162c4a2972e5e1da5b137a99d0e592
RA_IDS_SHA=22854cfca61cb61f48619d33a4d9e968b59b4b5d8bfea5c5ebceff882d2826f7
REL_ROWS=56141 REL_SHA=1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
SD=$R/m13/data/2b/teacher-sd.jsonl SD_SHA=2b9858d89fb650db9bc3e7fa608755135b81d7b20ff7b1b6fa186c86bf2e49f2
OWN=$R/m11/data/2b/teacher.jsonl OWN_SHA=3f92e8c108c5a9949fda4b08638907fcc80175f68efec18bece912d0edc8eea1
IB3=$M/inputs/ib3r2/m6/ib3/ib3.train.jsonl IB3_SHA=9d92d92a207109a585ceead7fa5e1bdb0b2c61859de8227a2a6058428f852dea
UP=choice=1.5,noul=1.5,score=2.0
SEED=20261002

check "$RA/train.jsonl" $RA_SHA
check "$RA/train.ids.jsonl" $RA_IDS_SHA
check "$SD" $SD_SHA
check "$OWN" $OWN_SHA
check "$IB3" $IB3_SHA
REL=$M/inputs/2b-released/train.jsonl
if [ ! -f "$REL" ]; then
  mkdir -p "$(dirname "$REL")"
  head -n $REL_ROWS "$RA/train.jsonl" > "$REL.tmp" && mv "$REL.tmp" "$REL"
fi
check "$REL" $REL_SHA
log "M18 inputs verified"

D=$M/data/2b
mkdir -p "$D"
if [ -e "$M/data/2b.FAILED" ]; then
  log "build failed earlier; not rerun"
  exit 1
fi
fail() { touch "$M/data/2b.FAILED"; log "FAILED: $*"; exit 1; }

S=$M/data/swap
if [ ! -f "$S/report.json" ]; then
  python3 -B "$OPS/m18/m18_swap.py" --base "$REL" --base-sha $REL_SHA --base-train "$RA/train.jsonl" \
    --base-train-sha $RA_SHA --base-ids "$RA/train.ids.jsonl" --pool-train "$RA/train.jsonl" --pool-sha $RA_SHA \
    --pool-ids "$RA/train.ids.jsonl" --teacher "$SD" --teacher-sha $SD_SHA --drop sentfin \
    --arm 2b-RS17UP=0.17 --seed $SEED --output "$S" > "$M/data/swap.log" 2>&1 || fail "swap build (see swap.log)"
  log "swap built: $(tail -1 "$M/data/swap.log" | cut -c1-900)"
fi
if [ ! -f "$D/2b-RS17UP/teacher.jsonl" ]; then
  a=$S/2b-RS17UP
  n=$(python3 -c 'import json,sys; print(sum(json.loads(l)["block"]=="base" for l in open(sys.argv[1])))' \
    "$a/train.ids.jsonl")
  python3 -B "$OPS/m14/m14_weights.py" --train "$a/train.jsonl" --train-sha "$(sha "$a/train.jsonl")" \
    --ids "$a/train.ids.jsonl" --ids-sha "$(sha "$a/train.ids.jsonl")" --released-rows "$n" \
    --released-sha "$(head -n "$n" "$a/train.jsonl" | sha256sum | cut -d' ' -f1)" --weight $UP --ib-weight 1.0 \
    --name 2b-RS17UP --output "$D" > "$M/data/weights-2b-RS17UP.log" 2>&1 || fail "2b-RS17UP weights"
  ln "$a/train.jsonl" "$D/2b-RS17UP/train.jsonl"
  ln "$a/train.ids.jsonl" "$D/2b-RS17UP/train.ids.jsonl"
  ln "$a/teacher-s.jsonl" "$D/2b-RS17UP/teacher.jsonl"
  log "2b-RS17UP: $(tail -1 "$M/data/weights-2b-RS17UP.log" | cut -c1-600)"
fi

B=$M/data/raupm
if [ ! -f "$D/2b-RAUPM/teacher.jsonl" ]; then
  rm -rf "$B" && mkdir -p "$B"
  cat "$RA/train.jsonl" "$IB3" > "$B/train.jsonl"
  python3 - "$RA/train.ids.jsonl" "$IB3" "$B/train.ids.jsonl" << 'EOF' || fail "2b-RAUPM ids"
import json, sys
ra_ids, ib3, out = sys.argv[1:]
seen = {json.loads(l)["id"] for l in open(ra_ids)}
with open(out, "x") as sink:
    for line in open(ra_ids):
        sink.write(line)
    for line in open(ib3):
        row = json.loads(line)
        if row["id"] in seen or row["task_type"] != "noul" or row["split"] != "train":
            raise SystemExit(f"bad IB3 row {row['id']}")
        seen.add(row["id"])
        sink.write(json.dumps({"id": row["id"], "block": "ib3", "tokens": None}) + "\n")
EOF
  python3 -B "$OPS/m14/m14_weights.py" --train "$B/train.jsonl" --train-sha "$(sha "$B/train.jsonl")" \
    --ids "$B/train.ids.jsonl" --ids-sha "$(sha "$B/train.ids.jsonl")" --released-rows $REL_ROWS \
    --released-sha $REL_SHA --weight $UP --ib-weight 1.0 --name 2b-RAUPM --output "$D" \
    > "$M/data/weights-2b-RAUPM.log" 2>&1 || fail "2b-RAUPM weights"
  ln "$B/train.jsonl" "$D/2b-RAUPM/train.jsonl"
  ln "$B/train.ids.jsonl" "$D/2b-RAUPM/train.ids.jsonl"
  cp "$OWN" "$D/2b-RAUPM/teacher.jsonl"
  log "2b-RAUPM: $(tail -1 "$M/data/weights-2b-RAUPM.log" | cut -c1-600)"
fi

mkdir -p "$M/triton-cache"
if [ ! -d "$M/triton-cache/2b-train" ]; then
  cp -a "$R/m13/triton-cache/2b-train" "$M/triton-cache/2b-train.tmp"
  mv "$M/triton-cache/2b-train.tmp" "$M/triton-cache/2b-train"
  log "Triton cache 2b-train copied from m13 ($(find "$M/triton-cache/2b-train" -type f | wc -l) files)"
fi

python3 - "$D" "$M/data/LOCK-candidate.json" << 'EOF'
import hashlib, json, sys
d, out = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
arms = {}
for arm in ("2b-RS17UP", "2b-RAUPM"):
    rep = json.load(open(f"{d}/{arm}/report.json"))
    arms[arm] = {"train": sha(f"{d}/{arm}/train.jsonl"), "weights": sha(f"{d}/{arm}/weights.jsonl"),
                 "teacher": sha(f"{d}/{arm}/teacher.jsonl"), "rows": rep["rows"],
                 "released_rows": rep["released_rows"], "ib_rows": rep["ib_rows"],
                 "released_weight_share": round(rep["released_weight_share"], 4)}
json.dump({"arms": arms}, open(out, "w"), indent=1)
print(json.dumps(arms))
EOF
log "prep finished"
