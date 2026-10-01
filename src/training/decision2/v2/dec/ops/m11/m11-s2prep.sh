#!/usr/bin/env bash
# Decoder M11 stage-2 data preparation on node E / F (prereg dec-m11-stage2-prereg-2026-10-01.md), CPU only, from an
# exact mirror, after IB1-r3 / IB2 TRAIN were relayed from node A (m11/inputs/ib1/, m11/inputs/ib2/). Idempotent.
#   1. input hashes: IB1-r3 / IB2 TRAIN, M10's 4B TRAIN and N4XF's own-Lux teacher;
#   2. the stage-2 Triton caches 4b-train / 4b-read, each a cp -a copy of this node's M10 decoder cache;
#   3. the stage-2 TRAIN files (m11_s2data.py in the decoder image) -> m11/data/s2/{4b-LHB,4b-LHBx}/train.jsonl and
#      report.json (both nodes build; the data lock compares them);
#   4. (node E) the retention probes' TRAIN overlap against both files (m10_probes.py overlap --kind train) ->
#      m11/probes/hits-4b.json (the union of the panel ids hit).
#
# usage: M11_NODE=e|f m11-s2prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M11_NODE:?set M11_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m11
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m11
mkdir -p "$M/data" "$M/triton-cache" "$M/probes"
log() { echo "$(date -u +%FT%TZ) s2prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
IB1_SHA=1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
IB2_SHA=ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
BASE_SHA=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
TEACHER_SHA=e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c

check "$M/inputs/ib1/ib1.train.jsonl" $IB1_SHA
check "$M/inputs/ib2/ib2.train.jsonl" $IB2_SHA
check "$R/m10/data/m10-4b-base/train.jsonl" $BASE_SHA
check "$R/m10/inputs/n4xf-teacher/m4-xl-full-29m/lux-teacher.jsonl" $TEACHER_SHA
log "stage-2 inputs verified"

for c in 4b-train 4b-read; do
  [ -d "$M/triton-cache/$c" ] && continue
  cp -a "$R/m10/triton-cache/dbe5f32b2263" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
  log "Triton cache $c copied from m10/triton-cache/dbe5f32b2263 ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
done

if [ ! -f "$M/data/s2/report.json" ]; then
  [ ! -e "$M/data/s2-build.launch.json" ] || { log "stage-2 build failed earlier; not rerun"; exit 1; }
  M11_NODE=$NODE bash "$OPS/m11-launch.sh" "s2data" "$SRC" "$M/data/s2-build" --cpu -- v2/dec/ops/m11/m11_s2data.py \
    --base /runs/m10/data/m10-4b-base/train.jsonl --base-sha $BASE_SHA \
    --ib1 /runs/m11/inputs/ib1/ib1.train.jsonl --ib1-sha $IB1_SHA \
    --ib2 /runs/m11/inputs/ib2/ib2.train.jsonl --ib2-sha $IB2_SHA \
    --tokenizer /models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b \
    --indist w2c,isarc,hover,gsm2 --output /out/s2 --workers 32
  mv "$M/data/s2-build/s2" "$M/data/s2"
fi
log "stage-2 TRAIN: $(tail -1 "$M/data/s2-build.stdout.log" | cut -c1-400)"

[ "$NODE" = e ] || { log "stage-2 prep finished (node F: no overlap step)"; exit 0; }
if [ ! -f "$M/probes/hits-4b.json" ]; then
  for arm in 4b-LHB 4b-LHBx; do
    python3 "$CODE/v2/dec/ops/m10/m10_probes.py" overlap --candidates "$R/m10/probes/candidates/candidates.jsonl" \
      --corpus "$M/data/s2/$arm/train.jsonl" --kind train --output "$M/probes/overlap-train-$arm.json" \
      > "$M/probes/overlap-train-$arm.log" 2>&1
  done
  python3 - "$M/probes/overlap-train-4b-LHB.json" "$M/probes/overlap-train-4b-LHBx.json" \
    "$R/panels/m10-probes.prompts.jsonl" "$M/probes/hits-4b.json" << 'EOF'
import hashlib, json, sys
a, b, panel_path, out = sys.argv[1:]
panel = [json.loads(line)["id"] for line in open(panel_path)]
hits = set()
files = {}
for path in (a, b):
    d = json.load(open(path))
    hits |= set(d["hit_ids"]) & set(panel)
    files[path] = {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "candidates_hit": d["candidates_hit"]}
json.dump({"overlaps": files, "panel_items": len(panel), "panel_hits": sorted(hits)}, open(out, "x"), indent=1)
EOF
fi
log "probe overlap 4b stage 2: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["panel_hits"]), "of", d["panel_items"], "panel items hit")' "$M/probes/hits-4b.json")"
log "stage-2 prep finished"
