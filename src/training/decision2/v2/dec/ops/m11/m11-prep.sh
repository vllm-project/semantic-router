#!/usr/bin/env bash
# Decoder M11 data preparation on node E / F (prereg "Data"), CPU only, from an exact mirror, after the inputs were
# relayed from node B (m11/inputs/, /data/dev2/models/). Idempotent; a step already done is skipped.
#   1. input hashes (TRAIN files, S2T's own-Sol targets) against the prereg's pins, and a per-file SHA-256 manifest of
#      every model directory (bases, Sol / Eos 1.0, the DEV2.0 starts) -> m11/inputs/MODELS.sha256;
#   2. the M11 Triton caches: <tier>-{train,read}, each a cp -a copy of this node's M10 decoder cache;
#   3. the 2B teacher: v2.dec.compose_teacher (S2T's targets 947bc65b onto the 2B TRAIN; a missing row is an error)
#      -> m11/data/2b/teacher.jsonl;
#   4. (node E) label-prompt lengths per tier (m11_lengths.py, the 1.0 tokenizer) -> m11/data/lengths-<tier>.json;
#   5. (node E) the retention probes' TRAIN overlap per tier (M10's m10_probes.py overlap --kind train) ->
#      m11/probes/overlap-train-<tier>.json and the probe-panel hit ids -> m11/probes/hits-<tier>.json.
#
# usage: M11_NODE=e|f m11-prep.sh <mirror-dir>
set -euo pipefail
SRC=$1
NODE=${M11_NODE:?set M11_NODE=e or f}
R=/data/dev2/runs/dec
M=$R/m11
CODE=/data/dev2/src/$SRC/src/training/decision2
OPS=$CODE/v2/dec/ops/m11
H=/data/dev2/models
mkdir -p "$M/data/2b" "$M/triton-cache" "$M/probes" "$M/logs"
log() { echo "$(date -u +%FT%TZ) prep-$NODE $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }

check "$M/inputs/m4-v2m-ret-r2/train.jsonl" 1527b38b1ba888695fe48dd43e92827d1719d57674009cfc29d5ab759b08dd2c
check "$M/inputs/m6-e8f-r2clean/train.jsonl" f9f3c0229551357ad1cb8e6923c64c636ee697e71d162f059548ae223e78d8ae
check "$M/inputs/sol-teacher/sol-teacher.jsonl" 947bc65b15f881aafaae4e3b0fda0885a78f752451dc1187663854d03d0ae4d4
if [ ! -f "$M/inputs/MODELS.sha256" ]; then
  (cd "$H" && find Qwen--Qwen3.5-2B-Base Qwen--Qwen3.5-0.8B-Base Decision-1.0-Sol-2B Decision-1.0-Eos-0.8B DEV2.0-2B \
    DEV2.0-0.8B -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$M/inputs/MODELS.sha256.tmp"
  mv "$M/inputs/MODELS.sha256.tmp" "$M/inputs/MODELS.sha256"
fi
log "inputs verified; model manifest $(sha "$M/inputs/MODELS.sha256") ($(wc -l < "$M/inputs/MODELS.sha256") files)"

for c in 2b-train 2b-read 08b-train 08b-read; do
  [ -d "$M/triton-cache/$c" ] && continue
  cp -a "$R/m10/triton-cache/dbe5f32b2263" "$M/triton-cache/$c.tmp" && mv "$M/triton-cache/$c.tmp" "$M/triton-cache/$c"
  log "Triton cache $c copied from m10/triton-cache/dbe5f32b2263 ($(find "$M/triton-cache/$c" -type f | wc -l) files)"
done

if [ ! -f "$M/data/2b/teacher.jsonl" ]; then
  M11_NODE=$NODE bash "$OPS/m11-launch.sh" "teacher-2b" "$SRC" "$M/data/2b/compose" --cpu -- -m v2.dec.compose_teacher \
    compose --train /runs/m11/inputs/m4-v2m-ret-r2/train.jsonl --source /runs/m11/inputs/sol-teacher/sol-teacher.jsonl \
    --output /out/teacher.jsonl
  mv "$M/data/2b/compose/teacher.jsonl" "$M/data/2b/teacher.jsonl"
fi
log "2B teacher $(sha "$M/data/2b/teacher.jsonl") ($(wc -l < "$M/data/2b/teacher.jsonl") rows)"

[ "$NODE" = e ] || { log "prep finished (node F: no length / overlap steps)"; exit 0; }
declare -A TOK=([2b]=/models/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
  [08b]=/models/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab)
declare -A TR=([2b]=m4-v2m-ret-r2 [08b]=m6-e8f-r2clean)
for t in 2b 08b; do
  if [ ! -f "$M/data/lengths-$t.json" ]; then
    M11_NODE=$NODE bash "$OPS/m11-launch.sh" "lengths-$t" "$SRC" "$M/data/lengths-$t" --cpu -- \
      v2/dec/ops/m11/m11_lengths.py --tokenizer "${TOK[$t]}" --workers 32 \
      --file "train=/runs/m11/inputs/${TR[$t]}/train.jsonl:train" --file "select=/data/select.jsonl:select" \
      --file "cal=/data/cal.jsonl:cal" --output /out/lengths.json
    mv "$M/data/lengths-$t/lengths.json" "$M/data/lengths-$t.json"
  fi
  log "label-prompt lengths $t: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({"longest": d["longest"], "nt_max_length": d["nt_max_length"], "over_8192": {k: v["over_8192"] for k, v in d["files"].items()}})' "$M/data/lengths-$t.json")"
  if [ ! -f "$M/probes/hits-$t.json" ]; then
    python3 "$CODE/v2/dec/ops/m10/m10_probes.py" overlap --candidates "$R/m10/probes/candidates/candidates.jsonl" \
      --corpus "$M/inputs/${TR[$t]}/train.jsonl" --kind train --output "$M/probes/overlap-train-$t.json" \
      > "$M/probes/overlap-train-$t.log" 2>&1
    python3 - "$M/probes/overlap-train-$t.json" "$R/panels/m10-probes.prompts.jsonl" "$M/probes/hits-$t.json" << 'EOF'
import hashlib, json, sys
overlap = json.load(open(sys.argv[1]))
panel = [json.loads(line)["id"] for line in open(sys.argv[2])]
hit = set(overlap["hit_ids"]) & set(panel)
json.dump({"tier_overlap": sys.argv[1], "overlap_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
           "panel_items": len(panel), "candidates_hit": overlap["candidates_hit"], "panel_hits": sorted(hit)},
          open(sys.argv[3], "x"), indent=1)
EOF
  fi
  log "probe overlap $t: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["candidates_hit"], "candidates hit,", len(d["panel_hits"]), "of", d["panel_items"], "panel items")' "$M/probes/hits-$t.json")"
done
log "prep finished"
