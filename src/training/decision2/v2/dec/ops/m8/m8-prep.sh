#!/usr/bin/env bash
# Decoder M8 data build on node B, CPU only (prereg dec-m8-prereg-2026-09-30.md, "Data"):
#   1. m8_data.py: the top-up slice of N4XF's mixture, the human-rated split, the C teacher (N4XF's own-Lux records),
#      the gold-free label prompts and their node-A shards -> m8/data/build;
#   2. the exposure receipt of the slice TRAIN against the r2 payload (2194716a) -> m8/exposure/exposure-slice.json.
# READY files are written by hand only after the data-lock record is committed. A finished step is skipped on a
# rerun; a failed step is not rerun. Each job leaves its launch receipt.
# usage: m8-prep.sh <mirror-dir>
set -u
SRC=$1
M=/data/dev2/runs/dec/m8
CODE=/data/dev2/src/$SRC/src/training/decision2
L=$CODE/v2/dec/launch.sh
H=/hf/datasets--llm-semantic-router--decision-2.0-training-data/snapshots
TOK=/hf/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/363c4a5e56afc115b1c78c837633956d0bbb63ab
B4=/runs/m4/data/m4-xl-full-29m/train.jsonl B4_SHA=c7d51219e9f0fdd107b914df5c7e9f81f2b53496bb43a94796857b8ea3fcfa60
N4XF_T=/runs/m4/teacher/m4-xl-full-29m/lux-teacher.jsonl N4XF_T_SHA=e2ff27ce2fc397c8c203e37133dcfac52ac3d1f172c6f3e9412ec28c2ffe296c
SPEC=/code/v2/dec/specs/m4-xl-full-29m.json SPEC_SHA=cf31117830d886af417da0b77be19a83ab0063e748da8e242818ab9ef41a06ae
PAYLOAD_DIR=/data/dev2/runs/eval/m5/overlap-effects/final
PAYLOAD_SHA=2194716a179b2e6c3ba529dee3952c5dd62c0f3281638b7236029d5d542f4914
mkdir -p "$M/logs"
export DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
cpu() {  # <job> <out dir under m8> <python args...>
  local job=$1 out=$M/$2
  shift 2
  if [ -f "$out.launch.json" ]; then
    grep -q '"exit_status": 0' "$out.launch.json" && return 0
    log "$job failed earlier (receipt $out.launch.json); not rerun"
    return 1
  fi
  if bash "$L" "m8-$job" "$SRC" "$out" --cpu -- "$@"; then
    log "$job done: $(tail -c 600 "$out.stdout.log" | tr '\n' ' ')"
  else
    log "$job FAILED: $(tail -c 800 "$out.stderr.log" | tr '\n' ' ')"
    return 1
  fi
}
for f in dev css-pilot hs1-dev ht-dev2 score5t-dev; do
  [ -f "/data/dev2/runs/dec/panels/$f.prompts.jsonl" ] || { log "missing panel prompts $f (m8-relay.sh panels)"; exit 1; }
done
[ "$(sha256sum "$PAYLOAD_DIR/excluded-groups.json" | cut -d' ' -f1)" = "$PAYLOAD_SHA" ] || { log "payload hash MISMATCH"; exit 1; }
DEC_DATA=/data/dev2/runs/dec/m3/data-sel700-cal698 cpu data data v2/dec/ops/m8/m8_data.py \
  --base "$B4" --base-sha "$B4_SHA" --spec "$SPEC" --spec-sha "$SPEC_SHA" \
  --root "mx=$H/5c602c5ae76c3930d5ed191facef4ac710863b6d" --root "xl=$H/100536133e192c54ec57c2599a5e4706f6d334ff" \
  --quarantine /code/v2/dec/ops/m7/specs/m7-quarantine-groups.json --lux-teacher "$N4XF_T" --lux-teacher-sha "$N4XF_T_SHA" \
  --rule /code/v2/dec/ops/m8/specs/m8-human-rated-rule.json --tokenizer "$TOK" --budget 6200000 --seed dec-m8-topup-v1 \
  --isolation select=/data/select.jsonl --isolation cal=/data/cal.jsonl \
  --panel-prompts dev=/panels/dev.prompts.jsonl --panel-prompts css-pilot=/panels/css-pilot.prompts.jsonl \
  --panel-prompts hs1-dev=/panels/hs1-dev.prompts.jsonl --panel-prompts ht-dev2=/panels/ht-dev2.prompts.jsonl \
  --panel-prompts score5t-dev=/panels/score5t-dev.prompts.jsonl --shards 3 --workers 40 --output /out/build || exit 1
sha=$(sha256sum "$M/data/build/slice/train.jsonl" | cut -d' ' -f1)
DEC_DATA=$PAYLOAD_DIR cpu exposure exposure -m v2.eval.overlap_effects exposure --groups /data/excluded-groups.json \
  --train /runs/m8/data/build/slice/train.jsonl --expect-sha256 "$sha" --label "decoder M8 top-up slice" \
  --output /out/exposure-slice.json || exit 1
log "prep finished: slice $sha"
